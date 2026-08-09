#!/usr/bin/env python3
"""
m4_study.py — near-memory input-side gating, measured on a REAL transformer.

Runs on an Apple-silicon Mac (M-series GPU via PyTorch MPS). Retires the
tiny-proxy caveat and produces reviewer-ready quality-vs-energy curves for:

  1. FFN read-mask  — gate the SwiGLU/GeLU hidden, reuse the survivor set as a
     down-projection weight READ-MASK. Skips weight reads (1st-order), not
     result writes. Turnkey.
  2. KV read-skip   — StreamingLLM-style sink+window restriction (content-
     independent, robust): each query reads only the first S "sink" tokens and
     the last W of context. Attacks the largest energy term (KV). Best-effort;
     content-aware policies (H2O / Deja-Vu) skip more for the same quality but
     need a scoring pass.

Everything is measured (WikiText-2 perplexity) and joined to a first-order
decode-energy budget derived from the model's own config (GQA-aware). This is
the hardware-side counterpart of the contextual-sparsity literature
(Deja Vu 2023; the lazy-neuron phenomenon 2022; StreamingLLM; H2O).

USAGE
  pip install -r requirements.txt
  python3 m4_study.py --mode all                     # Qwen2.5-1.5B, ~tens of min
  python3 m4_study.py --model meta-llama/Llama-3.2-3B --mode all   # needs HF login
  python3 m4_study.py --mode ffn                     # just the turnkey FFN result
Notes: Qwen2.5 is open (no login). Llama is gated (huggingface-cli login first).
"""
import argparse, json, math, os, time
import torch
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

# ---------------- energy model (first-order; mirrors inference_energy_sim) ----
DRAM_ARRAY, BASE_HOP, HBM_IO = 6.0, 3.0, 26.0     # pJ/byte
WBYTES = 0.5                                        # deployment weight precision (int4)

def energy_budget(cfg, n_params, seq_len, kv_bits=16):
    """Per-token near-memory decode energy, split into weights / KV / (small) act."""
    d   = cfg.hidden_size
    L   = cfg.num_hidden_layers
    n_attn = cfg.num_attention_heads
    n_kv   = getattr(cfg, "num_key_value_heads", n_attn)   # GQA-aware
    head_d = d // n_attn
    per_byte = DRAM_ARRAY + BASE_HOP                        # near-memory read (array+hop)
    w  = n_params * WBYTES * per_byte
    kv = 2 * L * n_kv * head_d * seq_len * (kv_bits/8) * per_byte
    # weight split per layer: attention (qkvo) vs FFN (up/gate + down)
    inter = getattr(cfg, "intermediate_size", 4*d)
    attn_w = (2*d*d + 2*n_kv*head_d*d)                     # q,o ~ d*d ; k,v ~ n_kv*head_d*d
    ffn_up   = (2 if _is_gated(cfg) else 1) * d * inter    # gate+up (SwiGLU) or just up
    ffn_down = inter * d
    tot_w = attn_w + ffn_up + ffn_down
    frac_down = ffn_down / tot_w                           # down-proj share of weight reads
    return dict(w=w, kv=kv, total=w+kv, frac_down=frac_down, inter=inter, n_kv=n_kv)

def _is_gated(cfg):
    a = getattr(cfg, "hidden_act", "") or ""
    return "silu" in a.lower() or "swish" in a.lower() or "glu" in a.lower()

# ---------------- perplexity (standard sliding window) ------------------------
@torch.no_grad()
def perplexity(model, enc, device, max_len, stride=512, limit=None, desc="ppl"):
    ids = enc.input_ids.to(device)
    seq = ids.size(1) if limit is None else min(ids.size(1), limit)
    nll, ntok, prev = 0.0, 0, 0
    begins = list(range(0, seq, stride))
    for begin in tqdm(begins, desc=desc, leave=False):
        end = min(begin + max_len, seq)
        tl  = end - prev
        inp = ids[:, begin:end]
        tgt = inp.clone(); tgt[:, :-tl] = -100
        out = model(inp, labels=tgt)
        nll += out.loss.float().item() * tl
        ntok += tl; prev = end
        if end == seq: break
    return math.exp(nll / max(ntok, 1))

# ---------------- FFN read-mask (down_proj input gate) ------------------------
_ffn_state = {"keep": 1.0, "zeroed": None, "total": 0}   # zeroed accumulates ON DEVICE (sync once/pass)
def _ffn_prehook(module, args):
    if _ffn_state["keep"] >= 1.0: return None
    h = args[0]                                            # [B,T,I] input to down_proj
    I = h.shape[-1]; k = max(1, int(round(_ffn_state["keep"] * I)))
    kth = torch.topk(h.abs(), k, dim=-1).values[..., -1:]  # k-th largest |h| per row (MPS-supported)
    mask = h.abs() >= kth
    z = (~mask).sum()                                      # stays on device — no host sync here
    _ffn_state["zeroed"] = z if _ffn_state["zeroed"] is None else _ffn_state["zeroed"] + z
    _ffn_state["total"] += mask.numel()                    # python int, no sync
    return (h * mask,) + args[1:]

def attach_ffn_hooks(model):
    hs = []
    for m in model.modules():
        if m.__class__.__name__.endswith("MLP") and hasattr(m, "down_proj"):
            hs.append(m.down_proj.register_forward_pre_hook(_ffn_prehook))
    return hs

# ---------------- KV read-skip (sink+window via sdpa wrapper) -----------------
_kv = {"sink": 0, "window": 10**9, "kept": None, "total": None}   # accumulate ON DEVICE
_orig_sdpa = torch.nn.functional.scaled_dot_product_attention
def _sdpa_sink_window(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kw):
    Lq, Lk = q.shape[-2], k.shape[-2]
    if _kv["window"] >= Lk and _kv["sink"] == 0:
        return _orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kw)
    dev = q.device
    qi = torch.arange(Lq, device=dev).view(Lq, 1)
    ki = torch.arange(Lk, device=dev).view(1, Lk)
    off = Lk - Lq                                          # query i is absolute pos off+i
    causal = ki <= (qi + off)
    keep = (ki < _kv["sink"]) | (ki > (qi + off - _kv["window"]))
    allow = causal & keep
    ka, ca = allow.sum(), causal.sum()                    # stay on device — sync once/pass
    _kv["kept"]  = ka if _kv["kept"]  is None else _kv["kept"]  + ka
    _kv["total"] = ca if _kv["total"] is None else _kv["total"] + ca
    bias = torch.zeros(Lq, Lk, device=dev, dtype=q.dtype)
    bias.masked_fill_(~allow, float("-inf"))
    return _orig_sdpa(q, k, v, attn_mask=bias, dropout_p=dropout_p, is_causal=False, **kw)

# ---------------- driver ------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--mode", choices=["baseline","ffn","kv","all"], default="all")
    ap.add_argument("--max-len", type=int, default=2048, help="measurement window (quality)")
    ap.add_argument("--energy-ctx", type=int, default=32768,
                    help="deployment context for the ENERGY budget (KV grows with it; quality still measured at --max-len)")
    ap.add_argument("--limit", type=int, default=60000, help="tokens of WikiText-2 to score (speed)")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    dev = args.device if torch.backends.mps.is_available() or args.device!="mps" else "cpu"
    print(f"device={dev}  model={args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="sdpa").to(dev).eval()
    n_params = sum(p.numel() for p in model.parameters())
    cfg = model.config

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tok("\n\n".join(ds["text"]), return_tensors="pt")
    print(f"params={n_params/1e9:.2f}B  layers={cfg.num_hidden_layers}  hidden={cfg.hidden_size}  "
          f"gated_ffn={_is_gated(cfg)}  scoring {min(args.limit, enc.input_ids.size(1))} tokens")

    bud = energy_budget(cfg, n_params, args.energy_ctx)
    kv_frac = bud["kv"]/bud["total"]; w_frac = bud["w"]/bud["total"]
    print(f"energy budget (near-mem, int4, DEPLOYMENT ctx={args.energy_ctx}; quality measured at {args.max_len}): "
          f"weights {100*w_frac:.0f}% · KV {100*kv_frac:.0f}%  | down-proj = {100*bud['frac_down']*w_frac:.1f}% of total\n")

    results = {"model": args.model, "params_B": n_params/1e9, "energy": {
        "weight_pct": 100*w_frac, "kv_pct": 100*kv_frac, "downproj_pct_total": 100*bud['frac_down']*w_frac}}

    t0 = time.time()
    base = perplexity(model, enc, dev, args.max_len, limit=args.limit)
    results["baseline_ppl"] = base
    print(f"[baseline] WikiText-2 perplexity = {base:.3f}   ({time.time()-t0:.0f}s)\n")

    if args.mode in ("ffn","all"):
        hs = attach_ffn_hooks(model)
        print("FFN read-mask (gate hidden -> skip down-proj reads):")
        print(f"{'keep%':>6} {'FFN_sparse%':>11} {'ppl':>8} {'Δppl%':>7} {'energy_saved%':>13}")
        rows=[]
        for keep in (1.0, 0.5, 0.3, 0.2, 0.1):
            _ffn_state.update(keep=keep, zeroed=None, total=0)
            ppl = perplexity(model, enc, dev, args.max_len, limit=args.limit, desc=f"ffn keep={keep}")
            zt = _ffn_state["zeroed"]
            sp = (float(zt.item()) if zt is not None else 0.0)/max(_ffn_state["total"],1)
            saved = sp * bud["frac_down"] * w_frac * 100          # % of total decode energy
            dp = 100*(ppl-base)/base
            rows.append(dict(keep=keep, sparsity=sp, ppl=ppl, dppl=dp, energy_saved_pct=saved))
            print(f"{100*keep:6.0f} {100*sp:11.1f} {ppl:8.3f} {dp:7.2f} {saved:13.1f}")
        for h in hs: h.remove()
        _ffn_state.update(keep=1.0)
        results["ffn"] = rows; print()

    if args.mode in ("kv","all"):
        torch.nn.functional.scaled_dot_product_attention = _sdpa_sink_window
        print("KV read-skip (StreamingLLM sink+window; content-independent):")
        print(f"{'sink':>5} {'window':>7} {'KV_kept%':>9} {'ppl':>8} {'Δppl%':>7} {'energy_saved%':>13}")
        rows=[]
        for sink, win in ((4, args.max_len), (4, 1024), (4, 512), (4, 256), (4, 128)):
            _kv.update(sink=sink, window=win, kept=None, total=None)
            ppl = perplexity(model, enc, dev, args.max_len, limit=args.limit, desc=f"kv win={win}")
            kept = float(_kv["kept"].item())/max(float(_kv["total"].item()), 1.0)
            saved = (1-kept) * kv_frac * 100                      # % of total decode energy
            dp = 100*(ppl-base)/base
            rows.append(dict(sink=sink, window=win, kv_kept=kept, ppl=ppl, dppl=dp, energy_saved_pct=saved))
            print(f"{sink:5d} {win:7d} {100*kept:9.1f} {ppl:8.3f} {dp:7.2f} {saved:13.1f}")
        torch.nn.functional.scaled_dot_product_attention = _orig_sdpa
        results["kv"] = rows
        print("\nContent-aware KV (H2O / Deja-Vu) would skip more for the same quality — needs a scoring pass.")

    json.dump(results, open("m4_results.json","w"), indent=2)
    print("\nsaved m4_results.json  — quality-vs-energy on a REAL model, reviewer-ready.")

if __name__ == "__main__":
    main()
