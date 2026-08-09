#!/usr/bin/env python3
"""
pytorch_cosim.py — NMGR near-memory gating, co-simulated on a real trained transformer.
============================================================================

Purpose. The RTL/golden demonstrator proves the near-memory gated-reduction
datapath is bit-exact on synthetic vectors. This script answers the question a
systems reviewer actually asks: *on a real model doing real inference, how much
total quality do you lose at each gate threshold — and how much data movement
does that buy back?*

We train a small GPT (char-level, tiny-shakespeare) as a transparent, fully
reproducible proxy — pretrained-LLM downloads are network-blocked here, and the
θ → quality → movement relationship is model-agnostic (it depends on the
heavy-tailed statistics of FFN activations, which hold across scale). Nothing
about the mechanism is specific to this model; the small model just makes the
whole experiment runnable and auditable end to end.

The NMGR gate, applied in the FFN. In the datapath, the gate keeps large
reduction outputs and zeros the rest, returning only survivors (compressed).
In a transformer that maps to the FFN intermediate activations: after the
nonlinearity we zero every activation with |a| < θ before it would move off the
compute tile, and only the survivors travel. θ = 0 is lossless by construction;
θ > 0 trades quality for movement. We sweep θ and report, at the SYSTEM level:

  * validation perplexity vs θ            (total inference quality lost)
  * FFN-activation sparsity vs θ          (movement removed on that path)
  * rel-L2 / SQNR of gated activations    (signal fidelity)
  * energy / token vs θ                   (sparsity folded into the energy model)

Usage:
  python3 pytorch_cosim.py train  --minutes 8      # train + checkpoint
  python3 pytorch_cosim.py sweep  --thetas 0,0.5,1,2,3,4,6,8
"""
import argparse, math, os, time, json
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(1337)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "cosim", "input.txt")
CKPT = os.path.join(HERE, "cosim", "gpt_ckpt.pt")

# ------------------------------------------------------------------ model ---
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd); self.ln2 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.fc   = nn.Linear(n_embd, 4*n_embd)
        self.proj = nn.Linear(4*n_embd, n_embd)
        self.register_buffer("mask", torch.triu(torch.ones(block, block)*float("-inf"), diagonal=1))
        # co-sim hook state
        self.gate_theta = 0.0
        self.last_total = 0
        self.last_gated = 0
        self.last_relL2 = 0.0

    def forward(self, x):
        T = x.size(1)
        a,_ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                        attn_mask=self.mask[:T,:T], need_weights=False)
        x = x + a
        h = F.gelu(self.fc(self.ln2(x)))            # FFN intermediate activations
        if self.gate_theta > 0.0:                   # ---- NMGR gate: keep survivors ----
            keep = h.abs() >= self.gate_theta
            hg = h * keep
            # bookkeeping for the sweep (system-level movement + fidelity)
            self.last_total += h.numel()
            self.last_gated += (~keep).sum().item()
            num = (hg - h).pow(2).sum().item(); den = h.pow(2).sum().item() + 1e-12
            self.last_relL2 += num  # accumulate; normalized later with den via _den
            self._den = getattr(self, "_den", 0.0) + den
            h = hg
        x = x + self.proj(h)
        return x

    def reset_stats(self):
        self.last_total = 0; self.last_gated = 0; self.last_relL2 = 0.0; self._den = 0.0

class GPT(nn.Module):
    def __init__(self, vocab, n_embd=192, n_head=4, n_layer=3, block=128):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)

    def set_gate(self, theta):
        for b in self.blocks:
            b.gate_theta = theta; b.reset_stats()

    def gate_stats(self):
        tot = sum(b.last_total for b in self.blocks)
        gat = sum(b.last_gated for b in self.blocks)
        num = sum(b.last_relL2 for b in self.blocks)
        den = sum(getattr(b, "_den", 0.0) for b in self.blocks)
        spars = gat/tot if tot else 0.0
        relL2 = math.sqrt(num/den) if den else 0.0
        sqnr = (10*math.log10(den/num)) if num > 0 else float("inf")
        return spars, relL2, sqnr

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks: x = b(x)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ------------------------------------------------------------------ data ----
def load_data():
    if not os.path.exists(DATA):                     # fetch the standard corpus once
        os.makedirs(os.path.dirname(DATA), exist_ok=True)
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"downloading corpus -> {DATA}")
        urllib.request.urlretrieve(url, DATA)
    text = open(DATA, "r").read()
    chars = sorted(list(set(text)))
    stoi = {c:i for i,c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9*len(data))
    return data[:n], data[n:], len(chars)

def get_batch(data, block, bs):
    ix = torch.randint(len(data)-block-1, (bs,))
    x = torch.stack([data[i:i+block] for i in ix])
    y = torch.stack([data[i+1:i+block+1] for i in ix])
    return x, y

@torch.no_grad()
def eval_ppl(model, data, block, bs=32, iters=40, batches=None):
    """Perplexity = exp(mean NLL). If `batches` is given, evaluate on that
    fixed set so every θ is compared on identical data (θ=0 == baseline exactly)."""
    model.eval(); tot=0.0
    if batches is None:
        batches = [get_batch(data, block, bs) for _ in range(iters)]
    for x,y in batches:
        _,loss = model(x,y); tot += loss.item()
    model.train()
    return math.exp(tot/len(batches))

def make_eval_batches(data, block, bs, iters, seed=2024):
    g = torch.Generator().manual_seed(seed)
    out=[]
    for _ in range(iters):
        ix = torch.randint(len(data)-block-1, (bs,), generator=g)
        x = torch.stack([data[i:i+block] for i in ix])
        y = torch.stack([data[i+1:i+block+1] for i in ix])
        out.append((x,y))
    return out

# ------------------------------------------------------------------ train ---
def train(args):
    torch.set_num_threads(os.cpu_count() or 4)
    tr, va, vocab = load_data()
    model = GPT(vocab)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M  vocab={vocab}")
    t0=time.time(); it=0; budget=args.minutes*60
    while time.time()-t0 < budget:
        x,y = get_batch(tr, model.block, 32)
        _,loss = model(x,y); opt.zero_grad(); loss.backward(); opt.step(); it+=1
        if it % 200 == 0:
            ppl = eval_ppl(model, va, model.block)
            print(f"  it {it:5d}  {time.time()-t0:5.0f}s  train_loss {loss.item():.3f}  val_ppl {ppl:.2f}")
    ppl = eval_ppl(model, va, model.block, iters=80)
    torch.save({"model":model.state_dict(),"vocab":vocab}, CKPT)
    print(f"DONE it={it} final val_ppl={ppl:.3f}  -> {CKPT}")

# ------------------------------------------------------------------ sweep ---
# energy-model constants (mirror inference_energy_sim.py; illustrative ~5nm-class)
DRAM_ARRAY=6.0; BASE_HOP=3.0; HBM_IO=26.0     # pJ/byte
def energy_per_ffn_byte_moved(sparsity, near_memory=True):
    """First-order: FFN activation-return energy per byte, conventional vs near-memory+gated.
       Conventional: array + off-die HBM I/O for every byte.
       Near-mem+gate: array + in-stack hop, and only survivors move."""
    conv = DRAM_ARRAY + HBM_IO
    near = (DRAM_ARRAY + BASE_HOP) * (1.0 - sparsity)   # survivors only cross the hop
    return conv, near

def sweep(args):
    tr, va, vocab = load_data()
    ck = torch.load(CKPT, map_location="cpu")
    model = GPT(ck["vocab"]); model.load_state_dict(ck["model"]); model.eval()
    thetas = [float(t) for t in args.thetas.split(",")]
    batches = make_eval_batches(va, model.block, 32, args.iters)   # fixed set for all θ
    model.set_gate(0.0)
    base_ppl = eval_ppl(model, va, model.block, batches=batches)
    conv = DRAM_ARRAY + HBM_IO                     # conventional off-die move, pJ/byte
    near0 = DRAM_ARRAY + BASE_HOP                   # near-memory, no gating (locality only)
    locality_x = conv/near0                         # gating-independent locality win
    rows=[]
    for th in thetas:
        model.set_gate(th)
        ppl = eval_ppl(model, va, model.block, batches=batches)
        spars, relL2, sqnr = model.gate_stats()
        near = near0 * (1.0 - spars)                # survivors only cross the hop
        total_x = conv/near if near>0 else float("inf")
        rows.append(dict(theta=th, ppl=ppl,
                         ppl_delta_pct=100*(ppl-base_ppl)/base_ppl,
                         ffn_sparsity=spars, relL2=relL2, sqnr_db=sqnr,
                         locality_x=locality_x, total_move_x=total_x))
    out = dict(base_ppl=base_ppl, locality_x=locality_x, rows=rows)
    json.dump(out, open(os.path.join(HERE,"cosim","sweep.json"),"w"), indent=2)
    print(f"\nbaseline val perplexity (θ=0, lossless): {base_ppl:.3f}")
    print(f"near-memory locality (gating-independent): {locality_x:.2f}x  "
          f"({conv:.0f} -> {near0:.0f} pJ/byte on the FFN activation path)\n")
    print(f"{'θ':>5} {'val_ppl':>9} {'Δppl%':>8} {'FFN_spars%':>11} {'SQNR_dB':>8} {'FFNmove_total':>13}")
    for r in rows:
        print(f"{r['theta']:5.2f} {r['ppl']:9.3f} {r['ppl_delta_pct']:8.2f} "
              f"{100*r['ffn_sparsity']:11.1f} {r['sqnr_db']:8.1f} {r['total_move_x']:12.1f}x")
    print("\nΔppl% = total, SYSTEM-LEVEL quality lost (whole-model perplexity) at that gate.")
    print("FFNmove_total = movement reduction on the FFN activation-return path:")
    print(f"  {locality_x:.1f}x is locality (near-memory, any θ); the rest is gating (θ-dependent).")
    print("This path is a COMPONENT of decode movement (weights/KV dominate) — not total system energy.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--minutes", type=float, default=8.0)
    s = sub.add_parser("sweep")
    s.add_argument("--thetas", default="0,0.5,1,2,3,4,6,8")
    s.add_argument("--iters", type=int, default=60)
    a = ap.parse_args()
    (train if a.cmd=="train" else sweep)(a)
