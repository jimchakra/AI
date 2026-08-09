# Real-model gating study — run on your Apple M4 (Terminal)

Measures, on a **real pretrained transformer**, the two input-side gating levers
and their quality-vs-energy curves — the result that retires the tiny-proxy
caveat and connects the work to the contextual-sparsity literature.

## Setup (once)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Qwen2.5 is open (no login). For Llama-3.2 you must first:
```bash
huggingface-cli login          # accept the model license on huggingface.co first
```

## Run

```bash
# turnkey headline (Qwen2.5-1.5B): baseline PPL + FFN read-mask + KV skip
python3 m4_study.py --mode all

# just the rock-solid part (FFN read-mask), fastest
python3 m4_study.py --mode ffn

# a larger, gated model
python3 m4_study.py --model meta-llama/Llama-3.2-3B --mode all
```

Speed knobs: `--limit 30000` (fewer WikiText tokens = faster), `--max-len 1024`.
KV stakes: `--energy-ctx 32768` (default) models KV at a realistic long-context deployment so the
KV-skip lever shows its true share; quality is still measured at `--max-len`.

If any op is missing on MPS (Apple adds them over time), prepend a CPU fallback:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 m4_study.py --mode all
```

## What to expect

- **First run** downloads the model (~1–6 GB) and WikiText-2 (small).
- **Runtime** (scoring ~60K tokens): Qwen2.5-1.5B ≈ 5–15 min for `--mode all`;
  Llama-3.2-3B ≈ 15–40 min. `--mode ffn` alone is ~⅓ of that.
- Output prints three blocks and writes `m4_results.json`:
  1. **baseline** WikiText-2 perplexity (calibrates the quality axis).
  2. **FFN read-mask**: keep% → FFN sparsity → ppl → Δppl% → **% total decode
     energy saved** (down-proj reads skipped via the survivor bitmap).
  3. **KV read-skip**: sink+window → KV kept% → ppl → Δppl% → **% total decode
     energy saved** (KV is the biggest term).

## Honesty notes (keep these with the numbers)

- **Turnkey vs best-effort.** The baseline and FFN read-mask paths are robust
  (a pre-hook on the down-projection input — version-stable). The KV path wraps
  `scaled_dot_product_attention` to impose a sink+window mask; it needs the model
  to use the SDPA attention backend (default for Qwen2.5/Llama on recent
  transformers). If KV errors on your version, run `--mode ffn` — the FFN result
  stands on its own.
- **KV policy is content-independent** (StreamingLLM sink+window). It's the
  *conservative* end: content-aware predictors (H2O heavy-hitter, Deja-Vu) skip
  more KV for the same quality. So the KV energy number here is a **floor**, not
  a ceiling.
- **Energy is first-order**, derived from the model's own config (GQA-aware),
  assuming int4 deployment weights and near-memory reads (array + in-stack hop).
  Quality (perplexity) is **measured**. Absolute power still needs a foundry PDK.
- The eval runs in fp16 on MPS; the energy accounting assumes int4 *deployment*
  (stated), since that's the near-memory target precision.

## Why this matters (one line for the reviewer)

The gate's survivor bitmap, reused as a down-projection **read-mask**, and a
sink+window **KV read-skip**, are the near-memory hardware realization of the
contextual-sparsity that large transformers exhibit (Deja Vu, ICML 2023; the
lazy-neuron phenomenon, 2022) — measured here on a real model with a real
benchmark, not a proxy.
