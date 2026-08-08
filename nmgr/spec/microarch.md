# NMGR tile — microarchitecture specification

**Near-memory gated-reduction tile**, `nmgr_tile_v1`. A small, synthesizable
datapath that completes a cross-bank reduction and applies an in-situ
activation gate *on the memory side*, returning only the surviving outputs in
compressed form. It is the concrete, checkable core of the "compute at the
memory boundary" argument — one tile, taken from PyTorch-equivalent reference
all the way to area/power/$-per-token.

> Scope note: this public demonstrator implements the **exact** gate (ReLU /
> hard threshold) only. The bounded-approximate / error-feedback gating modes
> are out of scope here.

---

## 1. Function

For each output lane `m` in a tile of `tile_m` lanes:

```
acc[m] = Σ_bank Σ_k  W[bank, m, k] · x[bank, k]      # distributed reduction
y[m]   = gate( saturate_out( relu( acc[m] ) ) )      # in-situ activation gate
```

The reduction is completed across all banks *before* the gate, so the
keep-or-discard decision is valid at the memory and only survivors ever move.
This is the whole point: evaluating the gate where the data lives is what makes
the transport saving real (moving a value only to discard it pays the bill the
gate was meant to save).

Output is encoded as **bitmap + survivor values**, not a dense vector.

## 2. Interfaces

| Port        | Dir | Width / shape                     | Notes                          |
|-------------|-----|-----------------------------------|--------------------------------|
| `x_in`      | in  | `banks × k_per_bank × x_bits`     | streamed activation (per token)|
| `w_banks`   | in  | `banks × tile_m × k_per_bank × w_bits` | resident weights          |
| `bitmap`    | out | `tile_m × 1`                      | 1 = survivor                   |
| `surv_val`  | out | `n_survivors × out_bits`          | compressed survivor stream     |
| `valid/ready` | — | handshake                         | back-pressured survivor stream |

Weights are **resident** (read from the local array over the base-die hop);
activations **stream**; only the compressed survivor stream crosses the host
link.

## 3. Datapath / pipeline

```
 bank 0 ─┐  per-bank MAC   ┐
 bank 1 ─┤  (k_per_bank    │   cross-bank      ReLU +      threshold     bitmap +
 bank 2 ─┤   MACs/lane)    ├─► adder tree  ─► saturate ─►   gate     ─►  survivor
 bank 3 ─┘                 ┘   (acc_bits)                 (|y|≤θ→0)      encoder
```

- **PE array:** `pe_count` lanes (one per output `m`), each a `k_per_bank`-deep
  MAC over its bank slice; `macs_per_cycle_per_pe` sets the fold.
- **Reduction:** balanced adder tree across `banks`, `acc_bits` wide (checked
  for overflow by the golden).
- **Gate:** ReLU → saturate to `out_bits` → threshold compare.
- **Encoder:** priority/prefix over the bitmap to pack survivors.

Parameters live in [`config/tile.yaml`](../config/tile.yaml) and drive the
golden, the RTL, and the scale-up — one source of truth.

## 4. Numerics (bit-exact contract)

Integer, deterministic, so RTL == golden bit-for-bit:

- `x`: signed `x_bits`, `W`: signed `w_bits`.
- accumulate in signed `acc_bits` (no intermediate rounding).
- `relu` = `max(0, acc)`, then **requantize**: `>> out_shift`, saturate to
  `[0, 2^(out_bits-1)−1]` (standard quantized back-end).
- gate: `out = sat if sat > threshold else 0`.

**Threshold = the lossless/lossy knob.** `threshold = 0` zeros only what ReLU
already kills → exact, no accuracy cost. `threshold > 0` also drops small
positives → more sparsity and less return traffic, but lossy (valid only within
a calibrated bound). See it directly:
`python3 golden/golden.py --sweep 0,1,2,3,5,8,12,20,32` prints sparsity,
compression, and discarded-mass at each θ.

The reference is a plain quantized `relu(x @ Wᵀ)`; a `torch.nn.functional`
version yields identical values. See [`golden/golden.py`](../golden/golden.py).

## 5. Verification plan

1. **Golden vectors** (done): `golden.py` emits `weights.hex`, `x_*.hex`,
   `expected_*.hex`, `bitmap_*.hex`, `meta.json`.
2. **RTL bit-exact** (next): Verilator/iverilog testbench replays `x_*.hex`
   against resident `weights.hex`, checks dense-equivalent output and bitmap
   match `expected_*` exactly across all vectors.
3. **Coverage:** sparsity sweep (threshold), accumulator-overflow guard,
   all-zero and all-survive corner vectors.

## 6. Physical + cost scale-up

1. **Synthesis** (next): Yosys → open PDK (sky130 / Nangate45) for gate count,
   area, and `fmax`; activity-annotated power from the testbench VCD.
2. **Tile → model:** a tile does `banks·tile_m·k_per_bank` MACs per pass; a
   `d_model×d_model` projection needs `(d_model/tile_m)·(d_model/K)` tiles;
   scale by projections × layers for tiles/token.
3. **→ $/token:** feed per-tile area/power × tile count into
   [`inference_energy_sim.py`](../../inference_energy_sim.py) for tokens/s,
   watts, and first-order die-cost/$-per-token — grounded in a real synthesized
   block rather than assumed constants.

## 7. Status

| Stage                         | State |
|-------------------------------|-------|
| Config / parameterization     | done  |
| Golden reference + vectors    | done  |
| RTL (Verilog)                 | next  |
| Bit-exact testbench (Verilator) | next |
| Synthesis (Yosys + open PDK)  | next  |
| Power + $/token scale-up      | next  |
| Accuracy: perplexity/accuracy vs θ on a real model (Tier-3) | next |
| PyTorch custom-op cosim (opt) | later |

**Two validation axes, both needed for the full story:** (a) *PPA* — RTL →
synthesis → area/power → $/token; (b) *Accuracy* — insert the gated-requant
activation into a real pretrained model (GPT-2 / Pythia class) and measure
perplexity/accuracy vs θ. The golden's rel-L2 / SQNR are per-layer proxies;
only end-to-end eval shows whether the loss compounds or washes out across
layers (literature says activation sparsity is usually tolerated far past where
the proxy looks scary — the favorable case for the gate).
