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
 bank 0 ─┐  base-die MAC   ┐
 bank 1 ─┤  lanes (operands│   cross-bank      ReLU +      threshold     bitmap +
 bank 2 ─┤  streamed per   ├─► adder tree  ─► saturate ─►   gate     ─►  survivor
 bank 3 ─┘  bank)          ┘   (acc_bits)                 (|y|≤θ→0)      encoder
```

- **MAC lanes (on the base logic die):** `pe_count` lanes (one per output `m`).
  Operands are *streamed up from each bank* over the short in-stack hop — there
  is **no compute inside the DRAM banks**; all multiply / reduce / gate / encode
  is on the base (aggregation) die.
- **Reduction:** balanced adder tree across `banks`, `acc_bits` wide (checked
  for overflow by the golden).
- **Gate:** ReLU → saturate to `out_bits` → threshold compare.
- **Encoder:** priority/prefix over the bitmap to pack survivors.

Parameters live in [`config/tile.yaml`](../config/tile.yaml) and drive the
golden, the RTL, and the scale-up — one source of truth.

## 3a. Physical mapping & host interface (design decisions)

- **All compute lives on the base (aggregation) logic die.** Per-bank /
  in-DRAM-die MAC is explicitly rejected: DRAM is a poor logic process (area,
  speed, yield, thermal). MAC, cross-bank reduction, gate, and survivor encoding
  all sit on the base logic die of the stack (or an equivalent memory-side
  buffer / CXL controller). The banks only supply operands over the short
  in-stack hop.
- **The host interface is a packetized link, not a DDR-transparent burst.**
  Gating + compression makes the return *variable-length and data-dependent*,
  which DDR's fixed-burst, fixed-latency contract cannot carry. The base die is
  therefore a protocol boundary: conventional fixed-burst upstream to the DRAM
  array, and a framed, flow-controlled **packet link downstream to the host —
  realized on SerDes (CXL / PCIe-class), not a DDR PHY.** The fixed-transfer PHY
  (DDR, or HBM's parallel protocol) is **retired or demoted to a legacy
  passthrough**; a *single* packetized link carries **both** ordinary load/store
  and the variable compute returns, QoS-arbitrated. Tradeoff: framing +
  serialization add latency to plain access — minimized when compute die and
  memory are on-package (accelerator-attached HBM).
- **Consequences to model:** variable completion latency (the host scheduler
  must tolerate out-of-order, credited returns); per-packet framing overhead vs
  a small survivor payload (batch survivors into right-sized packets); flow
  control and ordering across the link.

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
| RTL (Verilog) — `rtl/nmgr_pe.v` | **done** |
| Bit-exact testbench (Icarus) — `tb/tb_nmgr.sv` | **done — 1024/1024 outputs match golden (θ=0)** |
| Synthesis complexity (Yosys) | **done — ~1,360 gate cells + 48 FF per PE** |
| Survivor / compression encoder — `rtl/nmgr_encoder.v` | **done — 1510/1510 assertions match golden (bitmap + packed survivors)** |
| Packetizer / link-layer framing + framing-overhead model (SerDes / CXL) | next |
| Synthesis area/power on an open PDK (sky130 / Nangate45) | next |
| Power + $/token scale-up      | next  |
| Accuracy: perplexity/accuracy vs θ on a real model (Tier-3) | next |
| PyTorch custom-op cosim (opt) | later |

Run the check: `iverilog -g2012 -o /tmp/tb rtl/nmgr_pe.v tb/tb_nmgr.sv && vvp /tmp/tb`
(after `python3 golden/golden.py` and concatenating the per-vector hex into
`vectors/all_x.hex` and `vectors/all_expected.hex`).

**Two validation axes, both needed for the full story:** (a) *PPA* — RTL →
synthesis → area/power → $/token; (b) *Accuracy* — insert the gated-requant
activation into a real pretrained model (GPT-2 / Pythia class) and measure
perplexity/accuracy vs θ. The golden's rel-L2 / SQNR are per-layer proxies;
only end-to-end eval shows whether the loss compounds or washes out across
layers (literature says activation sparsity is usually tolerated far past where
the proxy looks scary — the favorable case for the gate).
