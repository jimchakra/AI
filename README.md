# AI inference architecture

[![nmgr-verify](https://github.com/jimchakra/AI/actions/workflows/ci.yml/badge.svg)](https://github.com/jimchakra/AI/actions/workflows/ci.yml)

Writing, working code, and early silicon results on real-time and memory-bound
AI inference — from building baseband silicon, applied to where inference is
heading.

**Site:** https://jimchakra.github.io/AI/

## Verification & sign-off — read the badge, not my word for it

The badge above is the source of truth. It links to a GitHub Actions run that
re-executes **`cd nmgr && make all` on a clean Ubuntu runner for the current
commit** — installing the open-source EDA tools from scratch and running the
whole flow. "Clean" is *reproduced by CI on every push*, not asserted in this
README. Click the badge → open the latest run → each stage prints its own
`RESULT: PASS`.

| Check | What it proves | Runs in | Status |
|-------|----------------|---------|--------|
| **SSOT consistency** | one `tile.yaml` == RTL params == FW header == DV config | **CI** (every push) | ✅ a drift fails CI |
| **Lint** (Verilator `-Wall`) | RTL style / width / hygiene, all 4 modules | **CI** | ✅ clean |
| **Simulation** (Icarus) | RTL == bit-exact golden | **CI** | ✅ PE 1024/1024 · encoder 1510/1510 |
| **CDC** (dual-clock sim) | memory-clk ↔ link-clk crossing (packetizer) | **CI** | ✅ 6/6 async + backpressure; safe by construction (Gray-pointer async FIFO + 2-flop sync) |
| **Synthesis** (Yosys) | maps to a gate netlist | **CI** | ✅ PE/encoder/packetizer |
| **LEC** (Yosys `equiv_opt`) | RTL == synthesized netlist (formal) | **CI** | ✅ encoder 608/608 |
| **Area** (Yosys + Nangate45) | grounded gate area | local `make area` | ✅ ≈ 0.017 mm² @ 45 nm |
| **HITL cosim** (iverilog + PyTorch) | RTL runs a *real* transformer FFN tile | local `make cosim` | ✅ 64/64 bit-exact, cosine 0.99 |

**Honest boundaries** (a silicon reviewer will ask, so they're stated up front):

- The **PE datapath LEC** and a **formal CDC sign-off** are commercial-tool steps
  (Cadence Conformal / Synopsys Formality; SpyGlass CDC / Questa CDC). Here the PE
  is covered by the exhaustive simulation regression, and CDC by construction plus
  the dual-clock testbench — open tools don't close those formally.
- **Absolute power** needs a foundry PDK + a PrimeTime-PX-class tool; area is
  grounded (45 nm), the 5 nm figure is a density-scaled estimate.
- `make area` and `make cosim` are **not** in CI only because they need a 6.7 MB
  PDK download / PyTorch + a trained checkpoint — both reproduce locally in one
  command.

## Contents

- **[Small Frees Big](https://jimchakra.github.io/AI/small-frees-big.html)** —
  essay: data movement, not arithmetic, is the binding constraint on inference;
  small targeted compute pushed toward the memory is where it's going.
- **[Compute at the memory boundary](https://jimchakra.github.io/AI/results.html)**
  — early results & roadmap: energy model, the θ gate sweep, a real-model
  accuracy co-sim, grounded area/$·token, and hardware-in-the-loop. Patent pending.
- **[`inference_energy_sim.py`](inference_energy_sim.py)** — first-order
  compute-vs-data-movement, $/token, and link-framing model (honest array-access /
  transport split). `python3 inference_energy_sim.py --dollars --framing`.
- **[`pytorch_cosim.py`](pytorch_cosim.py)** — near-memory gate co-simulated on a
  trained transformer: whole-model perplexity vs gate threshold, movement saved.
- **[`hitl_cosim.py`](hitl_cosim.py)** — hardware-in-the-loop: a real FFN tile
  computed by the actual RTL simulation, bit-exact to golden.
- **[`nmgr/`](nmgr/)** — the near-memory gated-reduction tile: parameterized
  [spec](nmgr/spec/microarch.md), a bit-exact [golden reference](nmgr/golden/golden.py),
  and synthesizable RTL verified against it:
  - [`rtl/nmgr_pe.v`](nmgr/rtl/nmgr_pe.v) — compute core (MAC → cross-bank
    reduction → requantize → gate).
  - [`rtl/nmgr_encoder.v`](nmgr/rtl/nmgr_encoder.v) — compressed return
    (bitmap + packed survivors).
  - [`rtl/nmgr_packetizer.v`](nmgr/rtl/nmgr_packetizer.v) +
    [`rtl/async_fifo.v`](nmgr/rtl/async_fifo.v) — link-layer framer across the
    real memory-clk ↔ link-clk crossing.
  - **Flow:** `cd nmgr && make all` (in CI) — SSOT check → lint → sim → synth →
    LEC. `make area` and `make cosim` add grounded area and the HITL cosim. See
    [`nmgr/FLOW.md`](nmgr/FLOW.md) for per-stage status and a candid
    AI-assisted-flow effectiveness write-up.

The core mechanism — near-memory distributed reduction with in-situ output
gating and compressed return — is the subject of a U.S. provisional patent filed
August 2026.
