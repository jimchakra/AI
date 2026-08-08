# AI inference architecture

Writing, working code, and early silicon results on real-time and memory-bound
AI inference — from building baseband silicon, applied to where inference is
heading.

**Site:** https://jimchakra.github.io/AI/

## Contents

- **[Small Frees Big](https://jimchakra.github.io/AI/small-frees-big.html)** —
  essay: data movement, not arithmetic, is the binding constraint on inference;
  small targeted compute pushed toward the memory is where it's going.
- **[Compute at the memory boundary](https://jimchakra.github.io/AI/results.html)**
  — early results & roadmap (Part II, in progress). Patent pending.
- **[`inference_energy_sim.py`](inference_energy_sim.py)** — a first-order
  compute-vs-data-movement and $/token model (honest array-access / transport
  split). Run: `python3 inference_energy_sim.py` (`--mem dram`, `--prec fp4`, …).
- **[`nmgr/`](nmgr/)** — near-memory gated-reduction tile demonstrator:
  microarchitecture [spec](nmgr/spec/microarch.md), a bit-exact
  [golden reference](nmgr/golden/golden.py), and test vectors.
  Try the gate sweep: `python3 nmgr/golden/golden.py --sweep 0,1,2,3,5,8,12,20,32`

The core mechanism — near-memory distributed reduction with in-situ output
gating and compressed return — is the subject of a U.S. provisional patent
filed August 2026. Figures here are first-order models pending RTL synthesis and
end-to-end accuracy evaluation.
