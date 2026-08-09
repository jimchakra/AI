# NMGR verification / signoff flow

A small, reproducible, **AI-assisted** flow around the near-memory tile — the
kind of methodology you stand up on an early silicon team. One command runs it;
GitHub Actions runs it on every push.

```
make all      # lint -> golden vectors -> simulation -> synthesis -> LEC
```

All tools are open-source (no license needed): **Verilator** (lint),
**Icarus Verilog** (simulation), **Yosys** (synthesis + logic-equivalence),
Python/NumPy (golden reference).

## Stages & current status

| Stage | Tool | What it proves | Status |
|-------|------|----------------|--------|
| **SSOT fan-out + check** | Python + `make check` | one YAML → RTL params / FW header / DV cfg; RTL defaults agree | ✅ consistency by construction — a hand-edit that drifts any artifact fails CI |
| **Lint** _(static)_ | Verilator `--lint-only -Wall` | RTL style/width/UNUSED hygiene | ✅ clean, all four modules (PE, encoder, packetizer, async FIFO) |
| **CDC** _(static + dynamic)_ | async FIFO structure + dual-clock sim | memory-clk ↔ link-clk crossings safe | ✅ Gray-pointer async FIFO + 2-flop synchronizers (async_fifo.v); verified dual-clock in tb_packetizer. Formal CDC sign-off (SpyGlass/Questa CDC) is a commercial-tool step — same honest boundary as PE LEC. |
| **Golden** | Python | bit-exact reference + θ sweep | ✅ |
| **Simulation** | Icarus | RTL == golden on every vector | ✅ PE 1024/1024 · encoder 1510/1510 |
| **Packetizer / link-layer** | Icarus (dual-clock) | packet framing across the crossing | ✅ 6/6 vectors, async, with backpressure |
| **Synthesis** | Yosys | maps to gates; complexity | ✅ PE ~1.36k cells+48 FF · enc ~530 cells · packetizer+FIFO ~3.06k cells |
| **LEC** | Yosys `equiv_opt` | RTL == synthesized netlist (formal) | ✅ encoder proven (608/608 cells) |
| **Area** | Yosys + Nangate45 | grounded gate area | ✅ `make area`: PE 1.6k µm² · enc 5.1k · packetizer+CDC 10.3k (45 nm) → ~0.0003 mm² tile scaled to 5 nm (est.) |
| **$/token** | energy model | movement-energy \$ floor | ✅ near-memory cuts modeled \$ / 1M-tok 3.5× (HBM) / 17.5× (off-pkg DRAM); floor on the data path, not full-system |
| **HITL cosim** | iverilog + PyTorch | RTL runs a real FFN tile | ✅ `make cosim`: 64/64 bit-exact on a real transformer tile; dequant cosine 0.99 |
| **Power (absolute)** | — | switching-activity power | needs a foundry PDK + PrimeTime-PX; area is grounded, absolute power is the remaining gap |

### Honest scope boundary on LEC
Formal logic-equivalence of the **PE datapath does not close with open SAT
tools** — a 64-lane, 8×4 signed-multiplier array with a 32-bit accumulator is
exactly the case commercial LEC (Cadence Conformal / Synopsys Formality) exists
to handle with structural matching. So: the **encoder** is proven equivalent
formally (control + memory, no multipliers); the **PE datapath**'s correctness
is carried by the exhaustive **simulation regression** (1024/1024) and would use
a commercial LEC tool at signoff. Claiming otherwise would be dishonest — and a
reviewer would know.

## AI-assisted: what it did, and how well

This flow — golden model, RTL, testbenches, synth/LEC scripts — was authored
with an AI assistant (Claude). A candid effectiveness read, because the JD asks
for exactly that:

**Worked well.** Scaffolding the parameterized golden, the first-pass RTL, the
Yosys/Verilator invocations, and this flow was fast — minutes, not hours. The
AI was strong at the "code-like" parts: boilerplate, sweep tooling, hex-vector
plumbing, and turning a spec into a first RTL draft.

**Did not work — and this is the important part.** The first RTL *simulated to
all-zeros*. Two bugs, both in the **testbench**, both classic:
1. a **poll race** — waiting on a one-cycle `done` pulse right at the clock edge
   hung the sim;
2. a **stimulus race** — deasserting `start` in the same timestep as the
   sampling edge, so the DUT sampled it already low and never launched.
The fix (drive stimulus on the negedge) is standard verification hygiene the AI
did *not* apply until the failure was diagnosed. Lesson: the AI accelerates
*generation* but does not replace *verification judgment* — the bit-exact
testbench and lint are what caught its mistakes. Lint also flagged real issues
(blocking-in-function, index-width truncation, an unused buffer) that were fixed
rather than waived.

**Verdict.** AI-assisted design is a genuine throughput multiplier on the
code-like layer, gated by a human-owned correctness harness. The value is the
*flow* — golden + lint + exhaustive sim + LEC — that makes the AI's output
trustworthy, not the generation itself.

## Reproduce

```
cd nmgr
make all          # full flow (check + lint + sim + synth + LEC)
make gen          # re-fan-out RTL params / FW header / DV cfg from config/tile.yaml
make check        # SSOT consistency: YAML == RTL == FW == DV
make lint         # Verilator lint only
make sim          # golden + both bit-exact regressions
make synth        # Yosys synthesis + gate stats
make lec          # formal equivalence (encoder)
```
Prereqs: `iverilog`, `verilator`, `yosys`, `python3` + `numpy`/`pyyaml`.
