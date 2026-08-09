#!/usr/bin/env python3
"""
input_gate_energy.py — the gate as a READ-MASK, not a return filter.

The θ gate produces a sparse FFN-hidden vector. That vector is the INPUT to the
down-projection matmul, so its survivor bitmap (already built by nmgr_encoder for
the compressed return) is exactly the set of down-projection weight columns worth
fetching. Reusing it as a read-mask skips WEIGHT reads — a 1st-order energy lever,
versus the ~0.01% the same gate saves when accounted only on the activation-return
path.

Quality (Δ perplexity) is the SAME curve measured by pytorch_cosim.py — zeroing a
hidden value is the same operation whether you call it "gate the output" or "skip
the column it would feed." Only the energy accounting changes. This script joins
the measured sparsity/quality (cosim/sweep.json) to a first-order decode-energy
budget (inference_energy_sim) and reports total energy saved vs θ.

Honest bounds: skips the down-projection only (~1/3 of weight reads for FFN ratio
4); the up-projection input is dense and attention weights are untouched, so the
ceiling is ~15% of total decode energy here. The bigger term is KV reads (~54% of
this budget) — attacking those needs a predictor (norm-bound / heavy-hitter),
not a bitmap. First-order model; quality measured on the proxy co-sim.
"""
import json, os
import inference_energy_sim as S

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(HERE, "cosim", "sweep.json")
PROJ_FRAC = 1.0/3.0          # down-proj share of weight-read energy (FFN ratio 4)

def main():
    sw = json.load(open(SWEEP))
    m  = S.ModelSpec()                       # 7B / int4 / HBM / 8K
    b  = S.near_memory(m)
    base = b.total
    proj_e = b.weights * PROJ_FRAC
    print(f"near-memory baseline: {base/1e9:.1f} mJ/token "
          f"(weights {b.weights/1e9:.1f}, kv {b.kv/1e9:.1f})")
    print(f"down-proj read energy (skippable via bitmap): {proj_e/1e9:.1f} mJ "
          f"= {100*proj_e/base:.1f}% of total\n")
    print(f"{'theta':>6} {'Δppl%':>8} {'FFN_sparse%':>12} {'TOTAL_energy_saved%':>20}")
    rows=[]
    for r in sw["rows"]:
        s = r["ffn_sparsity"]; saved = s*proj_e
        rows.append(dict(theta=r["theta"], dppl=r["ppl_delta_pct"],
                         sparsity=s, total_saved_pct=100*saved/base))
        print(f"{r['theta']:6.2f} {r['ppl_delta_pct']:8.2f} {100*s:12.1f} {100*saved/base:20.1f}")
    json.dump({"base_mJ":base/1e9,"proj_pct_of_total":100*proj_e/base,"rows":rows},
              open(os.path.join(HERE,"cosim","input_gate.json"),"w"), indent=2)
    print("\nFor comparison, the same gate accounted on the return path saves ~0.01% of total.")

if __name__ == "__main__":
    main()
