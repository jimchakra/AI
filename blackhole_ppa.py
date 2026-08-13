#!/usr/bin/env python3
"""
blackhole_ppa.py — $/token and J/token (energy) from PRODUCT-SHEET numbers, and an
HONEST read on the "king of PPA" question.

Every input is a stated, editable assumption. Cross-vendor absolute throughput is the
uncertain part (benchmarks are regime- and stack-dependent, and some are unverified);
the ROBUST part is the ratio on a fixed box — our technique roughly HALVES $/token and
J/token because both scale as 1/throughput at fixed CapEx and power.

Sources (2026): Tenstorrent cards page (p150a $1,399, 300 W, 664 BLOCKFP8 TFLOPS;
Galaxy from $110k). Spheron TT-vs-NVIDIA (Llama-70B batch-32 throughput; unverified TT).
NVIDIA DGX node power/price are public ballparks.
"""
# ---------------- shared economic assumptions (edit freely) ----------------
ELEC   = 0.10      # $/kWh (datacenter)
PUE    = 1.3       # facility overhead for ENERGY COST only
LIFE_H = 3*365*24  # 3-year amortization, 24/7
UTIL   = 0.85      # realistic sustained utilization (not 100%)

def econ(name, capex, node_kw, tok_s):
    """Return $/M-token and J/token at NODE power (fair, excludes facility for J/token)."""
    cap_per_h = capex/LIFE_H
    pwr_per_h = node_kw*PUE*ELEC
    usd_h = cap_per_h + pwr_per_h
    eff_tok_s = tok_s*UTIL
    usd_Mtok = usd_h/(eff_tok_s*3600)*1e6
    j_tok = node_kw*1000/tok_s                 # W / (tok/s) = J/token  (node power, peak tput)
    return dict(name=name, capex=capex, node_kw=node_kw, tok_s=tok_s,
                usd_h=usd_h, usd_Mtok=usd_Mtok, j_tok=j_tok)

# ---------------- throughput derived from datasheet AGGREGATE bandwidth ----------------
# Same bandwidth-bound model + same eta for ALL vendors (internally consistent, physics-based,
# not cherry-picked benchmarks). Llama-3-70B, batch 32, 8K context.
ETA = 0.48
B, S = 32, 8192
W_FP8 = 68e9                     # 70B weights, FP8, read once/step (shared over batch)
KV_SEQ = S * (2*8*128*80*2)      # per-user KV at 8K, FP16 (Llama-70B GQA), all layers
STEP_BYTES = W_FP8 + B*KV_SEQ
def tput(agg_bw): return B*ETA*agg_bw/STEP_BYTES   # tok/s aggregate
# datasheet aggregate DRAM bandwidth per system
BW_TT   = 32*512e9      # Galaxy Blackhole: 32 chips x 512 GB/s = 16.4 TB/s
BW_H100 = 8*3.35e12     # DGX H100: 8 x 3.35 TB/s HBM3  = 26.8 TB/s
BW_B200 = 8*8.0e12      # DGX B200: 8 x 8.0 TB/s HBM3e  = 64 TB/s
SYS = [
  econ("TT Galaxy Blackhole (baseline)",   110_000, 12.0, round(tput(BW_TT))),
  econ("NVIDIA DGX H100 (8 GPU)",          300_000, 10.2, round(tput(BW_H100))),
  econ("NVIDIA DGX B200 (8 GPU)",          500_000, 14.3, round(tput(BW_B200))),
]
# our technique on TT (long-context serving regime): ~2.07x throughput on the SAME box
TT = SYS[0]
TT_ENH = econ("TT Galaxy Blackhole + mem-reduction/KV-throttle", 110_000, 12.0, round(tput(BW_TT)*2.07))

if __name__ == "__main__":
    print("="*90)
    print("$/TOKEN and J/TOKEN from product sheets  (Llama-3-70B, batch-32; assumptions stated)")
    print(f"  elec ${ELEC}/kWh · PUE {PUE} · {LIFE_H/8760:.0f}-yr amortize · util {UTIL:.0%}")
    print("="*90)
    print(f"{'system':>46} | {'$/hr':>7} | {'tok/s':>6} | {'$/Mtok':>7} | {'J/token':>8}")
    print("-"*90)
    for s in SYS+[TT_ENH]:
        print(f"{s['name']:>46} | {s['usd_h']:6.1f} | {s['tok_s']:6d} | {s['usd_Mtok']:6.2f} | {s['j_tok']:7.2f}")

    print("\n"+"-"*90)
    print("ROBUST (ratio on the SAME box — no cross-vendor uncertainty):")
    print(f"  our technique: $/token x{TT['usd_Mtok']/TT_ENH['usd_Mtok']:.2f}  (${TT['usd_Mtok']:.2f} -> ${TT_ENH['usd_Mtok']:.2f} /M)")
    print(f"                 J/token x{TT['j_tok']/TT_ENH['j_tok']:.2f}  ({TT['j_tok']:.1f} -> {TT_ENH['j_tok']:.1f} J/tok)")
    print("  -> ~2x better $/token AND J/token, everything else equal. This is the defensible claim.")

    print("\n"+"-"*90)
    print("CROSS-VENDOR VERDICT (throughput bandwidth-derived, same eta; still a first-order read):")
    b200=SYS[2]
    print(f"  $/token : TT+ours ${TT_ENH['usd_Mtok']:.2f}  <  B200 ${b200['usd_Mtok']:.2f}  <  H100 ${SYS[1]['usd_Mtok']:.2f}"
          f"   -> TT+ours LEADS $/token ({b200['usd_Mtok']/TT_ENH['usd_Mtok']:.1f}x cheaper than best NVIDIA)")
    print(f"  J/token : B200 {b200['j_tok']:.1f}  <  TT+ours {TT_ENH['j_tok']:.1f}  <  H100 {SYS[1]['j_tok']:.1f}"
          f"   -> B200 LEADS energy; TT+ours beats H100 but NOT B200 ({TT_ENH['j_tok']/b200['j_tok']:.1f}x higher)")
    print("\n  So: KING of $/token, plausibly. NOT king of PPA overall. Honest caveats:")
    print("   - NVIDIA/AMD do their OWN KV reduction in software (paged-attn, KV-quant, spec-decode);")
    print("     fair fight is our HW-ENFORCED reduction vs their BEST, not vs naive.")
    print("   - NVIDIA wins raw Performance and likely PERF/AREA (advanced node, HBM 3-8 TB/s).")
    print("   - TT edge is $/token & J/token via cheap GDDR6 + low CapEx + open stack; our technique ~2x's it.")
    print("   - 'A' in PPA (area/node) is NOT ours to claim; our added logic is ~0 mm^2 but the base die isn't.")
    print("   - The 2x is the LONG-CONTEXT serving regime; short-context gets less.")
