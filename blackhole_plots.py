#!/usr/bin/env python3
"""blackhole_plots.py — figures for the Blackhole system model.
Palette: dataviz reference categorical hues (validated), light surface.
One y-axis per panel; thin marks; recessive grid; selective direct labels."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from blackhole_sysmodel import SystemSpec, M8B, step, bytes_per_step, flops_per_step, Cut, max_batch, serving_throughput
from capacity_sim import qcost
import capacity_sim as Q

# --- palette (dataviz reference, light) ---
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e2"
mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 10.5, "font.family": "DejaVu Sans", "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
})
sys, m = SystemSpec(), M8B


# ============================================================ FIG 1: ROOFLINE
def fig_roofline():
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ridge = sys.ridge
    ai = np.logspace(-0.5, 3.7, 400)
    perf = np.minimum(sys.fp8_flops, sys.bw_bytes_s * ai) / 1e12   # TFLOPS
    ax.loglog(ai, perf, color=BLUE, lw=2.2, zorder=3)
    ax.axvline(ridge, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.text(ridge*0.92, 2.3, f"ridge  AI*={ridge:,.0f}", rotation=90,
            ha="right", va="bottom", color=MUTED, fontsize=9)
    ax.fill_betweenx([1, sys.fp8_flops/1e12], 0.3, ridge, color=BLUE, alpha=0.05, zorder=0)
    ax.text(0.42, 330, "memory-bound region\n(all decode lives here)", color=BLUE,
            fontsize=9.5, ha="left", va="top")

    # batch-1 operating points
    for S in (2048, 8192, 32768, 131072):
        r = step(sys, m, 1, S)
        ax.plot(r['ai'], (r['flops']/r['t'])/1e12, 'o', color=ORANGE, ms=7, zorder=5, mec=SURF, mew=1.2)
    # label the batch-1 cluster once (text sits above-left, arrow down to cluster)
    rb = step(sys, m, 1, 8192)
    ax.annotate("batch-1 decode\n2–5 FLOP/byte\n(~500× below ridge)",
                xy=(rb['ai'], (rb['flops']/rb['t'])/1e12), xytext=(0.6, 6.0),
                color=ORANGE, fontsize=9.2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.0))

    # batched serving: baseline (full KV, B0) and throttled (57% cut, B1)
    op = Cut(0.57, 0.13, 16)
    g = serving_throughput(sys, m, 8192, op)
    r0 = step(sys, m, g['B0'], 8192)
    r1 = step(sys, m, g['B1'], 8192, op)
    for r, c, lab in ((r0, YELLOW, f"serving, full KV (B={g['B0']})"),
                      (r1, AQUA, f"serving, 57% cut (B={g['B1']})")):
        p = (r['flops']/r['t'])/1e12
        ax.plot(r['ai'], p, 'D', color=c, ms=9, zorder=6, mec=SURF, mew=1.3)
        ax.annotate(lab, xy=(r['ai'], p), xytext=(r['ai']*1.15, p*0.55),
                    color=c, fontsize=9.5, arrowprops=dict(arrowstyle="-", color=c, lw=1.0))

    ax.set_xlabel("arithmetic intensity  (FLOP / byte)")
    ax.set_ylabel("achievable compute  (TFLOPS)")
    ax.set_title("Blackhole roofline — decode is bandwidth-bound; throttling walks it up, never off")
    ax.set_ylim(1, sys.fp8_flops/1e12*1.6)
    ax.set_xlim(0.3, 5000)
    fig.text(0.5, -0.02, "512 GB/s · 745 TFLOPS FP8 · ridge 1,455 FLOP/byte. Even a 95% KV cut reaches only AI≈47 — the throttle keeps paying because the ridge is never reached.",
             ha="center", color=MUTED, fontsize=8.2)
    fig.tight_layout()
    fig.savefig("fig_roofline.png", dpi=150, bbox_inches="tight")
    print("wrote fig_roofline.png")


# ================================================ FIG 2: CUT -> THROUGHPUT vs QUALITY
def fig_cut():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    fkv = np.linspace(0, 0.95, 40)
    gains, dppl = [], []
    for f in fkv:
        g = serving_throughput(sys, m, 8192, Cut(float(f), 0.13, 16))
        gains.append(g['gain']); dppl.append(qcost(1 - f))
    gains, dppl = np.array(gains), np.array(dppl)

    a1.plot(fkv*100, gains, color=BLUE, lw=2.2, zorder=3)
    for f, lab, col in ((0.57, "safe ~1% ppl", AQUA), (0.70, "aggressive ~3%", ORANGE)):
        g = serving_throughput(sys, m, 8192, Cut(f, 0.13, 16))['gain']
        a1.plot(f*100, g, 'o', color=col, ms=9, mec=SURF, mew=1.3, zorder=5)
        a1.annotate(f"{lab}\n{g:.2f}×", (f*100, g), (f*100-3, g+0.15), color=col, fontsize=9.2, ha="right")
    a1.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    a1.set_xlabel("KV read cut  (%)"); a1.set_ylabel("serving throughput  (× baseline)")
    a1.set_title("What the cut buys (single chip)")
    a1.set_ylim(0.9, gains.max()*1.08)

    a2.plot(fkv*100, dppl, color=ORANGE, lw=2.2, zorder=3)
    a2.axhspan(0, 1.0, color=AQUA, alpha=0.10, zorder=0)
    a2.text(2, 0.55, "safe band (≤1% ppl)", color=AQUA, fontsize=9)
    for f, col in ((0.57, AQUA), (0.70, ORANGE)):
        a2.plot(f*100, qcost(1-f), 'o', color=col, ms=9, mec=SURF, mew=1.3, zorder=5)
    a2.set_xlabel("KV read cut  (%)"); a2.set_ylabel("perplexity cost  (Δppl %)")
    a2.set_title("The ceiling is quality (measured, Quest / Qwen2.5-1.5B)")
    a2.set_ylim(0, 25)
    fig.suptitle("Cut → throughput is bounded by the quality curve, not the compute wall",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("fig_cut_quality.png", dpi=150, bbox_inches="tight")
    print("wrote fig_cut_quality.png")


# ================================================ FIG 3: CAPACITY + QoS COLLAPSE
def fig_cap_qos():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    # capacity
    ctx = [4096, 8192, 16384, 32768, 65536, 131072]
    full = [max_batch(sys, m, S, 1.0) for S in ctx]
    cut = [max_batch(sys, m, S, 0.43) for S in ctx]
    x = np.arange(len(ctx))
    a1.plot(x, full, '-o', color=BLUE, lw=2, ms=6, mec=SURF, mew=1.2, label="full KV")
    a1.plot(x, cut, '-o', color=AQUA, lw=2, ms=6, mec=SURF, mew=1.2, label="57% KV cut")
    for xi, (f, c) in enumerate(zip(full, cut)):
        if xi in (1, 3):
            a1.annotate(f"{c/max(1,f):.1f}×", (x[xi], cut[xi]), (x[xi], cut[xi]+3), color=AQUA, fontsize=9, ha="center")
    a1.set_xticks(x); a1.set_xticklabels([f"{c//1024}K" for c in ctx])
    a1.set_xlabel("context length"); a1.set_ylabel("max sequences that fit 32 GB")
    a1.set_title("KV pruning enlarges the feasible batch")
    a1.legend(frameon=False, fontsize=9.5, loc="upper right")

    # QoS capacity vs percentile
    S = 8192; B = max(1, max_batch(sys, m, S, 1.0))
    from blackhole_qos import derive_service
    c, k0 = derive_service(sys, m, S, B, 32); D = 4.0*(c+k0); mu = 1.0/(c+k0)
    lams = [0.15*mu + 0.06*mu*i for i in range(22)]
    A = Q.sweep('A', lams, D=D, c=c, k0=k0, sigma=0.7)
    Bs = Q.sweep('B', lams, D=D, c=c, k0=k0, sigma=0.7)
    targets = [0.90, 0.95, 0.99, 0.995, 0.998, 0.999]
    capA = [Q.capacity_at(A, t)*1000 for t in targets]
    capB = [Q.capacity_at(Bs, t)*1000 for t in targets]
    xt = np.arange(len(targets))
    a2.plot(xt, capA, '-o', color=BLUE, lw=2, ms=6, mec=SURF, mew=1.2, label="hard quality (Policy A)")
    a2.plot(xt, capB, '-o', color=ORANGE, lw=2, ms=6, mec=SURF, mew=1.2, label="anytime KV-truncate (Policy B)")
    a2.fill_between(xt, capA, capB, color=ORANGE, alpha=0.08)
    a2.annotate("A collapses:\ncan't meet the tail", (xt[-2], capA[-2]), (xt[-2]-1.6, max(capB)*0.5),
                color=BLUE, fontsize=9, arrowprops=dict(arrowstyle="-", color=BLUE, lw=1))
    a2.set_xticks(xt); a2.set_xticklabels([f"P{t*100:g}" for t in targets])
    a2.set_xlabel("latency-percentile guarantee"); a2.set_ylabel("sustainable load  (chunks/s)")
    a2.set_title("Tightening the tail collapses hard-quality capacity")
    a2.legend(frameon=False, fontsize=9.5, loc="upper right")
    fig.suptitle("Two more system effects: capacity relief (left) and tail-robust throughput (right)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("fig_capacity_qos.png", dpi=150, bbox_inches="tight")
    print("wrote fig_capacity_qos.png")


if __name__ == "__main__":
    fig_roofline(); fig_cut(); fig_cap_qos()
