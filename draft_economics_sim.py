#!/usr/bin/env python3
"""
draft_economics_sim.py — tiny-model draft vs KV-selected draft: who drafts better?

Question (from the §05 reframe): the classic draft is a tiny second model (~2% of
the target's cost); a KV-selected draft (the target on a selected KV subset) costs
~20–50%. Is the expensive draft ever rational?

Method — exact economics, labeled assumptions:
  The speculative-decoding speedup formula (Leviathan et al.) is EXACT given a
  per-token acceptance rate a and a draft cost c (fraction of one target pass):
      E[accepted per block] = (1 - a^(k+1)) / (1 - a)      (incl. the bonus token)
      cost per block        = k*c + 1                       (k drafts + 1 verify)
      speedup vs target-alone = E / (k*c + 1),  k chosen optimally per (c, a).
  What is NOT exact is where each draft family sits on the (c, a) plane — those
  operating points are stated assumptions:
    * tiny draft, short context:  c ~ 0.03 (weights ratio), a ~ 0.75 (published range)
    * tiny draft, long context:   c grows (its own KV or a window) and a FALLS on
      context-dependent tokens (it cannot see the long context): c ~ 0.08, a ~ 0.55
    * KV-draft (target + selector): c ~ f * (KV share) + weight share amortized;
      a = stylized proxy off the measured Quest budget->Δppl curve (deeper trim,
      lower acceptance). Long context is exactly where it lives.
  EQUAL-KV-BYTE framing (the sharp comparison): at the same KV bytes read per
  drafted token, the tiny draft brings small-model capacity with a blind window;
  the KV-draft brings full-model weights with the IMPORTANT context selected.
  The acceptance gap between those two is the whole contest.

Deadline tie-in (spec_deadline_sim): higher acceptance also means fewer deferred
corrections per burst — backlog growth rate scales with (1 - a) — so the KV-draft
does not just match throughput, it shrinks the unverified window the §05 bound
depends on.

DIRECTIONAL: the formula is exact; the operating points are assumptions to react
to, not measurements. Swap in your own (c, a) and the chart re-answers.
"""
import math

def speedup(c, a, kmax=16):
    best = (1.0, 1)
    for k in range(1, kmax + 1):
        E = (1 - a ** (k + 1)) / (1 - a) if a < 1 else k + 1
        s = E / (k * c + 1)
        if s > best[0]:
            best = (s, k)
    return best  # (speedup, optimal block length)

# acceptance proxy off the Quest curve (same as spec_deadline_sim.p_acc_from_budget)
_QF = [1.000, 0.749, 0.434, 0.229, 0.115]
_QQ = [0.00, 0.12, 0.97, 4.09, 12.97]
def a_of_budget(f, alpha0=0.98, beta=0.035):
    f = max(_QF[-1], min(1.0, f))
    for i in range(len(_QF) - 1):
        if _QF[i] >= f >= _QF[i + 1]:
            t = (_QF[i] - f) / (_QF[i] - _QF[i + 1])
            d = _QQ[i] + t * (_QQ[i + 1] - _QQ[i]); break
    else:
        d = _QQ[-1]
    return alpha0 * math.exp(-beta * d)

# KV-draft cost model at long context: KV share of a target pass ~ 0.7 at 32K+,
# weights ~0.3 but amortized/streamed regardless; draft reads f of the KV.
def c_kv_draft(f, kv_share=0.7):
    return (1 - kv_share) * 1.0 * 0 + kv_share * f + 0.05   # +0.05 overhead; weights reuse resident tile

CASES = [
    # label                          c        a                  regime note
    ("tiny draft · short ctx",       0.03,    0.75,              "weights-bound; classic sweet spot"),
    ("tiny draft · long ctx",        0.08,    0.55,              "context-blind: a collapses"),
    ("KV-draft f=23% · long ctx",    c_kv_draft(0.229), a_of_budget(0.229), "deep trim; Quest 23%-kept point"),
    ("KV-draft f=43% · long ctx",    c_kv_draft(0.434), a_of_budget(0.434), "mid trim"),
    ("KV-draft f=75% · long ctx",    c_kv_draft(0.749), a_of_budget(0.749), "shallow trim (Layer-1 grade)"),
]

print("NOTE: formula exact; operating points are stated assumptions. Long context = KV-bound.")
print("=" * 88)
print(f"{'draft':<28}{'cost c':>8}{'accept a':>10}{'speedup':>9}{'k*':>4}{'corr/tok (1-a)':>16}")
print("-" * 88)
for label, c, a, note in CASES:
    s, k = speedup(c, a)
    print(f"{label:<28}{c:8.3f}{a:10.3f}{s:9.2f}x{k:4d}{(1-a):16.3f}   {note}")

print()
print("READING:")
print(" - Short context: the tiny draft wins — that is its home regime; nothing to fix.")
print(" - Long context: the tiny draft's acceptance collapse costs more than its cheapness")
print("   saves; the KV-draft at a deep trim matches or beats it on throughput AND cuts")
print("   correction traffic (1-a) by ~3x -> smaller unverified backlog under bursts,")
print("   which is the quantity the Sec.05 bound actually depends on.")
print(" - Equal-KV-bytes framing: same bytes buy 'small model + blind window' or")
print("   'full model + selected context'. The contest is acceptance; selection wins it")
print("   wherever long-range context matters — measure a on a real pair to go beyond proxy.")

# contour plot: speedup on the (c, a) plane with the cases marked
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    C = np.linspace(0.01, 0.8, 160); A = np.linspace(0.45, 0.99, 160)
    G = np.array([[speedup(c, a)[0] for c in C] for a in A])
    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=150)
    cs = ax.contourf(C, A, G, levels=[1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4], cmap="Greens", alpha=.85)
    fig.colorbar(cs, label="speculative speedup vs target-alone (×)")
    lines = ax.contour(C, A, G, levels=[1.5, 2, 2.5, 3], colors="#2f3e50", linewidths=.7)
    ax.clabel(lines, fmt="%.1fx", fontsize=8)
    marks = {"tiny draft · short ctx": ("o", "#8a5a1a"), "tiny draft · long ctx": ("x", "#a4443a"),
             "KV-draft f=23% · long ctx": ("^", "#2e7d5b"), "KV-draft f=43% · long ctx": ("s", "#2e7d5b"),
             "KV-draft f=75% · long ctx": ("D", "#2e7d5b")}
    for label, c, a, note in CASES:
        m, col = marks[label]
        ax.scatter([c], [a], marker=m, s=70, color=col, zorder=5)
        ax.annotate(label, (c, a), textcoords="offset points", xytext=(8, 6), fontsize=8.5, color=col)
    ax.set_xlabel("draft cost c (fraction of one target pass)")
    ax.set_ylabel("acceptance rate a")
    ax.set_title("Draft economics: cost vs acceptance (formula exact; points are assumptions)\n"
                 "Tiny drafts sit left-low at long context; KV-selected drafts sit right-high.")
    fig.tight_layout(); fig.savefig("draft_economics.png")
    print("\nsaved plot -> draft_economics.png")
except Exception as e:
    print("plot skipped:", e)
