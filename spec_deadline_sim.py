#!/usr/bin/env python3
"""
spec_deadline_sim.py — speculative decoding as the *bounded* NACK.

Layer 1 (KV-trim) held the deadline by reading less KV — but that quality loss is
SILENT: nothing checks the degraded token, so an error propagates uncaught. The
tighter analog to a HARQ NACK is speculative decoding under a deadline, because it
adds a VERIFY step — the loss is detected and corrected, exactly like a
retransmission recovers a frame.

Setup: a real-time decoder owes one token every D ms. A cheap DRAFT model proposes
a token quickly (t_d, low jitter — think guaranteed resources); the expensive
TARGET verifies it (t_v = base * contention, heavy-tailed). The draft is right with
probability p_acc; when it's wrong the target would reject and correct it.

Three policies under a contention burst:

  A. verify-always (rigid): emit only after the target verify. Perfect quality
     (every token verified) but MISSES the deadline whenever t_d + t_v > D.

  B. KV-trim (silent degrade, Layer 1's lever): shrink the target compute so the
     verify fits D. It fits — but a trimmed verify sometimes fails to catch a wrong
     draft token, and that error is emitted with NO flag and NEVER revisited. Live
     uncorrected error ACCUMULATES.

  C. speculative-NACK (this build): if the verify won't fit D, emit the DRAFT token
     unverified, on time, and push it to a verification backlog. On any later slot
     that has slack, one target pass verifies the whole pending block (parallel
     verification — the speculative-decoding trick) and corrects what was wrong.
     Live uncorrected error SPIKES during the burst, then DRAINS to ~0 — bounded and
     self-correcting, the HARQ property.

The point is not throughput; it is that C's loss is *detectable and correctable*
while B's is *silent and compounding*, at the same (near-100%) deadline-hit.
First-order, illustrative; contention is the same heavy-tailed control-plane term
used across the study.

WHAT THE DRAFT IS (reframe, matching the article's §05 note): at long context the
effective draft is not a tiny second model (that attacks WEIGHT bytes, the wrong
term here) but THE TARGET ITSELF on a content-selected KV subset (TriForce-style)
— i.e. the study's own KV-selector run at a deeper budget. Consistently, this
sim's draft is ~30% of the target's cost (t_d=3 vs base_v=10), not ~2%, and it is
low-jitter because the selected working set fits the guaranteed (SRAM) partition
while the full-KV verify rides the contended pool. Under that reading, Layer 1's
KV-trim and this draft emit the SAME token; B is C with the verify deleted.
p_acc is then not arbitrary — it is a function of the selection budget f
(read more, propose better): see p_acc_from_budget() below, a stylized proxy off
the measured Quest budget->Δppl curve. Defaults keep p_acc fixed at 0.75 so the
published curves are unchanged.
"""
import random, json, os

# Quest budget -> Δppl% (measured, from the study); stylized mapping to acceptance.
_QF = [1.000, 0.749, 0.434, 0.229, 0.115]
_QQ = [0.00,  0.12,  0.97,  4.09,  12.97]
def p_acc_from_budget(f, alpha0=0.98, beta=0.035):
    """Stylized proxy: acceptance decays with the draft's quality gap to the target.
    alpha0 = self-agreement ceiling at full budget; beta sets decay per Δppl%.
    The SHAPE (deeper trim -> lower acceptance) is the claim, not the constants."""
    f = max(_QF[-1], min(1.0, f))
    for i in range(len(_QF) - 1):
        if _QF[i] >= f >= _QF[i + 1]:
            t = (_QF[i] - f) / (_QF[i] - _QF[i + 1])
            dppl = _QQ[i] + t * (_QQ[i + 1] - _QQ[i])
            break
    else:
        dppl = _QQ[-1]
    import math
    return alpha0 * math.exp(-beta * dppl)

random.seed(20260810)

def contention(sigma, boost=1.0):
    mu = -sigma * sigma / 2.0          # E=1 at boost=1
    return boost * random.lognormvariate(mu, sigma)

def run(N=900, D=18.0, t_d=3.0, base_v=10.0, sigma=0.30,
        p_acc=0.75, burst=(200, 500), burst_boost=2.2, drain_k=6, seed=None):
    """Returns per-slot time series and summary for policies A, B, C."""
    if seed is not None:
        random.seed(seed)
    A = dict(miss=0, live=[], uncorr=0)
    B = dict(miss=0, live=[], uncorr=0)          # silent uncorrected accumulates
    C = dict(miss=0, live=[], uncorr=0, backlog=0, backlog_wrong=0, detected=0)
    for i in range(N):
        boost = burst_boost if burst[0] <= i < burst[1] else 1.0
        M = contention(sigma, boost)
        t_v = base_v * M
        draft_wrong = (random.random() > p_acc)

        # ---- A: verify-always ----
        if t_d + t_v > D:
            A['miss'] += 1
        A['live'].append(0)                      # always verified before emit

        # ---- B: KV-trim to fit; trimmed verify may silently miss a wrong token ----
        need = t_d + t_v
        if need > D:
            # fraction of the verify we had to cut to fit
            cut = min(1.0, (need - D) / max(1e-9, t_v))
            miss_detect = 0.55 * cut             # deeper trim -> more silent misses
            if draft_wrong and random.random() < miss_detect:
                B['uncorr'] += 1                 # silent, never revisited
        # (if it fit, verify catches the wrong token; if draft right, nothing to catch)
        B['live'].append(B['uncorr'])

        # ---- C: speculative-NACK with deferred parallel verify ----
        if t_d + t_v <= D:
            # slack this slot: run a full verify pass -> verify current + drain backlog block
            if draft_wrong:
                pass                             # caught now, corrected before emit
            # drain up to drain_k pending, detecting (and correcting) the wrong ones
            drain = min(C['backlog'], drain_k)
            # wrong ones among the drained are now DETECTED/corrected
            if C['backlog'] > 0:
                frac_wrong = C['backlog_wrong'] / C['backlog']
                det = round(drain * frac_wrong)
                C['detected'] += det
                C['backlog_wrong'] -= det
                C['backlog'] -= drain
                if C['backlog_wrong'] < 0: C['backlog_wrong'] = 0
        else:
            # spike: emit draft unverified on time, defer verification
            C['backlog'] += 1
            if draft_wrong:
                C['backlog_wrong'] += 1
        C['live'].append(C['backlog_wrong'])     # wrong & not-yet-verified = live uncorrected

    for P in (A, B, C):
        P['hit'] = 100.0 * (N - P['miss']) / N
    C['end_uncorr'] = C['backlog_wrong']
    return dict(A=A, B=B, C=C, N=N, D=D, burst=burst)

if __name__ == "__main__":
    r = run()
    A, B, C = r['A'], r['B'], r['C']
    print("=" * 74)
    print("SPECULATIVE DECODING as the bounded NACK  (burst slots %d–%d)" % r['burst'])
    print("=" * 74)
    print(f"{'policy':<24}{'deadline-hit':>13}{'end uncorrected':>17}{'detected/corrected':>20}")
    print(f"{'A verify-always':<24}{A['hit']:12.1f}%{0:17d}{'n/a (misses deadlines)':>20}")
    print(f"{'B KV-trim (silent)':<24}{B['hit']:12.1f}%{B['uncorr']:17d}{'0 (no verify)':>20}")
    print(f"{'C speculative-NACK':<24}{C['hit']:12.1f}%{C['end_uncorr']:17d}{C['detected']:20d}")
    print()
    print(f"A missed {A['miss']} deadlines; B & C met ~all.")
    print(f"B leaves {B['uncorr']} SILENT uncorrected errors in the stream (never revisited).")
    print(f"C detected+corrected {C['detected']} draft errors via deferred verify; "
          f"{C['end_uncorr']} still pending at end (drains to 0 with more slack).")
    # peak live uncorrected during burst
    peakB = max(B['live']); peakC = max(C['live'])
    print(f"Peak LIVE uncorrected during burst — B: {peakB}  C: {peakC} (C then drains; B does not).")

    json.dump(dict(A={k: A[k] for k in ('hit', 'miss', 'live')},
                   B={k: B[k] for k in ('hit', 'uncorr', 'live')},
                   C={k: C[k] for k in ('hit', 'end_uncorr', 'detected', 'live')},
                   N=r['N'], D=r['D'], burst=r['burst']),
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "spec_deadline_ref.json"), "w"))
    print("\nwrote spec_deadline_ref.json")
