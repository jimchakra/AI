#!/usr/bin/env python3
"""
capacity_sim.py — the throughput payoff of graceful degradation, and tiered QoS.

Jim's argument, made concrete:

  Holding a LATENCY-PERCENTILE guarantee fixed, an anytime-degrade policy lets you
  run at far higher load. Tightening the guarantee (P99 -> P99.8) collapses the
  load you can sustain, because you are buying insurance against a heavy tail with
  idle headroom. Keep the SAME percentile but truncate the tail with an anytime
  degrade, and you reclaim that headroom as throughput — the cost migrates from
  MISSED DEADLINES to slightly DEGRADED OUTPUTS.

Not advocating an unstable system with a band-aid: the first line of defense is
mixed-criticality partitioning (tight tasks on guaranteed-resource cores — tiny
cores + SRAM scratchpad, no cache, deterministic WCET). Graceful degradation is
the BACKSTOP for the residual tail that partitioning can't kill — and that
backstop is what unlocks throughput and, with priority, tiered QoS.

Model: single server (M/G/1). Service time S = c + k0*M, where c is irreducible
compute, k0 is the full KV-read time, and M is a heavy-tailed CONTENTION multiplier
(shared memory-BW / scheduler jitter — the control-plane uncertainty). Deadline D
is on total latency (queue wait + service).

  Policy A (hard / full quality): always read full KV. A request is on-time if
    wait + (c + k0*M) <= D.  Its high-percentile latency is dominated by the tail
    of M and by queueing near capacity.

  Policy B (anytime truncate): the server reads KV until the deadline, then stops.
    Given wait W it has budget = D - W; it does the mandatory compute c, then reads
    a fraction f = clamp((budget - c)/(k0*M), 0..1) of KV, finishing at or before D.
    Latency is capped at D by construction; the cost is quality loss q(f), measured
    off the Quest curve. A true miss only happens when W + c > D (queue so deep the
    request can't even start in time) — pushed to much higher load because degrading
    shrinks service and thus the queue.

  Tiered QoS (priority + degrade): a fraction `prem` of load is PREMIUM (never
    degrade, hard SLO) and is served ahead of ECONOMY (degrade-first). Shows the
    premium percentile held while economy absorbs the degradation — the sellable
    version of the throughput gain.

Quality axis is the real measured Quest curve (Qwen2.5-1.5B) — read_fraction -> Δppl%.
Absolute times are calibrated; the SHAPES are the claim.
"""
import random, json, os, bisect

random.seed(20260810)

# Quest: read_fraction -> quality cost (Δppl %). Piecewise-linear; f<0.115 extrapolates to ~30.
_QF = [1.000, 0.749, 0.434, 0.229, 0.115, 0.0]
_QQ = [0.00,  0.12,  0.97,  4.09,  12.97, 30.0]
def qcost(f):
    f = max(0.0, min(1.0, f))
    for i in range(len(_QF) - 1):
        hi, lo = _QF[i], _QF[i + 1]
        if f <= hi and f >= lo:
            t = (hi - f) / (hi - lo) if hi > lo else 0.0
            return _QQ[i] + t * (_QQ[i + 1] - _QQ[i])
    return 0.0

def lognorm_contention(sigma):
    mu = -sigma * sigma / 2.0           # E[M] = 1, median = exp(mu) < 1
    return random.lognormvariate(mu, sigma)

def simulate(lam, policy, N=120000, D=50.0, c=3.0, k0=7.0, sigma=0.7, warm=2000):
    """lam in requests/ms. Returns dict: ontime, degraded, mean_q (over all), p999 latency (A)."""
    dep_prev = 0.0; t = 0.0
    ontime = 0; deg = 0; qsum = 0.0; lat_tail = []
    for i in range(N):
        t += random.expovariate(lam)
        start = t if t > dep_prev else dep_prev
        W = start - t
        M = lognorm_contention(sigma)
        full = c + k0 * M
        if policy == 'A':
            svc = full
        else:  # B: anytime truncate at deadline
            budget = D - W
            if budget <= c:
                f = 0.0; svc = c            # can't fit mandatory compute -> miss
            else:
                f = min(1.0, (budget - c) / (k0 * M))
                svc = c + f * k0 * M
            if f < 0.999: deg += 1
            qsum += qcost(f)
        dep = start + svc
        lat = dep - t
        dep_prev = dep
        if i < warm:  # discard warmup
            continue
        if lat <= D + 1e-9:
            ontime += 1
        if policy == 'A':
            lat_tail.append(lat)
    n = N - warm
    out = dict(lam=lam, ontime=ontime / n, degraded=deg / N, mean_q=qsum / N)
    if lat_tail:
        lat_tail.sort()
        out['p99'] = lat_tail[int(0.99 * len(lat_tail))]
        out['p999'] = lat_tail[min(len(lat_tail) - 1, int(0.999 * len(lat_tail)))]
    return out

def sweep(policy, lams, **kw):
    return [simulate(lam, policy, **kw) for lam in lams]

def capacity_at(sweeprows, target):
    """Max lam (req/ms) whose on-time fraction >= target, by linear interp between sweep points."""
    best = 0.0
    prev = None
    for r in sweeprows:
        if r['ontime'] >= target:
            best = r['lam']
        if prev is not None and prev['ontime'] >= target > r['ontime']:
            # interpolate the crossing
            f = (prev['ontime'] - target) / (prev['ontime'] - r['ontime'] + 1e-12)
            best = prev['lam'] + f * (r['lam'] - prev['lam'])
            break
        prev = r
    return best

# ---------- Tiered QoS: priority (premium served first, never degrades) ----------
def simulate_tiers(lam, prem_frac, N=120000, D=50.0, c=3.0, k0=7.0,
                   sigma=0.7, sigma_prem=0.15, warm=2000):
    """Two priority classes sharing one server. Premium: full quality, served ahead,
    AND on low-contention (guaranteed) resources (sigma_prem << sigma) — point 1.
    Economy: degrade-first (anytime truncate) on the contended pool.
    Returns premium on-time, economy on-time, economy degraded frac, economy mean_q."""
    # generate arrivals; contention multiplier depends on class (premium = guaranteed resources)
    t = 0.0; arr = []
    for i in range(N):
        t += random.expovariate(lam)
        if random.random() < prem_frac:
            arr.append((t, 'P', lognorm_contention(sigma_prem)))
        else:
            arr.append((t, 'E', lognorm_contention(sigma)))
    # non-preemptive priority queue simulation
    busy_until = 0.0
    # process in arrival order but when server frees, pick highest-priority waiting job.
    # simpler: event loop over arrivals with a waiting pool
    pool = []  # waiting (premium first). store (prio, seq, arrtime, M, cls)
    seq = 0
    pmet = pe = 0; pn = en = 0; edeg = 0; eq = 0.0
    idx = 0; now = 0.0
    # We simulate by advancing through arrivals and draining when server is free.
    events = sorted(arr, key=lambda x: x[0])
    ei = 0
    server_free = 0.0
    # discrete: we push arrivals into pool at their time, and serve greedily
    import collections
    while ei < len(events) or pool:
        if pool and (ei >= len(events) or server_free <= events[ei][0]):
            # serve next (premium prio 0 before economy prio 1), FIFO within class
            pool.sort(key=lambda x: (x[0], x[1]))
            prio, s, at, M, cls = pool.pop(0)
            start = max(server_free, at); W = start - at
            if cls == 'P':
                svc = c + k0 * M
                dep = start + svc; lat = dep - at; server_free = dep
                if s >= warm:
                    pn += 1
                    if lat <= D + 1e-9: pmet += 1
            else:
                budget = D - W
                if budget <= c: f = 0.0; svc = c
                else: f = min(1.0, (budget - c) / (k0 * M)); svc = c + f * k0 * M
                dep = start + svc; lat = dep - at; server_free = dep
                if s >= warm:
                    en += 1; eq += qcost(f)
                    if f < 0.999: edeg += 1
                    if lat <= D + 1e-9: pe += 1
        else:
            at, cls, M = events[ei]; ei += 1
            pool.append((0 if cls == 'P' else 1, seq, at, M, cls)); seq += 1
    return dict(lam=lam, prem_ontime=pmet / max(1, pn), econ_ontime=pe / max(1, en),
                econ_degraded=edeg / max(1, en), econ_q=eq / max(1, en), prem_frac=prem_frac)

# =====================================================================
if __name__ == "__main__":
    D, c, k0, sigma = 50.0, 3.0, 7.0, 0.7
    mu = 1.0 / (c + k0)  # nominal service rate (req/ms) at M=1  -> 0.1/ms = 100/s
    lams = [round(0.02 + 0.006 * i, 4) for i in range(20)]  # 0.02..0.134 /ms
    A = sweep('A', lams, D=D, c=c, k0=k0, sigma=sigma)
    B = sweep('B', lams, D=D, c=c, k0=k0, sigma=sigma)

    print("=" * 78)
    print("CAPACITY vs PERCENTILE GUARANTEE  (service ~10 ms nominal, D=50 ms, heavy tail)")
    print("=" * 78)
    print(f"{'target':>8} | {'A: hard (req/s)':>16} | {'B: anytime (req/s)':>19} | {'B gain':>7} | {'B degraded%@cap':>16}")
    for target in (0.90, 0.95, 0.99, 0.995, 0.998, 0.999):
        ca = capacity_at(A, target) * 1000
        cb = capacity_at(B, target) * 1000
        # degradation tax at B's capacity
        capB = capacity_at(B, target)
        # nearest sweep row
        row = min(B, key=lambda r: abs(r['lam'] - capB))
        print(f"P{target*100:<6.1f} | {ca:16.0f} | {cb:19.0f} | {cb/max(1,ca):6.2f}x | {row['degraded']*100:15.1f}%")

    print("\nInterpretation: as the percentile tightens, A's sustainable load collapses")
    print("(insurance against the heavy contention tail). B holds the SAME percentile at")
    print("much higher load by truncating the tail into bounded quality loss.\n")

    print("=" * 78)
    print("TIERED QoS  (priority: premium never degrades & is served first; economy degrades)")
    print("=" * 78)
    print(f"{'load(req/s)':>11} | {'premium on-time':>15} | {'economy on-time':>15} | {'econ degraded%':>14} | {'econ Δppl':>9}")
    for lam in (0.06, 0.09, 0.11, 0.13, 0.15):
        r = simulate_tiers(lam, prem_frac=0.3, D=D, c=c, k0=k0, sigma=sigma)
        print(f"{lam*1000:11.0f} | {r['prem_ontime']*100:14.2f}% | {r['econ_ontime']*100:14.2f}% | "
              f"{r['econ_degraded']*100:13.1f}% | {r['econ_q']:9.2f}")

    # save reference
    out = dict(D=D, c=c, k0=k0, sigma=sigma, nominal_rate_rps=mu * 1000,
               sweepA=A, sweepB=B,
               capacity=[dict(target=t,
                              A_rps=capacity_at(A, t) * 1000,
                              B_rps=capacity_at(B, t) * 1000)
                         for t in (0.90, 0.95, 0.99, 0.995, 0.998, 0.999)])
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "capacity_sim_ref.json"), "w"), indent=2)
    print("\nwrote capacity_sim_ref.json")
