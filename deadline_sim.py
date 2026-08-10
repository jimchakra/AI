#!/usr/bin/env python3
"""
deadline_sim.py — "NACK, don't hang": graceful degradation under a hard deadline,
illustrated at two layers of an AI inference stack.

The modem principle (from HARQ): you cannot meet a hard deadline by
over-provisioning against an unbounded worst case (WCET / P100). The channel
tail is unbounded, so a robust receiver detects the impending miss EARLY and
emits a cheap, protocol-legal, bounded-loss fallback (a NACK -> one retransmit)
that the larger system is designed to absorb — trading a small known loss to
eliminate the unbounded, cascading failure (a stall that times out up the stack
into an RRC re-establishment / dropped call).

This script reproduces that pattern in two inference regimes and prints the
numbers the interactive page renders live.

LAYER 1 — token-deadline anytime KV degradation (single real-time stream).
  A streaming decoder (voice / live-translation / agent-in-the-loop) owes one
  token every D ms. Per-token time = fixed compute + KV read (grows with
  context) + heavy-tailed jitter. The full-KV read eventually cannot fit D
  (the P100 wall) and jitter blows even mid-context tokens.
    - RIGID: always read full KV. Late tokens accumulate lag -> the stream
      freezes (the "hang").
    - ROBUST: a controller predicts the token's time and, if it won't fit,
      tightens the KV budget to the largest that fits — emitting a slightly
      worse token ON TIME. The quality cost is your MEASURED Quest curve
      (budget -> Δppl). The degraded on-time token is the NACK.

LAYER 2 — serving-level admission control (bursty request queue).
  Requests arrive with a TTFT SLO. Over-provisioning P100 is impossible under
  bursts.
    - RIGID: unbounded queue; under overload latency explodes, every request
      misses SLO, and timed-out clients RETRY -> congestion collapse: ~100%
      utilization doing ~0 useful goodput (the "hang").
    - ROBUST: admission control sheds (503 + backoff = a NACK) when the
      predicted wait exceeds the SLO; admitted work meets SLO; goodput stays
      high. The shed request is the NACK.

First-order, illustrative models; the Layer-1 quality axis is grounded in the
real measured Quest data (m4_results.json). Absolute times are calibrated,
not measured.
"""
import random, math, json, os

random.seed(20260810)  # deterministic; no Date/random-at-import surprises

# ----- Quest budget -> (read_fraction, quality_cost Δppl%) from the M4 run -----
# (Qwen2.5-1.5B, page=16; kv_kept is the fraction of KV actually read.)
QUEST_LEVELS = [
    # budget, read_fraction (kv_kept), quality_cost (Δppl %)
    (2048, 1.000, 0.00),
    (1024, 0.749, 0.12),
    ( 512, 0.434, 0.97),
    ( 256, 0.229, 4.09),
    ( 128, 0.115, 12.97),
]

# =====================================================================
# LAYER 1: token-deadline anytime KV degradation
# =====================================================================
def draw_jitter(sev):
    """Multiplicative heavy-tailed jitter: mostly ~0, occasional large spike.
    sev in [0,1] scales both the everyday noise and the spike probability."""
    base = abs(random.gauss(0, 0.06 * sev))          # everyday memory-contention noise
    if random.random() < 0.04 * sev:                 # rare co-tenant / thermal spike
        base += random.expovariate(1.0 / (1.2 * sev + 1e-9))
    return base

def token_time(c_base, alpha, f, L, jit):
    """ms for a token: fixed compute + KV read (∝ read_fraction × context) , scaled by jitter."""
    return (c_base + alpha * f * L) * (1.0 + jit)

def layer1(D=20.0, c_base=8.0, alpha=0.012, L0=200, Lmax=4000, n=1600,
           jitter_sev=0.6, guard=0.15, verbose=False):
    """
    D        : per-token deadline (ms), = 1000/target_tok_per_s
    c_base   : fixed per-token compute (ms)   (weights + MAC + non-KV)
    alpha    : KV read time per token-of-context at full budget (ms/token)
    L0..Lmax : context grows linearly across the n decoded tokens
    guard    : safety margin fraction the controller reserves for jitter
    Returns dict of metrics for RIGID and ROBUST policies.
    """
    lvl = QUEST_LEVELS
    ewma = 0.0                     # controller's running estimate of jitter
    res = {p: dict(miss=0, lag=0.0, qcost=0.0, times=[], budgets=[]) for p in ("rigid", "robust")}
    for i in range(n):
        L = L0 + (Lmax - L0) * i / max(1, n - 1)
        jit = draw_jitter(jitter_sev)          # the realized jitter this token
        # --- RIGID: always full KV ---
        t = token_time(c_base, alpha, 1.0, L, jit)
        res["rigid"]["times"].append(t); res["rigid"]["budgets"].append(2048)
        if t > D:
            res["rigid"]["miss"] += 1
            res["rigid"]["lag"] += (t - D)     # real-time: lost time is not recoverable
        # --- ROBUST: pick largest read_fraction whose PREDICTED time fits D ---
        jguard = ewma + guard                  # deadline-monotonic style guard band
        chosen = lvl[-1]                        # fallback: smallest budget
        for (b, f, q) in lvl:                   # largest budget first
            t_pred = token_time(c_base, alpha, f, L, jguard)
            if t_pred <= D:
                chosen = (b, f, q); break
        b, f, q = chosen
        t_r = token_time(c_base, alpha, f, L, jit)   # actual time at chosen budget
        res["robust"]["times"].append(t_r); res["robust"]["budgets"].append(b)
        res["robust"]["qcost"] += q
        if t_r > D:
            res["robust"]["miss"] += 1
            res["robust"]["lag"] += (t_r - D)
        ewma = 0.9 * ewma + 0.1 * jit          # update jitter estimate AFTER the fact
    for p in res:
        m = res[p]
        m["hit_rate"] = 100.0 * (n - m["miss"]) / n
        m["avg_qcost"] = m["qcost"] / n        # mean Δppl% paid per token
        m["stall_ms"] = m["lag"]
        del m["times"]; del m["budgets"]
    return res

def layer1_sweep(**kw):
    """Sweep context length (Lmax) -> the cliff-vs-graceful money shot."""
    rows = []
    for Lmax in (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000):
        r = layer1(Lmax=Lmax, **kw)
        rows.append(dict(Lmax=Lmax,
                         rigid_hit=r["rigid"]["hit_rate"], rigid_stall=r["rigid"]["stall_ms"],
                         robust_hit=r["robust"]["hit_rate"], robust_q=r["robust"]["avg_qcost"]))
    return rows

# =====================================================================
# LAYER 2: serving-level admission control vs congestion collapse
# =====================================================================
def layer2(rho=1.0, mu=100.0, slo=100.0, burst=0.5, retry=0.6, shed=False,
           T=10000, dt=1.0, guard=0.25, seed=None):
    if seed is not None:
        random.seed(seed)
    """
    rho    : offered load = lambda/mu (MEAN-preserving under burst)
    mu     : server capacity (req/s); mean service = 1000/mu ms
    slo    : TTFT deadline (ms); a request is 'good' if wait+service <= slo
    burst  : 0..1 arrival burstiness (variance only; the time-mean stays = rho)
    retry  : fraction of TIMED-OUT clients that retry (adds future load)
    shed   : admission control on? (True = ROBUST)
    T, dt  : sim horizon and step (ms)
    Returns goodput (req/s meeting SLO), p99 latency (served), shed_rate.
    """
    lam = rho * mu / 1000.0 * dt        # base arrivals per step (mean)
    mean_s = 1000.0 / mu                # mean service (ms)
    timeout = 3.0 * slo                 # a client abandons past this, then may retry
    q = []                              # queue of enqueue-times (ms)
    busy_until = 0.0
    served_good = 0; served = 0; shed_ct = 0; offered = 0
    lat = []
    retry_pool = []                     # ready-times of retrying/backed-off clients
    # mean-preserving 2-state burst: hi/lo symmetric around lam, ~50/50 occupancy
    hi, lo = lam * (1 + 1.5 * burst), max(0.0, lam * (1 - 1.5 * burst))
    in_burst = False
    steps = int(T / dt)
    for s in range(steps):
        now = s * dt
        if burst > 0:                   # symmetric switching -> equal occupancy -> mean = lam
            if random.random() < 0.03: in_burst = not in_burst
            rate = hi if in_burst else lo
        else:
            rate = lam
        # Poisson arrivals this step (Knuth) + matured retries/backoffs
        Lp = math.exp(-rate); p = 1.0; kk = 0
        while p > Lp:
            kk += 1; p *= random.random()
        arrivals = (kk - 1) + sum(1 for rt in retry_pool if rt <= now)
        retry_pool = [rt for rt in retry_pool if rt > now]
        for _ in range(arrivals):
            offered += 1
            if shed:
                pred_wait = max(0.0, busy_until - now) + len(q) * mean_s
                if pred_wait > slo * (1 - guard):
                    shed_ct += 1
                    retry_pool.append(now + 150.0)   # 503 + backoff: returns calmly, later
                    continue
            q.append(now)
        # serve (single server, one job in flight)
        if q and busy_until <= now:
            enq = q.pop(0)
            svc = max(0.5, random.gauss(mean_s, mean_s * 0.35))
            start = max(now, busy_until)
            busy_until = start + svc
            latency = (start - enq) + svc
            served += 1; lat.append(latency)
            if latency <= slo:
                served_good += 1
            elif latency > timeout and random.random() < retry:
                retry_pool.append(now + 80.0)        # abandoned -> retry storm (no-shed case)
    secs = T / 1000.0
    lat.sort()
    p99 = lat[int(0.99 * len(lat)) - 1] if lat else 0.0
    return dict(goodput=served_good / secs, p99=p99,
                shed_rate=100.0 * shed_ct / max(1, offered),
                served=served, offered=offered)

def layer2_sweep(shed, seeds=(1, 2, 3, 4), **kw):
    rows = []
    for rho in (0.5, 0.7, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6):
        acc = dict(goodput=0.0, p99=0.0, shed=0.0)
        for sd in seeds:
            r = layer2(rho=rho, shed=shed, seed=sd, **kw)
            acc["goodput"] += r["goodput"]; acc["p99"] += r["p99"]; acc["shed"] += r["shed_rate"]
        n = len(seeds)
        rows.append(dict(rho=rho, goodput=acc["goodput"]/n, p99=acc["p99"]/n, shed=acc["shed"]/n))
    return rows

# =====================================================================
if __name__ == "__main__":
    print("=" * 74)
    print("LAYER 1 — token-deadline anytime KV degradation (single real-time stream)")
    print("=" * 74)
    r = layer1()
    print(f"deadline D=20ms (50 tok/s), context 200->4000, jitter_sev=0.6, guard=0.15")
    for p in ("rigid", "robust"):
        m = r[p]
        print(f"  {p:7s}: deadline-hit {m['hit_rate']:6.1f}%   "
              f"cumulative stall {m['stall_ms']:8.0f} ms   "
              f"mean quality cost {m['avg_qcost']:.3f} Δppl%/tok")
    print("\n  context-length sweep (the cliff vs graceful):")
    print(f"  {'Lmax':>6} {'rigid_hit%':>11} {'rigid_stall_ms':>15} {'robust_hit%':>12} {'robust_Δppl/tok':>16}")
    l1 = layer1_sweep()
    for row in l1:
        print(f"  {row['Lmax']:6d} {row['rigid_hit']:11.1f} {row['rigid_stall']:15.0f} "
              f"{row['robust_hit']:12.1f} {row['robust_q']:16.3f}")

    print("\n" + "=" * 74)
    print("LAYER 2 — serving-level admission control vs congestion collapse")
    print("=" * 74)
    print("  offered-load sweep (goodput = req/s meeting the 100ms SLO):")
    print(f"  {'rho':>5} | {'RIGID goodput':>14} {'p99ms':>7} | {'ROBUST goodput':>15} {'p99ms':>7} {'shed%':>7}")
    rig = layer2_sweep(shed=False)
    rob = layer2_sweep(shed=True)
    for a, b in zip(rig, rob):
        print(f"  {a['rho']:5.2f} | {a['goodput']:14.1f} {a['p99']:7.0f} | "
              f"{b['goodput']:15.1f} {b['p99']:7.0f} {b['shed']:7.1f}")

    out = dict(layer1_default=r, layer1_sweep=l1,
               layer2_rigid=rig, layer2_robust=rob,
               quest_levels=QUEST_LEVELS)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "deadline_sim_ref.json"), "w"), indent=2)
    print("\nwrote deadline_sim_ref.json")
