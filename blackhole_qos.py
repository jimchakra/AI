#!/usr/bin/env python3
"""
blackhole_qos.py — the QoS / queueing layer of the Blackhole system model.

This does NOT re-derive queueing theory. It reuses the validated M/G/1 heavy-tailed
machinery from capacity_sim.py (Policy A hard vs Policy B anytime-KV-truncate, and the
tiered-priority sim), but it DERIVES the service-time constants (c, k0) from the physical
byte model in blackhole_sysmodel.py, so the queue and the roofline are one coherent model:

    c   = mandatory per-chunk time  = (weight bytes / batch) / BW   [amortized weights]
    k0  = truncatable per-chunk time = (KV bytes for the chunk)  / BW   [the degrade lever]

At a batched serving operating point the amortized weight term is small and the KV term
dominates -- which is exactly why KV-read truncation is the effective graceful-degradation
actuator here. The contention multiplier M (heavy-tailed) is the shared-BW / scheduler
jitter of the control plane -- the unbounded term you cannot provision away.

Absolute times are directional (a cycle-accurate model refines constants, not conclusions);
the SHAPES -- capacity collapse as the percentile tightens, and its recovery under anytime
degrade -- are the claim.
"""
import json, os
from blackhole_sysmodel import SystemSpec, M8B, step, max_batch, Cut, bytes_per_step
import capacity_sim as Q


def derive_service(sys: SystemSpec, m, S: int, B: int, tokens_per_chunk: int = 32):
    """Return (c, k0) in ms for a 'chunk' of decode tokens at a serving operating point."""
    by = bytes_per_step(sys, m, B, S, Cut(0, 0))
    w_time_tok = (by['w_full'] / B) / sys.bw_bytes_s      # amortized weights, per token [s]
    kv_time_tok = by['kv_full'] / B / sys.bw_bytes_s      # KV per sequence, per token [s]
    c = w_time_tok * tokens_per_chunk * 1e3               # -> ms
    k0 = kv_time_tok * tokens_per_chunk * 1e3             # -> ms
    return c, k0


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys = SystemSpec()
    m = M8B
    S = 8192
    B = max(1, max_batch(sys, m, S, kv_keep=1.0))          # largest full-KV batch that fits
    c, k0 = derive_service(sys, m, S, B, tokens_per_chunk=32)
    D = 4.0 * (c + k0)                                      # deadline = 4x nominal chunk service

    print("=" * 82)
    print("BLACKHOLE QoS LAYER  --  service constants derived from the byte model")
    print("=" * 82)
    print(f"operating point: {m.name} on {sys.name}, S={S}, batch={B} (fills 32 GB), chunk=32 tok")
    print(f"  c  (mandatory, amortized weights) = {c:6.2f} ms/chunk")
    print(f"  k0 (truncatable, KV reads)        = {k0:6.2f} ms/chunk   <- the degrade lever")
    print(f"  KV is {k0/(c+k0)*100:.0f}% of nominal service -> KV-truncation is the effective actuator")
    print(f"  deadline D = {D:.1f} ms")

    # --- capacity vs percentile guarantee: hard (A) vs anytime-KV-truncate (B) ---
    lams = [round(0.02 + 0.01 * i, 4) for i in range(24)]   # req/ms (chunks/ms)
    # rescale arrival grid to this operating point's service rate
    mu = 1.0 / (c + k0)
    lams = [round(0.15 * mu + 0.06 * mu * i, 6) for i in range(22)]
    A = Q.sweep('A', lams, D=D, c=c, k0=k0, sigma=0.7)
    Bsw = Q.sweep('B', lams, D=D, c=c, k0=k0, sigma=0.7)

    print("\n" + "-" * 82)
    print("A. CAPACITY vs PERCENTILE  (hard-quality collapses at the tail; anytime holds it)")
    print("-" * 82)
    print(f"{'target':>8} | {'A hard (chunks/s)':>17} | {'B anytime (chunks/s)':>20} | {'B gain':>7} | {'B deg%@cap':>10}")
    caps = []
    for target in (0.90, 0.95, 0.99, 0.995, 0.998, 0.999):
        ca = Q.capacity_at(A, target) * 1000
        cb = Q.capacity_at(Bsw, target) * 1000
        capB = Q.capacity_at(Bsw, target)
        row = min(Bsw, key=lambda r: abs(r['lam'] - capB))
        gain = (cb / ca) if ca > 0.05 else float('inf')
        gtxt = f"{gain:6.2f}x" if gain != float('inf') else "  A->0 "
        caps.append(dict(target=target, A_cps=ca, B_cps=cb,
                         gain=(gain if gain != float('inf') else None), degraded=row['degraded']))
        print(f"P{target*100:<6.1f} | {ca:17.1f} | {cb:20.1f} | {gtxt:>7} | {row['degraded']*100:9.1f}%")

    # --- tiered QoS: premium never degrades & runs on guaranteed resources; economy degrades ---
    print("\n" + "-" * 82)
    print("B. TIERED QoS  (radiologist tier held; photo-touchup tier absorbs the degradation)")
    print("-" * 82)
    print(f"{'load(chunks/s)':>14} | {'premium on-time':>15} | {'economy on-time':>15} | {'econ deg%':>9} | {'econ Δppl':>9}")
    tiers = []
    for lam in (0.5 * mu, 0.8 * mu, 1.0 * mu, 1.2 * mu, 1.4 * mu):
        r = Q.simulate_tiers(lam, prem_frac=0.3, D=D, c=c, k0=k0, sigma=0.7)
        tiers.append(dict(lam_cps=lam * 1000, **{k: r[k] for k in
                     ('prem_ontime', 'econ_ontime', 'econ_degraded', 'econ_q')}))
        print(f"{lam*1000:14.1f} | {r['prem_ontime']*100:14.2f}% | {r['econ_ontime']*100:14.2f}% | "
              f"{r['econ_degraded']*100:8.1f}% | {r['econ_q']:9.2f}")

    json.dump(dict(op=dict(model=m.name, system=sys.name, S=S, batch=B, c_ms=c, k0_ms=k0, D_ms=D,
                           kv_frac=k0/(c+k0)),
                   capacity=caps, tiers=tiers),
              open(os.path.join(HERE, "blackhole_qos_ref.json"), "w"), indent=2)
    print("\nwrote blackhole_qos_ref.json")
