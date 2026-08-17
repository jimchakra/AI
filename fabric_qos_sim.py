#!/usr/bin/env python3
"""
fabric_qos_sim.py — first-order fluid contention model of a shared AI fabric.

Question (thesis #3, aimed at NVIDIA JR2016101's charter):
  On one contended interconnect carrying BOTH
    (a) collective all-reduce traffic  — BARRIER-synchronized, mandatory
    (b) KV-cache transfers (disaggregated prefill->decode) — independent, approximable
  how much does a barrier-aware, value-per-cost control law beat fair-share QoS?

Model (deliberately first-order, calibrated, not cycle-accurate):
  - A set of shared inter-node links, each capacity C (GB/s), oversubscribed.
  - Ring all-reduce = P sub-flows, each pinned to a subset of links.
      completion = MAX over sub-flows (the barrier) -> a stalled sub-flow stalls the step.
  - KV flows = independent transfers, each has bytes, a path (hops = #links), and a
      VALUE (attention/importance). cost = bytes * hops. value-per-cost = value/(bytes*hops).
  - Time-stepped fluid sim; per link, capacity split among active flows (equal-share ~ max-min).

Policies:
  BASELINE  : fair per-flow share, no barrier priority, no gating.
  CONTROL   : (1) protect the barrier — collective sub-flows get priority bandwidth;
              (2) under congestion raise threshold theta -> DROP independent KV flows with
                  value-per-cost < theta (far + low-value shed first);
              (3) inside the collective, degrade by COMPRESSING payload (never drop a sub-flow).
No randomness in the physics; one fixed seed builds the workload.
"""
import random, statistics

# ---------------- calibrated constants ----------------
LINK_BW      = 50.0     # GB/s per inter-node link (~400 Gb/s RoCE/IB class)
N_LINKS      = 32
P_COLL       = 16       # all-reduce participants (ring)
COLL_BYTES   = 8.0      # GB moved per participant per all-reduce step (model-parallel grads/activations)
COLL_PERIOD  = 1.0      # a new all-reduce step is READY every this many "seconds" (offered)
KV_BYTES_MED = 0.05     # GB per KV transfer (~50 MB: a long-context request's KV page set)
DT           = 0.002    # sim time step (s)
SEED         = 7

def make_workload(kv_rate, seconds=6.0):
    """Build the event list: periodic collectives + Poisson-ish KV transfers."""
    rnd = random.Random(SEED)
    coll_steps = [i*COLL_PERIOD for i in range(int(seconds/COLL_PERIOD))]
    kv=[]
    t=0.0
    while t < seconds:
        t += rnd.expovariate(kv_rate)
        if t>=seconds: break
        hops  = rnd.choice([1,1,2,2,4,8])                 # near vs far transfers
        val   = rnd.random()                              # importance in (0,1]
        bytes_= KV_BYTES_MED * rnd.choice([0.5,1,1,2,4])  # size spread
        links = tuple(sorted(rnd.sample(range(N_LINKS), min(hops, N_LINKS))))
        kv.append(dict(t=t, bytes=bytes_, rem=bytes_, hops=hops, val=val, links=links,
                       vpc=val/(bytes_*hops), done=None, dropped=False, admit=True))
    # collective sub-flows: P sub-flows per step, each pinned to 'hops' consecutive links
    return coll_steps, kv

def run(kv_rate, policy, theta=0.0, seconds=6.0, use_gate=True, use_compress=True):
    coll_steps, kv = make_workload(kv_rate, seconds)
    # active collective: dict or None
    coll=None; coll_idx=0
    coll_done_times=[]; coll_stall=0.0
    steps_completed=0
    dropped_val=0.0; dropped_bytes=0.0; total_val=0.0; total_bytes=0.0
    compress_time=0.0; compress_steps=0
    t=0.0
    kv_by_t=sorted(kv, key=lambda f:f['t'])
    ki=0; active_kv=[]
    n=int(seconds/DT)
    for step in range(n):
        t=step*DT
        # release a new collective step if due and none active
        while coll is None and coll_idx < len(coll_steps) and coll_steps[coll_idx] <= t:
            # sub-flow f uses 'hops' consecutive links; here fixed span 4 for ring segment
            span=4
            subs=[dict(rem=COLL_BYTES, links=tuple((j*2+k)%N_LINKS for k in range(span)))
                  for j in range(P_COLL)]
            coll=dict(subs=subs, start=t, compress=1.0)
            coll_idx+=1
        # admit newly-arrived KV
        while ki < len(kv_by_t) and kv_by_t[ki]['t'] <= t:
            f=kv_by_t[ki]; total_val+=f['val']; total_bytes+=f['bytes']
            active_kv.append(f); ki+=1
        # ---- congestion estimate: aggregate demand / capacity on hottest links ----
        # count active flows per link
        load=[0]*N_LINKS
        for f in active_kv:
            if f['rem']>0 and not f['dropped']:
                for l in f['links']: load[l]+=1
        if coll:
            for s in coll['subs']:
                if s['rem']>0:
                    for l in s['links']: load[l]+=1
        congested = sum(1 for x in load if x>0 and x>=3)/max(1,sum(1 for x in load if x>0))

        # ---- CONTROL levers, DECOMPOSED so each effect is attributable ----
        # lever 1: barrier-priority (in link_share, always on for policy=='control')
        # lever 2: value-per-cost gating of independent KV traffic
        if policy=='control' and use_gate and congested>0.25:
            thr = theta*congested   # tighter under more congestion
            for f in active_kv:
                if f['rem']>0 and not f['dropped'] and f['vpc'] < thr:
                    f['dropped']=True; f['done']=t
                    dropped_val+=f['val']; dropped_bytes+=(f['rem'])
        # lever 3: under-stress payload compression of the collective (never drops a sub-flow)
        if policy=='control' and use_compress and coll and congested>0.5:
            if coll['compress']==1.0: compress_steps+=1
            coll['compress']=0.6       # send 60% of bytes (lower-precision grad/KV)

        # ---- allocate link bandwidth for this dt ----
        # per link, list of (flow, is_coll)
        link_flows=[[] for _ in range(N_LINKS)]
        for f in active_kv:
            if f['rem']>0 and not f['dropped']:
                for l in f['links']: link_flows[l].append(('kv',f))
        if coll:
            for s in coll['subs']:
                if s['rem']>0:
                    for l in s['links']: link_flows[l].append(('coll',s))
        # compute each flow's rate = min over its links of share
        # CONTROL protects barrier: collectives get priority; KV shares residual.
        # BASELINE: equal share for all.
        def link_share(l):
            flows=link_flows[l]
            if not flows: return {}
            if policy=='control':
                colls=[x for x in flows if x[0]=='coll']
                kvs  =[x for x in flows if x[0]=='kv']
                share={}
                if colls:
                    per=LINK_BW/len(colls) if len(colls)>0 else 0
                    # collectives take what they need first (cap at per, but they usually saturate)
                    for _,s in colls: share[id(s)]=LINK_BW/len(colls)
                    residual=0.0
                else:
                    residual=LINK_BW
                if kvs:
                    rk=residual/len(kvs) if len(kvs)>0 else 0
                    for _,f in kvs: share[id(f)]=rk
                return share
            else:
                per=LINK_BW/len(flows)
                return {id(o): per for _,o in flows}
        # each flow's achievable rate is the min share across its links
        rate={}
        for l in range(N_LINKS):
            sh=link_share(l)
            for _,o in link_flows[l]:
                r=sh[id(o)]
                rate[id(o)]=min(rate.get(id(o), 1e9), r)
        # advance
        cf=coll['compress'] if coll else 1.0
        for f in active_kv:
            if f['rem']>0 and not f['dropped']:
                f['rem']-=rate.get(id(f),0)*DT
                if f['rem']<=0: f['rem']=0; f['done']=t
        if coll:
            for s in coll['subs']:
                if s['rem']>0:
                    s['rem']-=rate.get(id(s),0)*DT / cf   # compression => fewer effective bytes to move
                    if s['rem']<=0: s['rem']=0
            if all(s['rem']<=0 for s in coll['subs']):
                coll_done_times.append(t-coll['start']); steps_completed+=1; coll=None
        else:
            coll_stall+=0.0
    # metrics
    kv_lat=[f['done']-f['t'] for f in kv if f['done'] is not None and not f['dropped']]
    p99 = (sorted(kv_lat)[int(0.99*len(kv_lat))-1] if kv_lat else float('nan'))
    kv_val_done=sum(f['val'] for f in kv if f['done'] is not None and not f['dropped'])
    return dict(
        steps=steps_completed,
        coll_time=(statistics.mean(coll_done_times) if coll_done_times else float('nan')),
        coll_p99=(sorted(coll_done_times)[int(0.99*len(coll_done_times))-1] if len(coll_done_times)>3 else
                  (max(coll_done_times) if coll_done_times else float('nan'))),
        kv_p99=p99,
        kv_val_done=kv_val_done, total_val=total_val,
        dropped_val=dropped_val, dropped_bytes=dropped_bytes, total_bytes=total_bytes,
        compress_steps=compress_steps,
    )

def fmt(x,n=2):
    return ('nan' if x!=x else f'{x:.{n}f}')

def base(kv):      return run(kv,'baseline',theta=1.5)
def prio_only(kv): return run(kv,'control',theta=1.5,use_gate=False,use_compress=False)
def full(kv):      return run(kv,'control',theta=1.5,use_gate=True, use_compress=True)

print("NOTE: first-order fluid contention model. Numbers are DIRECTIONAL, not a benchmark.")
print("="*82)
print("A. LOAD SWEEP — all-reduce (barrier) mean completion, three policies (decomposed)")
print("   %d links @ %.0f GB/s; all-reduce %.0f GB/participant; step ready every %.1fs"
      %(N_LINKS,LINK_BW,COLL_BYTES,COLL_PERIOD))
print("="*82)
print(f"{'KV rate':>8}  {'policy':<28}{'coll_time':>10}{'kv_p99':>9}{'kv_val_done%':>14}")
for kv in [20,120,300]:
    for name,fn in [('baseline (fair-share)',base),
                    ('control: priority-only',prio_only),
                    ('control: prio+gate+compress',full)]:
        r=fn(kv); vd=100*r['kv_val_done']/r['total_val'] if r['total_val'] else float('nan')
        print(f"{kv:>8}  {name:<28}{fmt(r['coll_time']):>10}{fmt(r['kv_p99']):>9}{fmt(vd,1):>14}")
    print()

print("="*82)
print("B. ATTRIBUTION — what each lever actually buys (at KV rate=300)")
print("="*82)
b=base(300); p=prio_only(300); f=full(300)
print(f"  baseline (fair-share)          all-reduce mean: {fmt(b['coll_time'])} s")
print(f"  + barrier priority   (lever 1) all-reduce mean: {fmt(p['coll_time'])} s"
      f"   ({b['coll_time']/p['coll_time']:.2f}x vs baseline, NO quality cost)")
print(f"  + gate + compress (levers 2/3) all-reduce mean: {fmt(f['coll_time'])} s"
      f"   (further {p['coll_time']/f['coll_time']:.2f}x, but SPENDS collective quality)")
print("  honest read: most barrier protection is PRIORITY (free); compression adds")
print("  a little more speed at a quality cost -> a knob, not a free lunch.")

print()
print("="*82)
print("C. VALUE-PER-COST GATING — bytes shed vs VALUE shed (independent KV traffic)")
print("="*82)
f=full(300)
pb=100*f['dropped_bytes']/f['total_bytes'] if f['total_bytes'] else 0
pv=100*f['dropped_val']/f['total_val'] if f['total_val'] else 0
print(f"  bytes shed: {pb:.1f}%   value shed: {pv:.1f}%"
      f"   -> ~{pb/max(pv,1e-9):.1f}x cheaper in value than in bytes")
print("  directional only: value-per-cost sheds the cheap stuff first; magnitude is workload-dependent.")
