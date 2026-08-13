#!/usr/bin/env python3
"""
blackhole_sysmodel.py — a high-level full-system (compute die + memory die) performance
model of content-aware KV read-selection on a Tenstorrent Blackhole-class chip.

WHAT THIS IS
------------
A first-order, roofline-based system model. Every input is either (a) a published
Blackhole spec, (b) closed-form byte/FLOP accounting for an 8B-class GQA transformer,
or (c) a measured quality curve from the Quest real-model study (Qwen2.5-1.5B / WikiText-2).
Nothing here is a fitted fudge factor. The SHAPES and the crossover points are the claim;
absolute tokens/s are directional (a cycle-accurate model would refine constants, not the
conclusion).

THE THESIS, MADE PHYSICAL
-------------------------
Autoregressive decode is memory-bandwidth-bound: per-token time ~ bytes-moved / BW, and the
compute units idle waiting on memory. Blackhole is *bandwidth-starved relative to its compute*
(512 GB/s feeding ~745 TFLOPS FP8 -> a roofline ridge at ~1455 FLOP/byte, while batch-1 decode
runs at ~2 FLOP/byte). So cutting the bytes moved per token by a fraction f raises throughput
by ~1/(1-f) at the SAME bandwidth and power and on the SAME die -- until the workload climbs
the roofline and hits the compute ridge, at which point more cutting buys nothing. This model
finds that ceiling and shows where the 2-3x lives and where it stops.

WHY BLACKHOLE IS THE RIGHT CANVAS
---------------------------------
A chip whose bandwidth is the acknowledged bottleneck is exactly where "move fewer bytes"
converts most directly into system throughput. The claim is strongest precisely on a part
like this -- which is worth being able to say out loud.

Specs (Blackhole p150a, single chip; sources in README / study page):
  512 GB/s GDDR6 (32 GB @ 16 GT/s) | ~745 TFLOPS FP8 (single-die official; ~664 post-firmware)
  ~300 W TDP | ~180 MB on-chip SRAM | 4x QSFP-DD 800G chip-to-chip
"""
from dataclasses import dataclass, field
import json, os, math
from capacity_sim import qcost   # measured Quest curve: read_fraction -> Δppl% (single source of truth)

# ----------------------------------------------------------------------------------
# 1. HARDWARE  (Tenstorrent Blackhole p150a, single chip -- published specs)
# ----------------------------------------------------------------------------------
@dataclass
class SystemSpec:
    name: str = "Blackhole p150a (1 chip)"
    bw_bytes_s: float = 512e9        # GDDR6 bandwidth [B/s]
    fp8_flops: float = 745e12        # peak dense FP8 [FLOP/s] (official single-die)
    dram_bytes: float = 32e9         # GDDR6 capacity [B]
    sram_bytes: float = 180e6        # on-chip SRAM [B]
    tdp_w: float = 300.0             # [W]
    # directional energy constants (pJ) -- data movement dominates (>90%, Finding 01)
    e_dram_pj_per_byte: float = 16.0 # off-chip GDDR6 read energy [pJ/byte]  (~2 pJ/bit)
    e_flop_pj: float = 0.4           # on-chip FP8 MAC energy [pJ/FLOP] (directional)

    @property
    def ridge(self) -> float:
        """Roofline ridge point: arithmetic intensity [FLOP/byte] above which the
        workload is compute-bound. AI* = peak_FLOPS / BW."""
        return self.fp8_flops / self.bw_bytes_s


# ----------------------------------------------------------------------------------
# 2. MODEL  (8B-class dense GQA, Llama-3.1-8B-like; and an MoE contrast)
# ----------------------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str = "8B dense GQA"
    L: int = 32                      # layers
    d: int = 4096                    # model dim
    n_q: int = 32                    # query heads
    n_kv: int = 8                    # KV heads (GQA)
    hd: int = 128                    # head dim
    d_ff: int = 14336                # SwiGLU intermediate
    bpw: float = 1.0                 # weight bytes/param (FP8=1, INT4=0.5, BF16=2)
    bpw_kv: float = 2.0              # KV bytes/element (FP16=2 default/conservative; FP8=1 knob)
    # MoE (dense model: experts=1, active=1)
    experts: int = 1
    active_experts: int = 1

    def params_per_layer_attn(self) -> int:
        # q, k, v, o projections
        q = self.d * (self.n_q * self.hd)
        k = self.d * (self.n_kv * self.hd)
        v = self.d * (self.n_kv * self.hd)
        o = (self.n_q * self.hd) * self.d
        return q + k + v + o

    def params_per_layer_ffn_one_expert(self) -> int:
        # SwiGLU: gate, up, down
        return 3 * self.d * self.d_ff

    def total_params(self) -> int:
        attn = self.params_per_layer_attn()
        ffn_total = self.params_per_layer_ffn_one_expert() * self.experts
        return (attn + ffn_total) * self.L

    def active_params(self) -> int:
        """Params actually read per token (MoE reads only active experts)."""
        attn = self.params_per_layer_attn()
        ffn_active = self.params_per_layer_ffn_one_expert() * self.active_experts
        return (attn + ffn_active) * self.L

    def kv_bytes_per_token_all_layers(self) -> int:
        # K and V, n_kv heads, head_dim, all layers
        return 2 * self.n_kv * self.hd * self.L * int(round(self.bpw_kv))

    def kv_bytes_per_seq(self, S: int) -> float:
        return S * self.kv_bytes_per_token_all_layers()


# 8B dense (FP8 weights + FP8 KV is Blackhole-native; FP16 KV also offered as a knob)
M8B = ModelSpec()

# MoE contrast: 8 experts, top-2 routed. Same attn; expert FFN sized so ACTIVE params ~ dense-ish,
# TOTAL params larger (stresses capacity). This is a Mixtral-shaped toy at ~8B-active scale.
MMOE = ModelSpec(name="MoE 8x, top-2", d_ff=14336, experts=8, active_experts=2)


# ----------------------------------------------------------------------------------
# 3. PER-STEP ACCOUNTING  (one decode step = one new token for each of B sequences)
# ----------------------------------------------------------------------------------
@dataclass
class Cut:
    """A degradation / read-selection operating point."""
    f_kv: float = 0.0        # fraction of KV reads eliminated (content-aware selection)
    f_w: float = 0.0         # fraction of WEIGHT reads eliminated (activation read-mask, down-proj)
    page: int = 16           # KV page size (tokens); summary tax ~ 1/page
    selecting: bool = True   # incur the summary-read tax when a cut is active

    @property
    def summary_tax(self) -> float:
        # summaries are scanned only to make the KV page selection; the weight read-mask
        # reuses the activation bitmap (a byproduct) and carries no summary read.
        return (1.0 / self.page) if (self.selecting and self.f_kv > 0) else 0.0


def bytes_per_step(sys: SystemSpec, m: ModelSpec, B: int, S: int, cut: Cut = Cut(0, 0)):
    """Returns dict of byte components moved off-chip in one decode step."""
    w_full = m.active_params() * m.bpw                 # weights read once per step (shared over batch)
    kv_full = B * m.kv_bytes_per_seq(S)                # KV read per sequence
    w_read = w_full * (1.0 - cut.f_w)
    kv_read = kv_full * (1.0 - cut.f_kv)
    summ = kv_full * cut.summary_tax                   # summaries scanned to make the selection
    total = w_read + kv_read + summ
    return dict(w_full=w_full, kv_full=kv_full, w_read=w_read, kv_read=kv_read,
                summaries=summ, total=total)


def flops_per_step(m: ModelSpec, B: int, S: int) -> float:
    """FLOPs in one decode step (2 FLOP per MAC)."""
    mm = 2.0 * m.active_params() * B                   # weight matmuls
    attn = 2.0 * (2.0 * B * m.n_q * m.hd * S * m.L)    # QK^T + AV
    return mm + attn


def kv_cache_bytes(m: ModelSpec, B: int, S: int) -> float:
    """Resident KV cache size (capacity, not per-step bandwidth)."""
    return B * m.kv_bytes_per_seq(S)


def step(sys: SystemSpec, m: ModelSpec, B: int, S: int, cut: Cut = Cut(0, 0)):
    """Roofline step: t = max(mem_time, compute_time) under perfect overlap.
    Returns timing, regime, arithmetic intensity, throughput, energy."""
    by = bytes_per_step(sys, m, B, S, cut)
    fl = flops_per_step(m, B, S)
    t_mem = by['total'] / sys.bw_bytes_s
    t_cmp = fl / sys.fp8_flops
    t = max(t_mem, t_cmp)
    ai = fl / by['total']
    mem_bound = t_mem >= t_cmp
    toks_s = B / t                                      # B tokens emitted per step
    # energy (directional): off-chip byte movement + compute + static
    e_move = by['total'] * sys.e_dram_pj_per_byte * 1e-12
    e_flop = fl * sys.e_flop_pj * 1e-12
    e_static = sys.tdp_w * t * 0.30                     # ~30% of TDP as static/leakage floor
    e_tok = (e_move + e_flop + e_static) / B
    return dict(t=t, t_mem=t_mem, t_cmp=t_cmp, mem_bound=mem_bound, ai=ai,
                toks_s=toks_s, bytes=by, flops=fl, e_tok_j=e_tok,
                kv_share=by['kv_full'] / (by['w_full'] + by['kv_full']))


def feasible(sys: SystemSpec, m: ModelSpec, B: int, S: int, weight_bytes=None) -> bool:
    """Does weights + resident KV cache fit in DRAM?"""
    w = (weight_bytes if weight_bytes is not None else m.total_params() * m.bpw)
    return (w + kv_cache_bytes(m, B, S)) <= sys.dram_bytes


def max_batch(sys: SystemSpec, m: ModelSpec, S: int, kv_keep: float = 1.0) -> int:
    """Largest batch whose weights + (kept) KV fit in 32 GB."""
    w = m.total_params() * m.bpw
    per = m.kv_bytes_per_seq(S) * kv_keep
    if per <= 0:
        return 10**9
    return max(0, int((sys.dram_bytes - w) // per))


def serving_throughput(sys: SystemSpec, m: ModelSpec, S: int, cut: Cut):
    """The single-chip SERVING throughput gain, coupling the two effects the full-system
    view exposes:
      (i)  bytes/token falls with the KV+weight cut  -> more tokens/s at fixed BW; and
      (ii) freed DRAM lets MORE sequences fit        -> the fixed weight read amortizes
           over a larger batch, pushing arithmetic intensity up the roofline.
    Both live under the same 512 GB/s ceiling (tokens/s = BW / bytes-per-token), so this
    is not double-counting -- it is one ceiling with two ways to lower the denominator.
    Baseline = largest feasible full-KV batch; throttled = largest feasible cut-KV batch."""
    B0 = max(1, max_batch(sys, m, S, kv_keep=1.0))
    B1 = max(1, max_batch(sys, m, S, kv_keep=1.0 - cut.f_kv))
    r0 = step(sys, m, B0, S, Cut(0, 0))
    r1 = step(sys, m, B1, S, cut)
    return dict(S=S, B0=B0, B1=B1, batch_gain=B1 / B0,
                tok0=r0['toks_s'], tok1=r1['toks_s'], gain=r1['toks_s'] / r0['toks_s'],
                ai0=r0['ai'], ai1=r1['ai'],
                e0=1 / r0['e_tok_j'], e1=1 / r1['e_tok_j'], energy_gain=r0['e_tok_j'] / r1['e_tok_j'])


# ----------------------------------------------------------------------------------
# 4. VALIDATION REPORT
# ----------------------------------------------------------------------------------
def _fmt_gb(x):  # bytes -> GB
    return f"{x/1e9:6.2f} GB"

if __name__ == "__main__":
    sys = SystemSpec()
    m = M8B
    HERE = os.path.dirname(os.path.abspath(__file__))

    print("=" * 82)
    print(f"BLACKHOLE SYSTEM MODEL  --  {sys.name}   |   model: {m.name}")
    print("=" * 82)
    print(f"BW = {sys.bw_bytes_s/1e9:.0f} GB/s   peak = {sys.fp8_flops/1e12:.0f} TFLOPS FP8   "
          f"DRAM = {sys.dram_bytes/1e9:.0f} GB   TDP = {sys.tdp_w:.0f} W")
    print(f"Roofline RIDGE  AI* = peak/BW = {sys.ridge:,.0f} FLOP/byte")
    print(f"Model: {m.total_params()/1e9:.2f} B params  |  weights@FP8 = {_fmt_gb(m.total_params()*m.bpw)}"
          f"  |  KV = {m.kv_bytes_per_token_all_layers()/1024:.0f} KB/token (all layers, {m.bpw_kv:.0f}B/elem)")

    print("\n" + "-" * 82)
    print("A. WHERE ON THE ROOFLINE?  batch-1 decode is deeply memory-bound")
    print("-" * 82)
    print(f"{'context S':>10} | {'AI (FLOP/B)':>12} | {'x below ridge':>13} | {'regime':>12} | {'tokens/s':>9}")
    for S in (2048, 8192, 32768, 131072):
        r = step(sys, m, 1, S)
        print(f"{S:10d} | {r['ai']:12.2f} | {sys.ridge/r['ai']:12.0f}x | "
              f"{'mem-bound' if r['mem_bound'] else 'compute':>12} | {r['toks_s']:9.1f}")

    print("\n" + "-" * 82)
    print("B. REGIME TABLE FROM FIRST PRINCIPLES  (KV/weight mix sets the achievable cut)")
    print("   cut op-point: KV -57% @<=1% ppl (or -70% @~3%); weight read-mask -13% of weight reads")
    print("-" * 82)
    print(f"{'regime':>26} | {'KV share':>8} | {'overall cut':>11} | {'speedup':>8} | {'feasible?':>9}")
    # operating point: 57% KV cut + ~8% of weights masked (down-proj), summary tax at page16
    op = Cut(f_kv=0.57, f_w=0.13, page=16)
    regimes = [("batch-1, short (S=2K)", 1, 2048),
               ("batch-1, 32K", 1, 32768),
               ("batched long-ctx (B=8, 32K)", 8, 32768),
               ("batched long-ctx (B=max, 8K)", max_batch(sys, m, 8192), 8192)]
    for label, B, S in regimes:
        base = step(sys, m, B, S)
        cut = step(sys, m, B, S, op)
        overall_cut = 1.0 - cut['bytes']['total'] / base['bytes']['total']
        speedup = base['t'] / cut['t']
        fe = feasible(sys, m, B, S)
        print(f"{label:>26} | {base['kv_share']*100:7.1f}% | {overall_cut*100:10.1f}% | "
              f"{speedup:7.2f}x | {'yes' if fe else 'NO (cap)':>9}")

    print("\n" + "-" * 82)
    print("C. THE REAL CEILING IS QUALITY, NOT THE COMPUTE WALL")
    print("   the ridge (1455) sits so far above decode that KV-cutting never reaches it --")
    print("   bandwidth stays the bottleneck throughout, so the throttle keeps paying; what")
    print("   stops you is the measured perplexity curve (Quest), not the roofline.")
    print("-" * 82)
    B, S = max(1, max_batch(sys, m, 8192)), 8192
    print(f"   at B={B}, S={S} (largest feasible batch @ 8K): sweep KV cut")
    print(f"{'KV cut':>8} | {'AI':>8} | {'x below ridge':>13} | {'speedup':>8} | {'Δppl%':>7} | {'verdict':>16}")
    base = step(sys, m, B, S)
    for fkv in (0.0, 0.2, 0.4, 0.57, 0.70, 0.85, 0.95):
        r = step(sys, m, B, S, Cut(f_kv=fkv, f_w=0.13, page=16))
        dppl = qcost(1.0 - fkv)   # read_fraction = 1 - cut
        verdict = ("safe (~1%)" if dppl <= 1.1 else
                   "aggressive (~3%)" if dppl <= 4.5 else "too lossy")
        print(f"{fkv*100:7.0f}% | {r['ai']:8.1f} | {sys.ridge/r['ai']:12.0f}x | "
              f"{base['t']/r['t']:7.2f}x | {dppl:6.2f}% | {verdict:>16}")

    print("\n" + "-" * 82)
    print("D. CAPACITY  --  pruning KV also enlarges the feasible batch (compounds the win)")
    print("-" * 82)
    print(f"{'context S':>10} | {'max batch (full KV)':>19} | {'max batch (57% cut)':>20} | {'gain':>6}")
    for S in (8192, 16384, 32768, 65536):
        b0 = max_batch(sys, m, S, kv_keep=1.0)
        b1 = max_batch(sys, m, S, kv_keep=0.43)
        print(f"{S:10d} | {b0:19d} | {b1:20d} | {b1/max(1,b0):5.2f}x")

    print("\n" + "-" * 82)
    print("E. SERVING THROUGHPUT  --  the honest single-chip 2-3x (BW cut x capacity-enabled batch)")
    print("   baseline = largest full-KV batch that fits 32 GB;  throttled = largest 57%-cut batch")
    print("-" * 82)
    print(f"{'context S':>10} | {'batch 0->1':>12} | {'tok/s 0->1':>16} | {'AI 0->1':>13} | {'gain':>6} | {'E/tok gain':>10}")
    for S in (8192, 16384, 32768):
        g = serving_throughput(sys, m, S, op)
        print(f"{S:10d} | {g['B0']:4d} ->{g['B1']:5d}  | {g['tok0']:6.0f} ->{g['tok1']:6.0f}  | "
              f"{g['ai0']:5.1f} ->{g['ai1']:5.1f}  | {g['gain']:5.2f}x | {g['energy_gain']:9.2f}x")

    print("\n" + "-" * 82)
    print("G. THE MAC ARRAY IS SIZED FOR PREFILL/TRAINING, NOT DECODE")
    print("   Prefill & training are COMPUTE-bound (AI >> ridge) -> the array runs near peak there;")
    print("   that is what it is sized for. DECODE is a different phase on the same silicon: it is")
    print("   BANDWIDTH-bound, and no amount of batching fills the array -- AI asymptotes ~34 << ridge,")
    print("   so decode MFU is capped ~2.4% BY BANDWIDTH, not by MAC count. Real serving FUSES the two")
    print("   (chunked-prefill / continuous batching): prefill compute backfills decode's idle MACs,")
    print("   decode bandwidth backfills prefill's idle BW -> the array is well-used in aggregate.")
    print("   Our throttle cuts decode bytes/token -> more decode streams per unit BW (and frees BW")
    print("   for prefill). It needs FEWER BYTES, not more MACs.")
    print("-" * 82)
    print(f"{'operating point':>32} | {'AI':>7} | {'MAC util':>9} | {'bound by':>10}")
    # prefill (compute-bound) computed inline: FLOPs ~ 2*params*P + attention P^2; bytes ~ weights + KV write
    P = 2048
    pf_flops = 2.0 * m.active_params() * P + 2.0 * (2.0 * m.n_q * m.hd * P * P * m.L)
    pf_bytes = m.total_params() * m.bpw + P * m.kv_bytes_per_token_all_layers()
    pf_ai = pf_flops / pf_bytes
    pf_util = min(1.0, pf_ai / sys.ridge)
    print(f"{'PREFILL (prompt=2K, batch-1)':>32} | {pf_ai:7.0f} | {pf_util*100:8.1f}% | {'COMPUTE':>10}")
    _pts = [("decode batch-1, S=8K", 1, 8192, Cut(0, 0)),
            ("decode serving (B=%d, full KV)" % max_batch(sys, m, 8192), max(1, max_batch(sys, m, 8192)), 8192, Cut(0, 0)),
            ("decode serving (B=%d, 57%% cut)" % max_batch(sys, m, 8192, 0.43), max(1, max_batch(sys, m, 8192, 0.43)), 8192, op),
            ("decode asymptote (B->inf)", 20000, 8192, op)]
    for lab, B, S, cc in _pts:
        r = step(sys, m, B, S, cc)
        u = r['t_cmp'] / r['t']
        print(f"{lab:>32} | {r['ai']:7.1f} | {u*100:8.2f}% | {'memory':>10}")

    print("\n" + "-" * 82)
    print("H. MoE CONTRAST  --  bottleneck shifts to (active) weights + capacity -> lever shifts")
    print("   to expert-selection (patent claim 11); KV-throttle alone matters less on MoE.")
    print("-" * 82)
    for mm in (M8B, MMOE):
        S, B = 32768, 4
        r = step(sys, mm, B, S)
        fits = mm.total_params() * mm.bpw + kv_cache_bytes(mm, B, S) <= sys.dram_bytes
        print(f"  {mm.name:16s}: active {mm.active_params()/1e9:5.2f}B / total {mm.total_params()/1e9:5.2f}B "
              f"| KV share @ B={B},S={S}: {r['kv_share']*100:4.1f}%  | fits 32GB @FP8: {'yes' if fits else 'NO -> needs INT4'}")

    # ---- dump reference JSON ----
    def sweep_cut(B, S):
        base = step(sys, m, B, S)
        rows = []
        for fkv in [i/100 for i in range(0, 100, 5)]:
            r = step(sys, m, B, S, Cut(f_kv=fkv, f_w=0.08, page=16))
            rows.append(dict(f_kv=fkv, ai=r['ai'], mem_bound=r['mem_bound'],
                             speedup=base['t']/r['t'], toks_s=r['toks_s'], tokens_per_j=1/r['e_tok_j']))
        return rows

    ref = dict(
        system=dict(name=sys.name, bw_GBs=sys.bw_bytes_s/1e9, fp8_TFLOPS=sys.fp8_flops/1e12,
                    dram_GB=sys.dram_bytes/1e9, tdp_W=sys.tdp_w, ridge_flop_per_byte=sys.ridge),
        model=dict(name=m.name, params_B=m.total_params()/1e9,
                   kv_KB_per_token=m.kv_bytes_per_token_all_layers()/1024),
        roofline=[dict(S=S, **{k: step(sys, m, 1, S)[k] for k in ('ai', 'mem_bound', 'toks_s')})
                  for S in (2048, 8192, 32768, 131072)],
        sweep_8k=sweep_cut(max(1, max_batch(sys, m, 8192)), 8192),
        capacity=[dict(S=S, max_batch_full=max_batch(sys, m, S, 1.0),
                       max_batch_cut=max_batch(sys, m, S, 0.43)) for S in (8192, 16384, 32768, 65536)],
    )
    json.dump(ref, open(os.path.join(HERE, "blackhole_sysmodel_ref.json"), "w"), indent=2)
    print("\nwrote blackhole_sysmodel_ref.json")
