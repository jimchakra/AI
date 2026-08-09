#!/usr/bin/env python3
"""
inference_energy_sim.py
=======================

A first-order energy model for transformer inference that makes one thing
visible: for memory-bound decode, *moving* the numbers costs far more than
*computing* with them -- and completing the reduction near the memory, so you
return only the survivors, attacks the dominant term.

Companion to the essay "Small Frees Big" (jimchakra.github.io/AI).

The model is deliberately simple and transparent. It is meant to expose
*relationships* and orders of magnitude, not to sign off a design. Every
constant is adjustable from the command line and cited below.

Energy constants (illustrative, ~5 nm-class node)
-------------------------------------------------
Arithmetic has scaled with each process node; off-chip data movement has not,
because its cost is set by the physics of driving bits across a package. So the
compute:movement ratio has *widened* over time.

A memory read is modeled as additive segments, which is the honest way to see
what near-memory actually buys you:

    total read = DRAM array access  +  transport to the compute site

The array access is *unavoidable* wherever the compute sits -- near-memory does
NOT make it free. What near-memory removes is the off-die transport (PHY +
package + interposer + host cache fill). Getting this right is the difference
between a defensible model and a marketing chart.

  * Low-precision MAC:            ~0.1 - 1.5 pJ   (raw arithmetic is tens of fJ)
  * DRAM array access (core):     ~6 pJ / byte   (unavoidable, any location)
  * Base-die hop (in-stack):      ~3 pJ / byte
  * HBM3 PHY to host:             ~26 pJ / byte  (array+PHY ~= 32 pJ/B total)
  * Off-package LPDDR/GDDR I/O:   ~154 pJ / byte (array+I/O ~= 160; ~640 pJ/word)
  * On-chip SRAM touch:           ~1 pJ / byte

Sources
-------
  * M. Horowitz, "Computing's Energy Problem (and what we can do about it),"
    ISSCC 2014.
  * W. Dally, "Energy-Efficient AI Hardware," Stanford AHA Retreat keynote, 2023
    (LPDDR still ~640 pJ/word; on-chip SRAM ~5 pJ/word; "an add is worth ~10 um
    of on-chip movement").
  * A. Boroumand et al., "Google Workloads for Consumer Devices: Mitigating
    Data-Movement Bottlenecks," ASPLOS 2018 (~62.7% of system energy is movement).

License: MIT.  Author: Jim Chou.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Energy model                                                                #
# --------------------------------------------------------------------------- #

# Additive movement segments, pJ/byte (see module docstring for sources).
DRAM_ARRAY = 6.0     # reading bytes out of the DRAM array -- unavoidable anywhere
BASE_HOP = 3.0       # DRAM die -> base logic die, short in-stack hop
IO_PJB = {           # transport from the memory stack out to the host compute
    "hbm": 26.0,     # HBM3 PHY across the stack        (array+I/O ~= 32 pJ/B)
    "dram": 154.0,   # off-package LPDDR/GDDR PHY+pkg    (array+I/O ~= 160 pJ/B)
}
SRAM_PJB = 1.0       # on-chip SRAM touch


def host_read(nbytes: float, mem: str) -> float:
    """Read from the memory stack all the way to host compute."""
    return nbytes * (DRAM_ARRAY + IO_PJB[mem])


def local_read(nbytes: float) -> float:
    """Read from the array to compute on the memory-side base die."""
    return nbytes * (DRAM_ARRAY + BASE_HOP)


def return_to_host(nbytes: float, mem: str) -> float:
    """Send a base-die result out to the host (no array access; it's a result)."""
    return nbytes * IO_PJB[mem]

# pJ per multiply-accumulate, by weight precision (~5 nm-class, illustrative).
MAC_PJ = {
    "fp16": 1.5,
    "int8": 0.40,
    "int4": 0.15,
    "fp4": 0.10,
}

# bytes per stored weight, by precision.
WEIGHT_BYTES = {"fp16": 2.0, "int8": 1.0, "int4": 0.5, "fp4": 0.5}


@dataclass
class ModelSpec:
    """A decoder-only transformer, described just enough to count work."""

    name: str = "llama-7b-ish"
    n_params: float = 7.0e9          # total weight parameters
    n_layers: int = 32               # transformer blocks
    d_model: int = 4096              # hidden size (proxy for KV width)
    weight_prec: str = "int4"        # fp16 | int8 | int4 | fp4
    kv_bytes: float = 2.0            # bytes per cached K/V element (fp16=2)
    seq_len: int = 8192              # context length resident in the KV cache
    act_sparsity: float = 0.50       # fraction of returned activations that gate to zero
    ffn_ratio: int = 4               # FFN hidden = ffn_ratio * d_model
    mem: str = "hbm"                 # conventional baseline memory: hbm | dram

    def weight_bytes(self) -> float:
        return self.n_params * WEIGHT_BYTES[self.weight_prec]

    def macs_per_token(self) -> float:
        # Decode: each parameter participates in ~1 MAC per generated token.
        return self.n_params

    def kv_bytes_per_token(self) -> float:
        # Read the full K and V cache for attention on each decode step.
        return 2.0 * self.n_layers * self.d_model * self.seq_len * self.kv_bytes

    def output_bytes_per_token(self) -> float:
        # Per-layer output activations that the host orchestrates between blocks
        # -- the stream that crosses the bus, and the stream gating acts on.
        return self.n_layers * self.d_model * self.kv_bytes


@dataclass
class Breakdown:
    """Per-token energy, in picojoules, split into its parts."""

    compute: float = 0.0
    weights: float = 0.0
    kv: float = 0.0
    activations: float = 0.0
    label: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def movement(self) -> float:
        return self.weights + self.kv + self.activations

    @property
    def total(self) -> float:
        return self.compute + self.movement

    @property
    def movement_frac(self) -> float:
        return self.movement / self.total if self.total else 0.0


# --------------------------------------------------------------------------- #
# Scenarios                                                                    #
# --------------------------------------------------------------------------- #

def conventional(m: ModelSpec) -> Breakdown:
    """
    Host-side compute. Weights and the KV cache are read from the memory stack
    all the way to the host every token (array access + transport). This is the
    memory-bound decode regime most inference lives in.
    """
    b = Breakdown(label=f"Conventional  (host compute, {m.mem.upper()})")
    b.compute = m.macs_per_token() * MAC_PJ[m.weight_prec]
    b.weights = host_read(m.weight_bytes(), m.mem)
    b.kv = host_read(m.kv_bytes_per_token(), m.mem)
    b.activations = m.output_bytes_per_token() * SRAM_PJB  # host-side, on-die
    return b


def near_memory(m: ModelSpec) -> Breakdown:
    """
    LOCALITY win. Compute runs on the memory-side base logic die: weights and KV
    are read from the array over a short in-stack hop (array access is still
    paid -- it is unavoidable), and only the per-layer output activations are
    returned to the host. The big term -- shipping every weight across the
    package -- disappears.
    """
    b = Breakdown(label="Near-memory  (locality, no gating)")
    b.compute = m.macs_per_token() * MAC_PJ[m.weight_prec]       # same math
    b.weights = local_read(m.weight_bytes())                    # array + hop
    b.kv = local_read(m.kv_bytes_per_token())                   # array + hop
    b.activations = return_to_host(m.output_bytes_per_token(), m.mem)
    return b


def near_memory_gated(m: ModelSpec) -> Breakdown:
    """
    GATING win, on top of locality. The output is gated at the memory (fraction
    `act_sparsity` discarded before it moves); only survivors are returned.

    Honest caveat: in weight-bound decode the returned-activation term is already
    small, so gating is a second-order effect here. Its payoff grows with the
    activation-return share -- prefill, MoE routing, long-context attention --
    where far more data crosses the bus and the gate decision must be made at
    the memory to be worth anything.
    """
    b = Breakdown(label="Near-memory + output gating")
    b.compute = m.macs_per_token() * MAC_PJ[m.weight_prec]
    b.weights = local_read(m.weight_bytes())
    b.kv = local_read(m.kv_bytes_per_token())
    survivors = (1.0 - m.act_sparsity) * m.output_bytes_per_token()
    b.activations = return_to_host(survivors, m.mem)
    b.extra["gated_away_bytes"] = m.act_sparsity * m.output_bytes_per_token()
    return b


# --------------------------------------------------------------------------- #
# Roofline                                                                     #
# --------------------------------------------------------------------------- #

def arithmetic_intensity(m: ModelSpec) -> dict:
    """MACs per byte moved, and whether the step is compute- or memory-bound."""
    macs = m.macs_per_token()
    bytes_moved = m.weight_bytes() + m.kv_bytes_per_token()
    ai = macs / bytes_moved
    # Ridge point ~ energy per byte moved / energy per MAC (balance crossover).
    ridge = (DRAM_ARRAY + IO_PJB[m.mem]) / MAC_PJ[m.weight_prec]
    return {
        "intensity_mac_per_byte": ai,
        "ridge_point": ridge,
        "bound": "memory-bound" if ai < ridge else "compute-bound",
    }


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #

def _fmt_pj(pj: float) -> str:
    for unit, scale in (("mJ", 1e9), ("uJ", 1e6), ("nJ", 1e3), ("pJ", 1.0)):
        if pj >= scale:
            return f"{pj / scale:8.2f} {unit}"
    return f"{pj:8.2f} pJ"


def _bar(frac: float, width: int = 28) -> str:
    fill = int(round(frac * width))
    return "#" * fill + "." * (width - fill)


def print_report(m: ModelSpec) -> None:
    a = conventional(m)
    b = near_memory(m)
    c = near_memory_gated(m)

    print("=" * 72)
    print(f" Inference energy model  --  {m.name}")
    print("=" * 72)
    print(f"  params={m.n_params/1e9:.0f}B  layers={m.n_layers}  d_model={m.d_model}"
          f"  weights={m.weight_prec}  context={m.seq_len}  baseline={m.mem.upper()}")
    print(f"  weight traffic/token : {m.weight_bytes()/1e6:9.1f} MB")
    print(f"  KV traffic/token     : {m.kv_bytes_per_token()/1e6:9.1f} MB")
    print(f"  layer outputs/token  : {m.output_bytes_per_token()/1e6:9.3f} MB"
          f"   (gate sparsity {m.act_sparsity:.0%})")
    print()

    for bd in (a, b, c):
        print(f"  {bd.label}")
        print(f"    compute     {_fmt_pj(bd.compute)}   {_bar(bd.compute/bd.total)}")
        print(f"    weights     {_fmt_pj(bd.weights)}   {_bar(bd.weights/bd.total)}")
        print(f"    kv-cache    {_fmt_pj(bd.kv)}   {_bar(bd.kv/bd.total)}")
        print(f"    outputs     {_fmt_pj(bd.activations)}   {_bar(bd.activations/bd.total)}")
        print(f"    --> total   {_fmt_pj(bd.total)}   "
              f"(data movement = {bd.movement_frac:.1%})")
        print()

    loc = (a.total - b.total) / a.total if a.total else 0.0
    gat = (b.total - c.total) / b.total if b.total else 0.0
    print("-" * 72)
    print(f"  Locality (conventional -> near-memory):  {loc:5.1%} less energy"
          f"   ({a.total / b.total:.1f}x)")
    print(f"  Gating   (near-memory -> + output gate): {gat:5.1%} less energy"
          f"   (2nd-order in weight-bound decode; grows with activation share)")

    r = arithmetic_intensity(m)
    print(f"  Arithmetic intensity: {r['intensity_mac_per_byte']:.2f} MAC/byte"
          f"  |  ridge {r['ridge_point']:.0f}  ->  {r['bound']}")
    print("=" * 72)


def save_plot(m: ModelSpec, path: str = "energy_breakdown.png") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed -- skipping plot; pip install matplotlib)")
        return

    scen = [conventional(m), near_memory(m), near_memory_gated(m)]
    parts = ["compute", "weights", "kv", "activations"]
    colors = ["#aec1d6", "#4c6f9c", "#3c597f", "#88a4c6"]
    labels = ["Conventional", "Near-memory", "+ gating"]
    data = [[getattr(s, p) / 1e6 for p in parts] for s in scen]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bottom = [0.0] * len(scen)
    for i, p in enumerate(parts):
        vals = [data[j][i] for j in range(len(scen))]
        ax.bar(labels, vals, bottom=bottom, label=p, color=colors[i], width=0.55)
        bottom = [bottom[j] + vals[j] for j in range(len(scen))]
    ax.set_ylabel("energy per token  (uJ)")
    ax.set_title(f"Per-token inference energy -- {m.name}")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"(wrote {path})")


# --------------------------------------------------------------------------- #
# Link framing overhead  (mirrors the RTL packetizer, rtl/nmgr_packetizer.v)   #
# --------------------------------------------------------------------------- #
# The compressed return is not free on the wire: each tile packet carries a
# fixed header + a survivor bitmap, then the packed survivors. This models the
# effective bytes actually crossing the SerDes/CXL link, so "return only
# survivors" is costed honestly against the fixed framing overhead.

def link_bytes(count, M=64, ow_bytes=1, flit_bytes=4):
    """Effective link bytes for one tile packet (header + bitmap + survivors),
    flit-granular, exactly as nmgr_packetizer frames it."""
    header  = flit_bytes                                   # 1 flit
    bitmap  = ((M + 8*flit_bytes - 1) // (8*flit_bytes)) * flit_bytes   # ceil(M bits / flit)
    lanes   = flit_bytes // ow_bytes
    payload = ((count + lanes - 1) // lanes) * flit_bytes  # packed, flit-granular
    return dict(header=header, bitmap=bitmap, payload=payload,
                total=header+bitmap+payload)

def framing_table(M=64, ow_bytes=1, flit_bytes=4):
    """Effective bytes vs a dense M-element return, across survivor counts."""
    dense = M * ow_bytes
    rows = []
    for count in (M, 32, 13, 8, 3, 0):
        lb = link_bytes(count, M, ow_bytes, flit_bytes)
        overhead = lb["header"] + lb["bitmap"]
        rows.append(dict(count=count, sparsity=1.0-count/M, **lb,
                         dense=dense, vs_dense=lb["total"]/dense,
                         overhead_frac=overhead/lb["total"]))
    return rows

def dollars_per_mtok(energy_pj_per_token, price_kwh=0.10, pue=1.3):
    """First-order electricity cost of the MODELED data-movement+compute energy
    for 1M tokens. This is the physics floor the near-memory locality attacks —
    NOT full system $/token (which also carries idle draw, host, networking)."""
    joules = energy_pj_per_token * 1e-12 * 1e6      # J per 1M tokens
    kwh = joules / 3.6e6
    return kwh * price_kwh * pue

def print_dollars(m, price_kwh=0.10, pue=1.3):
    a, b, c = conventional(m), near_memory(m), near_memory_gated(m)
    print(f"\nModeled energy cost per 1M tokens  (electricity ${price_kwh}/kWh, PUE {pue}, {m.mem.upper()} baseline):")
    print(f"  {'scenario':22s} {'energy/token':>13} {'$/1M tokens':>13}")
    for lab, bd in (("conventional", a), ("near-memory", b), ("near-memory+gated", c)):
        e = bd.total
        print(f"  {lab:22s} {e/1e9:10.2f} mJ {dollars_per_mtok(e):12.4f}")
    print(f"  -> near-memory cuts the modeled movement-energy bill "
          f"{a.total/b.total:.1f}x vs {m.mem.upper()}.")
    print("  (Floor on the data path only; not a full-system $/token — those carry host/idle/network.)")

def print_framing(M=64):
    print("\nLink framing overhead (compressed return, M=%d, 8-bit survivors, 32-bit flits):" % M)
    print(f"{'survivors':>9} {'spars%':>7} {'hdr+bmp':>8} {'payload':>8} {'link B':>7} {'vs dense':>9} {'overhead%':>10}")
    for r in framing_table(M):
        print(f"{r['count']:9d} {100*r['sparsity']:7.0f} {r['header']+r['bitmap']:8d} "
              f"{r['payload']:8d} {r['total']:7d} {r['vs_dense']:8.2f}x {100*r['overhead_frac']:9.0f}%")
    print("Fixed 12 B overhead per tile packet (4 B header + 8 B bitmap); amortized when")
    print("survivors are many, dominant when few — the honest floor on compressed return.")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="First-order transformer inference energy model "
                    "(compute vs. data movement; near-memory gating).")
    p.add_argument("--name", default="llama-7b-ish")
    p.add_argument("--params", type=float, default=7.0e9, help="weight params")
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--d-model", type=int, default=4096)
    p.add_argument("--prec", choices=list(MAC_PJ), default="int4",
                   help="weight precision")
    p.add_argument("--seq", type=int, default=8192, help="context length")
    p.add_argument("--sparsity", type=float, default=0.50,
                   help="fraction of returned activations gated to zero (0..1)")
    p.add_argument("--mem", choices=list(IO_PJB), default="hbm",
                   help="conventional baseline memory (hbm | dram)")
    p.add_argument("--plot", action="store_true", help="save energy_breakdown.png")
    p.add_argument("--dollars", action="store_true",
                   help="print modeled $/1M-token (electricity floor) per scenario")
    p.add_argument("--framing", action="store_true",
                   help="show link framing overhead (effective bytes vs dense return)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    m = ModelSpec(
        name=args.name, n_params=args.params, n_layers=args.layers,
        d_model=args.d_model, weight_prec=args.prec, seq_len=args.seq,
        act_sparsity=args.sparsity, mem=args.mem,
    )
    print_report(m)
    if args.dollars:
        print_dollars(m)
    if args.framing:
        print_framing()
    if args.plot:
        save_plot(m)


if __name__ == "__main__":
    main()
