#!/usr/bin/env python3
"""
DRAM read-granularity model — does content-aware selection actually save READ bandwidth,
given that DRAM cannot fetch one INT4 element? It fetches a minimum burst (access
granularity), from a row it must first activate.

The question Jim raised: you only "need" address 123 at INT4, but the array/bus moves a
minimum burst (32-64 B). So what is the REAL read saving? Answer depends entirely on the
GRANULARITY of the selection:
  - block/page-granular (Quest selects KV *pages* = contiguous KBs)  -> saving preserved
  - element-granular unstructured sparsity (a survivor bitmap over scattered zeros) -> saving destroyed
"""

import math

# ---- DRAM minimum access granularity (bytes moved per column READ command) ----
# HBM2/3 pseudo-channel ~32 B; GDDR6 16b pseudo-channel ~32 B, 32b channel ~64 B; DDR5 ~64 B.
DRAM = {
    "HBM3 (pseudo-ch)":   {"burst_B": 32,  "row_B": 2048},
    "GDDR6 (pseudo-ch)":  {"burst_B": 32,  "row_B": 2048},   # Blackhole class
    "GDDR6 (32b channel)":{"burst_B": 64,  "row_B": 2048},
    "DDR5 (sub-channel)": {"burst_B": 64,  "row_B": 1024},
}

FORMATS = {"INT16": 2.0, "INT8/FP8": 1.0, "INT4": 0.5}  # bytes per element

# ---- Representative KV geometry (Llama-3-70B-ish, GQA) ----
N_KV_HEADS = 8
HEAD_DIM   = 128
PAGE_TOKENS = 16          # Quest-style KV page (contiguous block of tokens)
# contiguous KV chunk per (page) for K or V, per head — the smallest contiguous run
# a page-selector actually reads:
def kv_chunk_bytes(bytes_per_elem, per_head=True):
    if per_head:   # worst-case layout: contiguous per head
        elems = HEAD_DIM * PAGE_TOKENS
    else:          # all heads contiguous per page
        elems = N_KV_HEADS * HEAD_DIM * PAGE_TOKENS
    return elems * bytes_per_elem

def block_efficiency(block_B, burst_B):
    """Fraction of fetched bytes that are useful when reading whole contiguous blocks.
       Overhead = partial bursts at the two block boundaries (worst case)."""
    fetched = math.ceil(block_B / burst_B) * burst_B
    # worst case: block not burst-aligned -> up to one extra burst at the far end
    fetched_worst = (math.floor(block_B / burst_B) + 1) * burst_B if block_B % burst_B else block_B
    return block_B / fetched_worst

def element_fetch_fraction(keep, burst_B, elem_B):
    """Unstructured random survivors at rate `keep`. A burst holds N=burst/elem elements;
       it must be fully fetched if it contains >=1 survivor.  E[touched bursts fraction]."""
    N = burst_B / elem_B
    return 1.0 - (1.0 - keep) ** N

def fmt(x): return f"{x:5.2f}x"

print("="*84)
print("PART 1 — the contiguous KV chunk a PAGE-selector reads dwarfs the DRAM burst")
print("="*84)
print(f"  KV page = {PAGE_TOKENS} tokens, GQA {N_KV_HEADS} kv-heads x {HEAD_DIM} dim")
for name,b in FORMATS.items():
    print(f"  {name:9}: per-head page chunk = {kv_chunk_bytes(b):6.0f} B  |  full page chunk = {kv_chunk_bytes(b,per_head=False):7.0f} B")
print("  -> smallest contiguous run (per-head, INT4) is ~1 KB; DRAM burst is 32-64 B.")
print("     The selected unit is 16-256x the burst, so it FILLS bursts. Granularity ~free.")

print()
print("="*84)
print("PART 2 — BLOCK-GRANULAR (Quest KV-page) selection: read saving is preserved")
print("  keep = 43% of KV (the 57% cut). Ideal saving = 1/0.43 = 2.33x")
print("="*84)
keep = 0.43
ideal = 1/keep
print(f"  {'DRAM':22}{'fmt':10}{'chunk_B':>9}{'efficiency':>12}{'realized_saving':>16}")
for dname,cfg in DRAM.items():
    for fname,b in FORMATS.items():
        chunk = kv_chunk_bytes(b, per_head=True)   # worst-case (smallest) contiguous run
        eff = block_efficiency(chunk, cfg["burst_B"])
        realized = 1/(keep/eff)   # fetched fraction = keep/eff
        print(f"  {dname:22}{fname:10}{chunk:9.0f}{eff*100:11.1f}%{fmt(realized):>16}")
print(f"  ideal (no granularity loss) = {ideal:.2f}x  -> block selection realizes ~all of it")

print()
print("="*84)
print("PART 3 — ELEMENT-GRANULAR unstructured sparsity (naive survivor bitmap): saving DESTROYED")
print("  survivors scattered at element granularity; a burst is wasted if it holds >=1 survivor")
print("="*84)
print(f"  {'DRAM':22}{'fmt':10}{'keep':>6}{'bursts_touched':>16}{'realized_saving':>16}")
for dname,cfg in DRAM.items():
    for fname,b in FORMATS.items():
        for keep in (0.43, 0.10, 0.01):
            ff = element_fetch_fraction(keep, cfg["burst_B"], b)
            realized = 1/ff
            print(f"  {dname:22}{fname:10}{keep*100:5.0f}%{ff*100:15.1f}%{fmt(realized):>16}")
        print()

print("="*84)
print("PART 4 — break-even: how big must a contiguous selected block be for >=95% efficiency?")
print("="*84)
for dname,cfg in DRAM.items():
    burst = cfg["burst_B"]
    # efficiency ~ block/(block+burst) worst case; solve >=0.95 -> block >= 19*burst
    be = 19*burst
    print(f"  {dname:22} burst={burst:3d} B  ->  block >= ~{be:4d} B  ( = {be/32:.0f} INT4 elems ).  A KV page is ~1-16 KB, so it clears this by 50-800x.")

print()
print("="*84)
print("VERDICT")
print("="*84)
print("  * The 2x serving claim rides on KV *page* selection (Quest). A page is a contiguous")
print("    ~1-16 KB run -> 50-800x the DRAM burst -> granularity overhead is a rounding error.")
print("    The read saving survives. 2x holds.")
print("  * The FAILURE mode is UNstructured element sparsity (a survivor bitmap over scattered")
print("    zeros): at INT4 a 32 B burst holds 64 elements, so even keeping just 1% touches ~47%")
print("    of bursts -> ~2x at best, ~0 at realistic keep rates. Fine sparsity is a COMPUTE/ENERGY")
print("    win, NOT a bandwidth win, unless it is made block-structured (>= ~1 burst per run).")
print("  * RETURN traffic (compute-die-ward) is always saved: survivors are packed contiguously")
print("    before crossing the interface, so it is burst-efficient by construction. The granularity")
print("    tax is a READ-side phenomenon only -> it hits provisional #2 (avoid the read), not #1")
print("    (compress the return).")
