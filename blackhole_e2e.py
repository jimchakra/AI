#!/usr/bin/env python3
"""
blackhole_e2e.py — is decode actually the bottleneck of AI inference?

Decomposes end-to-end request time into PREFILL (compute-bound, parallel, fast) and
DECODE (memory-bound, sequential, slow-per-token) for a range of prompt:generation
profiles, and applies Amdahl's law to show how a decode-only speedup flows through to
the end-to-end result.

Conclusion (see table): for any workload with non-trivial generation length, decode is
~all of the wall-clock and accelerator time -- so a 2-3x decode capacity gain is a 2-3x
gain on the dominant cost. Prefill wins only when the prompt is huge AND the output is
near-zero (encode/embedding, classify/rerank). This is an INFERENCE-serving statement;
training is compute-bound and out of scope.

First-order, batch-1 latency view on the Blackhole system model (blackhole_sysmodel.py).
Shapes, not datasheet absolutes.
"""
import json, os
from blackhole_sysmodel import SystemSpec, M8B

sys, m = SystemSpec(), M8B
KV_TOK = m.kv_bytes_per_token_all_layers()
W = m.total_params() * m.bpw


def t_prefill(P: int) -> float:
    """Whole prompt in one pass: compute-bound for realistic P (attention is P^2)."""
    flops = 2 * m.total_params() * P + 2 * (2 * m.n_q * m.hd * P * P * m.L)
    byts = W + P * KV_TOK
    return max(flops / sys.fp8_flops, byts / sys.bw_bytes_s)


def t_decode(P: int, G: int) -> float:
    """G sequential memory-bound steps; context grows from P to P+G (exact sum)."""
    # sum_{i=0}^{G-1} (W + (P+i)*KV_TOK)/BW  = (G*W + KV_TOK*(G*P + G(G-1)/2)) / BW
    total_bytes = G * W + KV_TOK * (G * P + G * (G - 1) / 2)
    return total_bytes / sys.bw_bytes_s


def amdahl(fd: float, s: float) -> float:
    """End-to-end speedup if decode (fraction fd of time) is accelerated by factor s."""
    return 1.0 / ((1 - fd) + fd / s)


PROFILES = [
    ("encode / embedding (G=0)", 4000, 1),
    ("classify / rerank (P>>G)", 4000, 20),
    ("RAG, short answer",        8000, 200),
    ("chat, typical",            1000, 500),
    ("code / long answer",        500, 2000),
    ("reasoning / agentic CoT",   500, 8000),
]

if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    print("=" * 108)
    print("IS DECODE THE BOTTLENECK?  end-to-end prefill+decode decomposition (8B GQA on Blackhole, batch-1)")
    print("=" * 108)
    print(f"{'workload':>26} | {'P':>6} | {'G':>6} | {'prefill':>9} | {'decode':>10} | "
          f"{'decode %':>8} | {'e2e @2.15x':>10} | {'e2e @2.94x':>10}")
    print("-" * 108)
    rows = []
    for name, P, G in PROFILES:
        tp, td = t_prefill(P), t_decode(P, G)
        fd = td / (tp + td)
        rows.append(dict(workload=name, P=P, G=G, prefill_ms=tp * 1e3, decode_ms=td * 1e3,
                         decode_frac=fd, e2e_2_15=amdahl(fd, 2.15), e2e_2_94=amdahl(fd, 2.94)))
        print(f"{name:>26} | {P:6d} | {G:6d} | {tp*1e3:8.1f}ms | {td*1e3:9.0f}ms | "
              f"{fd*100:7.1f}% | {amdahl(fd,2.15):9.2f}x | {amdahl(fd,2.94):9.2f}x")
    print("-" * 108)
    print("Prefill: compute-bound, parallel, ~100% MFU -> fast even for long prompts.")
    print("Decode : memory-bound, sequential, re-reads weights+KV every token -> dominates for any real G.")
    print("Prefill wins only at extreme prompt:generation ratios (encode/embedding, classify).")
    print("Trend: reasoning / agentic / code (G in the thousands) are ~100% decode AND fastest-growing.")
    json.dump(dict(model=m.name, system=sys.name, profiles=rows),
              open(os.path.join(HERE, "blackhole_e2e_ref.json"), "w"), indent=2)
    print("\nwrote blackhole_e2e_ref.json")
