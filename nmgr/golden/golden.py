#!/usr/bin/env python3
"""
golden.py -- reference model for the near-memory gated-reduction (NMGR) tile.

This is the *contract*: the bit-exact functional definition that the RTL must
match, plus the stimulus/expected vectors the RTL testbench replays, plus the
statistics (achieved sparsity, return-traffic compression) that feed the
energy/cost scale-up in inference_energy_sim.py.

The op (one output tile), for each output lane m:

    acc[m] = sum over banks b, over k in [0, K):  W[b, m, k] * x[b, k]
    y[m]   = activation_gate( saturate_to_out_bits( relu(acc[m]) ) )

The cross-bank sum is the "distributed reduction"; the gate is applied
*after* the full reduction, on the memory side, so only survivors are encoded
and returned. Weights are resident (generated once); activations stream per
token (one vector per generated token).

Reference numerics are integer and deterministic, so the RTL match is
bit-exact. The math is a plain quantized linear + ReLU -- a drop-in
`torch.nn.functional.relu(x @ W.T)` produces the same values; numpy is used
here only to avoid a heavy dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import numpy as np

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install pyyaml") from e

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CFG = os.path.join(ROOT, "config", "tile.yaml")
OUT = os.path.join(ROOT, "vectors")


def signed_range(bits: int) -> tuple[int, int]:
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def requantize(cfg: dict, partial: np.ndarray) -> np.ndarray:
    """relu, then requantize the wide accumulator down to out_bits with a
    right-shift and saturation -- the standard quantized-inference back-end."""
    out_hi = signed_range(cfg["out_bits"])[1]
    return np.clip(np.maximum(0, partial) >> cfg["out_shift"], 0, out_hi).astype(np.int64)


def load_cfg(path: str = CFG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def gen_weights(cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Resident weights, shape [banks, tile_m, k_per_bank], signed w_bits."""
    lo, hi = signed_range(cfg["w_bits"])
    return rng.integers(lo, hi + 1,
                        size=(cfg["banks"], cfg["tile_m"], cfg["k_per_bank"]),
                        dtype=np.int64)


def gen_activation(cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """One streamed activation vector, shape [banks, k_per_bank], signed x_bits.

    Biased slightly negative so that ReLU produces realistic sparsity rather
    than ~50% -- decode activations are not symmetric after the nonlinearity.
    """
    lo, hi = signed_range(cfg["x_bits"])
    a = rng.integers(lo, hi + 1, size=(cfg["banks"], cfg["k_per_bank"]),
                     dtype=np.int64)
    return a


def compute_tile(cfg: dict, W: np.ndarray, x: np.ndarray) -> dict:
    """Bit-exact reference for one tile pass. Returns dense out + encoding."""
    acc_lo, acc_hi = signed_range(cfg["acc_bits"])
    out_lo, out_hi = signed_range(cfg["out_bits"])

    # cross-bank distributed reduction: sum partial products over banks and k
    partial = np.einsum("bmk,bk->m", W, x)          # [tile_m], int64
    assert partial.min() >= acc_lo and partial.max() <= acc_hi, \
        "accumulator overflow -- widen acc_bits"

    sat = requantize(cfg, partial)                   # relu + requant to out_bits

    theta = int(cfg["threshold"])
    gated = np.where(sat > theta, sat, 0)            # in-situ gate
    survive = gated != 0
    n_surv = int(survive.sum())

    m = cfg["tile_m"]
    out_bytes = cfg["out_bits"] / 8.0
    dense_bytes = m * out_bytes                      # return everything
    encoded_bytes = (m / 8.0) + n_surv * out_bytes   # bitmap + survivors only
    return {
        "dense": gated,
        "bitmap": survive.astype(np.uint8),
        "survivors": gated[survive],
        "n_survivors": n_surv,
        "sparsity": 1.0 - n_surv / m,
        "dense_bytes": dense_bytes,
        "encoded_bytes": encoded_bytes,
        "compression": encoded_bytes / dense_bytes,
        "macs": cfg["banks"] * cfg["tile_m"] * cfg["k_per_bank"],
    }


def _hexdump(path: str, arr: np.ndarray, bits: int) -> None:
    mask = (1 << bits) - 1
    width = (bits + 3) // 4
    with open(path, "w") as f:
        for v in arr.reshape(-1).tolist():
            f.write(f"{int(v) & mask:0{width}x}\n")


def build_dataset(cfg: dict):
    """Weights (resident) + a fixed set of activation vectors, generated once
    so a threshold sweep evaluates the *same* data at every theta."""
    rng = np.random.default_rng(cfg["seed"])
    W = gen_weights(cfg, rng)
    xs = [gen_activation(cfg, rng) for _ in range(cfg["vectors"])]
    return W, xs


def evaluate(cfg: dict, W: np.ndarray, xs: list[np.ndarray], theta: int) -> dict:
    """Run the tile at a given threshold over the dataset and measure both the
    transport win (sparsity, compression) and the accuracy cost vs the
    lossless theta=0 reference."""
    m = cfg["tile_m"]
    out_bytes = cfg["out_bits"] / 8.0
    n_surv = dense = enc = abs_err = disc_mass = ref_mass = 0.0
    sig_pow = noise_pow = 0.0
    max_drop = 0
    for x in xs:
        partial = np.einsum("bmk,bk->m", W, x)
        sat = requantize(cfg, partial)           # relu + requant to out_bits
        ref = np.where(sat > 0, sat, 0)          # theta=0 lossless reference
        gated = np.where(sat > theta, sat, 0)    # this threshold
        surv = gated != 0
        dropped = ref - gated                    # >0 where theta zeroed a survivor
        n_surv += int(surv.sum())
        dense += m * out_bytes
        enc += (m / 8.0) + int(surv.sum()) * out_bytes
        abs_err += float(np.abs(gated - ref).sum())
        disc_mass += float(dropped[dropped > 0].sum())
        ref_mass += float(ref.sum())
        sig_pow += float((ref.astype(float) ** 2).sum())
        noise_pow += float(((gated - ref).astype(float) ** 2).sum())
        if dropped.size:
            max_drop = max(max_drop, int(dropped.max()))
    n = m * len(xs)
    rel_l2 = (noise_pow / sig_pow) ** 0.5 if sig_pow else 0.0
    sqnr_db = float(10.0 * np.log10(sig_pow / noise_pow)) if noise_pow > 0 else float("inf")
    return {
        "theta": theta,
        "sparsity": 1.0 - n_surv / n,
        "compression": enc / dense,
        "discarded_mass_frac": (disc_mass / ref_mass) if ref_mass else 0.0,
        "rel_l2": rel_l2,
        "sqnr_db": sqnr_db,
        "max_drop": max_drop,
        "lossless": noise_pow == 0.0,
    }


def run_sweep(cfg: dict, thetas: list[int]) -> None:
    W, xs = build_dataset(cfg)
    print("=" * 78)
    print(f" NMGR threshold sweep  --  {cfg['name']}  ({cfg['vectors']} vectors,"
          f" gate={cfg['gate']})")
    print("=" * 78)
    print(f"  {'theta':>5} | {'sparsity':>9} | {'return':>7} | "
          f"{'disc.mass':>9} | {'rel-L2':>7} | {'SQNR(dB)':>8} | mode")
    print("  " + "-" * 74)
    for t in thetas:
        r = evaluate(cfg, W, xs, t)
        mode = "LOSSLESS" if r["lossless"] else "lossy"
        sqnr = " inf" if r["sqnr_db"] == float("inf") else f"{r['sqnr_db']:>8.1f}"
        print(f"  {t:>5} | {r['sparsity']:>8.1%} | {r['compression']:>6.2f}x | "
              f"{r['discarded_mass_frac']:>8.2%} | {r['rel_l2']:>6.1%} | {sqnr:>8} | {mode}")
    print("  " + "-" * 74)
    print("  Layer-level signal fidelity vs the theta=0 (lossless) output:")
    print("    disc.mass = fraction of true output magnitude dropped")
    print("    rel-L2    = ||error|| / ||signal||   (lower is better)")
    print("    SQNR(dB)  = 10*log10(signal/noise)   (higher is better; inf = exact)")
    print("  NOTE: this is per-layer numerical loss, NOT task accuracy. Whether it")
    print("  compounds or washes out across layers is empirical -- only end-to-end")
    print("  eval (perplexity/accuracy vs theta) on a real model settles it.")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description="NMGR golden reference / threshold sweep")
    ap.add_argument("--sweep", metavar="T0,T1,...",
                    help="comma-separated thresholds to sweep, e.g. 0,1,2,4,8,16")
    args = ap.parse_args()

    cfg = load_cfg()
    if args.sweep:
        thetas = [int(t) for t in args.sweep.split(",")]
        run_sweep(cfg, thetas)
        return

    rng = np.random.default_rng(cfg["seed"])
    os.makedirs(OUT, exist_ok=True)

    K = cfg["banks"] * cfg["k_per_bank"]
    W = gen_weights(cfg, rng)
    _hexdump(os.path.join(OUT, "weights.hex"), W, cfg["w_bits"])

    agg = {"sparsity": [], "compression": [], "n_survivors": []}
    for i in range(cfg["vectors"]):
        x = gen_activation(cfg, rng)
        r = compute_tile(cfg, W, x)
        _hexdump(os.path.join(OUT, f"x_{i:02d}.hex"), x, cfg["x_bits"])
        _hexdump(os.path.join(OUT, f"expected_{i:02d}.hex"), r["dense"], cfg["out_bits"])
        _hexdump(os.path.join(OUT, f"bitmap_{i:02d}.hex"), r["bitmap"], 1)
        for k in agg:
            agg[k].append(r[k])

    meta = {
        "config": cfg,
        "contraction_K": K,
        "macs_per_tile": cfg["banks"] * cfg["tile_m"] * cfg["k_per_bank"],
        "mean_sparsity": float(np.mean(agg["sparsity"])),
        "mean_compression": float(np.mean(agg["compression"])),
        "mean_survivors": float(np.mean(agg["n_survivors"])),
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # -- human summary ------------------------------------------------------
    print("=" * 66)
    print(f" NMGR golden  --  {cfg['name']}")
    print("=" * 66)
    print(f"  tile: {cfg['tile_m']} outputs x K={K}  "
          f"({cfg['banks']} banks x {cfg['k_per_bank']})")
    print(f"  numerics: x=int{cfg['x_bits']} w=int{cfg['w_bits']} "
          f"acc=int{cfg['acc_bits']} out=int{cfg['out_bits']}  gate={cfg['gate']}")
    print(f"  MACs / tile pass        : {meta['macs_per_tile']:,}")
    print(f"  mean activation sparsity: {meta['mean_sparsity']:.1%}")
    print(f"  return-traffic vs dense : {meta['mean_compression']:.2f}x "
          f"(bitmap + survivors only)")
    print(f"  vectors written         : {cfg['vectors']}  -> {OUT}")
    print("-" * 66)

    # -- analytical scale-up (this is how a tile becomes a model) ----------
    d_model, n_layers = 4096, 32
    tiles_per_proj = (d_model / cfg["tile_m"]) * (d_model / K)
    projs = 4  # rough: qkv-ish + ffn passes per block
    tiles_per_token = tiles_per_proj * projs * n_layers
    print("  scale-up (illustrative, d_model=4096, 32 layers):")
    print(f"    tiles / token ~ {tiles_per_token:,.0f}")
    print(f"    -> feed tile area/power (from synthesis) x this count into")
    print(f"       inference_energy_sim.py for tokens/s, W, and $/token.")
    print("=" * 66)


if __name__ == "__main__":
    main()
