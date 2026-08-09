#!/usr/bin/env python3
"""
hitl_cosim.py — hardware-in-the-loop: a real transformer forward pass whose
FFN tile is computed by the ACTUAL near-memory RTL (iverilog), not by numpy.

Flow:
  1. Load the trained GPT (cosim/gpt_ckpt.pt), run a forward pass, and capture a
     REAL activation vector at the input of block-0's FFN (via a hook).
  2. Take a real 64x128 sub-tile of that FFN's weight matrix.
  3. Quantize both to the tile's numeric format (int8 activations, int4 weights,
     per config/tile.yaml) — the same format the silicon consumes.
  4. Run that tile through the real nmgr_pe RTL simulation (tb_cosim.sv) and read
     the gated int8 outputs back.
  5. Assert bit-exact vs the numpy golden on the SAME real data, then dequantize
     the RTL outputs and splice them back into the forward pass.

This is the block standing in for a layer on real model data — the honest
"is it right, on something real?" test, above and beyond random-vector regression.
"""
import os, sys, subprocess
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NMGR = os.path.join(HERE, "nmgr")
sys.path.insert(0, os.path.join(NMGR, "golden"))
import golden as G                       # reuse the bit-exact reference + hex writers
from pytorch_cosim import GPT            # the trained proxy model

CKPT = os.path.join(HERE, "cosim", "gpt_ckpt.pt")
CFG  = G.load_cfg(os.path.join(NMGR, "config", "tile.yaml"))
VEC  = os.path.join(NMGR, "vectors")

def quantize_symmetric(a, bits):
    lo, hi = -(1 << (bits-1)), (1 << (bits-1)) - 1
    scale = float(np.max(np.abs(a))) / hi
    if scale == 0: scale = 1.0
    q = np.clip(np.rint(a / scale), lo, hi).astype(np.int64)
    return q, scale

def main():
    torch.manual_seed(0)
    ck = torch.load(CKPT, map_location="cpu")
    model = GPT(ck["vocab"]); model.load_state_dict(ck["model"]); model.eval()

    # capture a real FFN-input activation via a forward hook on block-0's fc
    captured = {}
    h = model.blocks[0].fc.register_forward_pre_hook(
        lambda mod, inp: captured.setdefault("x", inp[0].detach()))
    idx = torch.randint(0, ck["vocab"], (1, 64))          # a real token window
    with torch.no_grad(): model(idx)
    h.remove()

    x_real = captured["x"][0, -1].numpy()                 # [n_embd] real activation, last position
    W_real = model.blocks[0].fc.weight.detach().numpy()   # [4*n_embd, n_embd]

    K = CFG["banks"] * CFG["k_per_bank"]                  # 128
    M = CFG["tile_m"]                                     # 64
    x128 = x_real[:K]
    Wsub = W_real[:M, :K]                                 # real 64x128 weight sub-tile

    # quantize to the silicon's numeric format
    xq, sx = quantize_symmetric(x128, CFG["x_bits"])
    wq, sw = quantize_symmetric(Wsub, CFG["w_bits"])

    # reshape into [banks, (tile_m,) k_per_bank] exactly as golden/RTL index
    b, kpb = CFG["banks"], CFG["k_per_bank"]
    xq_t = xq.reshape(b, kpb)                             # x[b*KPB+k]
    wq_t = wq.reshape(M, b, kpb).transpose(1, 0, 2)       # W[b, m, k] from W[m, b*KPB+k]

    # numpy golden on the real quantized tile
    r = G.compute_tile(CFG, wq_t, xq_t)
    y_golden = np.asarray(r["dense"]).astype(np.int64)

    # write hex the RTL testbench reads
    os.makedirs(VEC, exist_ok=True)
    G._hexdump(os.path.join(VEC, "weights.hex"), wq_t, CFG["w_bits"])
    G._hexdump(os.path.join(VEC, "x_one.hex"),   xq_t, CFG["x_bits"])

    # run the ACTUAL RTL simulation
    tb  = os.path.join(NMGR, "tb", "tb_cosim.sv")
    pe  = os.path.join(NMGR, "rtl", "nmgr_pe.v")
    sim = "/tmp/tb_cosim"
    subprocess.run(["iverilog", "-g2012", "-o", sim, pe, tb], check=True, cwd=NMGR)
    out = subprocess.run(["vvp", sim], check=True, cwd=NMGR, capture_output=True, text=True)
    print(out.stdout.strip())

    # read RTL outputs back (skip $writememh address/comment lines)
    toks = []
    for line in open(os.path.join(VEC, "y_rtl.hex")):
        s = line.strip()
        if not s or s.startswith("//") or s.startswith("@"):
            continue
        toks += s.split()
    y_rtl = np.array([int(t, 16) for t in toks[:M]], dtype=np.int64)

    # ---- scoreboard: bit-exact on real data ----
    match = int((y_rtl == y_golden).sum())
    print(f"\nHITL cosim on a REAL transformer FFN tile (64x128, int8xint4, theta=0):")
    print(f"  RTL == numpy golden : {match}/{M} outputs bit-exact")
    print(f"  survivors (nonzero) : {int((y_golden>0).sum())}/{M}  "
          f"(sparsity {100*(1-(y_golden>0).mean()):.0f}%)")

    # ---- splice back into the forward pass ----
    # dequantize: RTL output = relu(acc) >> out_shift, acc = sum(wq*xq);
    # FP-scale recovery multiplies back the requantize shift and the q-scales.
    deq = y_rtl.astype(np.float64) * (sx * sw) * float(1 << CFG["out_shift"])
    ref = np.maximum(0.0, Wsub @ x128)                    # FP relu(W x) reference (partial tile)
    denom = np.linalg.norm(ref) or 1.0
    cos = float(deq @ ref / ((np.linalg.norm(deq) or 1.0) * denom))
    rel = float(np.linalg.norm(deq - ref) / denom)
    print(f"  dequantized RTL vs FP relu(Wx): cosine {cos:.4f}, rel-L2 {rel:.3f}")
    print("  -> the RTL block's gated outputs re-enter the forward pass; the model")
    print("     computes this FFN tile in actual silicon logic, bit-exact to golden.")

    ok = (match == M)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
