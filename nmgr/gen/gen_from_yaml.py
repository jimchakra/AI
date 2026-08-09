#!/usr/bin/env python3
"""
gen_from_yaml.py — single-source-of-truth fan-out.

Reads the ONE micro-arch spec (config/tile.yaml) and emits the downstream
artifacts that would otherwise drift apart across teams:

    config/tile.yaml  (SSOT)
        ├─► rtl/nmgr_params.vh   Verilog params for the RTL
        ├─► fw/nmgr_regs.h       firmware register / parameter map
        └─► dv/nmgr_cfg.json     DV / testbench configuration

Consistency is guaranteed by construction: every artifact is generated, never
hand-authored. check/check_consistency.py enforces that the committed artifacts
(and the RTL module defaults) still agree with the YAML — so drift fails CI.
This is the concurrent-co-design methodology in miniature.
"""
import json, os, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CFG  = os.path.join(ROOT, "config", "tile.yaml")

BANNER_V = "// AUTO-GENERATED from config/tile.yaml by gen/gen_from_yaml.py — DO NOT EDIT.\n"
BANNER_C = "/* AUTO-GENERATED from config/tile.yaml by gen/gen_from_yaml.py — DO NOT EDIT. */\n"

# canonical parameter set shared across all artifacts (name -> yaml key)
PARAMS = [
    ("BANKS",      "banks"),
    ("TILE_M",     "tile_m"),
    ("K_PER_BANK", "k_per_bank"),
    ("X_BITS",     "x_bits"),
    ("W_BITS",     "w_bits"),
    ("ACC_BITS",   "acc_bits"),
    ("OUT_SHIFT",  "out_shift"),
    ("OUT_BITS",   "out_bits"),
    ("THRESHOLD",  "threshold"),
]


def load():
    with open(CFG) as f:
        return yaml.safe_load(f)


def emit_verilog(cfg):
    K = cfg["banks"] * cfg["k_per_bank"]
    lines = [BANNER_V, "`ifndef NMGR_PARAMS_VH\n`define NMGR_PARAMS_VH\n"]
    for name, key in PARAMS:
        lines.append(f"`define NMGR_{name} {cfg[key]}\n")
    lines.append(f"`define NMGR_CONTRACTION_K {K}\n")
    lines.append("`endif\n")
    with open(os.path.join(ROOT, "rtl", "nmgr_params.vh"), "w") as f:
        f.writelines(lines)


def emit_fw(cfg):
    K = cfg["banks"] * cfg["k_per_bank"]
    lines = [BANNER_C, "#ifndef NMGR_REGS_H\n#define NMGR_REGS_H\n\n",
             "/* micro-arch parameters (must match RTL by construction) */\n"]
    for name, key in PARAMS:
        lines.append(f"#define NMGR_{name:<11} {cfg[key]}\n")
    lines.append(f"#define NMGR_CONTRACTION_K (NMGR_BANKS * NMGR_K_PER_BANK)\n\n")
    lines.append("/* illustrative MMIO register map (word offsets) */\n")
    for i, r in enumerate(["CTRL", "STATUS", "THRESHOLD", "SURV_COUNT", "BITMAP_LO", "BITMAP_HI"]):
        lines.append(f"#define NMGR_REG_{r:<11} 0x{i*4:02X}\n")
    lines.append("\n#endif /* NMGR_REGS_H */\n")
    with open(os.path.join(ROOT, "fw", "nmgr_regs.h"), "w") as f:
        f.writelines(lines)


def emit_dv(cfg):
    K = cfg["banks"] * cfg["k_per_bank"]
    d = {name.lower(): cfg[key] for name, key in PARAMS}
    d["contraction_k"] = K
    d["_generated_from"] = "config/tile.yaml"
    with open(os.path.join(ROOT, "dv", "nmgr_cfg.json"), "w") as f:
        json.dump(d, f, indent=2)
        f.write("\n")


def main():
    cfg = load()
    emit_verilog(cfg); emit_fw(cfg); emit_dv(cfg)
    print("generated from config/tile.yaml:")
    print("  rtl/nmgr_params.vh   fw/nmgr_regs.h   dv/nmgr_cfg.json")


if __name__ == "__main__":
    main()
