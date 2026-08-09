#!/usr/bin/env python3
"""
check_consistency.py — enforce the single-source-of-truth.

Parses the micro-arch parameters out of EVERY artifact independently:
  - config/tile.yaml            (the source)
  - rtl/nmgr_params.vh          (generated Verilog)
  - fw/nmgr_regs.h              (generated firmware header)
  - dv/nmgr_cfg.json            (generated DV config)
  - rtl/nmgr_pe.v / nmgr_encoder.v  (RTL module DEFAULT parameters)

and asserts they all agree. If a hand-edit ever drifts one artifact from the
YAML (the classic cross-team spec gap), this fails — in CI, at commit time,
not at bring-up. Exit non-zero on any mismatch.
"""
import json, os, re, sys, yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

KEYS = ["BANKS", "TILE_M", "K_PER_BANK", "X_BITS", "W_BITS",
        "ACC_BITS", "OUT_SHIFT", "OUT_BITS", "THRESHOLD"]
YKEY = {k: k.lower() for k in KEYS}


def from_yaml():
    c = yaml.safe_load(open(os.path.join(ROOT, "config", "tile.yaml")))
    return {k: int(c[YKEY[k]]) for k in KEYS}


def from_defines(path, prefix):
    txt = open(path).read()
    out = {}
    for k in KEYS:
        m = re.search(rf"{prefix}{k}\b\s+(\d+)", txt)
        if m:
            out[k] = int(m.group(1))
    return out


def from_json(path):
    d = json.load(open(path))
    return {k: int(d[k.lower()]) for k in KEYS if k.lower() in d}


def from_rtl(path, mapping):
    # parse `parameter NAME = N` defaults from a module header
    txt = open(path).read()
    out = {}
    for key, pname in mapping.items():
        m = re.search(rf"parameter\s+{pname}\s*=\s*(\d+)", txt)
        if m:
            out[key] = int(m.group(1))
    return out


def main():
    src = from_yaml()
    sources = {
        "tile.yaml":        src,
        "nmgr_params.vh":   from_defines(os.path.join(ROOT, "rtl", "nmgr_params.vh"), "NMGR_"),
        "nmgr_regs.h":      from_defines(os.path.join(ROOT, "fw", "nmgr_regs.h"), "NMGR_"),
        "nmgr_cfg.json":    from_json(os.path.join(ROOT, "dv", "nmgr_cfg.json")),
        # RTL module defaults (only the params each module actually declares)
        "nmgr_pe.v":        from_rtl(os.path.join(ROOT, "rtl", "nmgr_pe.v"),
                                     {"BANKS": "BANKS", "K_PER_BANK": "KPB", "X_BITS": "XW",
                                      "W_BITS": "WW", "ACC_BITS": "ACCW", "OUT_SHIFT": "OSHIFT",
                                      "OUT_BITS": "OW"}),
        "nmgr_encoder.v":   from_rtl(os.path.join(ROOT, "rtl", "nmgr_encoder.v"),
                                     {"TILE_M": "M", "OUT_BITS": "OW"}),
    }

    print(f"{'param':<12}" + "".join(f"{n:<16}" for n in sources))
    mismatches = 0
    for k in KEYS:
        row = f"{k:<12}"
        vals = []
        for n, d in sources.items():
            v = d.get(k, None)
            vals.append(v)
            row += f"{('-' if v is None else v)!s:<16}"
        present = [v for v in vals if v is not None]
        ok = len(set(present)) <= 1
        row += "OK" if ok else "  <-- MISMATCH"
        if not ok:
            mismatches += 1
        print(row)

    print("-" * 60)
    if mismatches:
        print(f"RESULT: FAIL — {mismatches} parameter(s) drifted from config/tile.yaml")
        sys.exit(1)
    print("RESULT: PASS — YAML == RTL == FW header == DV config (consistency by construction)")


if __name__ == "__main__":
    main()
