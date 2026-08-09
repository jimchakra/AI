#!/usr/bin/env bash
# Regenerate golden vectors and prove the RTL bit-exact against them.
set -e
cd "$(dirname "$0")"
python3 golden/golden.py >/dev/null
cat $(ls vectors/x_*.hex | sort)        > vectors/all_x.hex
cat $(ls vectors/expected_*.hex | sort) > vectors/all_expected.hex
cat $(ls vectors/bitmap_*.hex | sort)   > vectors/all_bitmap.hex
echo "== compute core (nmgr_pe) =="
iverilog -g2012 -o /tmp/tb_nmgr rtl/nmgr_pe.v tb/tb_nmgr.sv && vvp /tmp/tb_nmgr | tail -3
echo "== compressed return (nmgr_encoder) =="
iverilog -g2012 -o /tmp/tb_enc rtl/nmgr_encoder.v tb/tb_encoder.sv && vvp /tmp/tb_enc | tail -3
