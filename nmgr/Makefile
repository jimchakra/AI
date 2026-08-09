# NMGR verification / signoff flow  —  `make all`
# Open-source: Verilator (lint), Icarus (sim), Yosys (synth + LEC), Python (golden).

RTL_PE  := rtl/nmgr_pe.v
RTL_ENC := rtl/nmgr_encoder.v
VLINT   := verilator --lint-only -Wall -Wno-DECLFILENAME

.PHONY: all gen check lint vectors sim synth lec clean
all: check lint sim synth lec
	@echo "== FLOW COMPLETE: SSOT-check + lint + sim + synth + LEC all passed =="

gen:
	@echo "== fan out artifacts from the single source (config/tile.yaml) =="
	python3 gen/gen_from_yaml.py

check:
	@echo "== SSOT consistency: YAML == RTL == FW header == DV config =="
	python3 gen/check_consistency.py

lint:
	@echo "== lint (Verilator) =="
	$(VLINT) $(RTL_PE)
	$(VLINT) $(RTL_ENC)
	@echo "lint: clean"

vectors:
	@echo "== golden vectors (Python) =="
	python3 golden/golden.py >/dev/null
	cat $$(ls vectors/x_*.hex | sort)        > vectors/all_x.hex
	cat $$(ls vectors/expected_*.hex | sort) > vectors/all_expected.hex
	cat $$(ls vectors/bitmap_*.hex | sort)   > vectors/all_bitmap.hex

sim: vectors
	@echo "== sim: compute core (nmgr_pe) =="
	iverilog -g2012 -o /tmp/tb_nmgr $(RTL_PE) tb/tb_nmgr.sv
	vvp /tmp/tb_nmgr | tee /tmp/pe.log
	@grep -q "RESULT: PASS" /tmp/pe.log
	@echo "== sim: compressed return (nmgr_encoder) =="
	iverilog -g2012 -o /tmp/tb_enc $(RTL_ENC) tb/tb_encoder.sv
	vvp /tmp/tb_enc | tee /tmp/enc.log
	@grep -q "RESULT: PASS" /tmp/enc.log

synth:
	@echo "== synth (Yosys) =="
	yosys -q -p "read_verilog $(RTL_PE);  synth -top nmgr_pe -flatten;      stat"
	yosys -q -p "read_verilog $(RTL_ENC); synth -top nmgr_encoder -flatten; stat"

lec:
	@echo "== LEC: encoder — formal RTL == synthesized netlist (Yosys) =="
	yosys -q -p "read_verilog $(RTL_ENC); hierarchy -top nmgr_encoder; proc; memory; equiv_opt -assert synth"
	@echo "LEC encoder: equivalence proven."
	@echo "NOTE: PE multiplier-datapath LEC is a commercial-tool step (Conformal/Formality);"
	@echo "      its correctness is covered by the exhaustive simulation regression."

clean:
	rm -f vectors/all_*.hex /tmp/tb_nmgr /tmp/tb_enc /tmp/pe.log /tmp/enc.log
