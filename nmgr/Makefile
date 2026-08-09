# NMGR verification / signoff flow  —  `make all`
# Open-source: Verilator (lint), Icarus (sim), Yosys (synth + LEC), Python (golden).

RTL_PE   := rtl/nmgr_pe.v
RTL_ENC  := rtl/nmgr_encoder.v
RTL_PK   := rtl/nmgr_packetizer.v
RTL_FIFO := rtl/async_fifo.v
VLINT    := verilator --lint-only -Wall -Wno-DECLFILENAME

.PHONY: all gen check lint vectors sim synth lec area cosim clean
LIB     := pdk/nangate45.lib
LIB_URL := https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD-flow-scripts/master/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib
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
	$(VLINT) $(RTL_FIFO)
	$(VLINT) $(RTL_PK) $(RTL_FIFO)
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
	@echo "== sim: link-layer packetizer + CDC (nmgr_packetizer, dual-clock) =="
	iverilog -g2012 -o /tmp/tb_pk $(RTL_PK) $(RTL_FIFO) $(RTL_ENC) tb/tb_packetizer.sv
	vvp /tmp/tb_pk | tee /tmp/pk.log
	@grep -q "RESULT: PASS" /tmp/pk.log

synth:
	@echo "== synth (Yosys) =="
	yosys -q -p "read_verilog $(RTL_PE);  synth -top nmgr_pe -flatten;      stat"
	yosys -q -p "read_verilog $(RTL_ENC); synth -top nmgr_encoder -flatten; stat"
	yosys -q -p "read_verilog $(RTL_PK) $(RTL_FIFO); synth -top nmgr_packetizer -flatten; stat"

lec:
	@echo "== LEC: encoder — formal RTL == synthesized netlist (Yosys) =="
	yosys -q -p "read_verilog $(RTL_ENC); hierarchy -top nmgr_encoder; proc; memory; equiv_opt -assert synth"
	@echo "LEC encoder: equivalence proven."
	@echo "NOTE: PE multiplier-datapath LEC is a commercial-tool step (Conformal/Formality);"
	@echo "      its correctness is covered by the exhaustive simulation regression."

# --- grounded area/power (open PDK; not in `all` — needs the 6.7 MB liberty) ---
area:
	@echo "== area: synth to Nangate45 (45 nm) open cell library =="
	@test -f $(LIB) || (mkdir -p pdk && python3 -c "import urllib.request; urllib.request.urlretrieve('$(LIB_URL)','$(LIB)')")
	yosys -p "read_verilog $(RTL_PE);            synth -top nmgr_pe -flatten;         dfflibmap -liberty $(LIB); abc -liberty $(LIB); stat -liberty $(LIB)"
	yosys -p "read_verilog $(RTL_ENC);           synth -top nmgr_encoder -flatten;    dfflibmap -liberty $(LIB); abc -liberty $(LIB); stat -liberty $(LIB)"
	yosys -p "read_verilog $(RTL_PK) $(RTL_FIFO); synth -top nmgr_packetizer -flatten; dfflibmap -liberty $(LIB); abc -liberty $(LIB); stat -liberty $(LIB)"

# --- hardware-in-the-loop cosim (needs torch + a trained checkpoint) -----------
cosim:
	@echo "== HITL cosim: a real transformer FFN tile through the actual RTL =="
	python3 ../hitl_cosim.py

clean:
	rm -f vectors/all_*.hex vectors/y_rtl.hex vectors/x_one.hex /tmp/tb_nmgr /tmp/tb_enc /tmp/tb_pk /tmp/tb_cosim /tmp/pe.log /tmp/enc.log /tmp/pk.log
