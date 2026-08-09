// tb_packetizer.sv — dual-clock verification of the link-layer packetizer.
//
// Memory domain: feed known gated lane values into nmgr_encoder, which builds
// the bitmap + survivor payload; pulse the packetizer.  Link domain (a
// DIFFERENT, asynchronous clock, with random backpressure): collect the flit
// stream and reconstruct the packet.  Assert bit-exact framing:
//   header {magic,seq,count}, bitmap flits, packed survivor flits, sop/eop.
// Exercises the real memory-clk <-> link-clk CDC and variable-length payloads.

`timescale 1ns/1ps
module tb_packetizer;
    localparam M=64, OW=8, FLIT_W=32, LANES=FLIT_W/OW;

    // clocks: memory 100 MHz-ish (#5), link asynchronous & slower (#7)
    reg mem_clk=0, link_clk=0;
    always #5 mem_clk = ~mem_clk;
    always #7 link_clk = ~link_clk;

    reg mem_rst=1, link_rst=1;

    // ---- encoder ----
    reg               start=0, in_valid=0;
    reg  [OW-1:0]     in_val=0;
    wire [M-1:0]      bitmap;
    wire              surv_valid;
    wire [OW-1:0]     surv_val;
    wire [$clog2(M+1)-1:0] count;
    wire              enc_done;
    wire [$clog2(M)-1:0]  pk_rd_addr;
    wire [OW-1:0]         enc_rd_data;

    nmgr_encoder #(.M(M), .OW(OW)) enc (
        .clk(mem_clk), .rst(mem_rst), .start(start),
        .in_valid(in_valid), .in_val(in_val),
        .bitmap(bitmap), .surv_valid(surv_valid), .surv_val(surv_val),
        .count(count), .done(enc_done),
        .rd_addr(pk_rd_addr), .rd_data(enc_rd_data)
    );

    // ---- packetizer ----
    reg  [7:0]        seq=8'h10;
    wire              busy, link_valid, link_sop, link_eop;
    wire [FLIT_W-1:0] link_data;
    reg               link_ready=0;

    nmgr_packetizer #(.M(M), .OW(OW), .FLIT_W(FLIT_W)) pk (
        .mem_clk(mem_clk), .mem_rst(mem_rst), .go(enc_done),
        .bitmap(bitmap), .count(count), .seq(seq),
        .rd_addr(pk_rd_addr), .rd_data(enc_rd_data), .busy(busy),
        .link_clk(link_clk), .link_rst(link_rst), .link_ready(link_ready),
        .link_valid(link_valid), .link_data(link_data),
        .link_sop(link_sop), .link_eop(link_eop)
    );

    // ---- link-side backpressure (random-ish) ----
    reg [15:0] lfsr = 16'hACE1;
    always @(posedge link_clk) begin
        if (link_rst) begin link_ready<=0; lfsr<=16'hACE1; end
        else begin
            lfsr <= {lfsr[14:0], lfsr[15]^lfsr[13]^lfsr[12]^lfsr[10]};
            link_ready <= lfsr[0] | lfsr[3];   // ~75% ready -> exercises stalls
        end
    end

    // ---- link-side collector ----
    reg [FLIT_W-1:0] rx [0:127];
    integer          rxn;
    reg              in_pkt;
    always @(posedge link_clk) begin
        if (link_rst) begin rxn<=0; in_pkt<=0; end
        else if (link_valid & link_ready) begin
            if (link_sop) begin rx[0]<=link_data; rxn<=1; in_pkt<=1; end
            else if (in_pkt) begin rx[rxn]<=link_data; rxn<=rxn+1; end
            if (link_eop) in_pkt<=0;
        end
    end

    // ---- scoreboard ----
    integer vec, i, k, pass=0, fail=0;
    reg [OW-1:0] vals   [0:M-1];
    reg [OW-1:0] exp_sv [0:M-1];
    integer exp_count;
    reg [M-1:0] exp_bitmap;

    task run_vector(input integer mode);
        integer n; reg [OW-1:0] v; integer nf, sf, byte_i;
        begin
            // build the vector
            exp_count = 0; exp_bitmap = 0;
            for (i=0;i<M;i=i+1) begin
                case (mode)
                    0: v = (i%2==0) ? (i+1)      : 8'd0;         // alternating
                    1: v = i+1;                                  // all survive (nonzero)
                    2: v = (i%8==0) ? (8'h20+i)  : 8'd0;         // sparse (8)
                    3: v = (i==5)   ? 8'h7F      : 8'd0;         // single
                    4: v = 8'd0;                                 // none
                    default: v = ((i*37+13)&8'hFF);              // pseudo-random-ish
                endcase
                if (v==0 && mode==1) v=8'h55;                    // ensure nonzero for mode1
                vals[i]=v;
                if (v!=0) begin exp_bitmap[i]=1'b1; exp_sv[exp_count]=v; exp_count=exp_count+1; end
            end

            // feed encoder (stimulus on negedge — avoids sampling races)
            @(negedge mem_clk); start=1;
            @(negedge mem_clk); start=0;
            for (i=0;i<M;i=i+1) begin
                in_valid=1; in_val=vals[i]; @(negedge mem_clk);
            end
            in_valid=0; in_val=0;

            // wait for packetizer to finish framing and link to drain
            wait (busy==1'b0);
            repeat (120) @(posedge link_clk);

            // ---- check ----
            nf = 1 + 2 + ((exp_count+LANES-1)/LANES);            // header+bitmap+survflits
            if (mode==4) nf = 1 + 2;                             // no survivors
            if (rxn !== nf) begin
                $display("VEC %0d FAIL: flit count %0d != expected %0d", vec, rxn, nf);
                fail=fail+1;
            end else begin
                // header
                if (rx[0][31:24]!==8'hA5)              begin fail=fail+1; $display("VEC %0d FAIL magic",vec); end
                else if (rx[0][23:16]!==seq)           begin fail=fail+1; $display("VEC %0d FAIL seq",vec); end
                else if (rx[0][15:8]!==exp_count[7:0]) begin fail=fail+1; $display("VEC %0d FAIL count %0d!=%0d",vec,rx[0][15:8],exp_count); end
                // bitmap
                else if (rx[1]!==exp_bitmap[31:0])     begin fail=fail+1; $display("VEC %0d FAIL bitmap[31:0]",vec); end
                else if (rx[2]!==exp_bitmap[63:32])    begin fail=fail+1; $display("VEC %0d FAIL bitmap[63:32]",vec); end
                else begin
                    // survivors
                    reg ok; ok=1;
                    for (k=0;k<exp_count;k=k+1) begin
                        sf = 3 + (k/LANES); byte_i = k % LANES;
                        if (rx[sf][byte_i*OW +: OW] !== exp_sv[k]) begin
                            ok=0; $display("VEC %0d FAIL surv[%0d] got %02x exp %02x",vec,k,rx[sf][byte_i*OW +: OW],exp_sv[k]);
                        end
                    end
                    if (ok) begin pass=pass+1; $display("VEC %0d PASS  count=%0d flits=%0d",vec,exp_count,rxn); end
                    else fail=fail+1;
                end
            end
            seq = seq + 8'h01;
        end
    endtask

    initial begin
        repeat (4) @(posedge mem_clk); mem_rst=0;
        repeat (6) @(posedge link_clk); link_rst=0;
        for (vec=0; vec<6; vec=vec+1) run_vector(vec);
        $display("PACKETIZER: %0d passed, %0d failed", pass, fail);
        if (fail==0) $display("RESULT: PASS"); else $display("RESULT: FAIL");
        $finish;
    end

    initial begin #200000; $display("RESULT: FAIL (timeout)"); $finish; end
endmodule
