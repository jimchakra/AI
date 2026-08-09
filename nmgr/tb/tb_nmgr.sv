// tb_nmgr.sv — bit-exact check of nmgr_pe against the golden vectors.
// Replays every output lane of every vector through the PE and compares to
// expected_*.hex. Reads weights.hex (resident) and concatenated all_x/all_e.

`timescale 1ns/1ps
module tb_nmgr;
    localparam BANKS=4, M=64, KPB=32, XW=8, WW=4, ACCW=32, OSHIFT=8, OW=8;
    localparam NVEC = 16;
    localparam THRESH = 0;                     // theta = 0 => lossless (matches default golden)

    localparam KFULL = BANKS*KPB;              // 128 activations per vector
    localparam WLEN  = BANKS*M*KPB;            // 8192 weights (resident)

    reg  [WW-1:0] wmem [0:WLEN-1];
    reg  [XW-1:0] xmem [0:NVEC*KFULL-1];
    reg  [OW-1:0] emem [0:NVEC*M-1];

    reg clk=0, rst=1, start=0;
    reg  [BANKS*XW-1:0] x_k;
    reg  [BANKS*WW-1:0] w_k;
    wire done;
    wire [OW-1:0] y;

    nmgr_pe #(BANKS,KPB,XW,WW,ACCW,OSHIFT,OW) dut (
        .clk(clk), .rst(rst), .start(start),
        .x_k(x_k), .w_k(w_k), .threshold(THRESH[OW-1:0]),
        .done(done), .y(y)
    );

    always #5 clk = ~clk;

    integer i, m, k, bb, errors, checks;
    // W index (golden reshape of [banks, tile_m, k_per_bank]): b*(M*KPB)+m*KPB+k
    // x index (golden reshape of [banks, k_per_bank]):         b*KPB+k  (+ vector offset)
    task set_operands(input integer vi, input integer mi, input integer ki);
        integer b2;
        begin
            for (b2=0; b2<BANKS; b2=b2+1) begin
                x_k[b2*XW +: XW] = xmem[vi*KFULL + b2*KPB + ki];
                w_k[b2*WW +: WW] = wmem[b2*(M*KPB) + mi*KPB + ki];
            end
        end
    endtask

    initial begin : watchdog
        #20000000; $display("RESULT: FAIL (timeout)"); $finish;
    end

    initial begin
        $readmemh("vectors/weights.hex", wmem);
        $readmemh("vectors/all_x.hex",   xmem);
        $readmemh("vectors/all_expected.hex", emem);
        errors=0; checks=0;
        @(negedge clk); rst=0; @(negedge clk);

        for (i=0; i<NVEC; i=i+1) begin
            for (m=0; m<M; m=m+1) begin
                // drive operands/start on negedge so they are stable at the sampling posedge
                for (k=0; k<KPB; k=k+1) begin
                    @(negedge clk);
                    set_operands(i,m,k);
                    start = (k==0);
                    @(posedge clk);          // DUT accumulates k
                end
                @(negedge clk); start = 0;
                @(posedge clk);              // finalize edge (cnt == KPB)
                @(negedge clk);              // settle, then sample
                checks = checks + 1;
                if (y !== emem[i*M+m]) begin
                    errors = errors + 1;
                    if (errors <= 6)
                        $display("MISMATCH v=%0d m=%0d  rtl=%0d golden=%0d", i, m, y, emem[i*M+m]);
                end
            end
        end

        $display("--------------------------------------------------");
        $display("NMGR bit-exact check: %0d/%0d outputs match (theta=%0d)",
                 checks-errors, checks, THRESH);
        if (errors==0) $display("RESULT: PASS (RTL == golden)");
        else           $display("RESULT: FAIL (%0d mismatches)", errors);
        $display("--------------------------------------------------");
        $finish;
    end
endmodule
