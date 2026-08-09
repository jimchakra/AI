// tb_cosim.sv — hardware-in-the-loop harness.
// Reads one real quantized tile (weights.hex + x_one.hex, written by
// hitl_cosim.py from a trained transformer's FFN), runs it through the ACTUAL
// nmgr_pe RTL, and dumps the gated outputs to y_rtl.hex for Python to read back
// into the forward pass. No self-check here — the scoreboard lives in Python.

`timescale 1ns/1ps
module tb_cosim;
    localparam BANKS=4, M=64, KPB=32, XW=8, WW=4, ACCW=32, OSHIFT=8, OW=8;
    localparam THRESH=0, KFULL=BANKS*KPB, WLEN=BANKS*M*KPB;

    reg  [WW-1:0] wmem [0:WLEN-1];
    reg  [XW-1:0] xmem [0:KFULL-1];
    reg  [OW-1:0] ymem [0:M-1];

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

    integer m, k, b2;
    task set_ops(input integer mi, input integer ki);
        begin
            for (b2=0; b2<BANKS; b2=b2+1) begin
                x_k[b2*XW +: XW] = xmem[b2*KPB + ki];
                w_k[b2*WW +: WW] = wmem[b2*(M*KPB) + mi*KPB + ki];
            end
        end
    endtask

    initial begin #2000000; $display("RESULT: FAIL (timeout)"); $finish; end

    initial begin
        $readmemh("vectors/weights.hex", wmem);
        $readmemh("vectors/x_one.hex",   xmem);
        @(negedge clk); rst=0; @(negedge clk);
        for (m=0; m<M; m=m+1) begin
            for (k=0; k<KPB; k=k+1) begin
                @(negedge clk); set_ops(m,k); start=(k==0); @(posedge clk);
            end
            @(negedge clk); start=0; @(posedge clk); @(negedge clk);
            ymem[m] = y;
        end
        $writememh("vectors/y_rtl.hex", ymem);
        $display("COSIM: dumped %0d RTL outputs to vectors/y_rtl.hex", M);
        $finish;
    end
endmodule
