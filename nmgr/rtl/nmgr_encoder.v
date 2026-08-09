// nmgr_encoder.v — survivor / compression encoder for the NMGR tile.
//
// Consumes the M gated lane outputs in order (0..M-1, one per cycle) and
// produces the compressed return: a bitmap (bit i = lane i survived) plus the
// packed non-zero survivor values, in order. This is the "compressed return"
// stage of the invention — only survivors cross the host link.
//
// surv_val / surv_valid is the real streaming interface; surv_mem is the
// assembled packet payload (also lets the testbench check the packed stream
// without cycle-accurate capture). Bit-exact to golden/golden.py.

module nmgr_encoder #(
    parameter M  = 64,      // lanes per tile
    parameter OW = 8        // output width
)(
    input  wire              clk,
    input  wire              rst,
    input  wire              start,      // pulse before feeding lane 0
    input  wire              in_valid,   // one gated lane value presented
    input  wire [OW-1:0]     in_val,     // gated output (0 => gated away)
    output reg  [M-1:0]      bitmap,     // bit i = (lane i survived)
    output reg               surv_valid, // survivor emitted this cycle (stream)
    output reg  [OW-1:0]        surv_val,   // the survivor value        (stream)
    output reg  [$clog2(M+1)-1:0] count,    // number of survivors so far
    output reg                  done,       // pulses after the last lane
    input  wire [$clog2(M)-1:0]  rd_addr,   // read the assembled payload...
    output wire [OW-1:0]         rd_data    // ...survivor value at rd_addr
);
    localparam IW = $clog2(M+1);
    localparam AW = $clog2(M);

    // assembled survivor payload (the packet body); read via rd_addr/rd_data
    reg [OW-1:0] surv_mem [0:M-1];
    reg [IW-1:0] idx;
    assign rd_data = surv_mem[rd_addr];

    always @(posedge clk) begin
        if (rst) begin
            bitmap <= 0; idx <= 0; count <= 0;
            surv_valid <= 0; surv_val <= 0; done <= 0;
        end else begin
            done <= 1'b0; surv_valid <= 1'b0;
            if (start) begin
                bitmap <= 0; idx <= 0; count <= 0;
            end else if (in_valid) begin
                bitmap[idx[AW-1:0]] <= (in_val != 0);
                if (in_val != 0) begin
                    surv_mem[count[AW-1:0]] <= in_val;   // pack into payload
                    surv_val   <= in_val;        // and emit on the stream
                    surv_valid <= 1'b1;
                    count      <= count + 1'b1;
                end
                idx <= idx + 1'b1;
                if (idx == M-1) done <= 1'b1;     // last lane consumed
            end
        end
    end
endmodule
