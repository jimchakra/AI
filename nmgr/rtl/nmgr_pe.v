// nmgr_pe.v — one processing element of the near-memory gated-reduction tile.
//
// Folds BANKS multiply-accumulates per cycle over KPB cycles to produce one
// output lane: the full cross-bank reduction, then relu -> requantize (>>>) ->
// saturate -> threshold gate. Bit-exact to golden/golden.py.
//
// A tile is M of these in parallel; the model scales one PE's area/power by
// M and by tiles/token (see spec). Weights/activations are streamed in per k
// (they live in the memory array; this is the compute datapath only).

module nmgr_pe #(
    parameter BANKS  = 4,
    parameter KPB    = 32,
    parameter XW     = 8,     // activation bits (signed)
    parameter WW     = 4,     // weight bits (signed)
    parameter ACCW   = 32,    // accumulator width (signed)
    parameter OSHIFT = 8,     // requantization right-shift
    parameter OW     = 8      // output bits (signed, saturating)
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    start,       // 1-cycle pulse; k=0 operands valid
    input  wire signed [BANKS*XW-1:0] x_k,      // activations for current k, packed by bank
    input  wire signed [BANKS*WW-1:0] w_k,      // weights   for current k, packed by bank
    input  wire        [OW-1:0]    threshold,
    output reg                     done,
    output reg         [OW-1:0]    y
);
    localparam CW = $clog2(KPB + 1);

    // combinational sum of BANKS signed products for the current k
    reg signed [ACCW-1:0] psum;
    integer b;
    always @* begin
        psum = 0;
        for (b = 0; b < BANKS; b = b + 1)
            psum = psum + $signed(x_k[b*XW +: XW]) * $signed(w_k[b*WW +: WW]);
    end

    // relu -> requant (>> OSHIFT) -> saturate to [0, OMAX] -> threshold gate.
    // Combinational; matches the golden exactly. Sampled into y at finalize.
    localparam [OW-1:0] OMAX = (1 << (OW-1)) - 1;                 // 127 for OW=8
    wire [ACCW-1:0] relu_v  = acc[ACCW-1] ? {ACCW{1'b0}} : $unsigned(acc);
    wire [ACCW-1:0] shifted = relu_v >> OSHIFT;
    wire            ovf     = |shifted[ACCW-1:OW-1];             // value > OMAX
    wire [OW-1:0]   sat_v   = ovf ? OMAX : {1'b0, shifted[OW-2:0]};
    wire [OW-1:0]   gated_v = (sat_v > threshold) ? sat_v : {OW{1'b0}};

    reg signed [ACCW-1:0] acc;
    reg        [CW-1:0]   cnt;
    reg                   running;

    always @(posedge clk) begin
        if (rst) begin
            running <= 1'b0; done <= 1'b0; cnt <= 0; acc <= 0; y <= 0;
        end else begin
            done <= 1'b0;
            if (start) begin
                acc     <= psum;      // accumulate k=0
                cnt     <= 1;
                running <= 1'b1;
            end else if (running) begin
                if (cnt == KPB) begin
                    y       <= gated_v;             // full sum complete
                    done    <= 1'b1;
                    running <= 1'b0;
                end else begin
                    acc <= acc + psum;              // accumulate k = cnt
                    cnt <= cnt + 1'b1;
                end
            end
        end
    end
endmodule
