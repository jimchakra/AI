// nmgr_packetizer.v — link-layer framer for the compressed return.
//
// Takes the encoder's compressed output in the MEMORY clock domain:
//   header  : {magic, seq, survivor-count}
//   bitmap  : M bits  (which lanes survived)          -> NBM flits
//   payload : count survivors, OW bits each, packed   -> ceil(count/LANES) flits
// and streams it as fixed-width flits across an async FIFO into the LINK clock
// domain (SerDes/CXL), with sop/eop framing and downstream backpressure.
//
// The variable-length payload (fewer survivors -> fewer flits) is the whole
// point of the compressed return: the link only carries what survived, plus a
// small fixed framing overhead this module makes explicit.
//
// The FIFO IS the memory-clk <-> link-clk crossing (see async_fifo.v).

module nmgr_packetizer #(
    parameter M      = 64,
    parameter OW     = 8,
    parameter FLIT_W = 32,
    parameter [7:0] MAGIC = 8'hA5
)(
    // ---- memory clock domain (producer) ----
    input  wire                        mem_clk,
    input  wire                        mem_rst,
    input  wire                        go,      // pulse: frame one packet (tie to encoder 'done')
    input  wire [M-1:0]                bitmap,
    input  wire [$clog2(M+1)-1:0]      count,   // survivors this packet
    input  wire [7:0]                  seq,     // packet sequence id
    output reg  [$clog2(M)-1:0]        rd_addr, // -> encoder surv_mem
    input  wire [OW-1:0]               rd_data, // <- encoder surv_mem
    output wire                        busy,
    // ---- link clock domain (consumer) ----
    input  wire                        link_clk,
    input  wire                        link_rst,
    input  wire                        link_ready,
    output wire                        link_valid,
    output wire [FLIT_W-1:0]           link_data,
    output wire                        link_sop,
    output wire                        link_eop
);
    localparam CW    = $clog2(M+1);          // count width
    localparam AW    = $clog2(M);            // survivor-addr width
    localparam LANES = FLIT_W / OW;          // survivors packed per flit
    localparam LW    = $clog2(LANES);        // lane index width
    localparam NBM   = (M + FLIT_W - 1) / FLIT_W;   // bitmap flits
    localparam NBW   = (NBM > 1) ? $clog2(NBM) : 1;
    // LANES and NBM are powers of two for this tile (FLIT_W/OW=4, ceil(M/FLIT_W)=2),
    // so "last index" is all-ones — avoids SV casts (Yosys/Verilator both happy).
    localparam [LW-1:0]  LANE_LAST = {LW {1'b1}};
    localparam [NBW-1:0] BM_LAST   = {NBW{1'b1}};

    localparam S_IDLE=2'd0, S_HDR=2'd1, S_BM=2'd2, S_SURV=2'd3;
    reg [1:0]        state;
    reg [CW-1:0]     count_q;
    reg [M-1:0]      bitmap_q;
    reg [7:0]        seq_q;
    reg [NBW-1:0]    bm_i;                    // bitmap flit index
    reg [CW-1:0]     j;                       // survivor index
    reg [LW-1:0]     lane;                    // lane within current flit
    reg [FLIT_W-1:0] sbuf;                    // partial survivor flit

    wire [NBM*FLIT_W-1:0] bm_pad = {{(NBM*FLIT_W - M){1'b0}}, bitmap_q};

    reg              wr_req;
    reg [FLIT_W-1:0] flit;
    reg              sop, eop;
    wire             full;
    wire             fifo_empty;
    wire             accept = wr_req & ~full;

    wire boundary = (lane == LANE_LAST) || (j == count_q - 1);

    reg [FLIT_W-1:0] surv_flit;               // buffered lanes + this cycle's survivor
    always @* begin
        surv_flit = sbuf;
        surv_flit[lane*OW +: OW] = rd_data;
    end

    // survivor read address — its OWN block, dependent only on state/j, so it
    // never forms a combinational loop through rd_data -> surv_flit -> flit.
    always @* rd_addr = (state == S_SURV) ? j[AW-1:0] : {AW{1'b0}};

    // combinational flit / handshake per state
    always @* begin
        wr_req = 1'b0; flit = {FLIT_W{1'b0}}; sop = 1'b0; eop = 1'b0;
        case (state)
            S_HDR: begin
                wr_req = 1'b1; sop = 1'b1;
                flit = {MAGIC, seq_q, {{(8-CW){1'b0}}, count_q}, 8'h00};
            end
            S_BM: begin
                wr_req = 1'b1;
                flit   = bm_pad[bm_i*FLIT_W +: FLIT_W];
                eop    = (bm_i == BM_LAST) && (count_q == 0);
            end
            S_SURV: begin
                if (boundary) begin
                    wr_req = 1'b1;
                    flit   = surv_flit;
                    eop    = (j == count_q - 1);
                end
            end
            default: ;
        endcase
    end

    always @(posedge mem_clk) begin
        if (mem_rst) begin
            state<=S_IDLE; count_q<=0; bitmap_q<=0; seq_q<=0;
            bm_i<=0; j<=0; lane<=0; sbuf<=0;
        end else begin
            case (state)
                S_IDLE: if (go) begin
                    count_q<=count; bitmap_q<=bitmap; seq_q<=seq;
                    bm_i<=0; j<=0; lane<=0; sbuf<=0; state<=S_HDR;
                end
                S_HDR: if (accept) state<=S_BM;
                S_BM: if (accept) begin
                    if (bm_i == BM_LAST) state <= (count_q==0) ? S_IDLE : S_SURV;
                    else               bm_i  <= bm_i + 1'b1;
                end
                S_SURV: begin
                    if (boundary) begin
                        if (accept) begin
                            if (j == count_q-1) state<=S_IDLE;
                            else begin j<=j+1'b1; lane<=0; sbuf<=0; end
                        end
                    end else begin
                        sbuf[lane*OW +: OW] <= rd_data;
                        lane <= lane + 1'b1;
                        j    <= j + 1'b1;
                    end
                end
                default: state<=S_IDLE;
            endcase
        end
    end

    assign busy = (state != S_IDLE);

    // ---- CDC: memory clock -> link clock ----
    wire [FLIT_W+1:0] rdd;
    async_fifo #(.DW(FLIT_W+2), .AW(5)) u_fifo (
        .wr_clk (mem_clk),  .wr_rst (mem_rst), .wr_en (wr_req),
        .wr_data({sop, eop, flit}), .wr_full (full),
        .rd_clk (link_clk), .rd_rst (link_rst), .rd_en (link_ready),
        .rd_data(rdd), .rd_empty (fifo_empty)
    );
    assign link_valid = ~fifo_empty;
    assign link_data  = rdd[FLIT_W-1:0];
    assign link_eop   = rdd[FLIT_W];
    assign link_sop   = rdd[FLIT_W+1];
endmodule
