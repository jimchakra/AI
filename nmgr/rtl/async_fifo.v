// async_fifo.v — dual-clock FIFO, the real memory-clk <-> link-clk crossing.
//
// Classic Cummings-style async FIFO: binary+Gray pointers, two-flop
// synchronizers across the domains, REGISTERED full/empty (which also breaks
// the pointer/flag combinational loop), combinational read. This is the CDC
// primitive that lets the near-memory packetizer (memory clock) hand the
// framed survivor stream to the SerDes/CXL link (link clock) safely — the
// clock-domain crossing the single-clock demonstrator did not yet exercise.
//
// Lint-clean under Verilator -Wall. Depth = 2**AW.

module async_fifo #(
    parameter DW = 34,      // flit width (payload + framing sidebands)
    parameter AW = 5        // address bits -> depth 32
)(
    // write domain (memory clock)
    input  wire            wr_clk,
    input  wire            wr_rst,
    input  wire            wr_en,
    input  wire [DW-1:0]   wr_data,
    output reg             wr_full,
    // read domain (link clock)
    input  wire            rd_clk,
    input  wire            rd_rst,
    input  wire            rd_en,
    output wire [DW-1:0]   rd_data,
    output reg             rd_empty
);
    localparam DEPTH = (1 << AW);

    reg  [DW-1:0] mem [0:DEPTH-1];

    reg  [AW:0] wbin, wgray;              // write pointers (binary, Gray)
    reg  [AW:0] rbin, rgray;              // read pointers
    reg  [AW:0] rgray_s1, rgray_s2;       // read Gray synced into write domain
    reg  [AW:0] wgray_s1, wgray_s2;       // write Gray synced into read domain

    // ---- write domain -----------------------------------------------------
    wire        do_wr      = wr_en & ~wr_full;
    wire [AW:0] wbin_next  = wbin  + {{AW{1'b0}}, do_wr};
    wire [AW:0] wgray_next = (wbin_next >> 1) ^ wbin_next;
    // full when next write Gray == read Gray with top two bits inverted
    wire        wfull_next = (wgray_next ==
                              {~rgray_s2[AW:AW-1], rgray_s2[AW-2:0]});

    always @(posedge wr_clk) begin
        if (wr_rst) begin
            wbin <= 0; wgray <= 0; wr_full <= 1'b0;
        end else begin
            wbin <= wbin_next; wgray <= wgray_next; wr_full <= wfull_next;
            if (do_wr) mem[wbin[AW-1:0]] <= wr_data;
        end
    end
    always @(posedge wr_clk) begin
        if (wr_rst) begin rgray_s1 <= 0; rgray_s2 <= 0; end
        else        begin rgray_s1 <= rgray; rgray_s2 <= rgray_s1; end
    end

    // ---- read domain ------------------------------------------------------
    wire        do_rd      = rd_en & ~rd_empty;
    wire [AW:0] rbin_next  = rbin  + {{AW{1'b0}}, do_rd};
    wire [AW:0] rgray_next = (rbin_next >> 1) ^ rbin_next;
    wire        rempty_next = (rgray_next == wgray_s2);

    always @(posedge rd_clk) begin
        if (rd_rst) begin
            rbin <= 0; rgray <= 0; rd_empty <= 1'b1;
        end else begin
            rbin <= rbin_next; rgray <= rgray_next; rd_empty <= rempty_next;
        end
    end
    always @(posedge rd_clk) begin
        if (rd_rst) begin wgray_s1 <= 0; wgray_s2 <= 0; end
        else        begin wgray_s1 <= wgray; wgray_s2 <= wgray_s1; end
    end

    assign rd_data = mem[rbin[AW-1:0]];
endmodule
