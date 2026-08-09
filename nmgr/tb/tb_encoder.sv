// tb_encoder.sv — bit-exact check of nmgr_encoder against the golden.
// Feeds the golden gated outputs (expected_*.hex) lane-by-lane and checks the
// resulting bitmap against golden bitmap_*.hex and the packed survivors against
// the non-zeros of the golden dense output.

`timescale 1ns/1ps
module tb_encoder;
    localparam M=64, OW=8, NVEC=16;

    reg [OW-1:0] emem [0:NVEC*M-1];   // golden dense gated outputs
    reg [0:0]    bmem [0:NVEC*M-1];   // golden bitmap (0/1)

    reg clk=0, rst=1, start=0, in_valid=0;
    reg [OW-1:0] in_val=0;
    wire [M-1:0] bitmap;
    wire surv_valid; wire [OW-1:0] surv_val;
    wire [$clog2(M+1)-1:0] count;
    wire done;
    reg  [$clog2(M)-1:0] rd_addr=0;
    wire [OW-1:0] rd_data;

    nmgr_encoder #(M,OW) dut (
        .clk(clk),.rst(rst),.start(start),.in_valid(in_valid),.in_val(in_val),
        .bitmap(bitmap),.surv_valid(surv_valid),.surv_val(surv_val),
        .count(count),.done(done),.rd_addr(rd_addr),.rd_data(rd_data));

    always #5 clk=~clk;

    integer i,j,s,errors,checks,ref_cnt;
    reg [OW-1:0] ref_surv [0:M-1];

    initial begin : wd #20000000; $display("RESULT: FAIL (timeout)"); $finish; end

    initial begin
        $readmemh("vectors/all_expected.hex", emem);
        $readmemh("vectors/all_bitmap.hex",   bmem);
        errors=0; checks=0;
        @(negedge clk); rst=0; @(negedge clk);

        for (i=0;i<NVEC;i=i+1) begin
            @(negedge clk); start=1; in_valid=0;
            @(posedge clk); @(negedge clk); start=0;
            ref_cnt=0;
            for (j=0;j<M;j=j+1) begin
                in_val = emem[i*M+j]; in_valid=1;
                @(posedge clk); @(negedge clk);
                if (emem[i*M+j]!=0) begin ref_surv[ref_cnt]=emem[i*M+j]; ref_cnt=ref_cnt+1; end
            end
            in_valid=0;

            // 1) survivor count
            checks=checks+1;
            if (count !== ref_cnt) begin errors=errors+1;
                if (errors<=6) $display("COUNT v=%0d rtl=%0d ref=%0d",i,count,ref_cnt); end
            // 2) bitmap vs golden
            for (j=0;j<M;j=j+1) begin
                checks=checks+1;
                if (bitmap[j] !== bmem[i*M+j]) begin errors=errors+1;
                    if (errors<=6) $display("BITMAP v=%0d lane=%0d rtl=%b gold=%b",i,j,bitmap[j],bmem[i*M+j]); end
            end
            // 3) packed survivors vs golden non-zeros
            for (s=0;s<ref_cnt;s=s+1) begin
                rd_addr = s[$clog2(M)-1:0]; #1;   // read payload via the port
                checks=checks+1;
                if (rd_data !== ref_surv[s]) begin errors=errors+1;
                    if (errors<=6) $display("SURV v=%0d s=%0d rtl=%0d ref=%0d",i,s,rd_data,ref_surv[s]); end
            end
        end

        $display("--------------------------------------------------");
        $display("NMGR encoder check: %0d/%0d assertions pass", checks-errors, checks);
        if (errors==0) $display("RESULT: PASS (bitmap + packed survivors == golden)");
        else           $display("RESULT: FAIL (%0d mismatches)", errors);
        $display("--------------------------------------------------");
        $finish;
    end
endmodule
