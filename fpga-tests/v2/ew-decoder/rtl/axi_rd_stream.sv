// axi_rd_stream.sv — AXI4 read master that turns a byte region into a stream
//
// Given a start address and a beat count, it issues INCR bursts of up to
// MAX_BURST beats, keeps up to MAX_OUT of them in flight, and pushes the
// returning beats into a FIFO that the consumer pops at its own pace.
//
// Space is reserved when a burst is issued (FIFO occupancy + beats still
// in flight + the new burst <= FIFO_DEPTH), so RREADY can be tied high: the
// FIFO always has room for every beat it asked for. That is what makes the
// stream safe to throttle from the consumer side without stalling the
// AXI R channel for other masters.
//
// Bursts never cross a 4 KB boundary as long as `addr` is aligned to
// MAX_BURST*DW/8 bytes; sw/ewformat.py lays every stream out 4 KB-aligned.
// RRESP is not checked (the board's DDR controller returns OKAY or the
// whole system is broken anyway); a sim assertion flags it.

module axi_rd_stream #(
    parameter int DW         = 64,
    parameter int AW         = 32,
    parameter int MAX_BURST  = 16,
    parameter int FIFO_DEPTH = 64,
    parameter int MAX_OUT    = 4,
    parameter int SRAM       = 0      // FIFO entries in SRAM (see sync_fifo)
) (
    input  logic          clk,
    input  logic          rst_n,
    input  logic          clear,

    input  logic          start,
    input  logic [AW-1:0] addr,
    input  logic [31:0]   nbeats,
    output logic          busy,

    output logic          out_valid,
    input  logic          out_ready,
    output logic [DW-1:0] out_data,

    output logic [AW-1:0] araddr,
    output logic [7:0]    arlen,
    output logic [2:0]    arsize,
    output logic [1:0]    arburst,
    output logic          arvalid,
    input  logic          arready,
    input  logic [DW-1:0] rdata,
    input  logic [1:0]    rresp,
    input  logic          rlast,
    input  logic          rvalid,
    output logic          rready
);

    localparam int CW = $clog2(FIFO_DEPTH) + 2;

    logic [31:0]                 remain;
    logic [AW-1:0]               next_addr;
    logic [CW-1:0]               reserved;       // beats asked for, not yet arrived
    localparam int OW = $clog2(MAX_OUT + 1) + 1;
    logic [OW-1:0]               outstanding;    // bursts asked for, last beat not yet arrived
    logic [$clog2(FIFO_DEPTH):0] fcount;
    logic [31:0]                 blen;
    logic                        issue, r_hs, fifo_in_ready;

    assign arsize  = 3'($clog2(DW / 8));
    assign arburst = 2'b01;                       // INCR
    assign rready  = 1'b1;
    assign r_hs    = rvalid;

    assign blen  = (remain > MAX_BURST) ? MAX_BURST : remain;
    assign issue = !start && !arvalid && (remain != 0) && (32'(outstanding) < MAX_OUT)
                   && (32'(fcount) + 32'(reserved) + blen <= FIFO_DEPTH);

    assign busy = (remain != 0) || arvalid || (outstanding != 0);

    always_ff @(posedge clk) begin
        if (!rst_n || clear) begin
            remain      <= '0;
            arvalid     <= 1'b0;
            reserved    <= '0;
            outstanding <= '0;
        end else begin
            if (start) begin
                remain    <= nbeats;
                next_addr <= addr;
            end else if (arvalid) begin
                if (arready) arvalid <= 1'b0;
            end else if (issue) begin
                arvalid   <= 1'b1;
                araddr    <= next_addr;
                arlen     <= 8'(blen - 1);
                remain    <= remain - blen;
                next_addr <= next_addr + AW'(blen * (DW / 8));
            end
            reserved    <= reserved + (issue ? CW'(blen) : '0) - (r_hs ? CW'(1) : '0);
            outstanding <= outstanding + OW'(issue) - OW'(r_hs && rlast);
        end
    end

    sync_fifo #(.WIDTH(DW), .DEPTH(FIFO_DEPTH), .SRAM(SRAM)) u_fifo (
        .clk, .rst_n, .clear,
        .in_valid(rvalid), .in_ready(fifo_in_ready), .in_data(rdata),
        .out_valid, .out_ready, .out_data,
        .count(fcount)
    );

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && !clear) begin
            if (rvalid && !fifo_in_ready) $fatal(1, "axi_rd_stream: beat arrived with the FIFO full");
            if (rvalid && rresp != 2'b00) $fatal(1, "axi_rd_stream: RRESP %0d", rresp);
        end
        if (rst_n && clear && outstanding != 0) $fatal(1, "axi_rd_stream: cleared with bursts in flight");
    end
`endif

endmodule
