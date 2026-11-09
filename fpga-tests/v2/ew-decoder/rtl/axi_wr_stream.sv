// axi_wr_stream.sv — AXI4 write master that drains a stream to a byte region
//
// Buffers incoming words; once a whole burst's worth is waiting it sends
// AW, then the W beats back to back, and counts B responses. One burst at a
// time on AW/W is plenty here: the only thing written is y, 4 bytes per row
// against ~1.4·K bytes of weights read per row.

module axi_wr_stream #(
    parameter int DW         = 64,
    parameter int AW         = 32,
    parameter int MAX_BURST  = 16,
    parameter int FIFO_DEPTH = 32
) (
    input  logic            clk,
    input  logic            rst_n,
    input  logic            clear,

    input  logic            start,
    input  logic [AW-1:0]   addr,
    input  logic [31:0]     nbeats,
    output logic            busy,

    input  logic            in_valid,
    output logic            in_ready,
    input  logic [DW-1:0]   in_data,

    output logic [AW-1:0]   awaddr,
    output logic [7:0]      awlen,
    output logic [2:0]      awsize,
    output logic [1:0]      awburst,
    output logic            awvalid,
    input  logic            awready,
    output logic [DW-1:0]   wdata,
    output logic [DW/8-1:0] wstrb,
    output logic            wlast,
    output logic            wvalid,
    input  logic            wready,
    input  logic [1:0]      bresp,
    input  logic            bvalid,
    output logic            bready
);

    typedef enum logic [1:0] {S_IDLE, S_AW, S_W} state_t;
    state_t state;

    logic [31:0]                 remain, blen;
    logic [8:0]                  left;           // beats left in the current burst
    logic [AW-1:0]               next_addr;
    logic [7:0]                  bout;           // bursts waiting for B
    logic [$clog2(FIFO_DEPTH):0] fcount;
    logic                        f_valid, w_hs;

    assign awsize  = 3'($clog2(DW / 8));
    assign awburst = 2'b01;
    assign wstrb   = '1;
    assign bready  = 1'b1;
    assign wvalid  = (state == S_W) && f_valid;
    assign wlast   = (left == 9'd1);
    assign w_hs    = wvalid && wready;

    assign blen = (remain > MAX_BURST) ? MAX_BURST : remain;
    assign busy = (remain != 0) || (state != S_IDLE) || (bout != 0);

    always_ff @(posedge clk) begin
        if (!rst_n || clear) begin
            state   <= S_IDLE;
            remain  <= '0;
            awvalid <= 1'b0;
            bout    <= '0;
        end else begin
            case (state)
                S_IDLE: if (start) begin
                    remain    <= nbeats;
                    next_addr <= addr;
                end else if (remain != 0 && 32'(fcount) >= blen) begin
                    awvalid   <= 1'b1;
                    awaddr    <= next_addr;
                    awlen     <= 8'(blen - 1);
                    left      <= 9'(blen);
                    remain    <= remain - blen;
                    next_addr <= next_addr + AW'(blen * (DW / 8));
                    state     <= S_AW;
                end
                S_AW: if (awready) begin
                    awvalid <= 1'b0;
                    state   <= S_W;
                end
                S_W: if (w_hs) begin
                    left <= left - 1'b1;
                    if (left == 9'd1) state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
            bout <= bout + 8'(w_hs && wlast) - 8'(bvalid);
        end
    end

    sync_fifo #(.WIDTH(DW), .DEPTH(FIFO_DEPTH)) u_fifo (
        .clk, .rst_n, .clear,
        .in_valid, .in_ready, .in_data,
        .out_valid(f_valid), .out_ready(w_hs), .out_data(wdata),
        .count(fcount)
    );

`ifndef SYNTHESIS
    always @(posedge clk)
        if (rst_n && !clear && bvalid && bresp != 2'b00) $fatal(1, "axi_wr_stream: BRESP %0d", bresp);
`endif

endmodule
