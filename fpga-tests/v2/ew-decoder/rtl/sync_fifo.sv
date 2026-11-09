// sync_fifo.sv — single-clock FIFO, show-ahead (head visible while valid)
//
// DEPTH must be a power of two. `clear` empties it in one cycle; callers
// only clear between jobs, when nothing is in flight.
//
// SRAM = 1 keeps the entries in sram_2p (on the ASIC, SRAM macros instead of
// flip-flops). Its read is synchronous, so the head comes from a two-entry
// output buffer that is refilled one read ahead: still one entry per cycle,
// two cycles more from push to head. `count` covers every entry, buffered or
// not, so a caller that reserves space against it sees the same FIFO.

module sync_fifo #(
    parameter int WIDTH = 64,
    parameter int DEPTH = 16,
    parameter int SRAM  = 0
) (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     clear,
    input  logic                     in_valid,
    output logic                     in_ready,
    input  logic [WIDTH-1:0]         in_data,
    output logic                     out_valid,
    input  logic                     out_ready,
    output logic [WIDTH-1:0]         out_data,
    output logic [$clog2(DEPTH):0]   count
);

    localparam int AW = $clog2(DEPTH);

    logic [AW:0] wp, rp;                 // rp: entries popped

    assign count    = wp - rp;
    assign in_ready = (count != (AW+1)'(DEPTH));

    if (SRAM == 0) begin : g_ff
        logic [WIDTH-1:0] mem [DEPTH];

        assign out_valid = (count != '0);
        assign out_data  = mem[rp[AW-1:0]];

        always_ff @(posedge clk) begin
            if (!rst_n || clear) begin
                wp <= '0;
                rp <= '0;
            end else begin
                if (in_valid && in_ready) begin
                    mem[wp[AW-1:0]] <= in_data;
                    wp <= wp + 1'b1;
                end
                if (out_valid && out_ready)
                    rp <= rp + 1'b1;
            end
        end
    end else begin : g_sram
        logic [AW:0]      fp;            // entries read out of the SRAM
        logic             pend;          // a read is landing this cycle
        logic [1:0]       ob_n;          // entries in the output buffer
        logic [WIDTH-1:0] ob [2];
        logic [WIDTH-1:0] rdata;
        logic             push, pop, fetch;
        logic [1:0]       keep;          // buffered entries left after this pop

        assign push      = in_valid && in_ready;
        assign out_valid = (ob_n != 2'd0);
        assign out_data  = ob[0];
        assign pop       = out_valid && out_ready;
        assign keep      = ob_n - 2'(pop);
        // only entries written at an earlier edge, and only if the read
        // will still have a place in the buffer when it lands
        assign fetch     = (fp != wp) && (32'(keep) + 32'(pend) < 2);

        sram_2p #(.WIDTH(WIDTH), .DEPTH(DEPTH)) u_mem (
            .clk, .we(push), .waddr(wp[AW-1:0]), .wdata(in_data),
            .re(fetch), .raddr(fp[AW-1:0]), .rdata);

        always_ff @(posedge clk) begin
            if (!rst_n || clear) begin
                wp   <= '0;
                rp   <= '0;
                fp   <= '0;
                pend <= 1'b0;
                ob_n <= '0;
            end else begin
                if (push)  wp <= wp + 1'b1;
                if (pop)   rp <= rp + 1'b1;
                if (fetch) fp <= fp + 1'b1;
                pend <= fetch;
                if (pop) ob[0] <= ob[1];
                if (pend) ob[keep[0]] <= rdata;
                ob_n <= keep + 2'(pend);
            end
        end
    end

endmodule
