// gpu_mem.sv: FPGA-only copy of rtl/gpu_mem.sv (fpga-tests/patches/; diff in gpu_mem.sv.diff)
//
// Global memory: shared by every core, one line per cycle, slow
//
// Stands in for DRAM. It accepts one line request per cycle from one core
// (round-robin when several ask), does the read or write immediately, and
// hands the response back LATENCY cycles later. Responses come back in the
// order requests were accepted, which the LSU relies on.
//
// FPGA adaptation. The original holds the 64 KiB as an array read
// combinationally twice per cycle (the granted line, the host's line) and
// written twice per edge (a core's line, the host's word): no FPGA block RAM
// has that, and as flip-flops it doesn't fit. Here it is 32 byte-wide true
// dual-port block RAMs (one per byte of a line), clocked by the replay
// harness's base clock `fpga_fclk`, 8 base cycles per DUT clock cycle, with
// `fpga_ph` the phase (0..7; the DUT clock edge E7 ends phase 7):
//
//   E7      capture this cycle's writes (core line on port A, host word on
//           port B): the values the original writes at this DUT edge
//   ph 0    port A performs the core write    (written at E0)
//   ph 1    port B performs the host write    (written at E1)
//   ph 4    both ports read: A the granted line, B the host's line (E4)
//   E5      hold the read data: a_old and host_line, steady until next E5
//
// so every read in a DUT cycle sees every write of earlier DUT edges, the
// host write lands after the core write (the original's order), and a_old /
// host_line are valid long before the DUT edge E7 that uses them, exactly as
// the combinational reads were. While the harness is idle fpga_ph stays 0
// and port A repeats its last write, which changes nothing.

import gpu_pkg::*;

module gpu_mem #(
    parameter int LATENCY = 8
) (
    input  logic                                  clk,
    input  logic                                  rst,

    input  logic [NUM_CORES-1:0]                  req_valid,
    output logic [NUM_CORES-1:0]                  req_ready,
    input  logic [NUM_CORES-1:0]                  req_we,
    input  logic [NUM_CORES-1:0][LINE_IDX-1:0]    req_line,
    input  logic [NUM_CORES-1:0][LINE_BITS-1:0]   req_wdata,
    input  logic [NUM_CORES-1:0][LINE_BYTES-1:0]  req_be,
    output logic [NUM_CORES-1:0]                  resp_valid,
    output logic [LINE_BITS-1:0]                  resp_rdata,

    input  logic                                  host_we,
    input  logic [31:0]                           host_addr,
    input  logic [31:0]                           host_wdata,
    output logic [31:0]                           host_rdata,

    output logic                                  transaction,  // a request was accepted

    input  logic                                  fpga_fclk,    // FPGA harness base clock
    input  logic [3:0]                            fpga_ph       // its phase, 0..7
);

    localparam int CORE_BITS = NUM_CORES > 1 ? $clog2(NUM_CORES) : 1;

    // ---- round-robin arbiter (unchanged)
    logic [CORE_BITS-1:0] last, cand, grant;
    logic                 any;
    always_comb begin
        any   = 1'b0;
        grant = '0;
        for (int i = 1; i <= NUM_CORES; i++) begin
            cand = CORE_BITS'((last + i) % NUM_CORES);
            if (!any && req_valid[cand]) begin
                any   = 1'b1;
                grant = cand;
            end
        end
        req_ready = '0;
        if (any) req_ready[grant] = 1'b1;
    end
    assign transaction = any;

    logic [LINE_IDX-1:0]   a_line;
    logic [LINE_BITS-1:0]  a_old, a_wdata;
    logic [LINE_BYTES-1:0] a_be;
    assign a_line  = req_line[grant];
    assign a_wdata = req_wdata[grant];
    assign a_be    = req_be[grant];

    // ---- the memory: 32 byte columns, phased over the base clock
    logic                  w_en;                        // core write captured at E7
    logic [LINE_IDX-1:0]   w_line;
    logic [LINE_BITS-1:0]  w_data;
    logic [LINE_BYTES-1:0] w_be;
    logic                  h_en;                        // host write captured at E7
    logic [31:0]           h_addr, h_data;
    logic [LINE_BITS-1:0]  qa, qb, a_old_r, host_line;

    initial begin
        w_en = 1'b0; h_en = 1'b0; w_line = '0; w_data = '0; w_be = '0; h_addr = '0; h_data = '0;
        a_old_r = '0; host_line = '0;
    end

    always_ff @(posedge fpga_fclk) begin
        if (fpga_ph == 4'd7) begin
            w_en   <= any && req_we[grant];
            w_line <= a_line;
            w_data <= a_wdata;
            w_be   <= a_be;
            h_en   <= host_we;
            h_addr <= host_addr;
            h_data <= host_wdata;
        end
        if (fpga_ph == 4'd5) begin
            a_old_r   <= qa;
            host_line <= qb;
        end
    end

    genvar b;
    generate
        for (b = 0; b < LINE_BYTES; b++) begin : col
            logic          wa, wb;
            logic [LINE_IDX-1:0] aa, ab;
            assign wa = fpga_ph == 4'd0 && w_en && w_be[b];
            assign aa = fpga_ph == 4'd0 ? w_line : a_line;
            assign wb = fpga_ph == 4'd1 && h_en && int'(h_addr[OFF_BITS-1:2]) == b / 4;
            assign ab = fpga_ph == 4'd1 ? h_addr[OFF_BITS +: LINE_IDX] : host_addr[OFF_BITS +: LINE_IDX];
            fpga_tdp_ram #(.DW(8), .AW(LINE_IDX)) u_ram (
                .clk(fpga_fclk),
                .we_a(wa), .addr_a(aa), .wdata_a(w_data[b*8 +: 8]), .q_a(qa[b*8 +: 8]),
                .we_b(wb), .addr_b(ab), .wdata_b(h_data[(b % 4)*8 +: 8]), .q_b(qb[b*8 +: 8]));
        end
    endgenerate

    assign a_old = a_old_r;

    // ---- answer LATENCY cycles later (unchanged)
    logic [LATENCY-1:0]                  p_valid;
    logic [LATENCY-1:0][CORE_BITS-1:0]   p_core;
    logic [LATENCY-1:0][LINE_BITS-1:0]   p_data;

    always_ff @(posedge clk) begin
        p_valid <= rst ? '0 : {p_valid[LATENCY-2:0], any};
        p_core  <= {p_core[LATENCY-2:0], grant};
        p_data  <= {p_data[LATENCY-2:0], a_old};
        if (rst) last <= '0;
        else if (any) last <= grant;
    end

    always_comb
        for (int c = 0; c < NUM_CORES; c++)
            resp_valid[c] = p_valid[LATENCY-1] && p_core[LATENCY-1] == CORE_BITS'(c);
    assign resp_rdata = p_data[LATENCY-1];

    assign host_rdata = host_line[host_addr[OFF_BITS-1:2]*32 +: 32];

endmodule
