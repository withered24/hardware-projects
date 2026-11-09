// gpu_mem.sv — global memory: shared by every core, one line per cycle, slow
//
// Stands in for DRAM. It accepts one line request per cycle from one core
// (round-robin when several ask), does the read or write immediately, and
// hands the response back LATENCY cycles later. LATENCY is the number the
// warps have to hide; real GPU DRAM is a few hundred cycles away. Responses
// come back in the order requests were accepted, which the LSU relies on.
//
// The host port reads and writes single words while the GPU is idle (a
// host CPU, or the testbench, filling inputs and reading results).

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

    output logic                                  transaction   // a request was accepted
);

    localparam int CORE_BITS = NUM_CORES > 1 ? $clog2(NUM_CORES) : 1;

    logic [LINE_BITS-1:0] mem [MEM_LINES];
    initial for (int i = 0; i < MEM_LINES; i++) mem[i] = '0;

    // ---- round-robin arbiter
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

    // ---- access now, answer LATENCY cycles later
    logic [LINE_IDX-1:0]   a_line;
    logic [LINE_BITS-1:0]  a_old, a_wdata;
    logic [LINE_BYTES-1:0] a_be;
    assign a_line  = req_line[grant];
    assign a_wdata = req_wdata[grant];
    assign a_be    = req_be[grant];
    assign a_old  = mem[a_line];

    logic [LATENCY-1:0]                  p_valid;
    logic [LATENCY-1:0][CORE_BITS-1:0]   p_core;
    logic [LATENCY-1:0][LINE_BITS-1:0]   p_data;

    always_ff @(posedge clk) begin
        if (any && req_we[grant])
            for (int b = 0; b < LINE_BYTES; b++)
                if (a_be[b]) mem[a_line][b*8 +: 8] <= a_wdata[b*8 +: 8];
        if (host_we)
            mem[host_addr[OFF_BITS +: LINE_IDX]][host_addr[OFF_BITS-1:2]*32 +: 32] <= host_wdata;

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

    logic [LINE_BITS-1:0] host_line;
    assign host_line  = mem[host_addr[OFF_BITS +: LINE_IDX]];
    assign host_rdata = host_line[host_addr[OFF_BITS-1:2]*32 +: 32];

endmodule
