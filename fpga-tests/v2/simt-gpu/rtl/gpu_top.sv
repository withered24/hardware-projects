// gpu_top.sv — the whole GPU: block dispatcher, NUM_CORES SIMT cores, memory
//
// Launching a kernel works like CUDA's kernel<<<grid_dim, block_dim>>>(args):
// the host loads the code into instruction memory and the inputs into
// global memory, sets kernel_pc / grid_dim / block_dim / args, and pulses
// start. The dispatcher hands blocks 0, 1, 2, ... to whichever core is idle,
// one launch per cycle, until grid_dim blocks have run; then `done` pulses.
// A block has at most THREADS (WARPS*LANES) threads and runs on one core.
//
// The counters (valid until the next start) say what happened:
//   cycles         start to done
//   warp_instrs    instructions issued, one per warp per instruction
//   thread_instrs  instructions executed by threads (sum of active lanes);
//                  thread_instrs / (warp_instrs * LANES) is the SIMD
//                  efficiency: 1.0 means no lane ever sat idle
//   mem_lines      global-memory line requests (coalescing makes this small)

import gpu_pkg::*;

module gpu_top #(
    parameter int MEM_LATENCY = 8             // >= 2
) (
    input  logic        clk,
    input  logic        rst,

    // host access while idle: kernel code, and global memory by the word
    input  logic        imem_we,
    input  logic [31:0] imem_addr,
    input  logic [31:0] imem_wdata,
    input  logic        dmem_we,
    input  logic [31:0] dmem_addr,
    input  logic [31:0] dmem_wdata,
    output logic [31:0] dmem_rdata,

    // kernel launch
    input  logic        start,
    input  logic [31:0] kernel_pc,
    input  logic [31:0] grid_dim,
    input  logic [31:0] block_dim,
    input  logic [31:0] args,
    output logic        busy,
    output logic        done,
    output logic        error,

    output logic [31:0] cycles,
    output logic [31:0] warp_instrs,
    output logic [31:0] thread_instrs,
    output logic [31:0] mem_lines
);

    localparam int IMEM_BITS = $clog2(IMEM_WORDS);

    // ---------------------------------------------------------------- instruction memory
    // One copy of the code, one read port per core.
    logic [31:0] imem [IMEM_WORDS];
    logic [NUM_CORES-1:0][31:0] c_imem_rdata;
    wire  [NUM_CORES-1:0][31:0] c_imem_addr;

    always_ff @(posedge clk)
        if (imem_we) imem[imem_addr[2 +: IMEM_BITS]] <= imem_wdata;

    for (genvar c = 0; c < NUM_CORES; c++) begin : iport
        wire [31:0] a = c_imem_addr[c];
        always @(posedge clk) c_imem_rdata[c] <= imem[a[2 +: IMEM_BITS]];
    end

    // ---------------------------------------------------------------- dispatcher
    logic [31:0] k_pc, k_grid, k_block, k_args, next_block;
    logic [NUM_CORES-1:0] c_launch;
    wire  [NUM_CORES-1:0] c_idle, c_issue, c_error;
    wire  [NUM_CORES-1:0][LANE_BITS:0] c_threads;

    always_comb begin
        c_launch = '0;
        if (busy && next_block < k_grid)
            for (int c = NUM_CORES - 1; c >= 0; c--)
                if (c_idle[c]) c_launch = NUM_CORES'(1) << c;    // lowest idle core
    end

    logic        transaction;
    logic [31:0] issued_threads;
    always_comb begin
        issued_threads = '0;
        for (int c = 0; c < NUM_CORES; c++)
            if (c_issue[c]) issued_threads = issued_threads + c_threads[c];
    end

    always_ff @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            busy <= 1'b0;
        end else if (start && !busy) begin
            busy          <= 1'b1;
            next_block    <= '0;
            k_pc          <= kernel_pc;
            k_grid        <= grid_dim;
            k_block       <= block_dim;
            k_args        <= args;
            cycles        <= '0;
            warp_instrs   <= '0;
            thread_instrs <= '0;
            mem_lines     <= '0;
        end else if (busy) begin
            cycles        <= cycles + 1;
            warp_instrs   <= warp_instrs + $countones(c_issue);
            thread_instrs <= thread_instrs + issued_threads;
            mem_lines     <= mem_lines + transaction;
            if (c_launch != '0) next_block <= next_block + 1;
            else if (next_block == k_grid && &c_idle) begin
                busy <= 1'b0;
                done <= 1'b1;
            end
        end
    end

    assign error = |c_error;

    // ---------------------------------------------------------------- cores + memory
    logic [NUM_CORES-1:0]                  m_req_ready, m_resp_valid;
    wire  [NUM_CORES-1:0]                  m_req_valid, m_req_we;
    wire  [NUM_CORES-1:0][LINE_IDX-1:0]    m_req_line;
    wire  [NUM_CORES-1:0][LINE_BITS-1:0]   m_req_wdata;
    wire  [NUM_CORES-1:0][LINE_BYTES-1:0]  m_req_be;
    logic [LINE_BITS-1:0]                  m_resp_rdata;

    for (genvar c = 0; c < NUM_CORES; c++) begin : core
        simt_core u_core (
            .clk(clk), .rst(rst),
            .launch(c_launch[c]), .launch_pc(k_pc), .launch_block(next_block),
            .block_dim(k_block), .grid_dim(k_grid), .args(k_args), .idle(c_idle[c]),
            .imem_addr(c_imem_addr[c]), .imem_rdata(c_imem_rdata[c]),
            .mem_req_valid(m_req_valid[c]), .mem_req_ready(m_req_ready[c]), .mem_req_we(m_req_we[c]),
            .mem_req_line(m_req_line[c]), .mem_req_wdata(m_req_wdata[c]), .mem_req_be(m_req_be[c]),
            .mem_resp_valid(m_resp_valid[c]), .mem_resp_rdata(m_resp_rdata),
            .issue(c_issue[c]), .issue_threads(c_threads[c]), .error(c_error[c])
        );
    end

    gpu_mem #(.LATENCY(MEM_LATENCY)) u_mem (
        .clk(clk), .rst(rst),
        .req_valid(m_req_valid), .req_ready(m_req_ready), .req_we(m_req_we),
        .req_line(m_req_line), .req_wdata(m_req_wdata), .req_be(m_req_be),
        .resp_valid(m_resp_valid), .resp_rdata(m_resp_rdata),
        .host_we(dmem_we), .host_addr(dmem_addr), .host_wdata(dmem_wdata), .host_rdata(dmem_rdata),
        .transaction(transaction)
    );

endmodule
