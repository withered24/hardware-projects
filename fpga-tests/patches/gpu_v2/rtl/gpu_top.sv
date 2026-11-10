// gpu_top.sv: FPGA-only copy of rtl/gpu_top.sv (fpga-tests/patches/; diff in gpu_top.sv.diff)
//
// The whole GPU: block dispatcher, NUM_CORES SIMT cores, memory
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
    output logic [31:0] mem_lines,

    // FPGA harness: base clock and phase for gpu_mem's block RAMs
    input  logic        fpga_fclk,
    input  logic [3:0]  fpga_ph
);

    localparam int IMEM_BITS = $clog2(IMEM_WORDS);

    // ---------------------------------------------------------------- instruction memory
    // One copy of the code per core (FPGA: a block RAM has one read port, so
    // each core gets its own simple dual-port copy, all written together).
    // Read-old-data on a same-edge write, as in the original.
    wire  [NUM_CORES-1:0][31:0] c_imem_rdata;
    wire  [NUM_CORES-1:0][31:0] c_imem_addr;

    genvar gc;
    generate
        for (gc = 0; gc < NUM_CORES; gc++) begin : iport
            logic [31:0] im [IMEM_WORDS];
            logic [31:0] q;
            wire  [31:0] a = c_imem_addr[gc];
            always @(posedge clk) begin
                if (imem_we) im[imem_addr[2 +: IMEM_BITS]] <= imem_wdata;
                q <= im[a[2 +: IMEM_BITS]];
            end
            assign c_imem_rdata[gc] = q;
        end
    endgenerate

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

    generate
    for (gc = 0; gc < NUM_CORES; gc++) begin : core
        simt_core u_core (
            .clk(clk), .rst(rst),
            .launch(c_launch[gc]), .launch_pc(k_pc), .launch_block(next_block),
            .block_dim(k_block), .grid_dim(k_grid), .args(k_args), .idle(c_idle[gc]),
            .imem_addr(c_imem_addr[gc]), .imem_rdata(c_imem_rdata[gc]),
            .mem_req_valid(m_req_valid[gc]), .mem_req_ready(m_req_ready[gc]), .mem_req_we(m_req_we[gc]),
            .mem_req_line(m_req_line[gc]), .mem_req_wdata(m_req_wdata[gc]), .mem_req_be(m_req_be[gc]),
            .mem_resp_valid(m_resp_valid[gc]), .mem_resp_rdata(m_resp_rdata),
            .issue(c_issue[gc]), .issue_threads(c_threads[gc]), .error(c_error[gc])
        );
    end
    endgenerate

    gpu_mem #(.LATENCY(MEM_LATENCY)) u_mem (
        .clk(clk), .rst(rst),
        .req_valid(m_req_valid), .req_ready(m_req_ready), .req_we(m_req_we),
        .req_line(m_req_line), .req_wdata(m_req_wdata), .req_be(m_req_be),
        .resp_valid(m_resp_valid), .resp_rdata(m_resp_rdata),
        .host_we(dmem_we), .host_addr(dmem_addr), .host_wdata(dmem_wdata), .host_rdata(dmem_rdata),
        .transaction(transaction),
        .fpga_fclk(fpga_fclk), .fpga_ph(fpga_ph)
    );

endmodule
