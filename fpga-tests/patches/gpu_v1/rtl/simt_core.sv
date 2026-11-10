// simt_core.sv: FPGA-only copy of rtl/simt_core.sv (fpga-tests/patches/; diff in simt_core.sv.diff)
//
// One SIMT core: WARPS warps of LANES threads, one front end
//
// What makes this a GPU core and not WARPS*LANES little CPUs: there is ONE
// fetch, ONE decoder and ONE scheduler, shared by every thread. Each cycle
// the scheduler picks a warp, fetches one instruction for it, and LANES
// copies of the riscv-core ALU execute that instruction for LANES threads at
// once, each on its own registers. That's SIMT: single instruction, multiple
// threads.
//
// Pipeline: F (pick a warp, fetch) -> D (decode, read LANES register files)
// -> X (LANES ALUs, write back). A warp has at most one instruction in
// flight, so there are no data hazards and no forwarding at all: while warp
// 0 waits for its result, warps 1-3 fill the pipeline. That's how GPUs hide
// latency, with other warps' work instead of forwarding paths and caches.
// Loads and stores leave the pipeline for the load/store unit (lsu.sv) and
// can take tens of cycles; the warp sleeps, the others keep issuing.
//
// Divergence: every thread keeps its own PC. The scheduler runs a warp at
// the SMALLEST PC among its live threads, with only the threads at that PC
// enabled (the active mask). After an if/else, the threads that took the
// branch sit at a higher PC and wait; the others run the "then" side, jump
// past the "else", and once they are all at the same PC again the warp is
// reconverged, with no compiler hints and no reconvergence stack. Loops
// work the same way: threads still looping are at lower PCs, so they run
// first, and the finished ones wait at the exit.
//
// Programming model (see gpu_pkg.sv): a thread reads tid/ctaid/ntid/nctaid
// and its kernel argument pointer with csrr, uses RV32I plus MUL, and exits
// with ECALL. EBREAK, MRET and illegal instructions kill the thread and set
// `error`. FENCE is a no-op (memory is in order).

import rv32i_pkg::*;
import gpu_pkg::*;

module simt_core (
    input  logic                  clk,
    input  logic                  rst,

    // block launch, from the dispatcher in gpu_top
    input  logic                  launch,
    input  logic [31:0]           launch_pc,
    input  logic [31:0]           launch_block,
    input  logic [31:0]           block_dim,
    input  logic [31:0]           grid_dim,
    input  logic [31:0]           args,
    output logic                  idle,

    // instruction memory: synchronous read, data arrives the next cycle
    output logic [31:0]           imem_addr,
    input  logic [31:0]           imem_rdata,

    // global memory: one line per request, responses in request order
    output logic                  mem_req_valid,
    input  logic                  mem_req_ready,
    output logic                  mem_req_we,
    output logic [LINE_IDX-1:0]   mem_req_line,
    output logic [LINE_BITS-1:0]  mem_req_wdata,
    output logic [LINE_BYTES-1:0] mem_req_be,
    input  logic                  mem_resp_valid,
    input  logic [LINE_BITS-1:0]  mem_resp_rdata,

    // performance and status
    output logic                  issue,          // a warp instruction was fetched
    output logic [LANE_BITS:0]    issue_threads,  // ...for this many threads
    output logic                  error,          // sticky

    // FPGA harness: base clock and phase for the register files' block RAMs
    input  logic                  fpga_fclk,
    input  logic [3:0]            fpga_ph
);

    localparam int NT = WARPS * LANES;

    // ---------------------------------------------------------------- state
    logic [NT-1:0][31:0]  pc;            // one PC per thread: index warp*LANES + lane
    logic [NT-1:0]        alive;         // thread is in this block and hasn't exited
    logic [WARPS-1:0]     busy;          // warp has an instruction in flight
    logic                 running;
    logic [31:0]          block_id, ntid, nctaid, kargs;

    // One register file per lane, holding that lane's 32 registers for each
    // warp at {warp, reg}. x0 is never written. FPGA: block RAM, see "register
    // files" below.

    assign idle = !running;

    // ---------------------------------------------------------------- F: schedule + fetch
    logic [WARPS-1:0]     ready;
    logic [WARP_BITS-1:0] rr, pick, cand;
    logic                 pick_valid;
    logic [LANES-1:0]     pick_alive, f_mask;
    logic [31:0]          f_pc;

    always_comb begin
        for (int w = 0; w < WARPS; w++)
            ready[w] = running && (|alive[w*LANES +: LANES]) && !busy[w];

        // round robin: the first ready warp after the one picked last
        pick_valid = 1'b0;
        pick = '0;
        for (int i = 1; i <= WARPS; i++) begin
            cand = rr + i[WARP_BITS-1:0];
            if (!pick_valid && ready[cand]) begin
                pick_valid = 1'b1;
                pick = cand;
            end
        end

        // the warp runs at its lowest live PC, with the threads sitting there
        pick_alive = alive[pick*LANES +: LANES];
        f_pc = 32'hFFFF_FFFF;
        for (int l = 0; l < LANES; l++)
            if (pick_alive[l] && pc[pick*LANES + l] < f_pc) f_pc = pc[pick*LANES + l];
        for (int l = 0; l < LANES; l++)
            f_mask[l] = pick_alive[l] && pc[pick*LANES + l] == f_pc;

        issue_threads = '0;
        for (int l = 0; l < LANES; l++) issue_threads = issue_threads + f_mask[l];
    end

    assign imem_addr = f_pc;
    assign issue     = pick_valid;

    logic                 d_valid;
    logic [WARP_BITS-1:0] d_warp;
    logic [31:0]          d_pc;
    logic [LANES-1:0]     d_mask;

    always_ff @(posedge clk) begin
        if (rst) begin
            d_valid <= 1'b0;
            rr      <= '0;
        end else begin
            d_valid <= pick_valid;
            if (pick_valid) rr <= pick;
        end
        d_warp <= pick;
        d_pc   <= f_pc;
        d_mask <= f_mask;
    end

    // ---------------------------------------------------------------- D: decode + register read
    // The riscv-core decoder, unchanged: one per core, shared by every thread.
    logic [31:0] d_instr;
    assign d_instr = imem_rdata;

    logic       dc_reg_write, dc_b_imm, dc_mem_read, dc_mem_write, dc_branch, dc_jal, dc_jalr;
    logic       dc_csr_en, dc_ecall, dc_ebreak, dc_mret, dc_fence_i, dc_uses_rs1, dc_uses_rs2, dc_illegal;
    logic [1:0] dc_a_sel, dc_wb_sel;
    logic [3:0] dc_alu_op;
    logic [2:0] dc_imm_sel;
    logic [31:0] d_imm;

    decoder u_dec (
        .instr(d_instr), .reg_write(dc_reg_write), .a_sel(dc_a_sel), .b_imm(dc_b_imm),
        .alu_op(dc_alu_op), .imm_sel(dc_imm_sel), .mem_read(dc_mem_read), .mem_write(dc_mem_write),
        .wb_sel(dc_wb_sel), .branch(dc_branch), .jal(dc_jal), .jalr(dc_jalr), .csr_en(dc_csr_en),
        .ecall(dc_ecall), .ebreak(dc_ebreak), .mret(dc_mret), .fence_i(dc_fence_i),
        .uses_rs1(dc_uses_rs1), .uses_rs2(dc_uses_rs2), .illegal(dc_illegal)
    );
    immgen u_imm (.instr(d_instr), .imm_sel(dc_imm_sel), .imm(d_imm));

    // MUL (from the M extension) is the one instruction added to RV32I: a GPU
    // without a multiplier can't do much. The rest of M is illegal here.
    logic d_is_m, d_mul;
    assign d_is_m = d_instr[6:0] == OP_REG && d_instr[31:25] == 7'b0000001;
    assign d_mul  = d_is_m && d_instr[14:12] == 3'b000;

    logic [4:0] d_rs1, d_rs2;
    assign d_rs1 = d_instr[19:15];
    assign d_rs2 = d_instr[24:20];

    logic [LANES-1:0][31:0] d_rs1v, d_rs2v;     // driven by the register files below

    logic                   x_valid, x_reg_write, x_b_imm, x_mem_read, x_mem_write;
    logic                   x_branch, x_jal, x_jalr, x_ecall, x_bad, x_mul;
    logic [1:0]             x_a_sel, x_wb_sel;
    logic [3:0]             x_alu_op;
    logic [WARP_BITS-1:0]   x_warp;
    logic [31:0]            x_pc, x_imm;
    logic [LANES-1:0]       x_mask;
    logic [4:0]             x_rd;
    logic [2:0]             x_funct3;
    logic [11:0]            x_csr;
    logic [LANES-1:0][31:0] x_rs1v, x_rs2v;

    always_ff @(posedge clk) begin
        x_valid     <= rst ? 1'b0 : d_valid;
        x_warp      <= d_warp;
        x_pc        <= d_pc;
        x_mask      <= d_mask;
        x_imm       <= d_imm;
        x_rd        <= d_instr[11:7];
        x_funct3    <= d_instr[14:12];
        x_csr       <= d_instr[31:20];
        x_rs1v      <= d_rs1v;
        x_rs2v      <= d_rs2v;
        x_reg_write <= dc_reg_write;
        x_a_sel     <= dc_a_sel;
        x_b_imm     <= dc_b_imm;
        x_alu_op    <= dc_alu_op;
        x_mem_read  <= dc_mem_read;
        x_mem_write <= dc_mem_write;
        x_wb_sel    <= dc_wb_sel;
        x_branch    <= dc_branch;
        x_jal       <= dc_jal;
        x_jalr      <= dc_jalr;
        x_ecall     <= dc_ecall;
        x_mul       <= d_mul;
        x_bad       <= dc_ebreak || dc_mret || dc_illegal || (d_is_m && !d_mul);
    end

    // ---------------------------------------------------------------- X: LANES ALUs + write back
    logic [LANES-1:0][31:0] x_a, x_b, x_csrv, x_wdata, x_next;
    wire  [LANES-1:0][31:0] x_alu;
    wire  [LANES-1:0]       x_taken;

    for (genvar g = 0; g < LANES; g++) begin : lane
        alu         u_alu (.a(x_a[g]), .b(x_b[g]), .op(x_alu_op), .y(x_alu[g]));
        branch_unit u_br  (.a(x_rs1v[g]), .b(x_rs2v[g]), .funct3(x_funct3), .taken(x_taken[g]));
    end

    always_comb
        for (int l = 0; l < LANES; l++) begin
            x_a[l] = x_a_sel == A_PC ? x_pc : x_a_sel == A_ZERO ? 32'd0 : x_rs1v[l];
            x_b[l] = x_b_imm ? x_imm : x_rs2v[l];

            case (x_csr)
                CSR_TID:    x_csrv[l] = 32'(x_warp) * LANES + l;
                CSR_CTAID:  x_csrv[l] = block_id;
                CSR_NTID:   x_csrv[l] = ntid;
                CSR_NCTAID: x_csrv[l] = nctaid;
                CSR_ARGS:   x_csrv[l] = kargs;
                default:    x_csrv[l] = 32'd0;
            endcase

            case (x_wb_sel)
                WB_PC4:  x_wdata[l] = x_pc + 32'd4;
                WB_CSR:  x_wdata[l] = x_csrv[l];
                default: x_wdata[l] = x_mul ? x_rs1v[l] * x_rs2v[l] : x_alu[l];
            endcase

            if (x_jal)                         x_next[l] = x_pc + x_imm;
            else if (x_jalr)                   x_next[l] = x_alu[l] & ~32'd1;
            else if (x_branch && x_taken[l])   x_next[l] = x_pc + x_imm;
            else                               x_next[l] = x_pc + 32'd4;
        end

    logic x_mem, x_exit;
    assign x_mem  = x_valid && (x_mem_read || x_mem_write);
    assign x_exit = x_valid && (x_ecall || x_bad);

    // ---------------------------------------------------------------- load/store unit
    logic                   lsu_wb_valid, lsu_done_valid;
    logic [WARP_BITS-1:0]   lsu_wb_warp, lsu_done_warp;
    logic [4:0]             lsu_wb_rd;
    logic [LANES-1:0]       lsu_wb_mask, lsu_done_mask;
    logic [LANES-1:0][31:0] lsu_wb_data;

    lsu u_lsu (
        .clk(clk), .rst(rst),
        .push(x_mem), .push_warp(x_warp), .push_mask(x_mask), .push_addr(x_alu),
        .push_wdata(x_rs2v), .push_funct3(x_funct3), .push_rd(x_rd), .push_store(x_mem_write),
        .req_valid(mem_req_valid), .req_ready(mem_req_ready), .req_we(mem_req_we),
        .req_line(mem_req_line), .req_wdata(mem_req_wdata), .req_be(mem_req_be),
        .resp_valid(mem_resp_valid), .resp_rdata(mem_resp_rdata),
        .wb_valid(lsu_wb_valid), .wb_warp(lsu_wb_warp), .wb_rd(lsu_wb_rd),
        .wb_mask(lsu_wb_mask), .wb_data(lsu_wb_data),
        .done_valid(lsu_done_valid), .done_warp(lsu_done_warp), .done_mask(lsu_done_mask)
    );

    // ---------------------------------------------------------------- state updates
    // Two writers per cycle at most (X and the LSU), always for different
    // warps: a warp in X has no load outstanding, and vice versa.
    always_ff @(posedge clk) begin
        if (x_valid && !x_mem && !x_exit)
            for (int l = 0; l < LANES; l++)
                if (x_mask[l]) pc[x_warp*LANES + l] <= x_next[l];

        if (lsu_done_valid)
            for (int l = 0; l < LANES; l++)
                if (lsu_done_mask[l]) pc[lsu_done_warp*LANES + l] <= pc[lsu_done_warp*LANES + l] + 32'd4;

        if (launch)
            for (int t = 0; t < NT; t++) pc[t] <= launch_pc;
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            running <= 1'b0;
            busy    <= '0;
            alive   <= '0;
            error   <= 1'b0;
        end else if (launch) begin
            running  <= 1'b1;
            busy     <= '0;
            for (int t = 0; t < NT; t++) alive[t] <= t < block_dim;
            block_id <= launch_block;
            ntid     <= block_dim;
            nctaid   <= grid_dim;
            kargs    <= args;
        end else if (running) begin
            if (pick_valid)                 busy[pick]          <= 1'b1;
            if (x_valid && !x_mem)          busy[x_warp]        <= 1'b0;
            if (lsu_done_valid)             busy[lsu_done_warp] <= 1'b0;
            if (x_exit)                     alive[x_warp*LANES +: LANES] <= alive[x_warp*LANES +: LANES] & ~x_mask;
            if (x_valid && x_bad)           error <= 1'b1;
            if (alive == '0 && busy == '0)  running <= 1'b0;
        end
    end

    // ---------------------------------------------------------------- register files
    // FPGA adaptation. The original register file is an array read
    // combinationally (rs1 and rs2 of every lane, in D) and written at the
    // DUT edge by two writers (X's result, the LSU's load data; different
    // warps). Here each lane's file is a true dual-port block RAM clocked by
    // the harness base clock `fpga_fclk` (8 base cycles per DUT cycle, phase
    // `fpga_ph`; the DUT edge E7 ends phase 7):
    //   E7     capture both writers' enables, addresses and data
    //   ph 0   port A writes X's result     ph 1   port B writes the LSU's
    //   ph 4   port A reads rs1, port B rs2  E5     hold them until next E5
    // so D's reads see every write of earlier DUT edges, as before. While
    // the harness is idle the phase stays 0 and port A repeats its last write.
    localparam int RA = WARP_BITS + 5;
    logic [LANES-1:0]       w1_en, w2_en;
    logic [RA-1:0]          w1_a, w2_a;
    logic [LANES-1:0][31:0] w1_d, w2_d, q1, q2, rs1_r, rs2_r;

    initial begin
        w1_en = '0; w2_en = '0; w1_a = '0; w2_a = '0; w1_d = '0; w2_d = '0; rs1_r = '0; rs2_r = '0;
    end

    always_ff @(posedge fpga_fclk) begin
        if (fpga_ph == 4'd7) begin
            for (int l = 0; l < LANES; l++) begin
                w1_en[l] <= x_valid && !x_mem && !x_exit && x_mask[l] && x_reg_write && x_rd != 5'd0;
                w2_en[l] <= lsu_wb_valid && lsu_wb_rd != 5'd0 && lsu_wb_mask[l];
            end
            w1_a <= {x_warp, x_rd};
            w2_a <= {lsu_wb_warp, lsu_wb_rd};
            w1_d <= x_wdata;
            w2_d <= lsu_wb_data;
        end
        if (fpga_ph == 4'd5) begin
            rs1_r <= q1;
            rs2_r <= q2;
        end
    end

    genvar gl;
    generate
        for (gl = 0; gl < LANES; gl++) begin : lane_rf
            fpga_tdp_ram #(.DW(32), .AW(RA)) u_rf (
                .clk(fpga_fclk),
                .we_a(fpga_ph == 4'd0 && w1_en[gl]),
                .addr_a(fpga_ph == 4'd0 ? w1_a : {d_warp, d_rs1}),
                .wdata_a(w1_d[gl]), .q_a(q1[gl]),
                .we_b(fpga_ph == 4'd1 && w2_en[gl]),
                .addr_b(fpga_ph == 4'd1 ? w2_a : {d_warp, d_rs2}),
                .wdata_b(w2_d[gl]), .q_b(q2[gl]));
            assign d_rs1v[gl] = d_rs1 == 5'd0 ? 32'd0 : rs1_r[gl];
            assign d_rs2v[gl] = d_rs2 == 5'd0 ? 32'd0 : rs2_r[gl];
        end
    endgenerate

endmodule
