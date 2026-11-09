// simt_core.sv — one SIMT core: WARPS warps of LANES threads, one front end
//
// What makes this a GPU core and not WARPS*LANES little CPUs: there is ONE
// fetch, ONE decoder and ONE scheduler, shared by every thread. Each cycle
// the scheduler picks a warp, fetches one instruction for it, and LANES
// copies of the riscv-core ALU execute that instruction for LANES threads at
// once, each on its own registers. That's SIMT: single instruction, multiple
// threads.
//
// Pipeline: F (pick a warp, fetch) -> D (decode, start the register reads)
// -> R (the reads land) -> X (LANES ALUs, write back). Each lane's register
// file is a synchronous SRAM (simt_lane.sv), hence R. A warp has at most one
// instruction in flight, so there are no data hazards and no forwarding at
// all: while warp 0 waits for its result, warps 1-3 fill the pipeline. That's
// how GPUs hide latency, with other warps' work instead of forwarding paths
// and caches.
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
    output logic                  error           // sticky
);

    localparam int NT = WARPS * LANES;

    // ---------------------------------------------------------------- state
    logic [NT-1:0][31:0]  pc;            // one PC per thread: index warp*LANES + lane
    logic [NT-1:0]        alive;         // thread is in this block and hasn't exited
    logic [WARPS-1:0]     busy;          // warp has an instruction in flight
    logic                 running;
    logic [31:0]          block_id, ntid, nctaid, kargs;

    assign idle = !running;

    // ---------------------------------------------------------------- F: schedule + fetch
    logic [WARPS-1:0]     ready;
    logic                 lq_ok;         // room for what the LSU may still send (LSU events below)
    logic [WARP_BITS-1:0] rr, pick, cand;
    logic                 pick_valid;
    logic [LANES-1:0]     pick_alive, f_mask;
    logic [31:0]          f_pc;

    always_comb begin
        for (int w = 0; w < WARPS; w++)
            ready[w] = running && (|alive[w*LANES +: LANES]) && !busy[w] && lq_ok;

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
    end

    assign imem_addr = f_pc;

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

    // The counters' view of issue comes from D, a cycle after F: driven from
    // F, these outputs sat at the end of the scheduler's long PC compare.
    assign issue = d_valid;
    always_comb begin
        issue_threads = '0;
        for (int l = 0; l < LANES; l++) issue_threads = issue_threads + d_mask[l];
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

    // ---------------------------------------------------------------- R: register reads land
    // (the reads are in each simt_lane, addressed from D)
    logic                   r_valid, r_reg_write, r_b_imm, r_mem_read, r_mem_write;
    logic                   r_branch, r_jal, r_jalr, r_ecall, r_bad, r_mul, r_z1, r_z2;
    logic [1:0]             r_a_sel, r_wb_sel;
    logic [3:0]             r_alu_op;
    logic [WARP_BITS-1:0]   r_warp;
    logic [31:0]            r_pc, r_imm;
    logic [LANES-1:0]       r_mask;
    logic [4:0]             r_rd;
    logic [2:0]             r_funct3;
    logic [11:0]            r_csr;

    always_ff @(posedge clk) begin
        r_valid     <= rst ? 1'b0 : d_valid;
        r_warp      <= d_warp;
        r_pc        <= d_pc;
        r_mask      <= d_mask;
        r_imm       <= d_imm;
        r_rd        <= d_instr[11:7];
        r_funct3    <= d_instr[14:12];
        r_csr       <= d_instr[31:20];
        r_z1        <= d_rs1 == 5'd0;
        r_z2        <= d_rs2 == 5'd0;
        r_reg_write <= dc_reg_write;
        r_a_sel     <= dc_a_sel;
        r_b_imm     <= dc_b_imm;
        r_alu_op    <= dc_alu_op;
        r_mem_read  <= dc_mem_read;
        r_mem_write <= dc_mem_write;
        r_wb_sel    <= dc_wb_sel;
        r_branch    <= dc_branch;
        r_jal       <= dc_jal;
        r_jalr      <= dc_jalr;
        r_ecall     <= dc_ecall;
        r_mul       <= d_mul;
        r_bad       <= dc_ebreak || dc_mret || dc_illegal || (d_is_m && !d_mul);
    end

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

    always_ff @(posedge clk) begin
        x_valid     <= rst ? 1'b0 : r_valid;
        x_warp      <= r_warp;
        x_pc        <= r_pc;
        x_mask      <= r_mask;
        x_imm       <= r_imm;
        x_rd        <= r_rd;
        x_funct3    <= r_funct3;
        x_csr       <= r_csr;
        x_reg_write <= r_reg_write;
        x_a_sel     <= r_a_sel;
        x_b_imm     <= r_b_imm;
        x_alu_op    <= r_alu_op;
        x_mem_read  <= r_mem_read;
        x_mem_write <= r_mem_write;
        x_wb_sel    <= r_wb_sel;
        x_branch    <= r_branch;
        x_jal       <= r_jal;
        x_jalr      <= r_jalr;
        x_ecall     <= r_ecall;
        x_mul       <= r_mul;
        x_bad       <= r_bad;
    end

    // ---------------------------------------------------------------- X: LANES lanes + write back
    logic [LANES-1:0][31:0] x_csrv, x_next, x_alu, x_rs2v;
    logic [LANES-1:0]       wr_en;
    logic                   wr_lsu;
    logic [RF_AW-1:0]       wr_a;
    logic [LANES-1:0][31:0] lq_hdata;

    always_comb
        for (int l = 0; l < LANES; l++)
            case (x_csr)
                CSR_TID:    x_csrv[l] = 32'(x_warp) * LANES + l;
                CSR_CTAID:  x_csrv[l] = block_id;
                CSR_NTID:   x_csrv[l] = ntid;
                CSR_NCTAID: x_csrv[l] = nctaid;
                CSR_ARGS:   x_csrv[l] = kargs;
                default:    x_csrv[l] = 32'd0;
            endcase

    for (genvar g = 0; g < LANES; g++) begin : lane
        simt_lane u_lane (
            .clk,
            .rd_en(d_valid), .rd_a1({d_warp, d_rs1}), .rd_a2({d_warp, d_rs2}),
            .r_z1, .r_z2,
            .a_sel(x_a_sel), .b_imm(x_b_imm), .alu_op(x_alu_op), .funct3(x_funct3), .wb_sel(x_wb_sel),
            .mul(x_mul), .jal(x_jal), .jalr(x_jalr), .branch(x_branch),
            .pc(x_pc), .imm(x_imm), .csr(x_csrv[g]),
            .rs2v(x_rs2v[g]), .alu_y(x_alu[g]), .next_pc(x_next[g]),
            .wr_en(wr_en[g]), .wr_lsu, .wr_a, .lsu_data(lq_hdata[g])
        );
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

    // ---------------------------------------------------------------- LSU events
    // A lane's register file has one write port, and X uses it whenever it
    // retires an instruction with a result. So each LSU response (load data
    // for a warp, and/or "this warp's memory instruction is done") waits in a
    // small queue and is applied, in order, in a cycle when X isn't writing;
    // a done event needs no write port. The warp stays busy until its done
    // event is applied, so it can't read a register its load hasn't filled.
    // Issue pauses while the queue is half full: X drains within three
    // cycles (the instructions already in D, R and X), the LSU adds at most
    // one event per cycle, so the queue never overflows.
    localparam int LQ  = 8;
    localparam int LQB = $clog2(LQ);

    logic [LQ-1:0]                  lq_wb, lq_done;
    logic [LQ-1:0][WARP_BITS-1:0]   lq_warp;
    logic [LQ-1:0][4:0]             lq_rd;
    logic [LQ-1:0][LANES-1:0]       lq_wmask, lq_dmask;
    logic [LQ-1:0][LANES-1:0][31:0] lq_data;
    logic [LQB-1:0]                 lq_head, lq_tail;
    logic [LQB:0]                   lq_count;
    logic                           lq_push, lq_port, lq_apply, x_wr;

    assign lq_ok    = lq_count < LQ / 2;
    assign lq_push  = lsu_wb_valid || lsu_done_valid;
    assign lq_port  = lq_wb[lq_head] && lq_rd[lq_head] != 5'd0;      // head needs the write port
    assign x_wr     = x_valid && !x_mem && !x_exit && x_reg_write && x_rd != 5'd0;
    assign lq_apply = lq_count != 0 && !(lq_port && x_wr);
    assign lq_hdata = lq_data[lq_head];

    logic [WARP_BITS-1:0] lq_hwarp;
    logic [LANES-1:0]     lq_hdmask;
    assign lq_hwarp  = lq_warp[lq_head];
    assign lq_hdmask = lq_dmask[lq_head];

    assign wr_lsu = lq_apply && lq_port;
    assign wr_a   = wr_lsu ? {lq_hwarp, lq_rd[lq_head]} : {x_warp, x_rd};
    assign wr_en  = wr_lsu ? lq_wmask[lq_head] : (x_wr ? x_mask : '0);

    always_ff @(posedge clk) begin
        if (rst) begin
            lq_head  <= '0;
            lq_tail  <= '0;
            lq_count <= '0;
        end else begin
            if (lq_push) begin
                lq_wb[lq_tail]    <= lsu_wb_valid;
                lq_done[lq_tail]  <= lsu_done_valid;
                lq_warp[lq_tail]  <= lsu_done_warp;          // same tag as wb_warp
                lq_rd[lq_tail]    <= lsu_wb_rd;
                lq_wmask[lq_tail] <= lsu_wb_mask;
                lq_dmask[lq_tail] <= lsu_done_mask;
                lq_data[lq_tail]  <= lsu_wb_data;
                lq_tail <= lq_tail + 1'b1;
            end
            if (lq_apply) lq_head <= lq_head + 1'b1;
            lq_count <= lq_count + (lq_push ? 1'b1 : 1'b0) - (lq_apply ? 1'b1 : 1'b0);
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk)
        if (!rst && lq_push && !lq_apply && lq_count == LQ) $fatal(1, "simt_core: LSU event queue overflow");
`endif

    // ---------------------------------------------------------------- state updates
    always_ff @(posedge clk) begin
        if (x_valid && !x_mem && !x_exit)
            for (int l = 0; l < LANES; l++)
                if (x_mask[l]) pc[x_warp*LANES + l] <= x_next[l];

        if (lq_apply && lq_done[lq_head])
            for (int l = 0; l < LANES; l++)
                if (lq_hdmask[l]) pc[lq_hwarp*LANES + l] <= pc[lq_hwarp*LANES + l] + 32'd4;

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
            if (lq_apply && lq_done[lq_head]) busy[lq_hwarp] <= 1'b0;
            if (x_exit)                     alive[x_warp*LANES +: LANES] <= alive[x_warp*LANES +: LANES] & ~x_mask;
            if (x_valid && x_bad)           error <= 1'b1;
            if (alive == '0 && busy == '0)  running <= 1'b0;
        end
    end

endmodule
