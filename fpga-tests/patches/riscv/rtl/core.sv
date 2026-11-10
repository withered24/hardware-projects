// core.sv — 5-stage pipelined RV32I core (machine mode only)
//
// IF -> ID -> EX -> MEM -> WB, up to five instructions in flight. Build
// guide: reference/pipelining.html in the course. The single-cycle core this
// grew out of is still in core_single.sv, and the regression runs on both.
//
// Pipeline registers are named for the stage they FEED: d_* is IF/ID, x_* is
// ID/EX, m_* is EX/MEM, w_* is MEM/WB. Each has a valid bit saying whether it
// holds a real instruction or a bubble, and bubbles carry all-zero controls,
// so a bubble can't write a register, touch memory, branch, or trap.
//
// Data hazards
//   Forwarding into EX from EX/MEM (one instruction ahead) and MEM/WB (two
//   ahead); EX/MEM wins when both match, because it's the newer write. A
//   value three ahead is being written to the register file in WB while ID
//   reads it, so ID's register read has a write-through bypass.
//   Loads and CSR reads produce their value in MEM, too late for the very
//   next instruction's EX, so the hazard unit stalls that pair one cycle
//   (the load-use stall): PC and IF/ID hold, a bubble goes into ID/EX.
//
// Control hazards
//   Predict not taken. Branches and jumps resolve in EX; a taken one flushes
//   the two younger instructions (in IF/ID and ID/EX). FENCE.I resolves the
//   same way, as a "jump" to pc+4, so everything after it is refetched after
//   every older store has landed.
//
// Traps are precise
//   Exceptions found in ID (illegal, ECALL, EBREAK) or EX (misaligned jump
//   target or data address) ride down the pipe and are taken when the
//   instruction reaches MEM, where the CSRs live. Everything older has left
//   MEM by then; everything younger is flushed. MRET works the same way.
//
// Memory
//   Both ports have a registered (synchronous) read, like FPGA block RAM.
//   The fetch address is pc_next, so the instruction at pc arrives during
//   IF. The data address is presented in EX, so load data arrives in MEM.
//   That puts the store's write at the end of EX, so a store only writes if
//   nothing can still cancel it: the store itself didn't fault, and no older
//   instruction is trapping or returning in MEM that cycle.
//
// Memory wait states
//   A slow device (an AXI4-Lite peripheral, say) raises dmem_wait while the
//   access of the instruction in MEM hasn't finished. That freezes IF, ID,
//   EX and MEM in place and feeds bubbles into WB. The data port's request
//   (dmem_addr/wdata/wstrb/re, driven from EX) counts as accepted only at a
//   clock edge where dmem_wait was low, so a frozen store in EX is not
//   written twice. While frozen, EX latches its operands after forwarding:
//   the instruction it was forwarding from in WB moves on, and its value
//   must not be lost. Plain block RAM just ties dmem_wait low.
//
// The trace_* outputs describe the instruction leaving WB (trace_valid = 0
// for a bubble). The testbench compares them against the golden model.
//
// Under `ifdef FORMAL, the end of the file has SVA properties for these
// rules (bubbles, forwarding, stalls, flushes, stores, traps), checked by
// formal/core.sby.

import rv32i_pkg::*;

module core #(
    parameter logic [31:0] RESET_PC = 32'h0000_0000
) (
    input  logic        clk,
    input  logic        rst,           // synchronous, active high
    input  logic        rf_clear,      // FPGA replay only: see patches/riscv/rtl/regfile.sv

    // instruction memory (registered read: address now, data next cycle)
    output logic [31:0] imem_addr,
    input  logic [31:0] imem_rdata,

    // data memory (registered read; byte-strobed synchronous write)
    output logic [31:0] dmem_addr,
    output logic [31:0] dmem_wdata,
    output logic [3:0]  dmem_wstrb,
    output logic        dmem_re,       // a real load is in EX (devices need this; RAM can ignore it)
    input  logic [31:0] dmem_rdata,
    input  logic        dmem_wait,     // MEM's access isn't done: freeze the pipeline

    // retirement trace for the testbench
    output logic        trace_valid,
    output logic [31:0] trace_pc,
    output logic [31:0] trace_instr,
    output logic        trace_rd_we,
    output logic [4:0]  trace_rd,
    output logic [31:0] trace_rd_wdata,
    output logic        trace_illegal,
    output logic        trace_trap,
    output logic [31:0] trace_cause,
    output logic [31:0] trace_mem_addr,
    output logic [3:0]  trace_mem_wstrb,
    output logic [31:0] trace_mem_wdata
);

    localparam logic [31:0] NOP = 32'h0000_0013;   // addi x0, x0, 0

    // Signals that flow backwards (from later stages to earlier ones).
    logic        hold;             // MEM: waiting on a slow memory access
    logic        stall;            // ID: load-use hazard
    logic        redirect_x;       // EX: taken branch/jump, or FENCE.I
    logic [31:0] redirect_pc;
    logic        trap_m, mret_m;   // MEM: take a trap / return from one
    logic [31:0] mtvec_base, mepc_q;

    // MEM/WB, needed by ID (bypass) and EX (forwarding)
    logic        w_valid, w_rd_we, w_trap, w_illegal;
    logic [4:0]  w_rd;
    logic [31:0] w_pc, w_instr, w_wdata, w_cause;
    logic [31:0] w_st_addr, w_st_data;
    logic [3:0]  w_st_strb;
    logic        w_rf_we;          // register-file write this cycle

    assign w_rf_we = w_valid && w_rd_we;
    assign hold    = dmem_wait && !rst;

    // ==================================================================
    // IF: fetch
    // ==================================================================
    // Priority: the oldest instruction's redirect wins. A trap or MRET in
    // MEM beats a branch in EX, which beats a stall in ID. A memory wait
    // beats everything: nothing moves until MEM's access is done (and the
    // fetch address must stay pc_f so imem_rdata stays its instruction).
    logic [31:0] pc_f, pc_next;

    always_comb begin
        if (rst)             pc_next = RESET_PC;
        else if (hold)       pc_next = pc_f;
        else if (trap_m)     pc_next = mtvec_base;
        else if (mret_m)     pc_next = mepc_q;
        else if (redirect_x) pc_next = redirect_pc;
        else if (stall)      pc_next = pc_f;
        else                 pc_next = pc_f + 32'd4;
    end

    always_ff @(posedge clk) pc_f <= pc_next;

    // The memory registers this address at the clock edge, so during the
    // next cycle imem_rdata is the instruction at pc_f.
    assign imem_addr = pc_next;

    // ---- IF/ID ----
    logic        d_valid;
    logic [31:0] d_pc, d_instr;
    logic        flush_d;

    assign flush_d = trap_m || mret_m || redirect_x;

    always_ff @(posedge clk) begin
        if (rst || (flush_d && !hold)) begin
            d_valid <= 1'b0;
            d_pc    <= 32'd0;
            d_instr <= NOP;
        end else if (!stall && !hold) begin
            d_valid <= 1'b1;
            d_pc    <= pc_f;
            d_instr <= imem_rdata;
        end
    end

    // ==================================================================
    // ID: decode, register read, hazard detection
    // ==================================================================
    logic [4:0] rs1_d, rs2_d, rd_d;
    assign rs1_d = d_instr[19:15];
    assign rs2_d = d_instr[24:20];
    assign rd_d  = d_instr[11:7];

    logic       dec_reg_write, dec_b_imm, dec_mem_read, dec_mem_write;
    logic       dec_branch, dec_jal, dec_jalr, dec_fence_i;
    logic       dec_csr_en, dec_ecall, dec_ebreak, dec_mret, dec_illegal;
    logic       uses_rs1_d, uses_rs2_d;
    logic [1:0] dec_a_sel, dec_wb_sel;
    logic [3:0] dec_alu_op;
    logic [2:0] dec_imm_sel;

    decoder u_decoder (
        .instr     (d_instr),
        .reg_write (dec_reg_write),
        .a_sel     (dec_a_sel),
        .b_imm     (dec_b_imm),
        .alu_op    (dec_alu_op),
        .imm_sel   (dec_imm_sel),
        .mem_read  (dec_mem_read),
        .mem_write (dec_mem_write),
        .wb_sel    (dec_wb_sel),
        .branch    (dec_branch),
        .jal       (dec_jal),
        .jalr      (dec_jalr),
        .csr_en    (dec_csr_en),
        .ecall     (dec_ecall),
        .ebreak    (dec_ebreak),
        .mret      (dec_mret),
        .fence_i   (dec_fence_i),
        .uses_rs1  (uses_rs1_d),
        .uses_rs2  (uses_rs2_d),
        .illegal   (dec_illegal)
    );

    // A bubble in IF/ID decodes to all-zero controls.
    ctrl_t ctrl_d;
    always_comb begin
        ctrl_d = '0;
        if (d_valid) begin
            ctrl_d.reg_write = dec_reg_write;
            ctrl_d.a_sel     = dec_a_sel;
            ctrl_d.b_imm     = dec_b_imm;
            ctrl_d.alu_op    = dec_alu_op;
            ctrl_d.mem_read  = dec_mem_read;
            ctrl_d.mem_write = dec_mem_write;
            ctrl_d.wb_sel    = dec_wb_sel;
            ctrl_d.branch    = dec_branch;
            ctrl_d.jal       = dec_jal;
            ctrl_d.jalr      = dec_jalr;
            ctrl_d.fence_i   = dec_fence_i;
            ctrl_d.csr_en    = dec_csr_en;
            ctrl_d.ecall     = dec_ecall;
            ctrl_d.ebreak    = dec_ebreak;
            ctrl_d.mret      = dec_mret;
            ctrl_d.illegal   = dec_illegal;
        end
    end

    logic [31:0] imm_d;
    immgen u_immgen (.instr(d_instr), .imm_sel(dec_imm_sel), .imm(imm_d));

    logic [31:0] rf_rs1, rf_rs2, rs1_val_d, rs2_val_d;

    regfile u_rf (
        .clk      (clk),
        .rs1_addr (rs1_d),
        .rs1_data (rf_rs1),
        .rs2_addr (rs2_d),
        .rs2_data (rf_rs2),
        .rd_we    (w_rf_we),
        .rd_addr  (w_rd),
        .rd_data  (w_wdata),
        .rf_clear (rf_clear)
    );

    // Write-through bypass: WB writes the register file at the end of this
    // cycle, but ID is reading it now. Without this, an instruction three
    // behind its producer reads the stale value.
    assign rs1_val_d = (w_rf_we && w_rd != 5'd0 && w_rd == rs1_d) ? w_wdata : rf_rs1;
    assign rs2_val_d = (w_rf_we && w_rd != 5'd0 && w_rd == rs2_d) ? w_wdata : rf_rs2;

    // ---- hazard unit: load-use stall ----
    // The instruction in EX gets its rd value only in MEM (a load's data, a
    // CSR's old value). If the instruction in ID reads that register, it
    // would reach EX one cycle before the value exists: hold it in ID.
    logic        x_valid;
    ctrl_t       x_ctrl;
    logic [4:0]  x_rs1, x_rs2, x_rd;
    logic        x_late;

    assign x_late = x_ctrl.mem_read || x_ctrl.csr_en;   // zero for a bubble
    assign stall  = d_valid && x_late && x_rd != 5'd0 &&
                    ((uses_rs1_d && rs1_d == x_rd) || (uses_rs2_d && rs2_d == x_rd));

    // ---- ID/EX ----
    logic [31:0] x_pc, x_instr, x_rs1_val, x_rs2_val, x_imm;
    logic [31:0] rs1_x, rs2_x;   // EX operand values after forwarding
    logic [2:0]  x_funct3;
    logic        flush_x;

    assign flush_x = trap_m || mret_m || redirect_x || stall;   // stall = insert bubble

    always_ff @(posedge clk) begin
        if (rst) begin
            x_valid   <= 1'b0;
            x_ctrl    <= '0;
            x_pc      <= 32'd0;
            x_instr   <= NOP;
            x_rs1     <= 5'd0;
            x_rs2     <= 5'd0;
            x_rd      <= 5'd0;
            x_rs1_val <= 32'd0;
            x_rs2_val <= 32'd0;
            x_imm     <= 32'd0;
            x_funct3  <= 3'd0;
        end else if (hold) begin
            // Frozen: keep the instruction, but latch its operands as
            // forwarded now (a WB producer won't be in WB next cycle).
            x_rs1_val <= rs1_x;
            x_rs2_val <= rs2_x;
        end else if (flush_x) begin
            x_valid   <= 1'b0;
            x_ctrl    <= '0;
            x_instr   <= NOP;
            x_rd      <= 5'd0;
        end else begin
            x_valid   <= d_valid;
            x_ctrl    <= ctrl_d;
            x_pc      <= d_pc;
            x_instr   <= d_instr;
            x_rs1     <= rs1_d;
            x_rs2     <= rs2_d;
            x_rd      <= rd_d;
            x_rs1_val <= rs1_val_d;
            x_rs2_val <= rs2_val_d;
            x_imm     <= imm_d;
            x_funct3  <= d_instr[14:12];
        end
    end

    // ==================================================================
    // EX: forwarding, ALU, branch resolution, address checks, store
    // ==================================================================
    logic        m_valid;
    ctrl_t       m_ctrl;
    logic [4:0]  m_rd;
    logic [31:0] m_result;

    // ---- forwarding unit ----
    // EX/MEM is checked first: if both match, it's the newer write.
    // (m_result is wrong for a load or CSR in EX/MEM, but the stall above
    // guarantees no instruction that reads it is in EX at that moment.)
    logic [1:0] fwd_a, fwd_b;

    always_comb begin
        if (m_ctrl.reg_write && m_rd != 5'd0 && m_rd == x_rs1)  fwd_a = FWD_EXMEM;
        else if (w_rf_we && w_rd != 5'd0 && w_rd == x_rs1)      fwd_a = FWD_MEMWB;
        else                                                    fwd_a = FWD_NONE;

        if (m_ctrl.reg_write && m_rd != 5'd0 && m_rd == x_rs2)  fwd_b = FWD_EXMEM;
        else if (w_rf_we && w_rd != 5'd0 && w_rd == x_rs2)      fwd_b = FWD_MEMWB;
        else                                                    fwd_b = FWD_NONE;
    end


    always_comb begin
        case (fwd_a)
            FWD_EXMEM: rs1_x = m_result;
            FWD_MEMWB: rs1_x = w_wdata;
            default:   rs1_x = x_rs1_val;
        endcase
        case (fwd_b)
            FWD_EXMEM: rs2_x = m_result;
            FWD_MEMWB: rs2_x = w_wdata;
            default:   rs2_x = x_rs2_val;
        endcase
    end

    // ---- ALU ----
    logic [31:0] alu_a, alu_b, alu_y;

    always_comb begin
        case (x_ctrl.a_sel)
            A_PC:    alu_a = x_pc;
            A_ZERO:  alu_a = 32'd0;
            default: alu_a = rs1_x;
        endcase
    end

    assign alu_b = x_ctrl.b_imm ? x_imm : rs2_x;

    alu u_alu (.a(alu_a), .b(alu_b), .op(x_ctrl.alu_op), .y(alu_y));

    // ---- branches and jumps ----
    logic        br_taken, taken_x;
    logic [31:0] pc_plus4_x, jump_target;

    branch_unit u_br (.a(rs1_x), .b(rs2_x), .funct3(x_funct3), .taken(br_taken));

    assign pc_plus4_x  = x_pc + 32'd4;
    assign jump_target = x_ctrl.jalr ? {alu_y[31:1], 1'b0} : x_pc + x_imm;
    assign taken_x     = x_ctrl.jal || x_ctrl.jalr || (x_ctrl.branch && br_taken);

    // ---- exceptions detectable in EX ----
    logic       mis_fetch_x, mis_load_x, mis_store_x, misaligned_x, exc_x;
    logic [1:0] byte_off_x;

    assign byte_off_x   = alu_y[1:0];
    assign misaligned_x = (x_funct3[1:0] == 2'b01 && byte_off_x[0]) ||     // halfword
                          (x_funct3[1:0] == 2'b10 && byte_off_x != 2'b00); // word
    assign mis_fetch_x  = taken_x && jump_target[1];
    assign mis_load_x   = x_ctrl.mem_read  && misaligned_x;
    assign mis_store_x  = x_ctrl.mem_write && misaligned_x;
    assign exc_x        = x_ctrl.illegal || x_ctrl.ecall || x_ctrl.ebreak ||
                          mis_fetch_x || mis_load_x || mis_store_x;

    // A faulting instruction doesn't redirect: its trap will, from MEM.
    assign redirect_x  = x_valid && !exc_x && (taken_x || x_ctrl.fence_i);
    assign redirect_pc = x_ctrl.fence_i ? pc_plus4_x : jump_target;

    // ---- data memory: address and store (the write lands at the end of EX) ----
    logic store_ok;
    assign store_ok  = x_valid && x_ctrl.mem_write && !exc_x && !trap_m && !mret_m && !rst;
    assign dmem_re   = x_valid && x_ctrl.mem_read  && !exc_x && !trap_m && !mret_m && !rst;
    assign dmem_addr = alu_y;

    always_comb begin
        dmem_wdata = rs2_x;
        dmem_wstrb = 4'b0000;
        if (store_ok) begin
            case (x_funct3[1:0])
                2'b00: begin                                   // SB
                    dmem_wdata = {4{rs2_x[7:0]}};
                    dmem_wstrb = 4'b0001 << byte_off_x;
                end
                2'b01: begin                                   // SH
                    dmem_wdata = {2{rs2_x[15:0]}};
                    dmem_wstrb = byte_off_x[1] ? 4'b1100 : 4'b0011;
                end
                default: begin                                 // SW
                    dmem_wdata = rs2_x;
                    dmem_wstrb = 4'b1111;
                end
            endcase
        end
    end

    // ---- EX/MEM ----
    logic [31:0] m_pc, m_instr, m_csr_wsrc, m_tval;
    logic [2:0]  m_funct3;
    logic [4:0]  m_rs1;
    logic        m_mis_fetch, m_mis_load, m_mis_store;
    logic [31:0] m_st_addr, m_st_data;
    logic [3:0]  m_st_strb;
    logic        flush_m;

    assign flush_m = trap_m || mret_m;

    always_ff @(posedge clk) begin
        if (rst || flush_m) begin
            m_valid     <= 1'b0;
            m_ctrl      <= '0;
            m_pc        <= 32'd0;
            m_instr     <= NOP;
            m_rd        <= 5'd0;
            m_rs1       <= 5'd0;
            m_funct3    <= 3'd0;
            m_result    <= 32'd0;
            m_csr_wsrc  <= 32'd0;
            m_tval      <= 32'd0;
            m_mis_fetch <= 1'b0;
            m_mis_load  <= 1'b0;
            m_mis_store <= 1'b0;
            m_st_addr   <= 32'd0;
            m_st_strb   <= 4'd0;
            m_st_data   <= 32'd0;
        end else if (!hold) begin
            m_valid     <= x_valid;
            m_ctrl      <= x_ctrl;
            m_pc        <= x_pc;
            m_instr     <= x_instr;
            m_rd        <= x_rd;
            m_rs1       <= x_rs1;
            m_funct3    <= x_funct3;
            m_result    <= (x_ctrl.wb_sel == WB_PC4) ? pc_plus4_x : alu_y;
            m_csr_wsrc  <= x_funct3[2] ? {27'd0, x_rs1} : rs1_x;  // CSRR*I: zero-extended uimm
            m_tval      <= mis_fetch_x ? jump_target : alu_y;
            m_mis_fetch <= mis_fetch_x;
            m_mis_load  <= mis_load_x;
            m_mis_store <= mis_store_x;
            m_st_addr   <= {alu_y[31:2], 2'b00};
            m_st_strb   <= dmem_wstrb;
            m_st_data   <= dmem_wdata;
        end
    end

    // ==================================================================
    // MEM: load data, CSRs, traps
    // ==================================================================
    logic [31:0] csr_rdata, trap_cause, trap_val;
    logic        csr_illegal;

    csr u_csr (
        .clk         (clk),
        .rst         (rst),
        .csr_en      (m_ctrl.csr_en),
        .funct3      (m_funct3),
        .addr        (m_instr[31:20]),
        .wsrc        (m_csr_wsrc),
        .rs1_field   (m_rs1),
        .rdata       (csr_rdata),
        .csr_illegal (csr_illegal),
        .trap        (trap_m),
        .trap_cause  (trap_cause),
        .trap_pc     (m_pc),
        .trap_val    (trap_val),
        .mret        (mret_m),
        .mtvec_base  (mtvec_base),
        .mepc_q      (mepc_q)
    );

    // Same causes and priority as the single-cycle core.
    always_comb begin
        trap_m     = 1'b1;
        trap_val   = 32'd0;
        trap_cause = 32'd0;
        if (m_ctrl.illegal || csr_illegal) begin
            trap_cause = CAUSE_ILLEGAL_INSTR;
            trap_val   = m_instr;
        end else if (m_ctrl.ecall) begin
            trap_cause = CAUSE_MACHINE_ECALL;
        end else if (m_ctrl.ebreak) begin
            trap_cause = CAUSE_BREAKPOINT;
            trap_val   = m_pc;
        end else if (m_mis_fetch) begin
            trap_cause = CAUSE_MISALIGNED_FETCH;
            trap_val   = m_tval;
        end else if (m_mis_load) begin
            trap_cause = CAUSE_MISALIGNED_LOAD;
            trap_val   = m_tval;
        end else if (m_mis_store) begin
            trap_cause = CAUSE_MISALIGNED_STORE;
            trap_val   = m_tval;
        end else begin
            trap_m = 1'b0;
        end
        if (rst || !m_valid) trap_m = 1'b0;
    end

    assign mret_m = m_valid && m_ctrl.mret && !rst;

    // Load data: the word the memory registered at the end of EX, shifted to
    // the right byte lane and sign- or zero-extended.
    logic [31:0] load_shifted, load_data, wdata_m;

    assign load_shifted = dmem_rdata >> {m_result[1:0], 3'b000};

    always_comb begin
        case (m_funct3)
            3'b000:  load_data = {{24{load_shifted[7]}},  load_shifted[7:0]};   // LB
            3'b001:  load_data = {{16{load_shifted[15]}}, load_shifted[15:0]};  // LH
            3'b100:  load_data = {24'd0, load_shifted[7:0]};                    // LBU
            3'b101:  load_data = {16'd0, load_shifted[15:0]};                   // LHU
            default: load_data = dmem_rdata;                                    // LW
        endcase
    end

    always_comb begin
        case (m_ctrl.wb_sel)
            WB_MEM:  wdata_m = load_data;
            WB_CSR:  wdata_m = csr_rdata;
            default: wdata_m = m_result;       // ALU result or pc + 4
        endcase
    end

    // ---- MEM/WB ----
    // A trapping instruction still goes to WB (as a trace record) but with
    // its register write cancelled.
    always_ff @(posedge clk) begin
        if (rst) begin
            w_valid   <= 1'b0;
            w_pc      <= 32'd0;
            w_instr   <= NOP;
            w_rd_we   <= 1'b0;
            w_rd      <= 5'd0;
            w_wdata   <= 32'd0;
            w_trap    <= 1'b0;
            w_cause   <= 32'd0;
            w_illegal <= 1'b0;
            w_st_addr <= 32'd0;
            w_st_strb <= 4'd0;
            w_st_data <= 32'd0;
        end else if (hold) begin
            // MEM is still waiting: WB gets a bubble.
            w_valid   <= 1'b0;
            w_rd_we   <= 1'b0;
            w_trap    <= 1'b0;
        end else begin
            w_valid   <= m_valid;
            w_pc      <= m_pc;
            w_instr   <= m_instr;
            w_rd_we   <= m_ctrl.reg_write && !trap_m;
            w_rd      <= m_rd;
            w_wdata   <= wdata_m;
            w_trap    <= trap_m;
            w_cause   <= trap_cause;
            w_illegal <= m_ctrl.illegal || csr_illegal;
            w_st_addr <= m_st_addr;
            w_st_strb <= m_st_strb;
            w_st_data <= m_st_data;
        end
    end

    // ==================================================================
    // WB: the register-file write happens in u_rf (rd_we = w_rf_we)
    // ==================================================================

    // ---- trace ----
    assign trace_valid     = w_valid;
    assign trace_pc        = w_pc;
    assign trace_instr     = w_instr;
    assign trace_rd_we     = w_rf_we && w_rd != 5'd0;
    assign trace_rd        = w_rd;
    assign trace_rd_wdata  = w_wdata;
    assign trace_illegal   = w_illegal;
    assign trace_trap      = w_trap;
    assign trace_cause     = w_cause;
    assign trace_mem_addr  = w_st_addr;
    assign trace_mem_wstrb = w_st_strb;
    assign trace_mem_wdata = w_st_data;

`ifdef FORMAL
    // ==================================================================
    // Formal properties (SymbiYosys: formal/core.sby)
    // ==================================================================
    // Proven by k-induction with imem_rdata and dmem_rdata left completely
    // free, so they hold in every reachable state for every possible
    // instruction stream and every possible load value, not only for the
    // programs the simulations happen to run.

    logic f_past_valid = 1'b0;
    always_ff @(posedge clk) f_past_valid <= 1'b1;

    // Start in reset.
    always_comb if (!f_past_valid) assume (rst);

    // Memory wait protocol: a device only makes MEM wait on a real access it
    // accepted, and a faulting access never reaches a device. Beyond that,
    // dmem_wait is free, except that a device finishes within 4 wait cycles
    // (without some bound, k-induction can start from an unreachable state
    // and stay frozen in it past its depth). Simulation covers longer waits.
    always_comb
        if (dmem_wait) assume (m_valid && (m_ctrl.mem_read || m_ctrl.mem_write) && !trap_m);

    logic [2:0] f_wait_run = 3'd0;
    always_ff @(posedge clk) f_wait_run <= dmem_wait ? f_wait_run + 3'd1 : 3'd0;
    always_comb assume (f_wait_run <= 3'd4);

    // Whether the instruction in ID/EX really reads rs1/rs2 (a shadow copy of
    // the decoder's uses_rs1/uses_rs2, kept the same way as x_ctrl).
    logic f_x_uses1, f_x_uses2;
    always_ff @(posedge clk) begin
        if (rst) begin
            f_x_uses1 <= 1'b0;
            f_x_uses2 <= 1'b0;
        end else if (hold) begin
            // frozen along with ID/EX
        end else if (flush_x) begin
            f_x_uses1 <= 1'b0;
            f_x_uses2 <= 1'b0;
        end else begin
            f_x_uses1 <= d_valid && uses_rs1_d;
            f_x_uses2 <= d_valid && uses_rs2_d;
        end
    end

    always_comb begin
        if (f_past_valid && !rst) begin
            // Bubbles really are bubbles: zero controls, so they can't act.
            if (!x_valid) assert (x_ctrl == '0);
            if (!m_valid) assert (m_ctrl == '0);

            // x0 is never forwarded, and reads as zero after forwarding.
            if (x_rs1 == 5'd0) assert (fwd_a == FWD_NONE && rs1_x == 32'd0);
            if (x_rs2 == 5'd0) assert (fwd_b == FWD_NONE && rs2_x == 32'd0);

            // Never forward a value that doesn't exist yet: a load's data or a
            // CSR's old value is produced in MEM, so EX/MEM can't supply it.
            // The load-use stall is what guarantees this.
            if (x_valid && f_x_uses1 && fwd_a == FWD_EXMEM)
                assert (!m_ctrl.mem_read && !m_ctrl.csr_en);
            if (x_valid && f_x_uses2 && fwd_b == FWD_EXMEM)
                assert (!m_ctrl.mem_read && !m_ctrl.csr_en);

            // Memory writes only come from a real, non-faulting store in EX,
            // and never while an older instruction traps or returns in MEM.
            if (dmem_wstrb != 4'b0000) begin
                assert (x_valid && x_ctrl.mem_write && !exc_x);
                assert (!trap_m && !mret_m);
            end
            assert (dmem_wstrb inside {4'b0000, 4'b0001, 4'b0010, 4'b0100, 4'b1000,
                                       4'b0011, 4'b1100, 4'b1111});

            // A trap and an MRET can't be in MEM at once (one instruction).
            assert (!(trap_m && mret_m));

            // The PC is always word-aligned (a misaligned target traps
            // instead of redirecting), and so is every PC stored anywhere.
            // (The stored ones are what lets k-induction prove the first.)
            assert (pc_f[1:0] == 2'b00);
            assert (d_pc[1:0] == 2'b00 && x_pc[1:0] == 2'b00 && m_pc[1:0] == 2'b00);
            assert (mtvec_base[1:0] == 2'b00 && mepc_q[1:0] == 2'b00);
        end
    end

    always_ff @(posedge clk) begin
        if (f_past_valid && !$past(rst)) begin
            // The load-use stall lasts exactly one (unfrozen) cycle.
            assert (!($past(stall) && !$past(hold) && stall));

            // A flush kills the younger instructions (once the pipe moves).
            if ($past(flush_d) && !$past(hold)) assert (!d_valid);
            if ($past(redirect_x || trap_m || mret_m) && !$past(hold)) assert (!x_valid);
            if ($past(trap_m || mret_m)) assert (!m_valid);

            // A stalled instruction stays put in ID.
            if ($past(stall) && !$past(flush_d))
                assert (d_valid == $past(d_valid) && d_instr == $past(d_instr) && d_pc == $past(d_pc));

            // A memory wait freezes IF..MEM exactly and puts a bubble in WB:
            // nothing is lost, duplicated, or retired twice.
            if ($past(hold)) begin
                assert (pc_f == $past(pc_f));
                assert (d_valid == $past(d_valid) && d_instr == $past(d_instr) && d_pc == $past(d_pc));
                assert (x_valid == $past(x_valid) && x_ctrl == $past(x_ctrl) && x_pc == $past(x_pc)
                        && x_instr == $past(x_instr) && x_rd == $past(x_rd));
                assert (m_valid == $past(m_valid) && m_ctrl == $past(m_ctrl) && m_pc == $past(m_pc)
                        && m_result == $past(m_result) && m_st_strb == $past(m_st_strb));
                assert (!w_valid && !w_rf_we);
                // The operands EX will use are the ones it had when it froze.
                if ($past(f_x_uses1)) assert (rs1_x == $past(rs1_x));
                if ($past(f_x_uses2)) assert (rs2_x == $past(rs2_x));
            end

            // A trapping instruction doesn't write its register...
            if ($past(trap_m)) assert (!w_rd_we);
            // ...and the next fetch is the trap vector (or the MRET target).
            if ($past(trap_m)) assert (pc_f == $past(mtvec_base));
            if ($past(mret_m)) assert (pc_f == $past(mepc_q));
        end
        if (f_past_valid && $past(rst)) begin
            // Reset empties the pipeline.
            assert (!d_valid && !x_valid && !m_valid && !w_valid);
            assert (pc_f == RESET_PC);
        end
    end

    // Reachability: make sure the interesting cases can actually happen
    // (a property about a case that can't happen proves nothing).
    always_comb begin
        if (f_past_valid && !rst) begin
            cover (stall);
            cover (fwd_a == FWD_EXMEM && fwd_b == FWD_MEMWB);
            cover (trap_m && redirect_x);
            cover (mret_m);
            cover (x_valid && x_ctrl.mem_write && trap_m);    // a store killed by an older trap
            cover (w_valid && w_trap);
            cover (hold && x_valid && f_x_uses1 && fwd_a == FWD_MEMWB);  // forwarded operand latched
            cover (hold && redirect_x);                                  // a branch waits behind MEM
        end
    end
`endif

endmodule
