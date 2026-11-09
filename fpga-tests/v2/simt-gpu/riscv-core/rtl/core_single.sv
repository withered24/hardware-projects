// core_single.sv — single-cycle RV32I core (machine mode only)
//
// Milestones 1-2. Kept as a reference next to the pipelined core (core.sv):
// the regression runs on both, and comparing them is the cleanest way to see
// what pipelining bought (clock rate) and cost (CPI, hazard logic).
//
// Every instruction is fetched, decoded, executed, and written back in one
// clock cycle. The state is the program counter, the register file, and the
// CSRs (memory lives outside the core). Read this file top to bottom: the
// sections follow the instruction's path through the datapath.
//
// Memory interface: both ports are combinational reads with a synchronous
// write, which is what a single-cycle core needs. (Real FPGA block RAM has a
// registered read; the pipelined core is built for that.)
//
// Traps: an instruction that traps (illegal, ECALL, EBREAK, misaligned
// access or jump target) has NO architectural effect: no register write, no
// store. Instead mepc/mcause/mtval are recorded and the next pc is mtvec.
// Misaligned loads/stores trap rather than being split into two accesses;
// the spec allows either.
//
// The trace_* outputs publish what the instruction in flight is doing. The
// testbench uses them to compare against the golden model each cycle; they
// will grow into an RVFI port for riscv-formal (Lesson 04 M7).

import rv32i_pkg::*;

module core_single #(
    parameter logic [31:0] RESET_PC = 32'h0000_0000
) (
    input  logic        clk,
    input  logic        rst,           // synchronous, active high

    // instruction memory
    output logic [31:0] imem_addr,
    input  logic [31:0] imem_rdata,

    // data memory
    output logic [31:0] dmem_addr,
    output logic [31:0] dmem_wdata,
    output logic [3:0]  dmem_wstrb,    // byte enables; nonzero = write this cycle
    input  logic [31:0] dmem_rdata,

    // per-instruction trace for the testbench
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

    // ------------------------------------------------------------------
    // Fetch
    // ------------------------------------------------------------------
    logic [31:0] pc, pc_next, pc_plus4;
    logic [31:0] instr;

    always_ff @(posedge clk) begin
        if (rst) pc <= RESET_PC;
        else     pc <= pc_next;
    end

    assign imem_addr = pc;
    assign instr     = imem_rdata;
    assign pc_plus4  = pc + 32'd4;

    // ------------------------------------------------------------------
    // Decode
    // ------------------------------------------------------------------
    logic [4:0] rs1, rs2, rd;
    logic [2:0] funct3;

    assign rs1    = instr[19:15];
    assign rs2    = instr[24:20];
    assign rd     = instr[11:7];
    assign funct3 = instr[14:12];

    logic       reg_write, b_imm, mem_read, mem_write;
    logic       branch, jal, jalr, illegal;
    logic       csr_en, ecall, ebreak, mret;
    logic       fence_i, uses_rs1, uses_rs2;     // only the pipeline needs these
    logic [1:0] a_sel, wb_sel;
    logic [3:0] alu_op;
    logic [2:0] imm_sel;

    decoder u_decoder (
        .instr     (instr),
        .reg_write (reg_write),
        .a_sel     (a_sel),
        .b_imm     (b_imm),
        .alu_op    (alu_op),
        .imm_sel   (imm_sel),
        .mem_read  (mem_read),
        .mem_write (mem_write),
        .wb_sel    (wb_sel),
        .branch    (branch),
        .jal       (jal),
        .jalr      (jalr),
        .csr_en    (csr_en),
        .ecall     (ecall),
        .ebreak    (ebreak),
        .mret      (mret),
        .fence_i   (fence_i),
        .uses_rs1  (uses_rs1),
        .uses_rs2  (uses_rs2),
        .illegal   (illegal)
    );

    logic [31:0] imm;
    immgen u_immgen (.instr(instr), .imm_sel(imm_sel), .imm(imm));

    logic [31:0] rs1_data, rs2_data, rd_data;
    logic        rd_we;

    regfile u_rf (
        .clk      (clk),
        .rs1_addr (rs1),
        .rs1_data (rs1_data),
        .rs2_addr (rs2),
        .rs2_data (rs2_data),
        .rd_we    (rd_we),
        .rd_addr  (rd),
        .rd_data  (rd_data)
    );

    // ------------------------------------------------------------------
    // Execute
    // ------------------------------------------------------------------
    logic [31:0] alu_a, alu_b, alu_y;

    always_comb begin
        case (a_sel)
            A_PC:    alu_a = pc;
            A_ZERO:  alu_a = 32'd0;
            default: alu_a = rs1_data;
        endcase
    end

    assign alu_b = b_imm ? imm : rs2_data;

    alu u_alu (.a(alu_a), .b(alu_b), .op(alu_op), .y(alu_y));

    logic br_taken;
    branch_unit u_br (.a(rs1_data), .b(rs2_data), .funct3(funct3), .taken(br_taken));

    // Jump/branch target. Branch/JAL targets use their own adder (pc + imm)
    // so the ALU stays free; JALR's target is rs1 + imm, which the ALU
    // already computed, with bit 0 cleared.
    logic [31:0] pc_target, jump_target;
    logic        jumping;

    assign pc_target   = pc + imm;
    assign jump_target = jalr ? {alu_y[31:1], 1'b0} : pc_target;
    assign jumping     = jal || jalr || (branch && br_taken);

    // ------------------------------------------------------------------
    // CSRs
    // ------------------------------------------------------------------
    logic [31:0] csr_rdata, mtvec_base, mepc_q;
    logic        csr_illegal;
    logic        trap;
    logic [31:0] trap_cause, trap_val;

    csr u_csr (
        .clk         (clk),
        .rst         (rst),
        .csr_en      (csr_en),
        .funct3      (funct3),
        .addr        (instr[31:20]),
        .wsrc        (funct3[2] ? {27'd0, rs1} : rs1_data),   // immediate forms use the rs1 field
        .rs1_field   (rs1),
        .rdata       (csr_rdata),
        .csr_illegal (csr_illegal),
        .trap        (trap),
        .trap_cause  (trap_cause),
        .trap_pc     (pc),
        .trap_val    (trap_val),
        .mret        (mret && !rst),
        .mtvec_base  (mtvec_base),
        .mepc_q      (mepc_q)
    );

    // ------------------------------------------------------------------
    // Memory (load/store unit)
    // ------------------------------------------------------------------
    // Memory is word-organized: the address picks a 32-bit word, the low two
    // bits pick a byte lane inside it. Stores replicate the data into every
    // lane and let the byte enables choose; loads shift the right lane down
    // and sign- or zero-extend.
    logic [1:0]  byte_off;
    logic [31:0] load_shifted, load_data;
    logic        misaligned;

    assign dmem_addr = alu_y;
    assign byte_off  = alu_y[1:0];

    // Halfwords need bit 0 clear; words need bits 1:0 clear. funct3[1:0] is
    // the access size for loads and stores alike.
    assign misaligned = (funct3[1:0] == 2'b01 && byte_off[0]) ||
                        (funct3[1:0] == 2'b10 && byte_off != 2'b00);

    // Stores are suppressed during reset, like register writes: while rst is
    // held, pc still points wherever the last program left it, so the
    // "instruction" being decoded is garbage (see docs/bugs.md #2). They're
    // also suppressed when the instruction traps.
    always_comb begin
        dmem_wdata = rs2_data;
        dmem_wstrb = 4'b0000;
        if (mem_write && !rst && !trap) begin
            case (funct3[1:0])
                2'b00: begin                                   // SB
                    dmem_wdata = {4{rs2_data[7:0]}};
                    dmem_wstrb = 4'b0001 << byte_off;
                end
                2'b01: begin                                   // SH
                    dmem_wdata = {2{rs2_data[15:0]}};
                    dmem_wstrb = byte_off[1] ? 4'b1100 : 4'b0011;
                end
                default: begin                                 // SW
                    dmem_wdata = rs2_data;
                    dmem_wstrb = 4'b1111;
                end
            endcase
        end
    end

    assign load_shifted = dmem_rdata >> {byte_off, 3'b000};

    always_comb begin
        case (funct3)
            3'b000:  load_data = {{24{load_shifted[7]}},  load_shifted[7:0]};   // LB
            3'b001:  load_data = {{16{load_shifted[15]}}, load_shifted[15:0]};  // LH
            3'b100:  load_data = {24'd0, load_shifted[7:0]};                    // LBU
            3'b101:  load_data = {16'd0, load_shifted[15:0]};                   // LHU
            default: load_data = dmem_rdata;                                    // LW
        endcase
    end

    // ------------------------------------------------------------------
    // Traps
    // ------------------------------------------------------------------
    // At most one cause per instruction; the order below is the priority.
    // A jump to an address that isn't 4-byte aligned traps ON the jump
    // (mepc = the jump, mtval = the bad target), since this core has no
    // compressed instructions.
    always_comb begin
        trap       = 1'b1;
        trap_val   = 32'd0;
        trap_cause = 32'd0;
        if (illegal || csr_illegal) begin
            trap_cause = CAUSE_ILLEGAL_INSTR;
            trap_val   = instr;
        end else if (ecall) begin
            trap_cause = CAUSE_MACHINE_ECALL;
        end else if (ebreak) begin
            trap_cause = CAUSE_BREAKPOINT;
            trap_val   = pc;
        end else if (jumping && jump_target[1]) begin
            trap_cause = CAUSE_MISALIGNED_FETCH;
            trap_val   = jump_target;
        end else if (mem_read && misaligned) begin
            trap_cause = CAUSE_MISALIGNED_LOAD;
            trap_val   = alu_y;
        end else if (mem_write && misaligned) begin
            trap_cause = CAUSE_MISALIGNED_STORE;
            trap_val   = alu_y;
        end else begin
            trap = 1'b0;
        end
        if (rst) trap = 1'b0;
    end

    // ------------------------------------------------------------------
    // Next PC
    // ------------------------------------------------------------------
    always_comb begin
        if (trap)          pc_next = mtvec_base;
        else if (mret)     pc_next = mepc_q;
        else if (jumping)  pc_next = jump_target;
        else               pc_next = pc_plus4;
    end

    // ------------------------------------------------------------------
    // Write-back
    // ------------------------------------------------------------------
    always_comb begin
        case (wb_sel)
            WB_MEM:  rd_data = load_data;
            WB_PC4:  rd_data = pc_plus4;
            WB_CSR:  rd_data = csr_rdata;
            default: rd_data = alu_y;
        endcase
    end

    assign rd_we = reg_write && !rst && !trap;

    // ------------------------------------------------------------------
    // Trace
    // ------------------------------------------------------------------
    assign trace_valid    = !rst;
    assign trace_pc       = pc;
    assign trace_instr    = instr;
    assign trace_rd_we    = rd_we && (rd != 5'd0);
    assign trace_rd       = rd;
    assign trace_rd_wdata = rd_data;
    assign trace_illegal  = illegal || csr_illegal;
    assign trace_trap     = trap;
    assign trace_cause    = trap_cause;
    assign trace_mem_addr  = dmem_addr;
    assign trace_mem_wstrb = dmem_wstrb;
    assign trace_mem_wdata = dmem_wdata;

endmodule
