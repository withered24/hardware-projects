// decoder.sv — main control: turns an instruction into datapath controls
//
// This is the "control table" from reference/datapath.html written as a case
// statement on the opcode. Every output gets a safe default first (no
// register write, no memory access, no jump), so each opcode only needs to
// list the controls it turns ON. Forgetting a default in a combinational
// block is how you get a latch; this pattern makes that impossible.
//
// SYSTEM instructions: CSR accesses, plus ECALL/EBREAK/MRET, which the core
// turns into traps and returns. FENCE and WFI are no-ops (one memory, no
// interrupts). FENCE.I is a no-op on the single-cycle core; the pipeline uses
// `fence_i` to refetch everything after it. Anything unrecognized raises
// `illegal`, which the core turns into an illegal-instruction trap.
//
// uses_rs1/uses_rs2 say whether the instruction actually reads that
// register. The pipeline's hazard unit needs them so it doesn't stall on a
// "dependency" through a field that's really an immediate (LUI, JAL, ...).

import rv32i_pkg::*;

module decoder (
    input  logic [31:0] instr,

    output logic        reg_write,  // write rd at the end of this cycle
    output logic [1:0]  a_sel,      // ALU operand A: rs1 / pc / zero
    output logic        b_imm,      // ALU operand B: 1 = immediate, 0 = rs2
    output logic [3:0]  alu_op,
    output logic [2:0]  imm_sel,
    output logic        mem_read,   // load
    output logic        mem_write,  // store
    output logic [1:0]  wb_sel,     // what gets written to rd
    output logic        branch,     // conditional branch (taken decided by branch_unit)
    output logic        jal,
    output logic        jalr,
    output logic        csr_en,     // CSRRW/S/C (and the immediate forms)
    output logic        ecall,
    output logic        ebreak,
    output logic        mret,
    output logic        fence_i,
    output logic        uses_rs1,
    output logic        uses_rs2,
    output logic        illegal
);

    localparam logic [31:0] INSTR_ECALL  = 32'h0000_0073;
    localparam logic [31:0] INSTR_EBREAK = 32'h0010_0073;
    localparam logic [31:0] INSTR_MRET   = 32'h3020_0073;
    localparam logic [31:0] INSTR_WFI    = 32'h1050_0073;

    logic [6:0] opcode;
    logic [2:0] funct3;
    logic       sub_or_sra;   // instr[30]: picks SUB over ADD, SRA over SRL

    assign opcode = instr[6:0];
    assign funct3 = instr[14:12];

    // The ALU operation implied by funct3 (shared by OP and OP-IMM).
    logic [3:0] funct_alu_op;
    always_comb begin
        case (funct3)
            3'b000:  funct_alu_op = sub_or_sra ? ALU_SUB : ALU_ADD;
            3'b001:  funct_alu_op = ALU_SLL;
            3'b010:  funct_alu_op = ALU_SLT;
            3'b011:  funct_alu_op = ALU_SLTU;
            3'b100:  funct_alu_op = ALU_XOR;
            3'b101:  funct_alu_op = sub_or_sra ? ALU_SRA : ALU_SRL;
            3'b110:  funct_alu_op = ALU_OR;
            default: funct_alu_op = ALU_AND;   // 3'b111
        endcase
    end

    // For register-register ops, instr[30] always means SUB/SRA. For
    // immediate ops, bit 30 is part of the immediate, so it only counts for
    // the shift (SRAI); otherwise "addi x1, x1, -1024" would decode as a SUB.
    assign sub_or_sra = instr[30] && (opcode == OP_REG || funct3 == 3'b101);

    always_comb begin
        reg_write = 1'b0;
        a_sel     = A_RS1;
        b_imm     = 1'b0;
        alu_op    = ALU_ADD;
        imm_sel   = IMM_I;
        mem_read  = 1'b0;
        mem_write = 1'b0;
        wb_sel    = WB_ALU;
        branch    = 1'b0;
        jal       = 1'b0;
        jalr      = 1'b0;
        csr_en    = 1'b0;
        ecall     = 1'b0;
        ebreak    = 1'b0;
        mret      = 1'b0;
        fence_i   = 1'b0;
        uses_rs1  = 1'b0;
        uses_rs2  = 1'b0;
        illegal   = 1'b0;

        case (opcode)
            OP_LUI: begin              // rd = imm
                reg_write = 1'b1;
                a_sel     = A_ZERO;
                b_imm     = 1'b1;
                imm_sel   = IMM_U;
            end
            OP_AUIPC: begin            // rd = pc + imm
                reg_write = 1'b1;
                a_sel     = A_PC;
                b_imm     = 1'b1;
                imm_sel   = IMM_U;
            end
            OP_JAL: begin              // rd = pc + 4; pc = pc + imm
                reg_write = 1'b1;
                wb_sel    = WB_PC4;
                imm_sel   = IMM_J;
                jal       = 1'b1;
            end
            OP_JALR: begin             // rd = pc + 4; pc = (rs1 + imm) & ~1
                reg_write = 1'b1;
                wb_sel    = WB_PC4;
                b_imm     = 1'b1;
                imm_sel   = IMM_I;
                jalr      = 1'b1;
                uses_rs1  = 1'b1;
                if (funct3 != 3'b000) illegal = 1'b1;
            end
            OP_BRANCH: begin           // if (rs1 ? rs2) pc = pc + imm
                imm_sel   = IMM_B;
                branch    = 1'b1;
                uses_rs1  = 1'b1;
                uses_rs2  = 1'b1;
                if (funct3[2:1] == 2'b01) illegal = 1'b1;   // 010, 011 unassigned
            end
            OP_LOAD: begin             // rd = mem[rs1 + imm]
                reg_write = 1'b1;
                b_imm     = 1'b1;
                imm_sel   = IMM_I;
                mem_read  = 1'b1;
                wb_sel    = WB_MEM;
                uses_rs1  = 1'b1;
                if (funct3 == 3'b011 || funct3[2:1] == 2'b11) illegal = 1'b1;  // only LB/LH/LW/LBU/LHU
            end
            OP_STORE: begin            // mem[rs1 + imm] = rs2
                b_imm     = 1'b1;
                imm_sel   = IMM_S;
                mem_write = 1'b1;
                uses_rs1  = 1'b1;
                uses_rs2  = 1'b1;
                if (funct3[2] || funct3[1:0] == 2'b11) illegal = 1'b1;     // only SB/SH/SW
            end
            OP_IMM: begin              // rd = rs1 op imm
                reg_write = 1'b1;
                b_imm     = 1'b1;
                imm_sel   = IMM_I;
                alu_op    = funct_alu_op;
                uses_rs1  = 1'b1;
            end
            OP_REG: begin              // rd = rs1 op rs2
                reg_write = 1'b1;
                alu_op    = funct_alu_op;
                uses_rs1  = 1'b1;
                uses_rs2  = 1'b1;
            end
            OP_FENCE: begin            // FENCE (000) and FENCE.I (001)
                if (funct3[2:1] != 2'b00) illegal = 1'b1;
                else fence_i = funct3[0];
            end
            OP_SYSTEM: begin
                if (funct3 == 3'b000) begin
                    case (instr)
                        INSTR_ECALL:  ecall  = 1'b1;
                        INSTR_EBREAK: ebreak = 1'b1;
                        INSTR_MRET:   mret   = 1'b1;
                        INSTR_WFI:    ;                  // no interrupts: no-op
                        default:      illegal = 1'b1;
                    endcase
                end else if (funct3 == 3'b100) begin
                    illegal = 1'b1;                      // reserved
                end else begin                           // rd = csr; csr = f(csr, src)
                    csr_en    = 1'b1;
                    reg_write = 1'b1;
                    wb_sel    = WB_CSR;
                    uses_rs1  = !funct3[2];              // CSRR*I: rs1 field is an immediate
                end
            end
            default: illegal = 1'b1;
        endcase
    end

endmodule
