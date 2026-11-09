// rv32i_pkg.sv — shared constants for the RV32I core
//
// Opcodes come straight from the RISC-V ISA manual (Vol. I, "RV32I Base
// Instruction Set" opcode map). Everything else (ALU op codes, mux selects)
// is our own internal encoding: the values don't matter, only that every
// module agrees on them, which is why they live in one place.

package rv32i_pkg;

    // ---- major opcodes: instr[6:0] ----
    localparam logic [6:0] OP_LUI    = 7'b0110111;
    localparam logic [6:0] OP_AUIPC  = 7'b0010111;
    localparam logic [6:0] OP_JAL    = 7'b1101111;
    localparam logic [6:0] OP_JALR   = 7'b1100111;
    localparam logic [6:0] OP_BRANCH = 7'b1100011;
    localparam logic [6:0] OP_LOAD   = 7'b0000011;
    localparam logic [6:0] OP_STORE  = 7'b0100011;
    localparam logic [6:0] OP_IMM    = 7'b0010011;
    localparam logic [6:0] OP_REG    = 7'b0110011;
    localparam logic [6:0] OP_FENCE  = 7'b0001111;
    localparam logic [6:0] OP_SYSTEM = 7'b1110011;

    // ---- ALU operations ----
    localparam logic [3:0] ALU_ADD  = 4'd0;
    localparam logic [3:0] ALU_SUB  = 4'd1;
    localparam logic [3:0] ALU_SLL  = 4'd2;
    localparam logic [3:0] ALU_SLT  = 4'd3;
    localparam logic [3:0] ALU_SLTU = 4'd4;
    localparam logic [3:0] ALU_XOR  = 4'd5;
    localparam logic [3:0] ALU_SRL  = 4'd6;
    localparam logic [3:0] ALU_SRA  = 4'd7;
    localparam logic [3:0] ALU_OR   = 4'd8;
    localparam logic [3:0] ALU_AND  = 4'd9;

    // ---- immediate formats (which bits of the instruction form the imm) ----
    localparam logic [2:0] IMM_I = 3'd0;
    localparam logic [2:0] IMM_S = 3'd1;
    localparam logic [2:0] IMM_B = 3'd2;
    localparam logic [2:0] IMM_U = 3'd3;
    localparam logic [2:0] IMM_J = 3'd4;

    // ---- ALU operand A source ----
    localparam logic [1:0] A_RS1  = 2'd0;  // normal register operations
    localparam logic [1:0] A_PC   = 2'd1;  // AUIPC: pc + imm
    localparam logic [1:0] A_ZERO = 2'd2;  // LUI:   0  + imm

    // ---- register write-back source ----
    localparam logic [1:0] WB_ALU = 2'd0;  // arithmetic, LUI, AUIPC
    localparam logic [1:0] WB_MEM = 2'd1;  // loads
    localparam logic [1:0] WB_PC4 = 2'd2;  // JAL/JALR link address
    localparam logic [1:0] WB_CSR = 2'd3;  // CSR instructions: old CSR value

    // ---- exception causes (mcause values, privileged spec table 3.6) ----
    localparam logic [31:0] CAUSE_MISALIGNED_FETCH = 32'd0;
    localparam logic [31:0] CAUSE_ILLEGAL_INSTR    = 32'd2;
    localparam logic [31:0] CAUSE_BREAKPOINT       = 32'd3;
    localparam logic [31:0] CAUSE_MISALIGNED_LOAD  = 32'd4;
    localparam logic [31:0] CAUSE_MISALIGNED_STORE = 32'd6;
    localparam logic [31:0] CAUSE_MACHINE_ECALL    = 32'd11;

    // ---- pipeline (core.sv) ----
    // Forwarding-mux selects for an EX operand.
    localparam logic [1:0] FWD_NONE  = 2'd0;  // value read in ID (regfile + WB bypass)
    localparam logic [1:0] FWD_EXMEM = 2'd1;  // result of the instruction one ahead
    localparam logic [1:0] FWD_MEMWB = 2'd2;  // result of the instruction two ahead

    // Every control signal the decoder produces, bundled so a pipeline
    // register can carry them as one field and a bubble can zero them as one.
    typedef struct packed {
        logic       reg_write;
        logic [1:0] a_sel;
        logic       b_imm;
        logic [3:0] alu_op;
        logic       mem_read;
        logic       mem_write;
        logic [1:0] wb_sel;
        logic       branch;
        logic       jal;
        logic       jalr;
        logic       fence_i;
        logic       csr_en;
        logic       ecall;
        logic       ebreak;
        logic       mret;
        logic       illegal;
    } ctrl_t;

endpackage
