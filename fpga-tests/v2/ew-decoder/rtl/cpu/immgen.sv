// immgen.sv — immediate generator
//
// RISC-V scatters immediate bits across the instruction so that the register
// fields (rs1, rs2, rd) always sit in the same place. This module gathers
// them back into a 32-bit, sign-extended value. The sign bit is ALWAYS
// instr[31], in every format; that's a deliberate ISA design choice, so sign
// extension can start before decode finishes.
//
// B and J immediates have an implied 0 in bit 0 (targets are 2-byte aligned),
// which is why they end in 1'b0 instead of using an instruction bit.

import rv32i_pkg::*;

module immgen (
    input  logic [31:0] instr,
    input  logic [2:0]  imm_sel,
    output logic [31:0] imm
);

    always_comb begin
        case (imm_sel)
            IMM_I:   imm = {{20{instr[31]}}, instr[31:20]};
            IMM_S:   imm = {{20{instr[31]}}, instr[31:25], instr[11:7]};
            IMM_B:   imm = {{19{instr[31]}}, instr[31], instr[7],
                            instr[30:25], instr[11:8], 1'b0};
            IMM_U:   imm = {instr[31:12], 12'b0};
            IMM_J:   imm = {{11{instr[31]}}, instr[31], instr[19:12],
                            instr[20], instr[30:21], 1'b0};
            default: imm = 32'd0;
        endcase
    end

endmodule
