// alu.sv — the arithmetic/logic unit
//
// Purely combinational. Shift amounts use only the low 5 bits of b, as the
// ISA requires (shifting a 32-bit value by 32 or more is not a thing in RV32I).
// SLT/SRA are the two signed operations; everything else is sign-agnostic,
// which is the point of two's complement: ADD and SUB don't care.

import rv32i_pkg::*;

module alu (
    input  logic [31:0] a,
    input  logic [31:0] b,
    input  logic [3:0]  op,
    output logic [31:0] y
);

    always_comb begin
        case (op)
            ALU_ADD:  y = a + b;
            ALU_SUB:  y = a - b;
            ALU_SLL:  y = a << b[4:0];
            ALU_SLT:  y = {31'd0, $signed(a) < $signed(b)};
            ALU_SLTU: y = {31'd0, a < b};
            ALU_XOR:  y = a ^ b;
            ALU_SRL:  y = a >> b[4:0];
            ALU_SRA:  y = $signed(a) >>> b[4:0];
            ALU_OR:   y = a | b;
            ALU_AND:  y = a & b;
            default:  y = 32'd0;
        endcase
    end

endmodule
