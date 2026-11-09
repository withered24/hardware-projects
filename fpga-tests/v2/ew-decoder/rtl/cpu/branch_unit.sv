// branch_unit.sv — decides whether a conditional branch is taken
//
// funct3 (instr[14:12]) selects the comparison. 3'b010 and 3'b011 are not
// assigned to any branch in RV32I, so they never take.

module branch_unit (
    input  logic [31:0] a,       // rs1
    input  logic [31:0] b,       // rs2
    input  logic [2:0]  funct3,
    output logic        taken
);

    always_comb begin
        case (funct3)
            3'b000:  taken = (a == b);                    // BEQ
            3'b001:  taken = (a != b);                    // BNE
            3'b100:  taken = ($signed(a) <  $signed(b));  // BLT
            3'b101:  taken = ($signed(a) >= $signed(b));  // BGE
            3'b110:  taken = (a <  b);                    // BLTU
            3'b111:  taken = (a >= b);                    // BGEU
            default: taken = 1'b0;
        endcase
    end

endmodule
