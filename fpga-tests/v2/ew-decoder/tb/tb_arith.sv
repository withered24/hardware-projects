// tb_arith.sv — the two arithmetic blocks side by side, for test_arith.py
module tb_arith (
    input  logic [15:0] ma, mb,
    output logic [31:0] my,
    input  logic [31:0] aa, ab,
    output logic [31:0] ay
);
    bf16_mul u_mul (.a(ma), .b(mb), .y(my));
    fp32_add u_add (.a(aa), .b(ab), .y(ay));
endmodule
