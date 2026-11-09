// bf16_mul.sv — BF16 x BF16 -> FP32, exact
//
// Purely combinational. Two 8-bit significands (hidden 1 + 7 bits) multiply
// to a 16-bit product, and FP32 has 24 bits of significand, so the product
// is always exact: no rounding logic at all. The only special cases are the
// range ends: subnormal inputs count as zero (DAZ), a product too small to
// be a normal FP32 becomes a signed zero (FTZ), a product too big becomes
// infinity. sw/golden.py mul() is the reference.

module bf16_mul (
    input  logic [15:0] a,
    input  logic [15:0] b,
    output logic [31:0] y
);

    logic       sa, sb, s;
    logic [7:0] ea, eb;
    logic [6:0] fa, fb;
    assign {sa, ea, fa} = a;
    assign {sb, eb, fb} = b;
    assign s = sa ^ sb;

    logic a_zero, b_zero, a_inf, b_inf, a_nan, b_nan;
    assign a_zero = (ea == 8'd0);
    assign b_zero = (eb == 8'd0);
    assign a_inf  = (ea == 8'hFF) && (fa == 7'd0);
    assign b_inf  = (eb == 8'hFF) && (fb == 7'd0);
    assign a_nan  = (ea == 8'hFF) && (fa != 7'd0);
    assign b_nan  = (eb == 8'hFF) && (fb != 7'd0);

    logic [15:0]       p;
    logic signed [9:0] e;
    logic [22:0]       frac;

    always_comb begin
        p = {1'b1, fa} * {1'b1, fb};                         // in [2^14, 2^16)
        e = $signed({2'b0, ea}) + $signed({2'b0, eb}) - 10'sd127;
        if (p[15]) begin
            frac = {p[14:0], 8'd0};
            e    = e + 10'sd1;
        end else begin
            frac = {p[13:0], 9'd0};
        end

        if (a_nan || b_nan || (a_inf && b_zero) || (b_inf && a_zero))
            y = 32'h7FC0_0000;
        else if (a_inf || b_inf)
            y = {s, 8'hFF, 23'd0};
        else if (a_zero || b_zero)
            y = {s, 31'd0};
        else if (e >= 10'sd255)
            y = {s, 8'hFF, 23'd0};                           // overflow
        else if (e <= 10'sd0)
            y = {s, 31'd0};                                  // flush to zero
        else
            y = {s, e[7:0], frac};
    end

endmodule
