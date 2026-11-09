// fp32_add.sv — FP32 + FP32, IEEE round-to-nearest-even, subnormals flushed
//
// Purely combinational, the textbook guard/round/sticky adder:
//   1. order the operands so |big| >= |small|
//   2. shift small right by the exponent difference into 3 extra bits
//      (guard, round, and a sticky bit that ORs everything shifted past)
//   3. add or subtract the significands
//   4. normalize: one step right after a carry, or left by the leading-zero
//      count after a cancellation
//   5. round to nearest, ties to even, and fix up a rounding carry
// Subnormal inputs count as signed zeros (DAZ); results below the smallest
// normal become signed zeros (FTZ). x + (-x) = +0. sw/golden.py add() is the
// reference, and the unit test runs it against numpy float32.

module fp32_add (
    input  logic [31:0] a,
    input  logic [31:0] b,
    output logic [31:0] y
);

    function automatic logic [4:0] lzc27(input logic [26:0] x);
        lzc27 = 5'd27;
        for (int i = 0; i < 27; i++)
            if (x[i]) lzc27 = 5'(26 - i);
    endfunction

    logic        sa, sb;
    logic [7:0]  ea, eb;
    logic [22:0] fa, fb;
    assign {sa, ea, fa} = a;
    assign {sb, eb, fb} = b;

    logic a_inf, b_inf, a_nan, b_nan;
    assign a_inf = (ea == 8'hFF) && (fa == 23'd0);
    assign b_inf = (eb == 8'hFF) && (fb == 23'd0);
    assign a_nan = (ea == 8'hFF) && (fa != 23'd0);
    assign b_nan = (eb == 8'hFF) && (fb != 23'd0);

    // magnitude order; a subnormal (exponent 0) loses to any normal
    logic        swap;
    logic        s_big, s_sml;
    logic [7:0]  e_big, e_sml;
    logic [23:0] m_big, m_sml;
    assign swap  = {eb, fb} > {ea, fa};
    assign s_big = swap ? sb : sa;
    assign s_sml = swap ? sa : sb;
    assign e_big = swap ? eb : ea;
    assign e_sml = swap ? ea : eb;
    assign m_big = (e_big == 8'd0) ? 24'd0 : {1'b1, swap ? fb : fa};
    assign m_sml = (e_sml == 8'd0) ? 24'd0 : {1'b1, swap ? fa : fb};

    logic [7:0]        d;
    logic [4:0]        dd;
    logic [53:0]       wide;
    logic [26:0]       ext_big, ext_sml;
    logic [27:0]       sum;
    logic [26:0]       norm;
    logic [4:0]        lz;
    logic signed [9:0] e;
    logic              up;
    logic [24:0]       mant;
    logic              s;
    logic              exact_zero;

    always_comb begin
        // align the small operand: {significand, g, r, s}
        d       = e_big - e_sml;
        dd      = (d > 8'd27) ? 5'd27 : d[4:0];
        wide    = {m_sml, 3'b000, 27'd0} >> dd;
        ext_sml = {wide[53:28], wide[27] | (|wide[26:0])};
        ext_big = {m_big, 3'b000};

        exact_zero = 1'b0;
        lz         = 5'd0;
        sum        = 28'd0;
        if (s_big == s_sml) begin
            sum = {1'b0, ext_big} + {1'b0, ext_sml};
            if (sum[27]) begin
                norm = {sum[27:2], sum[1] | sum[0]};
                e    = $signed({2'b0, e_big}) + 10'sd1;
            end else begin
                norm = sum[26:0];
                e    = $signed({2'b0, e_big});
            end
        end else begin
            norm = ext_big - ext_sml;                        // >= 0 by the ordering
            exact_zero = (norm == 27'd0);
            lz   = lzc27(norm);
            norm = norm << lz;
            e    = $signed({2'b0, e_big}) - $signed({5'd0, lz});
        end

        // round to nearest even on the 24-bit significand
        up   = norm[2] & (norm[1] | norm[0] | norm[3]);
        mant = {1'b0, norm[26:3]} + {24'd0, up};
        if (mant[24]) begin
            mant = mant >> 1;
            e    = e + 10'sd1;
        end

        s = s_big;
        if (a_nan || b_nan || (a_inf && b_inf && (sa != sb)))
            y = 32'h7FC0_0000;
        else if (a_inf)
            y = {sa, 8'hFF, 23'd0};
        else if (b_inf)
            y = {sb, 8'hFF, 23'd0};
        else if (m_big == 24'd0)
            y = {sa & sb, 31'd0};                            // both (flushed) zeros
        else if (exact_zero)
            y = 32'd0;                                       // x + (-x) = +0
        else if (e >= 10'sd255)
            y = {s, 8'hFF, 23'd0};
        else if (e <= 10'sd0)
            y = {s, 31'd0};
        else
            y = {s, e[7:0], mant[22:0]};
    end

endmodule
