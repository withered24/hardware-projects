// ew_lane.sv — one exponent-window decoder lane
//
// Purely combinational: a 3-bit (CODE_BITS-bit) code plus the raw
// sign/mantissa byte becomes the original BF16 bit pattern.
//   code != 0 : exponent = base + code, an 8-bit add that wraps on purpose
//               (a window starting at exponent 0 has base 255)
//   code == 0 : escape, the exponent is esc_byte (picked for this lane by
//               weight_decoder's prefix popcount)

module ew_lane #(
    parameter int CODE_BITS = 3
) (
    input  logic [CODE_BITS-1:0] code,
    input  logic [7:0]           sm,        // sign << 7 | mantissa
    input  logic [7:0]           base,
    input  logic [7:0]           esc_byte,
    output logic [15:0]          w,
    output logic                 is_esc
);

    logic [7:0] exp;
    assign is_esc = (code == '0);
    assign exp    = is_esc ? esc_byte : base + 8'(code);
    assign w      = {sm[7], exp, sm[6:0]};

endmodule
