// ew_pkg.sv — shared constants: custom-instruction ops and config fields
//
// The encodings are documented in docs/ARCH.md §3. All ops live in the
// custom-0 major opcode (0001011), R-type, funct7 = 0, op in funct3.

package ew_pkg;

    localparam logic [6:0] OP_CUSTOM0 = 7'b0001011;

    // funct3
    localparam logic [2:0] F3_EXP_BASE    = 3'd0;
    localparam logic [2:0] F3_WSTREAM_CFG = 3'd1;
    localparam logic [2:0] F3_GEMV_RUN    = 3'd2;
    localparam logic [2:0] F3_GEMV_WAIT   = 3'd3;
    localparam logic [2:0] F3_DEC_BYPASS  = 3'd4;
    localparam logic [2:0] F3_EW_STAT     = 3'd5;

    // WSTREAM.CFG field ids (rs1)
    localparam int CFG_CODE_ADDR  = 0;
    localparam int CFG_SMR_ADDR   = 1;
    localparam int CFG_ESC_ADDR   = 2;
    localparam int CFG_ESC_COUNT  = 3;
    localparam int CFG_TBASE_ADDR = 4;
    localparam int CFG_X_ADDR     = 5;
    localparam int CFG_Y_ADDR     = 6;
    localparam int N_CFG          = 7;

    // EW.STAT counter ids (rs1)
    localparam int ST_CYCLES     = 0;
    localparam int ST_CHUNKS     = 1;
    localparam int ST_STALL_ESC  = 2;
    localparam int ST_STALL_DATA = 3;
    localparam int ST_ESCAPES    = 4;

endpackage
