// ew_top.sv — the exponent-window GEMV engine
//
// One command port (the CPU's custom instructions, see ew_ctrl.sv) and six
// AXI4 masters, one per stream:
//
//   m_code  64-bit read   code bit-planes
//   m_smr   16L-bit read  sign/mantissa bytes, or raw BF16 in bypass mode
//   m_esc   64-bit read   escape exponent bytes
//   m_tb    64-bit read   per-tile base bytes (PER_TILE_BASE builds)
//   m_x     64-bit read   x vector (BF16)
//   m_y     64-bit write  y vector (FP32)
//
// On the FPGA these go through an AXI interconnect to the DDR controller.
// The smr port is 2L bytes wide so the raw baseline can take a full chunk per
// cycle too; DRAM bandwidth, not port width, is what limits both modes.

import ew_pkg::*;

module ew_top #(
    parameter int AW            = 32,
    parameter int L             = 8,
    parameter int CODE_BITS     = 3,
    parameter int PER_TILE_BASE = 0,
    parameter int KMAX          = 4864,
    parameter int ESC_FIFO      = 64,     // escape FIFO depth, 64-bit words
    parameter int MAX_OUT       = 4,
    parameter int SRAM          = 0       // big buffers in SRAM: code/smr/esc FIFOs, x (ASIC build)
) (
    input  logic          clk,
    input  logic          rst_n,

    input  logic          cmd_valid,
    output logic          cmd_ready,
    input  logic [2:0]    cmd_funct3,
    input  logic [31:0]   cmd_rs1,
    input  logic [31:0]   cmd_rs2,
    output logic [31:0]   cmd_rdata,

    // code
    output logic [AW-1:0] m_code_araddr,  output logic [7:0] m_code_arlen,  output logic [2:0] m_code_arsize,
    output logic [1:0]    m_code_arburst, output logic m_code_arvalid,      input  logic m_code_arready,
    input  logic [63:0]   m_code_rdata,   input  logic [1:0] m_code_rresp,  input  logic m_code_rlast,
    input  logic          m_code_rvalid,  output logic m_code_rready,
    // smr
    output logic [AW-1:0] m_smr_araddr,   output logic [7:0] m_smr_arlen,   output logic [2:0] m_smr_arsize,
    output logic [1:0]    m_smr_arburst,  output logic m_smr_arvalid,       input  logic m_smr_arready,
    input  logic [16*L-1:0] m_smr_rdata,  input  logic [1:0] m_smr_rresp,   input  logic m_smr_rlast,
    input  logic          m_smr_rvalid,   output logic m_smr_rready,
    // esc
    output logic [AW-1:0] m_esc_araddr,   output logic [7:0] m_esc_arlen,   output logic [2:0] m_esc_arsize,
    output logic [1:0]    m_esc_arburst,  output logic m_esc_arvalid,       input  logic m_esc_arready,
    input  logic [63:0]   m_esc_rdata,    input  logic [1:0] m_esc_rresp,   input  logic m_esc_rlast,
    input  logic          m_esc_rvalid,   output logic m_esc_rready,
    // tile bases
    output logic [AW-1:0] m_tb_araddr,    output logic [7:0] m_tb_arlen,    output logic [2:0] m_tb_arsize,
    output logic [1:0]    m_tb_arburst,   output logic m_tb_arvalid,        input  logic m_tb_arready,
    input  logic [63:0]   m_tb_rdata,     input  logic [1:0] m_tb_rresp,    input  logic m_tb_rlast,
    input  logic          m_tb_rvalid,    output logic m_tb_rready,
    // x
    output logic [AW-1:0] m_x_araddr,     output logic [7:0] m_x_arlen,     output logic [2:0] m_x_arsize,
    output logic [1:0]    m_x_arburst,    output logic m_x_arvalid,         input  logic m_x_arready,
    input  logic [63:0]   m_x_rdata,      input  logic [1:0] m_x_rresp,     input  logic m_x_rlast,
    input  logic          m_x_rvalid,     output logic m_x_rready,
    // y
    output logic [AW-1:0] m_y_awaddr,     output logic [7:0] m_y_awlen,     output logic [2:0] m_y_awsize,
    output logic [1:0]    m_y_awburst,    output logic m_y_awvalid,         input  logic m_y_awready,
    output logic [63:0]   m_y_wdata,      output logic [7:0] m_y_wstrb,     output logic m_y_wlast,
    output logic          m_y_wvalid,     input  logic m_y_wready,
    input  logic [1:0]    m_y_bresp,      input  logic m_y_bvalid,          output logic m_y_bready
);

    localparam int SDW = 16 * L;

    logic          bypass, clear, start;
    logic [7:0]    base;
    logic [15:0]   rows, cols;
    logic          en_code, en_smr, en_esc, en_tb;
    logic [AW-1:0] a_code, a_smr, a_esc, a_tb, a_x, a_y;
    logic [31:0]   n_code, n_smr, n_esc, n_tb, n_x, n_y;
    logic          b_code, b_smr, b_esc, b_tb, b_x, b_y, b_gemv;
    logic          chunk_fire, stall_esc, stall_data;
    logic [$clog2(L+1)-1:0] chunk_nesc;

    ew_ctrl #(.AW(AW), .L(L), .CODE_BITS(CODE_BITS), .PER_TILE_BASE(PER_TILE_BASE)) u_ctrl (
        .clk, .rst_n,
        .cmd_valid, .cmd_ready, .cmd_funct3, .cmd_rs1, .cmd_rs2, .cmd_rdata,
        .bypass, .base, .clear, .start, .rows, .cols,
        .en_code, .en_smr, .en_esc, .en_tb,
        .a_code, .a_smr, .a_esc, .a_tb, .a_x, .a_y,
        .n_code, .n_smr, .n_esc, .n_tb, .n_x, .n_y,
        .dp_busy(b_code || b_smr || b_esc || b_tb || b_x || b_y || b_gemv),
        .chunk_fire, .chunk_nesc, .stall_esc, .stall_data
    );

    // ------------------------------------------------------------ readers
    logic           code_v, code_r, smr_v, smr_r, esc_v, esc_r, tb_v, tb_r, x_v, x_r;
    logic [63:0]    code_d, esc_d, tb_d, x_d;
    logic [SDW-1:0] smr_d;

    axi_rd_stream #(.DW(64), .AW(AW), .FIFO_DEPTH(64), .MAX_OUT(MAX_OUT), .SRAM(SRAM)) u_rd_code (
        .clk, .rst_n, .clear, .start(start && en_code), .addr(a_code), .nbeats(n_code), .busy(b_code),
        .out_valid(code_v), .out_ready(code_r), .out_data(code_d),
        .araddr(m_code_araddr), .arlen(m_code_arlen), .arsize(m_code_arsize), .arburst(m_code_arburst),
        .arvalid(m_code_arvalid), .arready(m_code_arready), .rdata(m_code_rdata), .rresp(m_code_rresp),
        .rlast(m_code_rlast), .rvalid(m_code_rvalid), .rready(m_code_rready));

    axi_rd_stream #(.DW(SDW), .AW(AW), .FIFO_DEPTH(64), .MAX_OUT(MAX_OUT), .SRAM(SRAM)) u_rd_smr (
        .clk, .rst_n, .clear, .start(start && en_smr), .addr(a_smr), .nbeats(n_smr), .busy(b_smr),
        .out_valid(smr_v), .out_ready(smr_r), .out_data(smr_d),
        .araddr(m_smr_araddr), .arlen(m_smr_arlen), .arsize(m_smr_arsize), .arburst(m_smr_arburst),
        .arvalid(m_smr_arvalid), .arready(m_smr_arready), .rdata(m_smr_rdata), .rresp(m_smr_rresp),
        .rlast(m_smr_rlast), .rvalid(m_smr_rvalid), .rready(m_smr_rready));

    axi_rd_stream #(.DW(64), .AW(AW), .FIFO_DEPTH(ESC_FIFO), .MAX_OUT(MAX_OUT), .SRAM(SRAM)) u_rd_esc (
        .clk, .rst_n, .clear, .start(start && en_esc), .addr(a_esc), .nbeats(n_esc), .busy(b_esc),
        .out_valid(esc_v), .out_ready(esc_r), .out_data(esc_d),
        .araddr(m_esc_araddr), .arlen(m_esc_arlen), .arsize(m_esc_arsize), .arburst(m_esc_arburst),
        .arvalid(m_esc_arvalid), .arready(m_esc_arready), .rdata(m_esc_rdata), .rresp(m_esc_rresp),
        .rlast(m_esc_rlast), .rvalid(m_esc_rvalid), .rready(m_esc_rready));

    axi_rd_stream #(.DW(64), .AW(AW), .FIFO_DEPTH(16), .MAX_OUT(2)) u_rd_tb (
        .clk, .rst_n, .clear, .start(start && en_tb), .addr(a_tb), .nbeats(n_tb), .busy(b_tb),
        .out_valid(tb_v), .out_ready(tb_r), .out_data(tb_d),
        .araddr(m_tb_araddr), .arlen(m_tb_arlen), .arsize(m_tb_arsize), .arburst(m_tb_arburst),
        .arvalid(m_tb_arvalid), .arready(m_tb_arready), .rdata(m_tb_rdata), .rresp(m_tb_rresp),
        .rlast(m_tb_rlast), .rvalid(m_tb_rvalid), .rready(m_tb_rready));

    axi_rd_stream #(.DW(64), .AW(AW), .FIFO_DEPTH(32), .MAX_OUT(MAX_OUT)) u_rd_x (
        .clk, .rst_n, .clear, .start(start), .addr(a_x), .nbeats(n_x), .busy(b_x),
        .out_valid(x_v), .out_ready(x_r), .out_data(x_d),
        .araddr(m_x_araddr), .arlen(m_x_arlen), .arsize(m_x_arsize), .arburst(m_x_arburst),
        .arvalid(m_x_arvalid), .arready(m_x_arready), .rdata(m_x_rdata), .rresp(m_x_rresp),
        .rlast(m_x_rlast), .rvalid(m_x_rvalid), .rready(m_x_rready));

    // --------------------------------------------------- decode + compute
    logic           w_v, w_r;
    logic [SDW-1:0] w_d;
    logic           y_v, y_r;
    logic [63:0]    y_d;

`ifdef DECODER_MACRO
    // the ASIC build places the decoder as its own hardened block
    // (openlane/decoder: L = 8, 3-bit codes, one base per matrix)
    weight_decoder u_dec (
`else
    weight_decoder #(.L(L), .CODE_BITS(CODE_BITS), .PER_TILE_BASE(PER_TILE_BASE)) u_dec (
`endif
        .clk, .rst_n, .clear, .bypass, .base,
        .code_valid(code_v), .code_ready(code_r), .code_data(code_d),
        .tb_valid(tb_v), .tb_ready(tb_r), .tb_data(tb_d),
        .smr_valid(smr_v), .smr_ready(smr_r), .smr_data(smr_d),
        .esc_valid(esc_v), .esc_ready(esc_r), .esc_data(esc_d),
        .out_valid(w_v), .out_ready(w_r), .out_w(w_d), .out_nesc(chunk_nesc),
        .stall_esc, .stall_data
    );

    gemv_core #(.L(L), .KMAX(KMAX), .SRAM(SRAM)) u_gemv (
        .clk, .rst_n, .clear, .start, .rows, .cols, .busy(b_gemv),
        .x_valid(x_v), .x_ready(x_r), .x_data(x_d),
        .w_valid(w_v), .w_ready(w_r), .w_data(w_d), .chunk_fire,
        .y_valid(y_v), .y_ready(y_r), .y_data(y_d)
    );

    // ------------------------------------------------------------ writer
    axi_wr_stream #(.DW(64), .AW(AW)) u_wr_y (
        .clk, .rst_n, .clear, .start, .addr(a_y), .nbeats(n_y), .busy(b_y),
        .in_valid(y_v), .in_ready(y_r), .in_data(y_d),
        .awaddr(m_y_awaddr), .awlen(m_y_awlen), .awsize(m_y_awsize), .awburst(m_y_awburst),
        .awvalid(m_y_awvalid), .awready(m_y_awready),
        .wdata(m_y_wdata), .wstrb(m_y_wstrb), .wlast(m_y_wlast), .wvalid(m_y_wvalid), .wready(m_y_wready),
        .bresp(m_y_bresp), .bvalid(m_y_bvalid), .bready(m_y_bready));

endmodule
