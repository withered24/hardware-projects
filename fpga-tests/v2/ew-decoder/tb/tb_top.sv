// tb_top.sv — ew_top + testbench DRAM, driven by test_top.py
//
// cocotb drives the command port exactly as the CPU's custom instructions
// would (test_cpu.py does it with the real core instead).

module tb_top #(
    parameter int L             = 8,
    parameter int CODE_BITS     = 3,
    parameter int PER_TILE_BASE = 0,
    parameter int BW_NUM        = 16,
    parameter int BW_DEN        = 1,
    parameter int LAT           = 20,
    parameter int ESC_FIFO      = 64,
    parameter int MEM_WORDS     = 1 << 18,    // testbench DRAM, 64-bit words (2 MB)
    parameter int SRAM          = 0
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        cmd_valid,
    output logic        cmd_ready,
    input  logic [2:0]  cmd_funct3,
    input  logic [31:0] cmd_rs1,
    input  logic [31:0] cmd_rs2,
    output logic [31:0] cmd_rdata,
    input  logic        load,
    input  logic        dump,
    input  logic [31:0] dump_lo,
    input  logic [31:0] dump_hi
);

    localparam int AW   = 32;
    localparam int SDW  = 16 * L;
    localparam int MAXW = (SDW > 64) ? SDW : 64;
    localparam int NP   = 5;                  // code, smr, esc, tb, x

    logic [NP*AW-1:0]   araddr;
    logic [NP*8-1:0]    arlen;
    logic [NP-1:0]      arvalid, arready, rlast, rvalid, rready;
    logic [NP*MAXW-1:0] rdata;
    logic [2:0]         arsize [NP];
    logic [1:0]         arburst [NP];

    logic [AW-1:0] awaddr;
    logic [7:0]    awlen, wstrb;
    logic [2:0]    awsize;
    logic [1:0]    awburst;
    logic          awvalid, awready, wlast, wvalid, wready, bvalid, bready;
    logic [63:0]   wdata;

    ew_top #(.AW(AW), .L(L), .CODE_BITS(CODE_BITS), .PER_TILE_BASE(PER_TILE_BASE), .ESC_FIFO(ESC_FIFO), .SRAM(SRAM)) u_dut (
        .clk, .rst_n,
        .cmd_valid, .cmd_ready, .cmd_funct3, .cmd_rs1, .cmd_rs2, .cmd_rdata,

        .m_code_araddr(araddr[0*AW +: AW]), .m_code_arlen(arlen[0*8 +: 8]), .m_code_arsize(arsize[0]),
        .m_code_arburst(arburst[0]), .m_code_arvalid(arvalid[0]), .m_code_arready(arready[0]),
        .m_code_rdata(rdata[0*MAXW +: 64]), .m_code_rresp(2'b00), .m_code_rlast(rlast[0]),
        .m_code_rvalid(rvalid[0]), .m_code_rready(rready[0]),

        .m_smr_araddr(araddr[1*AW +: AW]), .m_smr_arlen(arlen[1*8 +: 8]), .m_smr_arsize(arsize[1]),
        .m_smr_arburst(arburst[1]), .m_smr_arvalid(arvalid[1]), .m_smr_arready(arready[1]),
        .m_smr_rdata(rdata[1*MAXW +: SDW]), .m_smr_rresp(2'b00), .m_smr_rlast(rlast[1]),
        .m_smr_rvalid(rvalid[1]), .m_smr_rready(rready[1]),

        .m_esc_araddr(araddr[2*AW +: AW]), .m_esc_arlen(arlen[2*8 +: 8]), .m_esc_arsize(arsize[2]),
        .m_esc_arburst(arburst[2]), .m_esc_arvalid(arvalid[2]), .m_esc_arready(arready[2]),
        .m_esc_rdata(rdata[2*MAXW +: 64]), .m_esc_rresp(2'b00), .m_esc_rlast(rlast[2]),
        .m_esc_rvalid(rvalid[2]), .m_esc_rready(rready[2]),

        .m_tb_araddr(araddr[3*AW +: AW]), .m_tb_arlen(arlen[3*8 +: 8]), .m_tb_arsize(arsize[3]),
        .m_tb_arburst(arburst[3]), .m_tb_arvalid(arvalid[3]), .m_tb_arready(arready[3]),
        .m_tb_rdata(rdata[3*MAXW +: 64]), .m_tb_rresp(2'b00), .m_tb_rlast(rlast[3]),
        .m_tb_rvalid(rvalid[3]), .m_tb_rready(rready[3]),

        .m_x_araddr(araddr[4*AW +: AW]), .m_x_arlen(arlen[4*8 +: 8]), .m_x_arsize(arsize[4]),
        .m_x_arburst(arburst[4]), .m_x_arvalid(arvalid[4]), .m_x_arready(arready[4]),
        .m_x_rdata(rdata[4*MAXW +: 64]), .m_x_rresp(2'b00), .m_x_rlast(rlast[4]),
        .m_x_rvalid(rvalid[4]), .m_x_rready(rready[4]),

        .m_y_awaddr(awaddr), .m_y_awlen(awlen), .m_y_awsize(awsize), .m_y_awburst(awburst),
        .m_y_awvalid(awvalid), .m_y_awready(awready),
        .m_y_wdata(wdata), .m_y_wstrb(wstrb), .m_y_wlast(wlast), .m_y_wvalid(wvalid), .m_y_wready(wready),
        .m_y_bresp(2'b00), .m_y_bvalid(bvalid), .m_y_bready(bready)
    );

    // port order and beat widths: x, tb, esc, smr, code (MSB first)
    axi_mem #(.NP(NP), .MAXW(MAXW), .AW(AW), .WORDS(MEM_WORDS), .LAT(LAT), .BW_NUM(BW_NUM), .BW_DEN(BW_DEN),
              .PBYTES({8'd8, 8'd8, 8'd8, 8'(SDW / 8), 8'd8})) u_mem (
        .clk, .rst_n,
        .araddr, .arlen, .arvalid, .arready, .rdata, .rlast, .rvalid, .rready,
        .awaddr, .awlen, .awvalid, .awready, .wdata, .wstrb, .wlast, .wvalid, .wready,
        .bvalid, .bready,
        .load, .dump, .dump_lo, .dump_hi
    );

endmodule
