// simt_lane.sv — one lane of a SIMT core: its register file and its ALU
//
// A core has LANES of these side by side, all driven by the same decoded
// instruction; only the register contents differ. That makes the lane the
// natural unit to lay out once and repeat, the way a GPU's lanes are tiled.
//
// The register file holds the lane's 32 registers for every warp, at
// {warp, register}. It is two copies of one synchronous SRAM (sram_2p), one
// per source register, written together: each copy has one read port and
// one write port, and an instruction reads two registers at once. A read
// asked for in D lands in R and is registered into X; x0 is never written
// and reads as 0. One write per cycle: X's result, or load data from the LSU.

import rv32i_pkg::*;
import gpu_pkg::*;

module simt_lane (
    input  logic             clk,

    // D: read two registers; the data lands in R
    input  logic             rd_en,
    input  logic [RF_AW-1:0] rd_a1,
    input  logic [RF_AW-1:0] rd_a2,
    // R: which of them is x0
    input  logic             r_z1,
    input  logic             r_z2,

    // X: the instruction, the same for every lane
    input  logic [1:0]       a_sel,
    input  logic             b_imm,
    input  logic [3:0]       alu_op,
    input  logic [2:0]       funct3,
    input  logic [1:0]       wb_sel,
    input  logic             mul,
    input  logic             jal,
    input  logic             jalr,
    input  logic             branch,
    input  logic [31:0]      pc,
    input  logic [31:0]      imm,
    input  logic [31:0]      csr,            // this lane's value of the CSR read
    output logic [31:0]      rs2v,           // store data
    output logic [31:0]      alu_y,          // load/store address
    output logic [31:0]      next_pc,

    // write back
    input  logic             wr_en,
    input  logic             wr_lsu,         // load data, not X's result
    input  logic [RF_AW-1:0] wr_a,
    input  logic [31:0]      lsu_data
);

    // ------------------------------------------------------------ register file
    logic [31:0] q1, q2, rs1v, xw, wdata;

    assign wdata = wr_lsu ? lsu_data : xw;

    sram_2p #(.WIDTH(32), .DEPTH(WARPS * 32)) u_rf1 (
        .clk, .we(wr_en), .waddr(wr_a), .wdata, .re(rd_en), .raddr(rd_a1), .rdata(q1));
    sram_2p #(.WIDTH(32), .DEPTH(WARPS * 32)) u_rf2 (
        .clk, .we(wr_en), .waddr(wr_a), .wdata, .re(rd_en), .raddr(rd_a2), .rdata(q2));

    always_ff @(posedge clk) begin
        rs1v <= r_z1 ? 32'd0 : q1;
        rs2v <= r_z2 ? 32'd0 : q2;
    end

    // ------------------------------------------------------------ X
    logic [31:0] a, b;
    logic        taken;

    assign a = a_sel == A_PC ? pc : a_sel == A_ZERO ? 32'd0 : rs1v;
    assign b = b_imm ? imm : rs2v;

    alu         u_alu (.a, .b, .op(alu_op), .y(alu_y));
    branch_unit u_br  (.a(rs1v), .b(rs2v), .funct3, .taken);

    always_comb begin
        case (wb_sel)
            WB_PC4:  xw = pc + 32'd4;
            WB_CSR:  xw = csr;
            default: xw = mul ? rs1v * rs2v : alu_y;
        endcase

        if (jal)                  next_pc = pc + imm;
        else if (jalr)            next_pc = alu_y & ~32'd1;
        else if (branch && taken) next_pc = pc + imm;
        else                      next_pc = pc + 32'd4;
    end

endmodule
