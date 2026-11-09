// tb_soc.sv — the full system: riscv-core running the driver, its custom-0
// instructions wired to the engine
//
//   core --imem/dmem--> sim_mem (64 KiB: program, job table, results)
//   core --cop_*------> tb_top's command port (ew_top + testbench DRAM)
//
// Two memories in simulation: the CPU's own RAM, and the DRAM the engine
// streams weights from. On a board they're one address space; the job
// table's addresses are already DRAM addresses either way.

module tb_soc #(
    parameter int L         = 8,
    parameter int CODE_BITS = 3,
    parameter int BW_NUM    = 16,
    parameter int BW_DEN    = 1
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        load,
    input  logic        dump,
    input  logic [31:0] dump_lo,
    input  logic [31:0] dump_hi
);

    logic [31:0] imem_addr, imem_rdata, dmem_addr, dmem_wdata, dmem_rdata;
    logic [3:0]  dmem_wstrb;
    logic        dmem_re;

    logic        cop_valid, cop_ready;
    logic [2:0]  cop_funct3;
    logic [31:0] cop_rs1, cop_rs2, cop_rdata;

    logic        trace_valid, trace_rd_we, trace_illegal, trace_trap;
    logic [31:0] trace_pc, trace_instr, trace_rd_wdata, trace_cause;
    logic [31:0] trace_mem_addr, trace_mem_wdata;
    logic [4:0]  trace_rd;
    logic [3:0]  trace_mem_wstrb;

    core u_core (
        .clk, .rst(!rst_n),
        .imem_addr, .imem_rdata,
        .dmem_addr, .dmem_wdata, .dmem_wstrb, .dmem_re, .dmem_rdata,
        .dmem_wait(1'b0),
        .cop_valid, .cop_ready, .cop_funct3, .cop_rs1, .cop_rs2, .cop_rdata,
        .trace_valid, .trace_pc, .trace_instr, .trace_rd_we, .trace_rd, .trace_rd_wdata,
        .trace_illegal, .trace_trap, .trace_cause,
        .trace_mem_addr, .trace_mem_wstrb, .trace_mem_wdata
    );

    // a coprocessor wait freezes the whole pipeline, so the data port must
    // freeze with it, exactly as it does for dmem_wait
    sim_mem #(.SYNC_READ(1'b1)) u_mem (
        .clk,
        .iaddr(imem_addr), .irdata(imem_rdata),
        .daddr(dmem_addr), .dwdata(dmem_wdata), .dwstrb(dmem_wstrb),
        .dhold(cop_valid && !cop_ready), .drdata(dmem_rdata)
    );

    tb_top #(.L(L), .CODE_BITS(CODE_BITS), .BW_NUM(BW_NUM), .BW_DEN(BW_DEN)) u_eng (
        .clk, .rst_n,
        .cmd_valid(cop_valid), .cmd_ready(cop_ready), .cmd_funct3(cop_funct3),
        .cmd_rs1(cop_rs1), .cmd_rs2(cop_rs2), .cmd_rdata(cop_rdata),
        .load, .dump, .dump_lo, .dump_hi
    );

endmodule
