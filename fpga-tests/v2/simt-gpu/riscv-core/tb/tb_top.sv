// tb_top.sv — simulation top level: a core wired to sim_mem
//
// cocotb drives clk/rst and reads the trace_* signals from here.
// Builds the pipelined core by default; define SINGLE_CYCLE for the
// single-cycle one. Both cores have the same trace.
//
// Define MEM_WAIT (pipelined core only) to make the data port slow: every
// accepted load or store is held in MEM for a pseudo-random number of wait
// cycles (0, 1, 2, ... with a falling probability), the way an AXI4-Lite
// device would. The same lockstep checks then prove the freeze logic.

`ifdef SINGLE_CYCLE
  `define CORE_MODULE core_single
  `define MEM_SYNC_READ 1'b0
`else
  `define CORE_MODULE core
  `define MEM_SYNC_READ 1'b1
`endif

module tb_top #(
    parameter logic [31:0] RESET_PC = 32'h0000_0000   // riscv-tests use 0x8000_0000
) (
    input  logic        clk,
    input  logic        rst,

    output logic        trace_valid,
    output logic [31:0] trace_pc,
    output logic [31:0] trace_instr,
    output logic        trace_rd_we,
    output logic [4:0]  trace_rd,
    output logic [31:0] trace_rd_wdata,
    output logic        trace_illegal,
    output logic        trace_trap,
    output logic [31:0] trace_cause,
    output logic [31:0] trace_mem_addr,
    output logic [3:0]  trace_mem_wstrb,
    output logic [31:0] trace_mem_wdata
);

    logic [31:0] imem_addr, imem_rdata, dmem_addr, dmem_wdata, dmem_rdata;
    logic [3:0]  dmem_wstrb;
    logic        dmem_re, dmem_wait;

`ifdef MEM_WAIT
    // An access is accepted at an edge where dmem_wait was low; it then sits
    // in MEM, and each cycle it waits again with probability 1/2.
    logic        pend;
    logic [15:0] lfsr;
    always_ff @(posedge clk) begin
        if (rst) begin
            pend <= 1'b0;
            lfsr <= 16'hACE1;
        end else begin
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
            if (!dmem_wait) pend <= dmem_re || (dmem_wstrb != 4'b0000);
        end
    end
    assign dmem_wait = pend && lfsr[0] && !rst;
`else
    assign dmem_wait = 1'b0;
`endif

    `CORE_MODULE #(.RESET_PC(RESET_PC)) u_core (
        .clk             (clk),
        .rst             (rst),
        .imem_addr       (imem_addr),
        .imem_rdata      (imem_rdata),
        .dmem_addr       (dmem_addr),
        .dmem_wdata      (dmem_wdata),
        .dmem_wstrb      (dmem_wstrb),
        .dmem_rdata      (dmem_rdata),
`ifndef SINGLE_CYCLE
        .dmem_re         (dmem_re),
        .dmem_wait       (dmem_wait),
`endif
        .trace_valid     (trace_valid),
        .trace_pc        (trace_pc),
        .trace_instr     (trace_instr),
        .trace_rd_we     (trace_rd_we),
        .trace_rd        (trace_rd),
        .trace_rd_wdata  (trace_rd_wdata),
        .trace_illegal   (trace_illegal),
        .trace_trap      (trace_trap),
        .trace_cause     (trace_cause),
        .trace_mem_addr  (trace_mem_addr),
        .trace_mem_wstrb (trace_mem_wstrb),
        .trace_mem_wdata (trace_mem_wdata)
    );

    sim_mem #(.SYNC_READ(`MEM_SYNC_READ)) u_mem (
        .clk    (clk),
        .iaddr  (imem_addr),
        .irdata (imem_rdata),
        .daddr  (dmem_addr),
        .dwdata (dmem_wdata),
        .dwstrb (dmem_wstrb),
        .dhold  (dmem_wait),
        .drdata (dmem_rdata)
    );

endmodule
