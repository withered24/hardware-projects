// tb_dec.sv — the weight decoder alone, for RTL and gate-level runs
//
// Built against rtl/weight_decoder.sv or against the routed SKY130 netlist
// (gl/weight_decoder.nl.v + the cell models), same ports either way. With
// +VCD=<file> it dumps the decoder's switching for the power report.
`timescale 1ns/1ps

module tb_dec (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         clear,
    input  logic         bypass,
    input  logic [7:0]   base,
    input  logic         code_valid,
    output logic         code_ready,
    input  logic [63:0]  code_data,
    input  logic         tb_valid,
    output logic         tb_ready,
    input  logic [63:0]  tb_data,
    input  logic         smr_valid,
    output logic         smr_ready,
    input  logic [127:0] smr_data,
    input  logic         esc_valid,
    output logic         esc_ready,
    input  logic [63:0]  esc_data,
    output logic         out_valid,
    input  logic         out_ready,
    output logic [127:0] out_w,
    output logic [3:0]   out_nesc,
    output logic         stall_esc,
    output logic         stall_data
);

    weight_decoder dut (.*);

    string vcd;
    initial begin
        if ($value$plusargs("VCD=%s", vcd)) begin
            $dumpfile(vcd);
            $dumpvars(0, dut);
        end
    end

endmodule
