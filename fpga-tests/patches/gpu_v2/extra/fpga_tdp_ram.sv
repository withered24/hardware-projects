// fpga_tdp_ram.sv: FPGA-only, a true dual-port RAM on one clock (M9K blocks)
//
// Written in Quartus's true-dual-port inference template. Each port reads
// with a registered address: q is the word at the address present at the last
// clock edge (the new word if that port wrote it then). Powers up to zero,
// like the M9K blocks it maps to. Used by the GPU's FPGA memory patches,
// which time-share each port within one DUT clock cycle (see gpu_mem.sv).

module fpga_tdp_ram #(
    parameter int DW = 8,
    parameter int AW = 11
) (
    input  logic          clk,
    input  logic          we_a,
    input  logic [AW-1:0] addr_a,
    input  logic [DW-1:0] wdata_a,
    output logic [DW-1:0] q_a,
    input  logic          we_b,
    input  logic [AW-1:0] addr_b,
    input  logic [DW-1:0] wdata_b,
    output logic [DW-1:0] q_b
);
    logic [DW-1:0] mem [0:(1<<AW)-1];
`ifdef SIM
    initial for (int i = 0; i < (1 << AW); i++) mem[i] = '0;
`endif

    always_ff @(posedge clk) begin
        if (we_a) begin
            mem[addr_a] <= wdata_a;
            q_a         <= wdata_a;
        end else
            q_a <= mem[addr_a];
    end

    always_ff @(posedge clk) begin
        if (we_b) begin
            mem[addr_b] <= wdata_b;
            q_b         <= wdata_b;
        end else
            q_b <= mem[addr_b];
    end
endmodule
