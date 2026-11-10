// regfile.sv — RV32I integer register file
//
// 32 registers x 32 bits. Two combinational read ports (rs1, rs2) feed the
// ALU; one synchronous write port (rd) is updated on the rising clock edge.
// Register x0 is hardwired to zero: writes to it are dropped, reads return 0.
//
// Single-cycle note: a read and a write to the same register in one cycle
// returns the OLD value (the write lands at the clock edge, after the read).
// That's correct for a single-cycle core. It stops being correct once we
// pipeline — keep that in mind for Milestone 3.

module regfile (
    input  logic        clk,

    // read port 1
    input  logic [4:0]  rs1_addr,
    output logic [31:0] rs1_data,

    // read port 2
    input  logic [4:0]  rs2_addr,
    output logic [31:0] rs2_data,

    // write port
    input  logic        rd_we,
    input  logic [4:0]  rd_addr,
    input  logic [31:0] rd_data,

    // FPGA replay only: zero every register at this edge (stands in for the
    // testbench's backdoor clear in cosim.load_and_reset)
    input  logic        rf_clear
);

    logic [31:0] regs [31:0];

    // x0 is never written, so regs[0] stays at its initial value of zero.
    // Simulation only: ASIC flops have no power-up value, and the read ports
    // below force x0 to zero anyway.
`ifndef SYNTHESIS
    initial begin
        for (int i = 0; i < 32; i++) regs[i] = 32'd0;
    end
`endif

    always_ff @(posedge clk) begin
        if (rf_clear)
            for (int r = 0; r < 32; r++) regs[r] <= 32'd0;
        if (rd_we && rd_addr != 5'd0)        // after the clear, as in simulation
            regs[rd_addr] <= rd_data;
    end

    // The explicit x0 check means x0 reads 0 even if regs[0] got corrupted.
    assign rs1_data = (rs1_addr == 5'd0) ? 32'd0 : regs[rs1_addr];
    assign rs2_data = (rs2_addr == 5'd0) ? 32'd0 : regs[rs2_addr];

endmodule
