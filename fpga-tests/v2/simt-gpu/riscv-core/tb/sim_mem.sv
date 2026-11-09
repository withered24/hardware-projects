// sim_mem.sv — simulation-only unified memory (code + data in one array)
//
// Byte-enabled synchronous write. Reads are combinational (SYNC_READ = 0,
// for the single-cycle core) or registered like FPGA block RAM
// (SYNC_READ = 1, for the pipeline): the address is captured at the clock
// edge and the data appears after it. A registered read in the same cycle
// as a write to that word returns the OLD data (read-before-write).
// dhold freezes the data port (the core's dmem_wait): no write lands and
// the registered read data stays put, so a waiting load keeps its value.
// The address is taken modulo the memory size, so a program linked at
// 0x8000_0000 (where riscv-tests live) lands at index 0 too.
// The testbench loads programs by writing `mem` directly through cocotb.

module sim_mem #(
    parameter int WORDS = 16384,          // 64 KiB
    parameter bit SYNC_READ = 1'b0
) (
    input  logic        clk,

    input  logic [31:0] iaddr,
    output logic [31:0] irdata,

    input  logic [31:0] daddr,
    input  logic [31:0] dwdata,
    input  logic [3:0]  dwstrb,
    input  logic        dhold,
    output logic [31:0] drdata
);

    localparam int AW = $clog2(WORDS);

    logic [31:0] mem [0:WORDS-1];

    initial begin
        for (int i = 0; i < WORDS; i++) mem[i] = 32'd0;
    end

    generate
        if (SYNC_READ) begin : g_sync
            always_ff @(posedge clk) begin
                irdata <= mem[iaddr[AW+1:2]];
                if (!dhold) drdata <= mem[daddr[AW+1:2]];
            end
        end else begin : g_comb
            assign irdata = mem[iaddr[AW+1:2]];
            assign drdata = mem[daddr[AW+1:2]];
        end
    endgenerate

    always_ff @(posedge clk) begin
        for (int b = 0; b < 4; b++)
            if (dwstrb[b] && !dhold) mem[daddr[AW+1:2]][8*b +: 8] <= dwdata[8*b +: 8];
    end

endmodule
