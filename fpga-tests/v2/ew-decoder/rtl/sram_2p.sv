// sram_2p.sv — one write port, one read port, synchronous read
//
// Read data is valid for the one cycle after `re`, and callers register it at
// the next clock edge. The ASIC build (`define SKY130_SRAM) tiles SkyWater's
// OpenRAM macros (sky130_sram_macros): 32-bit columns, 256 rows per macro up
// to DEPTH 256, else 512, with port 0 (rw) for writes and port 1 (r) for
// reads. A macro's output is only valid in the cycle after a read, so the
// behavioural model drives X in every other cycle to catch a caller that
// relies on it holding. Writing and reading the same address in one cycle is
// undefined in the macros; callers never do.

module sram_2p #(
    parameter int WIDTH = 128,
    parameter int DEPTH = 512
) (
    input  logic                     clk,
    input  logic                     we,
    input  logic [$clog2(DEPTH)-1:0] waddr,
    input  logic [WIDTH-1:0]         wdata,
    input  logic                     re,
    input  logic [$clog2(DEPTH)-1:0] raddr,
    output logic [WIDTH-1:0]         rdata
);

    localparam int AW = $clog2(DEPTH);

`ifdef SKY130_SRAM
    localparam int ROWS_PER = DEPTH <= 256 ? 256 : 512;    // words per macro
    localparam int MAW      = $clog2(ROWS_PER);
    localparam int NCOL     = (WIDTH + 31) / 32;
    localparam int NROW     = (DEPTH + ROWS_PER - 1) / ROWS_PER;
    localparam int RSW      = NROW > 1 ? $clog2(NROW) : 1;

    logic [NCOL*32-1:0] wd;
    assign wd = (NCOL*32)'(wdata);

    logic [NROW-1:0][NCOL*32-1:0] dout;
    logic [RSW-1:0]               rrow;                  // macro row of the last read

    always_ff @(posedge clk)
        if (re) rrow <= NROW > 1 ? RSW'(raddr >> MAW) : '0;
    assign rdata = WIDTH'(dout[rrow]);

    for (genvar r = 0; r < NROW; r++) begin : g_row
        logic wsel, rsel;
        assign wsel = we && (NROW == 1 || (waddr >> MAW) == r);
        assign rsel = re && (NROW == 1 || (raddr >> MAW) == r);
        for (genvar c = 0; c < NCOL; c++) begin : g_col
            if (ROWS_PER == 256) begin : g_1k
                sky130_sram_1kbyte_1rw1r_32x256_8 u_mem (
                    .clk0(clk), .csb0(!wsel), .web0(1'b0), .wmask0(4'hF),
                    .addr0(MAW'(waddr)), .din0(wd[32*c +: 32]), .dout0(),
                    .clk1(clk), .csb1(!rsel), .addr1(MAW'(raddr)), .dout1(dout[r][32*c +: 32]));
            end else begin : g_2k
                sky130_sram_2kbyte_1rw1r_32x512_8 u_mem (
                    .clk0(clk), .csb0(!wsel), .web0(1'b0), .wmask0(4'hF),
                    .addr0(MAW'(waddr)), .din0(wd[32*c +: 32]), .dout0(),
                    .clk1(clk), .csb1(!rsel), .addr1(MAW'(raddr)), .dout1(dout[r][32*c +: 32]));
            end
        end
    end
`else
    logic [WIDTH-1:0] mem [DEPTH];

    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
`ifdef SYNTHESIS
        if (re) rdata <= mem[raddr];
`else
        rdata <= re ? mem[raddr] : 'x;
`endif
    end
`endif

endmodule
