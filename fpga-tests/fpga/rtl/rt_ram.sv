// rt_ram.sv: a 32-bit-wide RAM the host can read and write over JTAG
//
// Quartus build: altsyncram in single-port mode with ENABLE_RUNTIME_MOD, so
// the In-System Memory Content Editor (quartus_stp's *_content_*_memory
// commands) reaches it through the M9K's second port, by INSTANCE_NAME (at
// most 4 characters). The design side reads with a registered address: q is
// mem[addr as of the last edge].
//
// Simulation (`define SIM): the same timing from a plain array, `mem`, which
// the simulation testbench loads and dumps with $readmemh / $writememh in
// place of JTAG.

module rt_ram #(
    parameter int    AW   = 11,
    parameter string NAME = "R000"
) (
    input  logic        clk,
    input  logic [AW-1:0] addr,
    input  logic        we,
    input  logic [31:0] wdata,
    output logic [31:0] q
);

`ifdef SIM
    logic [31:0] mem [0:(1<<AW)-1];
    initial for (int i = 0; i < (1 << AW); i++) mem[i] = 32'd0;
    logic [AW-1:0] a_r;
    always_ff @(posedge clk) begin
        if (we) mem[addr] <= wdata;
        a_r <= addr;
    end
    assign q = mem[a_r];
`else
    altsyncram #(
        .operation_mode                ("SINGLE_PORT"),
        .width_a                       (32),
        .widthad_a                     (AW),
        .numwords_a                    (1 << AW),
        .width_byteena_a               (1),
        .outdata_reg_a                 ("UNREGISTERED"),
        .outdata_aclr_a                ("NONE"),
        .clock_enable_input_a          ("BYPASS"),
        .clock_enable_output_a         ("BYPASS"),
        .read_during_write_mode_port_a ("NEW_DATA_NO_NBE_READ"),
        .power_up_uninitialized        ("FALSE"),
        .intended_device_family        ("Cyclone IV E"),
        .lpm_type                      ("altsyncram"),
        .lpm_hint                      ({"ENABLE_RUNTIME_MOD=YES,INSTANCE_NAME=", NAME})
    ) u_ram (
        .clock0         (clk),
        .address_a      (addr),
        .wren_a         (we),
        .data_a         (wdata),
        .q_a            (q),
        .aclr0          (1'b0),
        .aclr1          (1'b0),
        .address_b      (1'b1),
        .addressstall_a (1'b0),
        .addressstall_b (1'b0),
        .byteena_a      (1'b1),
        .byteena_b      (1'b1),
        .clock1         (1'b1),
        .clocken0       (1'b1),
        .clocken1       (1'b1),
        .clocken2       (1'b1),
        .clocken3       (1'b1),
        .data_b         (1'b1),
        .eccstatus      (),
        .q_b            (),
        .rden_a         (1'b1),
        .rden_b         (1'b1),
        .wren_b         (1'b0)
    );
`endif

endmodule
