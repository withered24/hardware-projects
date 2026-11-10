// dclk_gate.sv: the DUT's clock, the base clock with edges only when `ena`
//
// Quartus: a Cyclone IV clock-control block. Its enable register samples ena
// on the falling edge of inclk, so the output is glitch-free and an ena that
// is high for one whole base cycle lets exactly the next rising edge through.
// Simulation: the same behaviour from a negative-edge latch.

module dclk_gate (
    input  logic clk,
    input  logic ena,
    output logic gclk
);
`ifdef SIM
    logic ena_l = 1'b0;
    always @(negedge clk) ena_l <= ena;
    assign gclk = clk & ena_l;
`else
    cycloneive_clkctrl #(
        .clock_type        ("global clock"),
        .ena_register_mode ("falling edge"),
        .lpm_type          ("cycloneive_clkctrl")
    ) u_ctrl (
        .inclk     ({3'b000, clk}),
        .clkselect (2'b00),
        .ena       (ena),
        .outclk    (gclk)
    );
`endif
endmodule
