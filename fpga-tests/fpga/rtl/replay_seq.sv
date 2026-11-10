// replay_seq.sv: plays recorded DUT inputs one clock edge at a time
//
// Runs on the free-running base clock `clk`. The host (over JTAG) fills the
// stimulus RAMs S00.. (instantiated by the generated top) with one input
// vector per entry, writes the entry count to CTRL[1], then a new token to
// CTRL[0]. For each entry i the sequencer
//
//   phase 0      presents address i to the stimulus and response RAMs
//   phase 1      the stimulus RAM output is valid; in_vec <= it at the end
//   phase 2..P-2 the DUT inputs settle
//   phase P-1    dclk_en is high, so the gated DUT clock rises at the end of
//                this phase; at that same base edge r_we writes out_vec (the
//                DUT outputs just before its edge) to response entry i
//
// and when all entries are done it copies the token to STAT[0]. Between runs
// the DUT gets no edges and in_vec holds, so DUT state carries over from one
// chunk to the next.
//
// STAT, written continuously: [0] done token, [1] DUT edges since
// configuration, [2] MAGIC, [3] SUITE_ID (the host checks [2] and [3] before
// trusting the bitstream). `ph` is exported for DUTs with FPGA-only memory
// adaptations that time-multiplex a block RAM within one DUT cycle.

module replay_seq #(
    parameter int NIN   = 32,               // input vector bits (ports + sideband)
    parameter int NIW   = 1,                // (NIN + 31) / 32
    parameter int AW    = 11,               // stimulus / response depth = 2**AW
    parameter int PH    = 8,                // base cycles per DUT edge, 4..15
    parameter logic [31:0] SUITE_ID = 32'h0
) (
    input  logic              clk,
    output logic [AW-1:0]     s_addr,       // stimulus and response RAM address
    input  logic [NIW*32-1:0] s_q,
    output logic              r_we,
    output logic [NIN-1:0]    in_vec,
    output logic              dclk_en,      // to the clock-control block's ena
    output logic [3:0]        ph,
    output logic              busy,
    output logic [31:0]       edges
);
    localparam logic [31:0] MAGIC = 32'h5EB1_A701;
    localparam logic [3:0]  PH1   = 4'd1;
    localparam logic [3:0]  PHM2  = PH - 2;
    localparam logic [3:0]  PHM1  = PH - 1;

    // ---------------------------------------------------------------- control
    logic [31:0] ctrl_q, stat_d;
    logic [1:0]  ctrl_a, stat_a;
    logic        ctrl_a_r;                  // which CTRL word ctrl_q holds
    logic [31:0] last_tok, count;
    logic [AW:0] n, i;

    rt_ram #(.AW(2), .NAME("CTRL")) u_ctrl (.clk(clk), .addr(ctrl_a), .we(1'b0), .wdata(32'd0), .q(ctrl_q));
    rt_ram #(.AW(2), .NAME("STAT")) u_stat (.clk(clk), .addr(stat_a), .we(1'b1), .wdata(stat_d), .q());

    initial begin
        busy = 1'b0; ph = 4'd0; i = '0; n = '0; last_tok = 32'd0; edges = 32'd0; in_vec = '0;
        ctrl_a = 2'd0; ctrl_a_r = 1'b0; count = 32'd0; stat_a = 2'd0; dclk_en = 1'b0;
    end

    // STAT: cycle through the four words, one write per base cycle
    always_ff @(posedge clk) stat_a <= stat_a + 2'd1;
    always_comb
        case (stat_a)
            2'd0:    stat_d = last_tok;
            2'd1:    stat_d = edges;
            2'd2:    stat_d = MAGIC;
            default: stat_d = SUITE_ID;
        endcase

    // ---------------------------------------------------------------- sequencer
    assign s_addr = i[AW-1:0];
    assign r_we   = busy && ph == PHM1;

    always_ff @(posedge clk) begin
        dclk_en <= 1'b0;
        if (!busy) begin
            // poll CTRL[1] (count) and CTRL[0] (token) on alternate cycles
            ctrl_a   <= (ctrl_a == 2'd0) ? 2'd1 : 2'd0;
            ctrl_a_r <= ctrl_a[0];
            if (ctrl_a_r)
                count <= ctrl_q;
            else if (ctrl_q != last_tok && count != 32'd0) begin
                busy     <= 1'b1;
                last_tok <= ctrl_q;
                n        <= count[AW:0];
                i        <= '0;
                ph       <= 4'd0;
            end
        end else begin
            if (ph == PH1)  in_vec  <= s_q[NIN-1:0];
            if (ph == PHM2) dclk_en <= 1'b1;            // high during phase PH-1
            if (ph == PHM1) begin
                ph    <= 4'd0;
                edges <= edges + 32'd1;
                if (i + 1'b1 == n) busy <= 1'b0;
                i     <= i + 1'b1;
            end else
                ph <= ph + 4'd1;
        end
    end

endmodule
