// axi_mem.sv — testbench DRAM: NP AXI read ports + 1 write port, one memory
//
// Behavioural, not synthesizable. What it models:
//   * a fixed latency LAT (cycles from AR handshake to the first beat)
//   * a shared bandwidth budget of BW_NUM/BW_DEN bytes per cycle across all
//     ports (a token bucket, capped at two max-width beats), handed out
//     round-robin, so the decoded and raw-BF16 runs compete for the same
//     bytes. This budget is the stand-in for DRAM bandwidth; cycle counts
//     measured against it are labelled "simulated" everywhere.
//   * up to QD outstanding bursts per port, in order
// Ports are flattened vectors (Icarus is happier with those than with
// unpacked-array ports). Port p's beat is PBYTES[p] bytes, 8..MAXW/8.
//
// The memory is 64-bit words. `load` (rising edge) reads +MEMFILE=<hex>
// with $readmemh; `dump` writes words dump_lo..dump_hi to +DUMPFILE=<hex>.

module axi_mem #(
    parameter int NP     = 5,
    parameter int MAXW   = 128,
    parameter int AW     = 32,
    parameter int WORDS  = 1 << 18,
    parameter int LAT    = 20,
    parameter int QD     = 8,
    parameter int BW_NUM = 16,
    parameter int BW_DEN = 1,
    parameter logic [NP*8-1:0] PBYTES = {8'd8, 8'd8, 8'd8, 8'd16, 8'd8}
) (
    input  logic               clk,
    input  logic               rst_n,

    input  logic [NP*AW-1:0]   araddr,
    input  logic [NP*8-1:0]    arlen,
    input  logic [NP-1:0]      arvalid,
    output logic [NP-1:0]      arready,
    output logic [NP*MAXW-1:0] rdata,
    output logic [NP-1:0]      rlast,
    output logic [NP-1:0]      rvalid,
    input  logic [NP-1:0]      rready,

    input  logic [AW-1:0]      awaddr,
    input  logic [7:0]         awlen,
    input  logic               awvalid,
    output logic               awready,
    input  logic [63:0]        wdata,
    input  logic [7:0]         wstrb,
    input  logic               wlast,
    input  logic               wvalid,
    output logic               wready,
    output logic               bvalid,
    input  logic               bready,

    input  logic               load,
    input  logic               dump,
    input  logic [31:0]        dump_lo,
    input  logic [31:0]        dump_hi
);

    logic [63:0] mem [WORDS];
    string memfile, dumpfile;

    initial begin
        for (int i = 0; i < WORDS; i++) mem[i] = 64'd0;
        if (!$value$plusargs("MEMFILE=%s", memfile))  memfile  = "mem.hex";
        if (!$value$plusargs("DUMPFILE=%s", dumpfile)) dumpfile = "dump.hex";
    end
    always @(posedge load) $readmemh(memfile, mem);
    always @(posedge dump) $writememh(dumpfile, mem, dump_lo, dump_hi);

    // per-port burst queues
    logic [AW-1:0] q_addr  [NP][QD];
    int            q_beats [NP][QD];
    longint        q_t     [NP][QD];
    int            q_wp [NP], q_rp [NP], q_cnt [NP];
    logic [AW-1:0] c_addr [NP];
    int            c_left [NP];

    // write side
    logic [AW-1:0] aw_q_addr [QD];
    int            aw_wp, aw_rp, aw_cnt;
    logic [AW-1:0] w_addr;
    int            w_inburst;
    longint        b_t [QD];
    int            b_wp, b_rp, b_cnt;

    longint cyc;
    longint credit;
    int     rr;

    localparam longint CAP = 2 * (MAXW / 8) * BW_DEN;

    // ready outputs are registered (from the state after this edge), so the
    // DUT and this model always agree on which handshakes happened
    always @(posedge clk) begin
        if (!rst_n) begin
            cyc = 0; credit = 0; rr = 0;
            for (int p = 0; p < NP; p++) begin
                q_wp[p] = 0; q_rp[p] = 0; q_cnt[p] = 0; c_left[p] = 0;
            end
            aw_wp = 0; aw_rp = 0; aw_cnt = 0; w_inburst = 0;
            b_wp = 0; b_rp = 0; b_cnt = 0;
            rvalid  <= '0;
            bvalid  <= 1'b0;
            arready <= '0;
            awready <= 1'b0;
            wready  <= 1'b0;
        end else begin
            logic [NP-1:0]      nv;
            logic [NP*MAXW-1:0] nd;
            logic [NP-1:0]      nl;
            nv = rvalid; nd = rdata; nl = rlast;
            cyc = cyc + 1;
            credit = credit + BW_NUM;
            if (credit > CAP) credit = CAP;

            // accept AR
            for (int p = 0; p < NP; p++) begin
                if (arvalid[p] && arready[p]) begin
                    q_addr[p][q_wp[p]]  = araddr[p*AW +: AW];
                    q_beats[p][q_wp[p]] = int'(arlen[p*8 +: 8]) + 1;
                    q_t[p][q_wp[p]]     = cyc + LAT;
                    q_wp[p] = (q_wp[p] + 1) % QD;
                    q_cnt[p] = q_cnt[p] + 1;
                end
            end

            // R beats, round-robin over ports for the bandwidth budget
            for (int k = 0; k < NP; k++) begin
                int p, nb;
                p  = (rr + k) % NP;
                nb = int'(PBYTES[p*8 +: 8]);
                if (nv[p] && rready[p]) nv[p] = 1'b0;
                if (!nv[p] && c_left[p] == 0 && q_cnt[p] > 0 && q_t[p][q_rp[p]] <= cyc) begin
                    c_addr[p] = q_addr[p][q_rp[p]];
                    c_left[p] = q_beats[p][q_rp[p]];
                    q_rp[p]   = (q_rp[p] + 1) % QD;
                    q_cnt[p]  = q_cnt[p] - 1;
                end
                if (!nv[p] && c_left[p] > 0 && credit >= nb * BW_DEN) begin
                    for (int b = 0; b < nb / 8; b++)
                        nd[p*MAXW + 64*b +: 64] = mem[(c_addr[p] >> 3) + b];
                    nl[p]     = (c_left[p] == 1);
                    nv[p]     = 1'b1;
                    c_addr[p] = c_addr[p] + nb;
                    c_left[p] = c_left[p] - 1;
                    credit    = credit - nb * BW_DEN;
                end
            end
            rr = (rr + 1) % NP;
            rvalid <= nv;
            rdata  <= nd;
            rlast  <= nl;

            // writes
            if (awvalid && awready) begin
                aw_q_addr[aw_wp] = awaddr;
                aw_wp = (aw_wp + 1) % QD;
                aw_cnt = aw_cnt + 1;
            end
            if (wvalid && wready) begin
                if (w_inburst == 0) begin
                    w_addr = aw_q_addr[aw_rp];
                    w_inburst = 1;
                end
                for (int b = 0; b < 8; b++)
                    if (wstrb[b]) mem[w_addr >> 3][8*b +: 8] = wdata[8*b +: 8];
                w_addr = w_addr + 8;
                credit = credit - 8 * BW_DEN;
                if (wlast) begin
                    aw_rp = (aw_rp + 1) % QD;
                    aw_cnt = aw_cnt - 1;
                    w_inburst = 0;
                    b_t[b_wp] = cyc + LAT;
                    b_wp = (b_wp + 1) % QD;
                    b_cnt = b_cnt + 1;
                end
            end
            if (bvalid && bready) begin
                b_rp = (b_rp + 1) % QD;
                b_cnt = b_cnt - 1;
            end
            bvalid <= (b_cnt > 0) && (b_t[b_rp] <= cyc);
            for (int p = 0; p < NP; p++) arready[p] <= (q_cnt[p] < QD - 1);
            awready <= (aw_cnt < QD - 1);
            wready  <= (aw_cnt > 0 || (awvalid && awready)) && (credit >= 8 * BW_DEN);
        end
    end

endmodule
