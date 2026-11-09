// weight_decoder.sv — streams in, L decoded BF16 weights per cycle out
//
// Three (four with per-tile bases) input streams become one stream of
// L-weight chunks, in row-major order:
//
//   code   64-bit words; CODE_BITS words make one 64-weight tile (bit planes)
//   smr    2L bytes per word: sign/mantissa bytes (decoded mode, half a word
//          per chunk) or raw BF16 (bypass mode, a whole word per chunk)
//   esc    64-bit words of escaped exponents, in stream order
//   tbase  64-bit words of per-tile base bytes (PER_TILE_BASE only)
//
// Tiles: two register sets, the tile being decoded and the next one, which
// fills from the code stream while the current one drains, so moving to a
// new tile costs no cycle.
//
// Escapes: lane i escapes when its code is 0. Its escape byte is window
// byte s_i, where s_i counts the escaping lanes before it (a prefix
// popcount), and the window then shifts by the total count. The window
// holds WB = L + 8 bytes and refills 8 at a time, which sustains even a run
// of chunks where every lane escapes (for L <= 8).
//
// A chunk goes out whole or not at all: it waits until the tile, the
// sign/mantissa bytes and enough escape bytes are all there. stall_esc and
// stall_data say which one it waited for (they count only while the
// consumer is ready, i.e. while the decoder is the bottleneck).
//
// Bypass (the raw-BF16 baseline): the smr word is the weights themselves and
// the lanes are skipped. Nothing else changes.
//
// Output register: a chunk is decoded in one cycle into an output stage
// (out_valid/out_w/out_nesc are flops), which takes a new chunk whenever it
// is empty or being read, so throughput stays one chunk per cycle and only
// latency grows by one. Without it the escape-routing muxes drove out_w
// combinationally and missed 50 MHz at the slow corner by 0.54 ns.

module weight_decoder #(
    parameter int L             = 8,
    parameter int CODE_BITS     = 3,
    parameter int PER_TILE_BASE = 0
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    clear,
    input  logic                    bypass,
    input  logic [7:0]              base,         // per-matrix base (EXP.BASE)

    input  logic                    code_valid,
    output logic                    code_ready,
    input  logic [63:0]             code_data,
    input  logic                    tb_valid,
    output logic                    tb_ready,
    input  logic [63:0]             tb_data,
    input  logic                    smr_valid,
    output logic                    smr_ready,
    input  logic [16*L-1:0]         smr_data,
    input  logic                    esc_valid,
    output logic                    esc_ready,
    input  logic [63:0]             esc_data,

    output logic                    out_valid,
    input  logic                    out_ready,
    output logic [16*L-1:0]         out_w,
    output logic [$clog2(L+1)-1:0]  out_nesc,     // escapes in the chunk that just went out
    output logic                    stall_esc,
    output logic                    stall_data
);

    localparam int CHUNKS = 64 / L;               // chunks per tile
    localparam int JW     = (CHUNKS > 1) ? $clog2(CHUNKS) : 1;
    localparam int WB     = L + 8;                // escape window, bytes
    localparam int CW     = $clog2(WB + 1);       // window count width
    localparam int PW     = $clog2(L + 1);        // popcount width

    // ------------------------------------------------------------- tiles
    logic [63:0]                    stg [CODE_BITS];
    logic [$clog2(CODE_BITS+1)-1:0] stg_cnt;
    logic                           stg_has_base;
    logic [7:0]                     stg_base;
    logic [63:0]                    cur [CODE_BITS];
    logic                           cur_valid;
    logic [7:0]                     cur_base;
    logic [JW-1:0]                  j;            // chunk within the tile

    // per-tile base bytes, one word at a time
    logic [63:0] tbw;
    logic [3:0]  tbn;                             // bytes left in tbw
    logic        take_base;

    logic stg_full, last_chunk, load_cur;
    assign stg_full   = (32'(stg_cnt) == CODE_BITS) && (PER_TILE_BASE == 0 || stg_has_base);
    assign code_ready = !bypass && (32'(stg_cnt) < CODE_BITS);
    assign take_base  = (PER_TILE_BASE != 0) && !bypass && !stg_has_base && (tbn != 0);
    assign tb_ready   = (PER_TILE_BASE != 0) && !bypass && (tbn == 0);
    assign last_chunk = (j == JW'(CHUNKS - 1));

    // ------------------------------------------- codes, escapes, lanes
    logic [CODE_BITS-1:0] code [L];
    logic [L-1:0]         esc_mask;
    logic [PW-1:0]        slot [L];
    logic [PW-1:0]        need;
    logic [8*WB-1:0]      win;
    logic [CW-1:0]        wcnt;
    logic [7:0]           base_use;
    logic                 h;                      // which half of the smr word (decoded mode)
    logic [8*L-1:0]       sm;
    wire  [16*L-1:0]      dec_w;

    assign base_use = (PER_TILE_BASE != 0) ? cur_base : base;
    assign sm       = h ? smr_data[16*L-1 -: 8*L] : smr_data[8*L-1:0];

    always_comb begin
        need = '0;
        for (int i = 0; i < L; i++) begin
            for (int p = 0; p < CODE_BITS; p++)
                code[i][p] = cur[p][32'(j) * L + i];
            esc_mask[i] = (code[i] == '0);
            slot[i]     = need;                   // escapes in lanes 0..i-1
            need        = need + PW'(esc_mask[i]);
        end
    end

    genvar gi;
    generate
        for (gi = 0; gi < L; gi++) begin : g_lane
            logic is_esc_unused;
            ew_lane #(.CODE_BITS(CODE_BITS)) u_lane (
                .code     (code[gi]),
                .sm       (sm[8*gi +: 8]),
                .base     (base_use),
                .esc_byte (win[8*slot[gi] +: 8]),
                .w        (dec_w[16*gi +: 16]),
                .is_esc   (is_esc_unused)
            );
        end
    endgenerate

    // ------------------------------------------------------------- issue
    logic dec_ok, iv, i_ready, fire;
    logic [JW-1:0] out_j;                         // chunk index of out_w (for testbenches)
    assign dec_ok     = cur_valid && smr_valid && (CW'(need) <= wcnt);
    assign iv         = bypass ? smr_valid : dec_ok;
    assign i_ready    = !out_valid || out_ready;  // output stage empty or being read
    assign fire       = iv && i_ready;
    assign smr_ready  = fire && (bypass || h);
    assign stall_esc  = i_ready && !bypass && cur_valid && smr_valid && (CW'(need) > wcnt);
    assign stall_data = i_ready && !iv && !stall_esc;

    always_ff @(posedge clk) begin
        if (!rst_n || clear) begin
            out_valid <= 1'b0;
        end else if (i_ready) begin
            out_valid <= iv;
            out_w     <= bypass ? smr_data : dec_w;
            out_nesc  <= bypass ? '0 : need;
            out_j     <= j;
        end
    end

    assign load_cur   = stg_full && (!cur_valid || (fire && !bypass && last_chunk));
    assign esc_ready  = !bypass && (32'(wcnt) + 8 <= WB);

    // escape window: drop what this chunk used, append a word if one came in
    logic [PW-1:0]   pop;
    logic [CW-1:0]   cnt1;
    logic [8*WB-1:0] win_n;
    logic [8*WB-1:0] esc_ext;
    assign esc_ext = {{(8*WB-64){1'b0}}, esc_data};
    always_comb begin
        pop   = (fire && !bypass) ? need : '0;
        cnt1  = wcnt - CW'(pop);
        win_n = win >> (8 * pop);
        if (esc_valid && esc_ready)
            win_n = win_n | (esc_ext << (8 * cnt1));
    end

    always_ff @(posedge clk) begin
        if (!rst_n || clear) begin
            stg_cnt      <= '0;
            stg_has_base <= 1'b0;
            cur_valid    <= 1'b0;
            j            <= '0;
            h            <= 1'b0;
            tbn          <= '0;
            win          <= '0;
            wcnt         <= '0;
        end else begin
            if (code_valid && code_ready) begin
                stg[stg_cnt] <= code_data;
                stg_cnt      <= stg_cnt + 1'b1;
            end
            if (tb_valid && tb_ready) begin
                tbw <= tb_data;
                tbn <= 4'd8;
            end else if (take_base) begin
                tbw <= tbw >> 8;
                tbn <= tbn - 1'b1;
            end
            if (take_base) begin
                stg_base     <= tbw[7:0];
                stg_has_base <= 1'b1;
            end

            if (fire && !bypass) begin
                j <= last_chunk ? '0 : j + 1'b1;
                h <= ~h;
            end
            if (load_cur) begin
                for (int p = 0; p < CODE_BITS; p++) cur[p] <= stg[p];
                cur_base     <= stg_base;
                cur_valid    <= 1'b1;
                stg_cnt      <= '0;
                stg_has_base <= 1'b0;
            end else if (fire && !bypass && last_chunk) begin
                cur_valid <= 1'b0;
            end

            win  <= win_n;
            wcnt <= cnt1 + ((esc_valid && esc_ready) ? CW'(8) : CW'(0));
        end
    end

`ifndef SYNTHESIS
    initial begin
        if (64 % L != 0)  $fatal(1, "weight_decoder: L must divide 64");
        if (L < 4)        $fatal(1, "weight_decoder: L must be at least 4");
    end
`endif

endmodule
