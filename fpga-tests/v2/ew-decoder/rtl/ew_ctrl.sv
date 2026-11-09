// ew_ctrl.sv — custom-instruction front end and job sequencer
//
// The CPU side is one request/response handshake per custom instruction:
// cmd_valid with funct3/rs1/rs2; the instruction completes in the cycle
// cmd_ready is high, and cmd_rdata (combinational) is its rd result. The
// core holds the instruction in MEM while cmd_ready is low, which is how
// GEMV.WAIT blocks. Encodings: docs/ARCH.md §3.
//
// A job: GEMV.RUN latches rows/cols, then one cycle of `clear` (empties every
// FIFO), one cycle of `start` (each enabled reader/writer latches its address
// and beat count), then RUN until the whole datapath is idle again. Config
// writes wait for the job to finish, so nothing changes under a running job.

import ew_pkg::*;

module ew_ctrl #(
    parameter int AW            = 32,
    parameter int L             = 8,
    parameter int CODE_BITS     = 3,
    parameter int PER_TILE_BASE = 0
) (
    input  logic          clk,
    input  logic          rst_n,

    input  logic          cmd_valid,
    output logic          cmd_ready,
    input  logic [2:0]    cmd_funct3,
    input  logic [31:0]   cmd_rs1,
    input  logic [31:0]   cmd_rs2,
    output logic [31:0]   cmd_rdata,

    output logic          bypass,
    output logic [7:0]    base,
    output logic          clear,
    output logic          start,
    output logic [15:0]   rows,
    output logic [15:0]   cols,

    output logic          en_code, en_smr, en_esc, en_tb,
    output logic [AW-1:0] a_code, a_smr, a_esc, a_tb, a_x, a_y,
    output logic [31:0]   n_code, n_smr, n_esc, n_tb, n_x, n_y,

    input  logic          dp_busy,
    input  logic          chunk_fire,
    input  logic [$clog2(L+1)-1:0] chunk_nesc,
    input  logic          stall_esc,
    input  logic          stall_data
);

    typedef enum logic [1:0] {S_IDLE, S_CLEAR, S_START, S_RUN} state_t;
    state_t state;

    logic [31:0] cfg [N_CFG];
    logic [31:0] handle;
    logic [31:0] st_cycles, st_chunks, st_stall_esc, st_stall_data, st_escapes;

    logic idle;
    assign idle  = (state == S_IDLE);
    assign clear = (state == S_CLEAR);
    assign start = (state == S_START);

    // ------------------------------------------------------- commands
    always_comb begin
        case (cmd_funct3)
            F3_GEMV_WAIT: cmd_ready = idle || (cmd_rs1 != handle);
            F3_EW_STAT:   cmd_ready = 1'b1;
            default:      cmd_ready = idle;
        endcase
        case (cmd_funct3)
            F3_GEMV_RUN:  cmd_rdata = handle + 1'b1;
            F3_GEMV_WAIT: cmd_rdata = st_cycles;
            F3_EW_STAT: case (cmd_rs1)
                ST_CYCLES:     cmd_rdata = st_cycles;
                ST_CHUNKS:     cmd_rdata = st_chunks;
                ST_STALL_ESC:  cmd_rdata = st_stall_esc;
                ST_STALL_DATA: cmd_rdata = st_stall_data;
                ST_ESCAPES:    cmd_rdata = st_escapes;
                default:       cmd_rdata = 32'd0;
            endcase
            default:      cmd_rdata = 32'd0;
        endcase
    end

    // -------------------------------------------- stream lengths (beats)
    logic [31:0] nweights, ntiles;
    assign nweights = 32'(rows) * 32'(cols);
    assign ntiles   = nweights >> 6;

    assign en_code = !bypass;
    assign en_esc  = !bypass && (cfg[CFG_ESC_COUNT] != 0);
    assign en_tb   = !bypass && (PER_TILE_BASE != 0);
    assign en_smr  = 1'b1;

    assign a_code = AW'(cfg[CFG_CODE_ADDR]);
    assign a_smr  = AW'(cfg[CFG_SMR_ADDR]);
    assign a_esc  = AW'(cfg[CFG_ESC_ADDR]);
    assign a_tb   = AW'(cfg[CFG_TBASE_ADDR]);
    assign a_x    = AW'(cfg[CFG_X_ADDR]);
    assign a_y    = AW'(cfg[CFG_Y_ADDR]);

    assign n_code = ntiles * CODE_BITS;                         // 64-bit words
    assign n_smr  = (bypass ? 2 * nweights : nweights) / (2 * L); // 2L-byte words
    assign n_esc  = (cfg[CFG_ESC_COUNT] + 7) >> 3;
    assign n_tb   = (ntiles + 7) >> 3;
    assign n_x    = 32'(cols) >> 2;
    assign n_y    = 32'(rows) >> 1;

    // ------------------------------------------------------------- FSM
    logic cmd_fire;
    assign cmd_fire = cmd_valid && cmd_ready;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state  <= S_IDLE;
            handle <= '0;
            bypass <= 1'b0;
            base   <= '0;
            rows   <= '0;
            cols   <= '0;
            for (int i = 0; i < N_CFG; i++) cfg[i] <= '0;
            st_cycles <= '0; st_chunks <= '0; st_stall_esc <= '0; st_stall_data <= '0; st_escapes <= '0;
        end else begin
            if (cmd_fire) begin
                case (cmd_funct3)
                    F3_EXP_BASE:    base <= cmd_rs1[7:0];
                    F3_WSTREAM_CFG: if (cmd_rs1 < N_CFG) cfg[cmd_rs1] <= cmd_rs2;
                    F3_DEC_BYPASS:  bypass <= cmd_rs1[0];
                    F3_GEMV_RUN: begin
                        rows   <= cmd_rs1[15:0];
                        cols   <= cmd_rs2[15:0];
                        handle <= handle + 1'b1;
                        state  <= S_CLEAR;
                    end
                    default: ;
                endcase
            end
            case (state)
                S_CLEAR: begin
                    st_cycles <= '0; st_chunks <= '0; st_stall_esc <= '0; st_stall_data <= '0; st_escapes <= '0;
                    state <= S_START;
                end
                S_START: state <= S_RUN;
                S_RUN: begin
                    st_cycles <= st_cycles + 1'b1;
                    if (chunk_fire) begin
                        st_chunks  <= st_chunks + 1'b1;
                        st_escapes <= st_escapes + 32'(chunk_nesc);
                    end
                    if (stall_esc)  st_stall_esc  <= st_stall_esc + 1'b1;
                    if (stall_data) st_stall_data <= st_stall_data + 1'b1;
                    if (!dp_busy) state <= S_IDLE;
                end
                default: ;
            endcase
        end
    end

endmodule
