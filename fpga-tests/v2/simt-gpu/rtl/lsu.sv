// lsu.sv — load/store unit with a coalescer
//
// A warp's load or store is LANES separate addresses. Global memory moves
// whole lines (LINE_BYTES, 32 bytes), so the coalescer groups the lanes by
// line: each request serves every pending lane whose address falls in the
// same line as the lowest pending lane. Eight threads reading a[i..i+7]
// cost ONE request; eight threads reading a[8*i] cost eight. That is the
// whole reason CUDA code makes neighbouring threads touch neighbouring
// words, and it's visible in gpu_top's mem_lines counter.
//
// Memory instructions wait in a queue (one slot per warp: a warp can't have
// two in flight). The unit sends one request per cycle, up to TAGS of them
// in flight, without waiting for the answers. Each request carries a tag
// (warp, destination register, lanes, byte offsets) that comes back with
// its response, in order. When the last request of an instruction returns,
// the warp is told it's done and may issue again. Stores get a response too
// (an acknowledgement), so "done" always means "visible in memory".
//
// Loads support LB/LH/LW/LBU/LHU and stores SB/SH/SW. Misaligned accesses
// are not detected; their result is undefined. When several lanes store to
// the same bytes in one instruction, the highest lane wins.

import gpu_pkg::*;

module lsu #(
    parameter int TAGS = 16                  // requests in flight
) (
    input  logic                   clk,
    input  logic                   rst,

    // a memory instruction from X
    input  logic                   push,
    input  logic [WARP_BITS-1:0]   push_warp,
    input  logic [LANES-1:0]       push_mask,
    input  logic [LANES-1:0][31:0] push_addr,
    input  logic [LANES-1:0][31:0] push_wdata,
    input  logic [2:0]             push_funct3,
    input  logic [4:0]             push_rd,
    input  logic                   push_store,

    // global memory
    output logic                   req_valid,
    input  logic                   req_ready,
    output logic                   req_we,
    output logic [LINE_IDX-1:0]    req_line,
    output logic [LINE_BITS-1:0]   req_wdata,
    output logic [LINE_BYTES-1:0]  req_be,
    input  logic                   resp_valid,
    input  logic [LINE_BITS-1:0]   resp_rdata,

    // load data for the register files
    output logic                   wb_valid,
    output logic [WARP_BITS-1:0]   wb_warp,
    output logic [4:0]             wb_rd,
    output logic [LANES-1:0]       wb_mask,
    output logic [LANES-1:0][31:0] wb_data,

    // an instruction finished: its warp may issue again
    output logic                   done_valid,
    output logic [WARP_BITS-1:0]   done_warp,
    output logic [LANES-1:0]       done_mask
);

    localparam int TAG_BITS = $clog2(TAGS);

    // ---------------------------------------------------------------- instruction queue
    logic [WARPS-1:0][WARP_BITS-1:0]     q_warp;
    logic [WARPS-1:0][LANES-1:0]         q_mask;
    logic [WARPS-1:0][LANES-1:0][31:0]   q_addr, q_wdata;
    logic [WARPS-1:0][2:0]               q_funct3;
    logic [WARPS-1:0][4:0]               q_rd;
    logic [WARPS-1:0]                    q_store;
    logic [WARP_BITS-1:0]                q_head, q_tail;
    logic [WARP_BITS:0]                  q_count;

    // lanes of the head instruction still to be sent (all of q_mask until
    // its first request goes out)
    logic             started;
    logic [LANES-1:0] sent_left, pending, group, rest;
    assign pending = started ? sent_left : q_mask[q_head];

    // ---------------------------------------------------------------- tag FIFO
    logic [TAGS-1:0][WARP_BITS-1:0]          t_warp;
    logic [TAGS-1:0][4:0]                    t_rd;
    logic [TAGS-1:0][2:0]                    t_funct3;
    logic [TAGS-1:0][LANES-1:0]              t_group, t_mask;
    logic [TAGS-1:0][LANES-1:0][OFF_BITS-1:0] t_off;
    logic [TAGS-1:0]                         t_store, t_last;
    logic [TAG_BITS-1:0]                     t_head, t_tail;
    logic [TAG_BITS:0]                       t_count;

    // ---------------------------------------------------------------- coalescer
    logic [LANES-1:0][31:0]   h_addr, h_wdata;
    logic [2:0]               h_funct3;
    logic [LANE_BITS-1:0]     first;
    logic [31-OFF_BITS:0]     line;
    logic [OFF_BITS-1:0]      byte_at;
    logic [31:0]              la, wd;              // one lane's address / store data
    logic [LANES-1:0][OFF_BITS-1:0] h_off;

    assign h_addr   = q_addr[q_head];
    assign h_wdata  = q_wdata[q_head];
    assign h_funct3 = q_funct3[q_head];

    always_comb begin
        first = '0;
        for (int l = LANES - 1; l >= 0; l--)
            if (pending[l]) first = LANE_BITS'(l);
        la   = h_addr[first];
        line = la[31:OFF_BITS];
        for (int l = 0; l < LANES; l++) begin
            la       = h_addr[l];
            group[l] = pending[l] && la[31:OFF_BITS] == line;
            h_off[l] = la[OFF_BITS-1:0];
        end
        rest = pending & ~group;

        // place each lane's store data in the line, highest lane last
        // (byte_at and wd get defaults so no path holds an old value: in
        // simulation that's harmless, in synthesis it's a latch)
        req_wdata = '0;
        req_be    = '0;
        byte_at   = '0;
        wd        = '0;
        for (int l = 0; l < LANES; l++)
            if (group[l]) begin
                byte_at = h_off[l];
                wd      = h_wdata[l];
                case (h_funct3[1:0])
                    2'b00: begin                                   // SB
                        req_be[byte_at]            = 1'b1;
                        req_wdata[byte_at*8 +: 8]  = wd[7:0];
                    end
                    2'b01: begin                                   // SH
                        req_be[byte_at +: 2]       = 2'b11;
                        req_wdata[byte_at*8 +: 16] = wd[15:0];
                    end
                    default: begin                                 // SW
                        req_be[{byte_at[OFF_BITS-1:2], 2'b00} +: 4]       = 4'hF;
                        req_wdata[{byte_at[OFF_BITS-1:2], 2'b00}*8 +: 32] = wd;
                    end
                endcase
            end
    end

    assign req_valid = q_count != 0 && t_count != TAGS;
    assign req_we    = q_store[q_head];
    assign req_line  = line[LINE_IDX-1:0];

    logic fire;
    assign fire = req_valid && req_ready;

    always_ff @(posedge clk) begin
        if (rst) begin
            q_head <= '0; q_tail <= '0; q_count <= '0;
            t_head <= '0; t_tail <= '0; t_count <= '0;
            started <= 1'b0;
        end else begin
            if (push) begin
                q_warp[q_tail]   <= push_warp;
                q_mask[q_tail]   <= push_mask;
                q_addr[q_tail]   <= push_addr;
                q_wdata[q_tail]  <= push_wdata;
                q_funct3[q_tail] <= push_funct3;
                q_rd[q_tail]     <= push_rd;
                q_store[q_tail]  <= push_store;
                q_tail <= q_tail + 1'b1;
            end
            if (fire) begin
                t_warp[t_tail]   <= q_warp[q_head];
                t_rd[t_tail]     <= q_rd[q_head];
                t_funct3[t_tail] <= h_funct3;
                t_group[t_tail]  <= group;
                t_mask[t_tail]   <= q_mask[q_head];
                t_off[t_tail]    <= h_off;
                t_store[t_tail]  <= q_store[q_head];
                t_last[t_tail]   <= rest == '0;
                t_tail <= t_tail + 1'b1;
                if (rest == '0) begin                 // instruction fully sent
                    q_head  <= q_head + 1'b1;
                    started <= 1'b0;
                end else begin
                    sent_left <= rest;
                    started   <= 1'b1;
                end
            end
            q_count <= q_count + (push ? 1'b1 : 1'b0) - (fire && rest == '0 ? 1'b1 : 1'b0);
            if (resp_valid) t_head <= t_head + 1'b1;
            t_count <= t_count + (fire ? 1'b1 : 1'b0) - (resp_valid ? 1'b1 : 1'b0);
        end
    end

    // ---------------------------------------------------------------- responses
    logic [31:0]  word, whole;
    logic [OFF_BITS-1:0] off;
    logic [LANES-1:0][OFF_BITS-1:0] offs;

    always_comb begin
        wb_valid   = resp_valid && !t_store[t_head];
        wb_warp    = t_warp[t_head];
        wb_rd      = t_rd[t_head];
        wb_mask    = t_group[t_head];
        done_valid = resp_valid && t_last[t_head];
        done_warp  = t_warp[t_head];
        done_mask  = t_mask[t_head];
        offs = t_off[t_head];
        for (int l = 0; l < LANES; l++) begin
            off   = offs[l];
            whole = resp_rdata[off[OFF_BITS-1:2]*32 +: 32];
            word  = whole >> (8 * off[1:0]);
            case (t_funct3[t_head])
                3'b000:  wb_data[l] = {{24{word[7]}},  word[7:0]};    // LB
                3'b001:  wb_data[l] = {{16{word[15]}}, word[15:0]};   // LH
                3'b100:  wb_data[l] = {24'd0, word[7:0]};             // LBU
                3'b101:  wb_data[l] = {16'd0, word[15:0]};            // LHU
                default: wb_data[l] = whole;                         // LW
            endcase
        end
    end

endmodule
