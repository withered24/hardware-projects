// layer.c — run one decoder layer's linear projections on the engine
//
// The host (the testbench now, the PS or a loader on the board later) puts
// a job table in data memory: word 0 is the job count, then one 11-word
// record per projection (q, k, v, o, gate, up, down in a real layer). Each
// record is a matrix already laid out in DRAM by sw/image.py. For every job
// this configures the streams, runs the GEMV, waits, and stores the cycle
// count and counters. Activations between projections (RoPE, attention,
// SiLU) are the host's business; this measures the weight path.

#include "ew.h"

enum { J_ROWS, J_COLS, J_BYPASS, J_BASE, J_NESC, J_CODE, J_SMR, J_ESC, J_TBASE, J_X, J_Y, J_WORDS };

#define JOBS ((volatile uint32_t *)0x8000)
#define RES  ((volatile uint32_t *)0xC000)     // 4 words per job
#define DONE (*(volatile uint32_t *)0xFFFC)

int main(void) {
    uint32_t n = JOBS[0];
    for (uint32_t i = 0; i < n; i++) {
        volatile uint32_t *j = &JOBS[1 + i * J_WORDS];
        ew_wstream_cfg(EW_CODE_ADDR, j[J_CODE]);
        ew_wstream_cfg(EW_SMR_ADDR, j[J_SMR]);
        ew_wstream_cfg(EW_ESC_ADDR, j[J_ESC]);
        ew_wstream_cfg(EW_ESC_COUNT, j[J_NESC]);
        ew_wstream_cfg(EW_TBASE_ADDR, j[J_TBASE]);
        ew_wstream_cfg(EW_X_ADDR, j[J_X]);
        ew_wstream_cfg(EW_Y_ADDR, j[J_Y]);
        ew_exp_base(j[J_BASE]);
        ew_dec_bypass(j[J_BYPASS]);
        uint32_t h = ew_gemv_run(j[J_ROWS], j[J_COLS]);
        RES[4 * i + 0] = ew_gemv_wait(h);
        RES[4 * i + 1] = ew_stat(EW_ST_ESCAPES);
        RES[4 * i + 2] = ew_stat(EW_ST_STALL_ESC);
        RES[4 * i + 3] = ew_stat(EW_ST_STALL_DATA);
    }
    DONE = 1;
    for (;;) {}
}
