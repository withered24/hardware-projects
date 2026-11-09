// ew.h — the exponent-window engine's custom instructions, as C functions
//
// Each is one R-type instruction in the custom-0 opcode (docs/ARCH.md §3),
// emitted with GNU as's `.insn r`. GEMV.RUN and GEMV.WAIT stall the core
// (the instruction waits in MEM) until the engine can answer.

#pragma once
#include <stdint.h>

enum { EW_CODE_ADDR, EW_SMR_ADDR, EW_ESC_ADDR, EW_ESC_COUNT, EW_TBASE_ADDR, EW_X_ADDR, EW_Y_ADDR };
enum { EW_ST_CYCLES, EW_ST_CHUNKS, EW_ST_STALL_ESC, EW_ST_STALL_DATA, EW_ST_ESCAPES };

static inline void ew_exp_base(uint32_t base) {
    __asm__ volatile(".insn r CUSTOM_0, 0, 0, x0, %0, x0" :: "r"(base));
}
static inline void ew_wstream_cfg(uint32_t field, uint32_t value) {
    __asm__ volatile(".insn r CUSTOM_0, 1, 0, x0, %0, %1" :: "r"(field), "r"(value));
}
static inline uint32_t ew_gemv_run(uint32_t rows, uint32_t cols) {
    uint32_t h;
    __asm__ volatile(".insn r CUSTOM_0, 2, 0, %0, %1, %2" : "=r"(h) : "r"(rows), "r"(cols));
    return h;
}
static inline uint32_t ew_gemv_wait(uint32_t handle) {
    uint32_t cycles;
    __asm__ volatile(".insn r CUSTOM_0, 3, 0, %0, %1, x0" : "=r"(cycles) : "r"(handle) : "memory");
    return cycles;
}
static inline void ew_dec_bypass(uint32_t on) {
    __asm__ volatile(".insn r CUSTOM_0, 4, 0, x0, %0, x0" :: "r"(on));
}
static inline uint32_t ew_stat(uint32_t id) {
    uint32_t v;
    __asm__ volatile(".insn r CUSTOM_0, 5, 0, %0, %1, x0" : "=r"(v) : "r"(id));
    return v;
}
