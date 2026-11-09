// gpu_pkg.sv — the GPU's shape and its programming-model constants
//
// One GPU = NUM_CORES SIMT cores ("SMs" in NVIDIA's terms). Each core runs one
// thread block at a time, split into WARPS warps of LANES threads. A warp is
// the unit the hardware schedules: all its lanes share one fetch, one decode
// and one PC choice, and each lane has its own registers and its own ALU.
//
// Changing a number here reshapes the whole GPU; the testbench reads them
// back from here too (via tb/gpu.py), so they can't drift apart.

package gpu_pkg;

    localparam int NUM_CORES = 2;
    localparam int WARPS     = 4;                  // warps per core
    localparam int LANES     = 8;                  // threads per warp
    localparam int THREADS   = WARPS * LANES;      // max threads per block

    localparam int WARP_BITS = $clog2(WARPS);
    localparam int LANE_BITS = $clog2(LANES);
    localparam int RF_AW     = WARP_BITS + 5;      // a lane's register file: {warp, register}

    // Global memory is read and written in lines: one line is exactly what a
    // warp needs when its LANES threads touch LANES consecutive words.
    localparam int LINE_BYTES = 4 * LANES;         // 32 bytes
    localparam int LINE_BITS  = 8 * LINE_BYTES;
    localparam int OFF_BITS   = $clog2(LINE_BYTES);
    localparam int MEM_BYTES  = 64 * 1024;
    localparam int MEM_LINES  = MEM_BYTES / LINE_BYTES;
    localparam int LINE_IDX   = $clog2(MEM_LINES);
    localparam int IMEM_WORDS = 4096;              // 16 KiB of kernel code

    // Read-only CSRs a thread uses to find out who it is (CUDA's threadIdx.x,
    // blockIdx.x, blockDim.x, gridDim.x) and where its kernel arguments are.
    // 0xCC0-0xCFF is the custom read-only user range of the CSR space.
    localparam logic [11:0] CSR_TID    = 12'hCC0;
    localparam logic [11:0] CSR_CTAID  = 12'hCC1;
    localparam logic [11:0] CSR_NTID   = 12'hCC2;
    localparam logic [11:0] CSR_NCTAID = 12'hCC3;
    localparam logic [11:0] CSR_ARGS   = 12'hCC4;

endpackage
