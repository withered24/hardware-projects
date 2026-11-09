# DE2-115 FPGA smoke tests

Source commit [bf8f6ea](https://github.com/withered24/hardware-projects/commit/bf8f6ea964ae446341fb31d527e6ceb5699bc3af). Tests ran on a Terasic DE2-115 (Cyclone IV E EP4CE115F29C7) using Quartus Prime Lite 25.1std.0 build 1129. The harnesses derived a 12.5 MHz test clock from the board's 50 MHz oscillator. Each test used on-chip stimulus and checked results over USB-Blaster JTAG.

| Design | Directed check | Hardware result |
| --- | --- | --- |
| Pipelined RV32I core | A 14-word program exercised integer arithmetic, a store/load dependency, taken and untaken branches, a shift, and XOR. | Passed. The core wrote the expected signature `0x0000002B` after storing `0x00000004` in scratch memory. |
| 8×8 systolic array | Eight tagged signed-int8 input vectors ran back-to-back across both weight banks. | Passed. All 64 output sums matched independently calculated integer results. |
| Full BF16 decoder/GEMV engine (`ew_top`) | A 4×16 matrix ran through the command port and the AXI stream interfaces in compressed and raw BF16 modes. | Passed. Both modes returned the exact FP32 values `[174, -133, -203, -26]`. The compressed run reported all 18 expected escapes. |
| Two-core SIMT GPU | A kernel ran two 32-thread blocks with divergent branches and coalesced loads and stores. | Passed. All 64 output words matched. The GPU reported 672 thread instructions and 16 coalesced memory transactions over 69 kernel cycles. |

## JTAG readback

Each status word matched on two consecutive reads of the programmed FPGA.

| Test image | Status word |
| --- | --- |
| RV32I core and systolic array | `DE210000002B04FF85` |
| Decoder/GEMV engine | `E8240412001C001B01` |
| SIMT GPU | `6A244002A000100459` |

## Test scope

The CPU and systolic test used the upstream RTL. The engine used its default eight lanes and a synthesizable AXI memory fixture. The test drove the engine command port directly. It did not use the optional CPU integration or external DDR. The systolic test covered the array datapath. It did not exercise the accelerator's AXI-Lite controller.

The FPGA GPU build kept the two cores, four warps per core, eight lanes per warp, and 64 KiB of global memory. For Quartus, it used banked synchronous register-file RAM and a single-port global memory shared by the GPU and the idle host. Host reads take one registered cycle. Quartus-specific generate syntax and a synthesizable warp popcount were applied to the local test copy. The original and adapted GPU produced the same signature in Questa for this kernel.

These are directed tests at 12.5 MHz. They do not measure maximum frequency, DDR bandwidth, power, or performance on the full benchmark suites.
