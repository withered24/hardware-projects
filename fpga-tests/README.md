# FPGA regression replay (DE2-115)

The goal is to run every design's full simulation regression on a Terasic DE2-115 (Cyclone IV E EP4CE115F29C7) and show that the FPGA matches the simulator on every output, every cycle.

There are two halves, both implemented:

1. **Record** (any PC, no board). Each repo's own cocotb regression runs unchanged under Icarus. `tools/fpga_rec.py` logs the DUT's ports at every rising clock edge into `traces/<suite>/<build>.trc`.
2. **Replay** (the board). A generated Quartus top feeds the recorded inputs to the DUT one clock edge at a time and captures its outputs. A host script moves the data over JTAG and compares every output with the trace.

A pass means the design in FPGA fabric produced exactly the outputs the simulator did, for every recorded cycle where the simulator had no X.

## Run it (board plugged in)

From `hardware-projects/`, with Quartus Prime Lite 25.1 at `C:\altera_lite\25.1std` (or `QUARTUS_BIN` set), the traces recorded and the bitstreams built:

```bash
python fpga-tests/tools/run_all.py --quick     # first 2 chunks of every trace, a few minutes
python fpga-tests/tools/run_all.py             # everything
```

`run_all.py` checks the JTAG chain (`jtagconfig`), runs a selftest, then replays each suite (`replay.py hw`). It writes `results/summary.html` and `results/summary.json`, plus per-trace reports in `results/<suite>/hw/`. A suite fails on the first cycle where any checked output bit differs. The report names that cycle, the ports, and both values.

If the board isn't found: use the USB-Blaster USB port (not the USB device port), power the board on, and check that Device Manager lists the Altera USB-Blaster driver.

Single steps:

```bash
python fpga-tests/tools/replay.py selftest [suite]        # JTAG path + bitstream identity, no DUT edges
python fpga-tests/tools/replay.py hw gpu_v1 [--trace gpu_top] [--chunks N]
python fpga-tests/tools/replay.py sim gpu_v1              # same harness in Icarus, no board
```

## Prepare (no board)

```bash
source fpga-tests/../../fpga-tests/tools/env.sh   # or put OSS CAD Suite bin+lib, then Quartus, on PATH
bash fpga-tests/tools/record_all.sh               # record every suite (riscv with RAND_PROGRAMS=1000)
python fpga-tests/tools/gen.py                    # generate every harness (builds/<suite>/)
python fpga-tests/tools/replay.py sim <suite>     # prove the harness in simulation
python fpga-tests/tools/build.py                  # compile every bitstream, report fit and timing
```

`build.py` refuses to call a bitstream ready unless timing closes at every corner.

## Suites

Suites are listed in `tools/suites.py`. Each suite is one bitstream: a DUT top, its parameters, and the simulator builds that use exactly that configuration.

| suite | DUT | regressions | cycles (2026-11-09) |
|---|---|---|---|
| `riscv` | `core` (pipelined RV32I) | test_core, test_pipeline, test_random (1000 programs); plain memory and random wait states | 846,231 |
| `riscv_rvtests` | `core`, `RESET_PC = 0x8000_0000` | official riscv-tests, both memory models | 37,210 |
| `systolic_array` | `systolic_array`, N = 8 | test_array | 201 |
| `accel_axil` | `accel_axil` (array + AXI4-Lite) | test_accel | 21,001 |
| `gpu_v1` / `gpu_v2` | `gpu_top` | test_gpu (all kernels + 25 random kernels), test_bench at `MEM_LATENCY = 8` | 204,528 / 230,407 |
| `ew_v1` | `ew_top`, L = 8, 3-bit codes | test_top_default | 65,655 |
| `ew_v2` | `ew_top`, L = 8, 3-bit codes, `SRAM = 1` | test_top_sram | 65,746 |

`v1` is the RTL in this repo's design folders. `v2/` holds the unreleased v2 RTL and its testbenches. Builds define `` `SYNTHESIS `` so v2's `sram_2p` is a plain synchronous-read RAM (M9K). `SKY130_SRAM` (the ASIC macros) is never defined.

## How it works

### Record

`traces/<suite>/ports.json` lists the DUT ports (Yosys-slang, after parameters). Each line of a `.trc` file is one rising edge of the DUT clock:

```
<inputs hex> <outputs hex> <output X-mask hex>
```

The input vector is every input port except `clk`, in `ports.json` order with the first port in the most significant bits. Sideband bits follow, at the least significant end. Inputs that were X or Z in simulation are logged as 0. All values are as they stood just before the edge, which is what the flops sample. The outputs are compared only where the mask bit is 0.

When a test starts the clock, it writes its inputs (reset, memories) in the same instant as the clock's first rising edge. Outputs that depend combinationally on those inputs are then read before they settle, so the recorder masks all outputs at that one edge. The DUT's state after the edge is still well defined, and every later edge is checked.

On Windows, put OSS CAD Suite's `bin` and `lib` ahead of the RISC-V GCC on PATH. The other way round, Yosys loads the GCC's DLLs and fails with no message.

### Sideband bits: testbench backdoors

The FPGA can only see ports, so testbench writes into the DUT's internals are handled like this:

- **riscv:** `cosim.load_and_reset` zeroes the register file through the hierarchy. This is logged as sideband bit `rf_clear`, and the FPGA copy of `regfile.sv` clears all registers at that edge (`patches/riscv/`).
- **gpu:** the tests' backdoor loads and readbacks are rerouted through the GPU's own host ports. Code goes in through `imem_we` and data through `dmem_we`, before each launch. After each kernel, every word that changed is read back through `dmem_addr`/`dmem_rdata`, so the results appear at ports the FPGA checks.

### Replay harness (`tools/gen.py`, `fpga/rtl/`)

- `replay_top.sv` (generated) has `CLOCK_50` as the base clock, the sequencer, the trace RAMs, the gated DUT clock and the DUT. On the LEDs: LEDG0 is a heartbeat, LEDG1 means busy, LEDG3 means idle.
- **Trace RAMs** (`rt_ram.sv`) are 32-bit M9K RAMs with `ENABLE_RUNTIME_MOD`, named `S00..` (stimulus), `R00..` (responses), `CTRL` and `STAT`. They are reached by name through the In-System Memory Content Editor. Each chunk is 2^aw entries (`suites.py`).
- **Sequencer** (`replay_seq.sv`) runs 8 base cycles per DUT edge. The phases go as follows:
  - Phase 0 addresses entry *i*.
  - At the end of phase 1, the stimulus goes onto the DUT inputs.
  - Phase 7 gates exactly one clock edge through a `cycloneive_clkctrl` (`dclk_gate.sv`).
  - At that same base edge, the DUT outputs from just before its edge go into response entry *i*.

  Between chunks the DUT gets no edges and its inputs hold, so state carries over exactly.
- **Host protocol** (`replay.tcl`, under `quartus_stp`):
  1. Write the stimulus.
  2. Write `CTRL[1]` = count, then a new token to `CTRL[0]`.
  3. Wait for `STAT[0]` to echo the token.
  4. Check that `STAT[1]` (the DUT edge counter) advanced by exactly the count.
  5. Read the responses.

  Before any of that, it reads `STAT[2]` (magic `5EB1A701`) and `STAT[3]` (the suite id, a CRC of the port layout), so a stale or wrong bitstream is refused. It also calibrates the memory editor's word order once at start.
- **Timing** (`replay.sdc`, generated) has multicycle constraints for DUT to DUT (8 base cycles) and for the paths between the DUT and the base-clock logic. `build.py` checks for zero negative slack at every corner.
- **Simulation** of the same harness (`replay.py sim`) loads the chunk files straight into the RAM arrays and runs the whole flow in Icarus. It's the check that the harness, the patches and the trace agree, before the board is involved.

### FPGA-only adaptations (`patches/`)

Each is a full copy of the original file plus a `.diff`. All keep the ports cycle-identical.

- **riscv `regfile.sv` / `core.sv`:** the `rf_clear` sideband input (see above).
- **GPU `gpu_mem.sv` (v1, v2):** the 64 KiB global memory is read combinationally twice per cycle and written twice per edge, which no block RAM supports, and as flops it doesn't fit. On the FPGA it is 32 byte-wide true-dual-port M9K RAMs on the harness base clock. The original's writes (core line, host word) are captured at the DUT edge and performed at phases 0 and 1. Both reads happen at phase 4 and are held from phase 5, well before the next DUT edge. Every read sees every write of earlier edges, as in the original.
- **GPU `gpu_top.sv` (v1, v2):** one instruction-memory copy per core (block RAMs have one read port), with read-old-data, as before. It also threads `fpga_fclk`/`fpga_ph` through.
- **v1 `simt_core.sv`:** each lane's register file (128 × 32) becomes a true-dual-port M9K on the same phase scheme. X's write goes on port A, the LSU's on port B, and rs1/rs2 are read at phase 4. v2 already has its register files in `sram_2p`.
- **Quartus syntax gaps** (`tools/qcompat.py`, applied to every source on the way into `builds/<suite>/src/`): it wraps bare generate constructs, moves `for (genvar ...)` declarations out of the loop, and maps `$countones` to `qcompat_pkg::qc_popcount`. It keeps line numbers.

## Not covered by replay

These run natively, self-checking, at speed. They are not prepared yet:

- systolic-accel SoC: `soc/soc_top.sv` running `test_gemm` and `mnist` (`tb/test_soc.py`).
- ew-decoder SoC: `tb_soc` running the CPU driver program (`tb/test_cpu.py`).
