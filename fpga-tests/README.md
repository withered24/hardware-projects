# FPGA regression replay (DE2-115)

The goal is to run every design's full simulation regression on a Terasic DE2-115 (Cyclone IV E EP4CE115F29C7) and show that the FPGA matches the simulator on every output, every cycle.

There are two halves:

1. **Record** (`tools/`, done, runs on any PC). Each repo's own cocotb regression runs unchanged under Icarus. `tools/fpga_rec.py` logs the DUT's ports at every rising clock edge into `traces/<suite>/<build>.trc`.
2. **Replay** (to build, on the machine with the board). A generated Quartus top level feeds the recorded inputs to the DUT one clock edge at a time, captures its outputs, and a host script compares them with the trace.

A pass means the design in FPGA fabric produced exactly the outputs the simulator did, for every recorded cycle where the simulator had no X.

## Record

Requirements: Python 3.12, `cocotb==2.1.*`, pytest, numpy, Icarus Verilog 12+ and Yosys with the slang plugin. OSS CAD Suite provides both. Add the RISC-V GCC (`riscv-none-elf-gcc`, `-march=rv32i_zicsr_zifencei`) for `riscv_rvtests`, and the systolic/engine SoC programs if you build them.

```bash
git submodule update --init --recursive                   # riscv-tests (riscv_rvtests suite)
python fpga-tests/tools/record.py <suite>                 # every test the suite lists
python fpga-tests/tools/record.py gpu_v2 tb/test_gpu.py   # or pytest selectors
RAND_PROGRAMS=1000 python fpga-tests/tools/record.py riscv   # shell env overrides suite defaults
COCOTB_TEST_FILTER='test_vecadd' python ...               # a subset of cocotb tests
```

On Windows, put OSS CAD Suite's `bin` and `lib` ahead of the RISC-V GCC on PATH. The other way round, Yosys loads the GCC's DLLs and fails with no message.

Traces are small. On 2026-11-09, 100 random CPU programs came to about 40k cycles, the GPU v2 benchmark kernels to 30k cycles, and the riscv-tests to under 1k cycles per test.

Suites are listed in `tools/suites.py`. Each suite is one bitstream: a DUT top, its parameters, and the simulator builds that use exactly that configuration.

| suite | DUT | regressions |
|---|---|---|
| `riscv` | `core` (pipelined RV32I), path `tb_top.u_core` | test_core, test_pipeline, test_random (100 programs); plain memory and random wait states |
| `riscv_rvtests` | `core`, `RESET_PC = 0x8000_0000` | official riscv-tests |
| `systolic_array` | `systolic_array`, N = 8 | test_array |
| `accel_axil` | `accel_axil` (array + AXI4-Lite) | test_accel |
| `gpu_v1` / `gpu_v2` | `gpu_top` | test_gpu (all kernels + 25 random kernels), test_bench at `MEM_LATENCY = 8` |
| `ew_v1` | `ew_top`, L = 8, 3-bit codes | test_top_default |
| `ew_v2` | `ew_top`, L = 8, 3-bit codes, `SRAM = 1` | test_top_sram |

`v1` is the RTL in this repo's design folders. `v2/` holds the unreleased v2 RTL and its testbenches. In v2 the GPU register files and the engine's x buffer and FIFOs live in `sram_2p`. Build v2 with `` `define SYNTHESIS `` so `sram_2p` becomes a plain synchronous-read RAM, which Quartus maps to M9K. Don't define `SKY130_SRAM`: that selects the ASIC macros.

`traces/<suite>/ports.json` lists the DUT ports (Yosys-slang, after parameters). Each line of a `.trc` file is one rising edge of the DUT clock:

```
<inputs hex> <outputs hex> <output X-mask hex>
```

The input vector is every input port except `clk`, in `ports.json` order with the first port in the most significant bits. Sideband bits follow, at the least significant end. Inputs that were X or Z in simulation are logged as 0. All values are as they stood just before the edge, which is what the flops sample. The outputs are compared only where the mask bit is 0.

### Sideband bits: testbench backdoors

The FPGA can only see ports, so testbench writes into the DUT's internals are handled like this:

- **riscv:** `cosim.load_and_reset` zeroes the register file through the hierarchy while the core is held in reset. This is logged as sideband bit `rf_clear`, the LSB of the input vector. The FPGA copy of `regfile.sv` needs a synchronous clear on that bit. Thread it through `core` as an extra input. This is the only RTL change in that suite.
- **gpu:** the tests' backdoor loads and readbacks are rerouted through the GPU's own host ports. Code goes in through `imem_we` and data through `dmem_we`, one word a cycle, before each launch. After each kernel, every word that changed is read back through `dmem_addr`/`dmem_rdata`, so the results appear at ports the FPGA checks. The tests still compare everything against the model and pass.

## Replay (to implement)

The wrapper is generated per suite from `ports.json`:

- **Stimulus and response RAMs.** Store them in M9K, readable and writable from the host over JTAG. Use the In-System Memory Content Editor (`altsyncram` with `ENABLE_RUNTIME_MOD=YES`, 32-bit-wide instances) or a JTAG-to-Avalon bridge with System Console. The stimulus is the input vector, including sideband bits. The response is the output vector.
- **Sequencer on a free-running base clock.** For entry *i*: read the stimulus, register it onto the DUT inputs, let it settle, capture the DUT outputs into response entry *i*, then give the DUT one clock edge. Gate the edge with a Cyclone IV clock-control block (`cycloneive_clkctrl` / ALTCLKCTRL `ena`), so DUT state holds while the host reloads RAM between chunks. Use 4+ base cycles per DUT edge, with matching multicycle constraints on paths into, inside and out of the DUT.
- **Control and status.** In-System Sources and Probes is enough: chunk length, start, done, and an edge counter.
- **Host loop.** Per chunk: write stimulus, start, wait for done, read the response. Then compare offline (mask applied) and report the first mismatch cycle with the port name and both values. A chunk's last edge must be followed directly by the next chunk's first entry; the DUT never gets an extra or missing edge.

Self-check the replay machinery before trusting a pass: a loopback DUT (outputs = registered inputs) must replay perfectly, and a deliberately broken DUT bit must be caught at the right cycle.

### FPGA-only adaptations

Keep these minimal. Put each one in `patches/` with the reason. Port behaviour must stay cycle-identical to the RTL.

- **`gpu_mem`:** global memory is 64 KiB with an asynchronous host read. As flops it won't fit, so it needs block RAM. Because the DUT edge comes several base cycles after the inputs settle, a RAM clocked on the base clock can serve the host read and still look combinational to the DUT. Gate its writes to the DUT edge.
- **`gpu_top` imem:** one write port and one registered read port per core. Duplicate it per core in M9K.
- **v1 GPU register files:** need block-RAM banking (as done for the PR #1 smoke test). v2 already has them in `sram_2p`.
- **Quartus syntax gaps:** for example `$countones`, or generate syntax.

## Not covered by replay

Run these natively, self-checking, at speed:

- systolic-accel SoC: `soc/soc_top.sv` running `test_gemm` and `mnist` (`tb/test_soc.py`).
- ew-decoder SoC: `tb_soc` running the CPU driver program (`tb/test_cpu.py`).

Also report, for every suite, the Quartus fit: LEs, registers, M9K, multipliers, and TimeQuest fmax of the DUT at the Slow 1200 mV 85 °C and 0 °C models, with the critical path.
