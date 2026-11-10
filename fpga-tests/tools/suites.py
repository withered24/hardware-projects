"""suites.py: one entry per FPGA bitstream.

Each suite is a DUT (top module + parameters) and the repo regressions whose
simulator builds use exactly that configuration. record.py logs traces only
for builds whose name matches `builds`; other builds in the same pytest
function still run (and must pass) but are not replayed.

v1 = the RTL in this repo's design folders (what is published).
v2 = fpga-tests/v2/<design>: the unreleased v2 RTL + its testbenches
     (GPU register files and engine buffers in sram_2p, which the FPGA
     build maps to block RAM with `define SYNTHESIS).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # hardware-projects/
V2 = ROOT / "fpga-tests" / "v2"

CORE_RTL = [f"rtl/{f}.sv" for f in ("rv32i_pkg", "regfile", "immgen", "alu", "branch_unit",
                                   "decoder", "csr", "core")]
ACCEL_RTL = ["rtl/accel_pkg.sv", "rtl/pe.sv", "rtl/systolic_array.sv", "rtl/dpram64.sv",
             "rtl/accel_core.sv", "rtl/accel_axil.sv"]
GPU_RV = [f"riscv-core/rtl/{f}.sv" for f in ("rv32i_pkg", "alu", "branch_unit", "decoder", "immgen")]
GPU_V1_RTL = GPU_RV + [f"rtl/{f}.sv" for f in ("gpu_pkg", "lsu", "simt_core", "gpu_mem", "gpu_top")]
GPU_V2_RTL = GPU_RV + [f"rtl/{f}.sv" for f in ("gpu_pkg", "sram_2p", "simt_lane", "lsu", "simt_core",
                                              "gpu_mem", "gpu_top")]
EW_V1_RTL = [f"rtl/{f}.sv" for f in ("ew_pkg", "bf16_mul", "fp32_add", "ew_lane", "sync_fifo",
                                    "axi_rd_stream", "axi_wr_stream", "weight_decoder", "gemv_core",
                                    "ew_ctrl", "ew_top")]
EW_V2_RTL = EW_V1_RTL[:4] + ["rtl/sram_2p.sv"] + EW_V1_RTL[4:]

SUITES = {
    # RV32I pipelined core. The single-cycle teaching core (core_single) is not included.
    "riscv": dict(
        repo=ROOT / "riscv-core", top="core", sources=CORE_RTL, dut="u_core", patch="riscv",
        builds=r"tb_top_(pipeline|wait)|tb_top_pipeline_hazards|tb_top_random_(pipeline|wait)",
        env={"REGRESS_CORES": "pipeline,wait", "RAND_PROGRAMS": "100"},
        tests=["tb/test_core.py", "tb/test_pipeline.py", "tb/test_random.py"]),
    # The same core built with RESET_PC = 0x8000_0000 for the official riscv-tests (rv32ui etc.).
    "riscv_rvtests": dict(
        repo=ROOT / "riscv-core", top="core", params={"RESET_PC": "32'h80000000"}, sources=CORE_RTL,
        dut="u_core", patch="riscv", builds=r"tb_top_rvtests_(pipeline|wait)",
        env={"REGRESS_CORES": "pipeline,wait"}, tests=["tb/test_riscv_tests.py"]),
    "systolic_array": dict(
        repo=ROOT / "systolic-accel", top="systolic_array", params={"N": 8, "TAGW": 8},
        sources=["rtl/pe.sv", "rtl/systolic_array.sv"], builds=r"systolic_array",
        tests=["tb/test_array.py"]),
    # The accelerator behind its AXI4-Lite slave (the path the friend's smoke test skipped).
    "accel_axil": dict(
        repo=ROOT / "systolic-accel", top="accel_axil", sources=ACCEL_RTL, builds=r"accel_axil",
        tests=["tb/test_accel.py"]),
    "gpu_v1": dict(
        repo=ROOT / "simt-gpu", top="gpu_top", sources=GPU_V1_RTL, patch="gpu",
        builds=r"gpu_top|bench_gpu8", tests=["tb/test_gpu.py::test_gpu", "tb/test_bench.py"]),
    "gpu_v2": dict(
        repo=V2 / "simt-gpu", top="gpu_top", sources=GPU_V2_RTL, patch="gpu",
        builds=r"gpu_top|bench_gpu8", tests=["tb/test_gpu.py::test_gpu", "tb/test_bench.py"]),
    # Engine, default config: L = 8 lanes, 3-bit codes, per-matrix base.
    "ew_v1": dict(
        repo=ROOT / "ew-decoder", top="ew_top", params={"L": 8, "CODE_BITS": 3, "PER_TILE_BASE": 0},
        sources=EW_V1_RTL, dut="u_dut", builds=r"top_l8_b3",
        tests=["tb/test_top.py::test_top_default"]),
    # v2 as laid out: SRAM = 1 (x buffer and FIFOs in sram_2p).
    "ew_v2": dict(
        repo=V2 / "ew-decoder", top="ew_top",
        params={"L": 8, "CODE_BITS": 3, "PER_TILE_BASE": 0, "SRAM": 1},
        sources=EW_V2_RTL, dut="u_dut", builds=r"top_l8_b3_sram",
        tests=["tb/test_top.py::test_top_sram"]),
}

# FPGA-only patches: patches/<dir>/<path relative to the suite's repo>
# replaces that source file in the FPGA build (never in simulation
# recording); patches/<dir>/extra/*.sv are added before the RTL. Every
# patched file has a .diff beside it explaining the change.
PATCHES = Path(__file__).resolve().parents[1] / "patches"
PATCH_DIRS = {"riscv": "riscv", "riscv_rvtests": "riscv", "gpu_v1": "gpu_v1", "gpu_v2": "gpu_v2"}


def sources_for(name: str, patched: bool = True) -> list[Path]:
    s = SUITES[name]
    pd = PATCHES / PATCH_DIRS.get(name, "_none")
    out = []
    if patched and (pd / "extra").is_dir():
        out += sorted((pd / "extra").glob("*.sv"))
    for rel in s["sources"]:
        p = pd / rel
        out.append(p if patched and p.exists() else Path(s["repo"]) / rel)
    return out

# FPGA build settings per suite. aw: trace RAM depth 2**aw entries per chunk
# (bounded by M9K blocks: 2**aw/256 blocks per 32-bit RAM). sideband: extra
# DUT inputs logged by fpga_rec after the ports (see patches/). harness_ports:
# extra DUT ports of an FPGA-patched DUT, wired to harness signals.
FPGA = {
    "riscv":          dict(aw=12, sideband=["rf_clear"]),
    "riscv_rvtests":  dict(aw=12, sideband=["rf_clear"]),
    "systolic_array": dict(aw=10),
    "accel_axil":     dict(aw=12),
    "gpu_v1":         dict(aw=11, harness_ports={"fpga_fclk": "clk", "fpga_ph": "ph"}),
    "gpu_v2":         dict(aw=11, harness_ports={"fpga_fclk": "clk", "fpga_ph": "ph"}),
    "ew_v1":          dict(aw=11),
    "ew_v2":          dict(aw=10),
}
