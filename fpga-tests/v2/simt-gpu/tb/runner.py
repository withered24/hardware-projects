"""Shared cocotb runner: build an RTL top level with Icarus and run a test module.

Each tb/test_*.py ends with a pytest function that calls run(), so
`pytest tb/` from the repo root runs every cocotb test.
"""

import os
from pathlib import Path

from cocotb_tools.runner import get_runner

REPO = Path(__file__).resolve().parent.parent

# riscv-core's building blocks, reused unchanged, then the GPU itself
GPU_SOURCES = [f"riscv-core/rtl/{f}.sv" for f in ("rv32i_pkg", "alu", "branch_unit", "decoder", "immgen")] \
    + [f"rtl/{f}.sv" for f in ("gpu_pkg", "sram_2p", "simt_lane", "lsu", "simt_core", "gpu_mem", "gpu_top")]
# OpenRAM's model of the SkyWater SRAM macro the register files use (a PDK
# copy, see tb/sky130_sram), for builds with `define SKY130_SRAM
SRAM_MODELS = ["tb/sky130_sram/sky130_sram_1kbyte_1rw1r_32x256_8.v"]


def run(toplevel: str, sources: list[str], test_module: str, waves: bool = False,
        parameters: dict | None = None, build_name: str | None = None,
        extra_env: dict | None = None, testcase: str | None = None, defines: dict | None = None):
    """sources are relative to the repo root, in compile order (packages first)."""
    root = Path(os.environ.get("SIM_BUILD_ROOT", REPO / "sim_build"))
    build_dir = root / (build_name or toplevel)
    runner = get_runner("icarus")
    runner.build(
        sources=[REPO / s for s in sources],
        hdl_toplevel=toplevel,
        build_dir=build_dir,
        timescale=("1ns", "1ps"),
        parameters=parameters or {},
        defines=defines or {},
        waves=waves,
        always=True,
    )
    runner.test(
        hdl_toplevel=toplevel,
        test_module=test_module,
        test_dir=Path(__file__).resolve().parent,
        build_dir=build_dir,
        waves=waves,
        extra_env=extra_env or {},
        testcase=testcase,
        results_xml=str(build_dir / f"{test_module}.result.xml"),
    )
