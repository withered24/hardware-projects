"""Shared cocotb runner: build an RTL top-level with Icarus and run a test module.

Same pattern as riscv-core: every tb/test_*.py ends with a pytest function
that calls run(), so `pytest tb/` from the repo root runs the regression.
"""

import os
import sys
from pathlib import Path

from cocotb_tools.runner import get_runner

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sw"))         # tests import ewformat / golden

RTL = ["rtl/ew_pkg.sv", "rtl/bf16_mul.sv", "rtl/fp32_add.sv", "rtl/ew_lane.sv",
       "rtl/sram_2p.sv", "rtl/sync_fifo.sv", "rtl/axi_rd_stream.sv", "rtl/axi_wr_stream.sv",
       "rtl/weight_decoder.sv", "rtl/gemv_core.sv", "rtl/ew_ctrl.sv", "rtl/ew_top.sv"]
TB_TOP = RTL + ["tb/axi_mem.sv", "tb/tb_top.sv"]
# OpenRAM's models of the SkyWater SRAM macros (PDK copies, VERBOSE off), for
# builds with `define SKY130_SRAM: sram_2p then instantiates the real macros
SRAM_MODELS = ["tb/sky130_sram/sky130_sram_1kbyte_1rw1r_32x256_8.v",
               "tb/sky130_sram/sky130_sram_2kbyte_1rw1r_32x512_8.v"]
CPU = ["rtl/cpu/rv32i_pkg.sv", "rtl/cpu/alu.sv", "rtl/cpu/branch_unit.sv", "rtl/cpu/decoder.sv",
       "rtl/cpu/immgen.sv", "rtl/cpu/regfile.sv", "rtl/cpu/csr.sv", "rtl/cpu/core.sv"]
TB_SOC = CPU + TB_TOP + ["tb/sim_mem.sv", "tb/tb_soc.sv"]


def run(toplevel: str, sources: list[str], test_module: str, waves: bool = False,
        parameters: dict | None = None, build_name: str | None = None,
        defines: dict | None = None, extra_env: dict | None = None,
        plusargs: list[str] | None = None, testcase: str | None = None):
    """sources are paths relative to the repo root, in compile order.
    build_name keeps differently-configured builds of one top level apart."""
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
        plusargs=plusargs or [],
        testcase=testcase,
        extra_env={"PYTHONPATH": str(REPO / "sw"), **(extra_env or {})},
        results_xml=str(build_dir / f"{test_module}.result.xml"),
    )
