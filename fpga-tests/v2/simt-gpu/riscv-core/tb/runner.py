"""Shared cocotb runner: build an RTL top-level with Icarus and run a test module.

Every tb/test_*.py ends with a small pytest function that calls run(), so
`pytest tb/` from the repo root runs the whole regression.
"""

import os
from pathlib import Path

from cocotb_tools.runner import get_runner

REPO = Path(__file__).resolve().parent.parent


def run(toplevel: str, sources: list[str], test_module: str, waves: bool = True,
        parameters: dict | None = None, build_name: str | None = None,
        defines: dict | None = None, extra_env: dict | None = None):
    """sources are paths relative to the repo root, in compile order
    (packages first). build_name keeps differently-configured builds of the
    same top level apart. extra_env is passed to the simulation (cocotb
    tests read it with os.environ).

    SIM_BUILD_ROOT moves the build directories (and their result files)
    somewhere else, so tb/mutate.py can run next to a normal regression."""
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
        results_xml=str(build_dir / f"{test_module}.result.xml"),
    )
