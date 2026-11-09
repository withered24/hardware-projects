"""The weight decoder alone on real weights, RTL and routed gate-level netlist.

    pytest tb/test_dec.py            both: RTL, then the SKY130 netlist

Streams the first EW_DEC_ROWS rows (default 128) of layer 21's q_proj (the
matrix with the lowest window coverage) through weight_decoder at full
speed (every stream valid, output always ready) and checks every 8-weight
chunk against the original BF16 bit for bit. A second test repeats it with
random gaps on every input and on the output.

The gate-level build is the routed netlist from the layout
(gl/weight_decoder.nl.v) over the SKY130 cell models, functional with unit
delays: it proves the netlist the layout produced is still bit-exact. Timing
at the clock comes from static timing analysis in the layout run, not from
this simulation. The full-speed gate-level run dumps a VCD (+VCD) that
gl/power.tcl turns into a power report from real switching.
"""

import os
import random
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

from ewformat import encode

REPO = Path(__file__).resolve().parent.parent
L = 8


def real_rows(n_rows):
    from test_real import layer
    w = dict(layer(int(os.environ.get("EW_LAYER", "21"))))[os.environ.get("EW_DEC_MAT", "q_proj")]
    return np.ascontiguousarray(w[:n_rows])


def words(b: np.ndarray, nbytes: int):
    b = np.concatenate([b, np.zeros((-len(b)) % nbytes, np.uint8)])
    return [int.from_bytes(b[i:i + nbytes].tobytes(), "little") for i in range(0, len(b), nbytes)]


async def stream(dut, rng_gap):
    rows = int(os.environ.get("EW_DEC_ROWS", "128"))
    w = real_rows(rows)
    e = encode(w, 3)
    n = w.size
    srcs = {
        "code": words(e.code[: (n // 64) * 3 * 8], 8),
        "smr": words(e.sm[:n], 16),
        "esc": words(e.esc[: e.n_esc], 8),
    }
    want = [int.from_bytes(w.reshape(-1)[i:i + L].astype("<u2").tobytes(), "little") for i in range(0, n, L)]

    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())     # 50 MHz, the layout's target
    for s in ("clear", "bypass", "tb_valid", "code_valid", "smr_valid", "esc_valid", "out_ready"):
        getattr(dut, s).value = 0
    dut.base.value = e.base
    dut.tb_data.value = 0
    for s in ("code", "smr", "esc"):
        getattr(dut, f"{s}_data").value = 0
    dut.rst_n.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst_n.value = 1

    pos = {s: 0 for s in srcs}
    got, cycles = 0, 0
    while got < len(want):
        await FallingEdge(dut.clk)
        for s, ws in srcs.items():
            v = pos[s] < len(ws) and not rng_gap()
            getattr(dut, f"{s}_valid").value = int(v)
            getattr(dut, f"{s}_data").value = ws[pos[s]] if v else 0
        dut.out_ready.value = int(not rng_gap())
        await RisingEdge(dut.clk)
        cycles += 1
        for s in srcs:
            if getattr(dut, f"{s}_valid").value and getattr(dut, f"{s}_ready").value:
                pos[s] += 1
        if dut.out_valid.value and dut.out_ready.value:
            v = dut.out_w.value
            assert v.is_resolvable, f"chunk {got}: X/Z on out_w"
            assert int(v) == want[got], f"chunk {got}: got {int(v):032x} want {want[got]:032x}"
            got += 1
        assert cycles < 20 * len(want) + 1000, "decoder stopped making progress"
    dut._log.info(f"{rows}x{w.shape[1]} real weights, {e.n_esc} escapes: {got} chunks bit-exact "
                  f"in {cycles} cycles ({len(want) / cycles:.4f} chunks/cycle)")
    return cycles


@cocotb.test()
async def test_full_speed(dut):
    await stream(dut, lambda: False)


@cocotb.test()
async def test_random_gaps(dut):
    r = random.Random(1)
    await stream(dut, lambda: r.random() < 0.2)


def _gl_sources():
    lib = REPO / "sim_build" / "gl_lib"
    return [lib / "primitives.v", lib / "sky130_fd_sc_hd.v", REPO / "gl" / "weight_decoder.nl.v", REPO / "tb" / "tb_dec.sv"]


def _run(build, sources, defines=None, plusargs=None, testcase=None):
    from cocotb_tools.runner import Icarus
    from runner import REPO as R

    class IcarusVcd(Icarus):
        """cocotb passes vvp '-none' when waves are off, which turns $dumpfile
        into a no-op; '-vcd' keeps tb_dec's +VCD dump in plain VCD."""
        def _test_command(self):
            return [["-vcd" if a == "-none" else a for a in c] for c in super()._test_command()]

    bdir = R / "sim_build" / build
    r = IcarusVcd()
    r.build(sources=sources, hdl_toplevel="tb_dec", build_dir=bdir, timescale=("1ns", "1ps"),
            defines=defines or {}, always=True)
    r.test(hdl_toplevel="tb_dec", test_module="test_dec", test_dir=Path(__file__).resolve().parent,
           build_dir=bdir, plusargs=plusargs or [], testcase=testcase,
           extra_env={"PYTHONPATH": str(R / "sw")}, results_xml=str(bdir / "test_dec.result.xml"))


def test_dec_rtl():
    _run("dec_rtl", [REPO / "rtl" / "ew_pkg.sv", REPO / "rtl" / "ew_lane.sv",
                     REPO / "rtl" / "weight_decoder.sv", REPO / "tb" / "tb_dec.sv"])


def _need_lib():
    if not (REPO / "sim_build" / "gl_lib" / "sky130_fd_sc_hd.v").exists():
        import pytest
        pytest.skip("copy primitives.v + sky130_fd_sc_hd.v from the PDK into sim_build/gl_lib")


def test_dec_gl():
    """Routed netlist, both tests."""
    _need_lib()
    _run("dec_gl", _gl_sources(), defines={"FUNCTIONAL": 1, "UNIT_DELAY": "#1"})


def test_dec_gl_power():
    """Routed netlist at full speed only, dumping sim_build/dec_gl_pwr/dec.vcd
    for gl/power.tcl."""
    _need_lib()
    vcd = REPO / "sim_build" / "dec_gl_pwr" / "dec.vcd"
    _run("dec_gl_pwr", _gl_sources(), defines={"FUNCTIONAL": 1, "UNIT_DELAY": "#1"},
         plusargs=[f"+VCD={vcd.as_posix()}"], testcase="test_full_speed")
