"""fpga_rec: cocotb-side trace recorder, loaded as an extra test module.

record.py runs a repo's own pytest regression with this module listed first
in COCOTB_TEST_MODULES. It has no tests; importing it patches cocotb's Clock
so that whenever a test starts a clock on the DUT's clk, a sampler starts
too. At every rising edge the sampler logs the DUT's inputs and outputs as
they stand just before the edge (the values the flops see):

    <inputs hex> <outputs hex> <output X-mask hex>

inputs are the DUT's input ports (clk excluded) then any sideband bits, in
port order, LSB = last port's bit 0; X/Z inputs are logged as 0. The FPGA
replays the inputs one DUT clock edge at a time and checks the outputs
everywhere the simulator had no X.

Sideband bits stand in for testbench backdoor writes into the DUT, which
the FPGA can't see through ports (see PATCH below). Env:
  FPGA_TRACE   output file          FPGA_PORTS  ports JSON (ports.py)
  FPGA_DUT     dotted path from the toplevel to the DUT ("" = toplevel)
  FPGA_PATCH   riscv | gpu | ""     FPGA_MAX_CYCLES  stop logging after this many
"""
import json
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

TRACE = os.environ.get("FPGA_TRACE")
PORTS = json.loads(open(os.environ["FPGA_PORTS"]).read()) if TRACE else []
DUT_PATH = os.environ.get("FPGA_DUT", "")
PATCH = os.environ.get("FPGA_PATCH", "")
MAX_CYCLES = int(os.environ.get("FPGA_MAX_CYCLES", "0") or 0)

INS = [p for p in PORTS if p["dir"] == "input" and p["name"] != "clk"]
OUTS = [p for p in PORTS if p["dir"] == "output"]
SIDEBAND = {"riscv": ["rf_clear"]}.get(PATCH, [])
side = {k: 0 for k in SIDEBAND}       # set by patched helpers, logged at the next edge

_fh = None
_task = None
_count = 0


def _dut_handle():
    h = cocotb.top
    for part in filter(None, DUT_PATH.split(".")):
        h = getattr(h, part)
    return h


def _bits(v) -> tuple[str, str]:
    """(value bits with X/Z as 0, mask bits with 1 where X/Z)"""
    s = str(v).lower()
    return (s.replace("x", "0").replace("z", "0").replace("u", "0").replace("w", "0").replace("l", "0").replace("h", "1"),
            "".join("1" if c in "xzuw-" else "0" for c in s))


async def _sampler(clk):
    global _count
    dut = _dut_handle()
    ih = [getattr(dut, p["name"]) for p in INS]
    oh = [getattr(dut, p["name"]) for p in OUTS]
    for p, h in zip(INS + OUTS, ih + oh):
        assert len(h) == p["width"], f"{p['name']}: sim width {len(h)}, ports.json {p['width']}"
    nin = sum(p["width"] for p in INS) + len(SIDEBAND)
    nout = sum(p["width"] for p in OUTS)
    wi, wo = (nin + 3) // 4, (nout + 3) // 4
    while True:
        await RisingEdge(clk)
        if MAX_CYCLES and _count >= MAX_CYCLES:
            continue
        ib = "".join(_bits(h.value)[0] for h in ih) + "".join(str(side[k]) for k in SIDEBAND)
        for k in SIDEBAND:
            side[k] = 0
        ov, om = zip(*(_bits(h.value) for h in oh))
        _fh.write(f"{int(ib, 2):0{wi}x} {int(''.join(ov), 2):0{wo}x} {int(''.join(om), 2):0{wo}x}\n")
        _count += 1


_orig_start = Clock.start


def _start(self, *a, **kw):
    global _fh, _task
    t = _orig_start(self, *a, **kw)
    if TRACE and (_task is None or _task.done()):
        if _fh is None:
            _fh = open(TRACE, "a")
        _task = cocotb.start_soon(_sampler(self.signal))
    return t


Clock.start = _start


def _flush_at_exit():
    if _fh:
        _fh.flush()


import atexit  # noqa: E402

atexit.register(_flush_at_exit)

# ---------------------------------------------------------------- backdoor patches
if PATCH == "riscv" and TRACE:
    # cosim.load_and_reset zeroes the register file through the hierarchy
    # (dut.u_core.u_rf.regs[r]) while the core is held in reset. The FPGA
    # copy of regfile.sv clears all registers at the edge where the
    # rf_clear sideband bit is set.
    import cosim

    class _RegsProxy:
        def __init__(self, regs):
            self._r = regs

        def __getitem__(self, i):
            side["rf_clear"] = 1
            return self._r[i]

        def __len__(self):
            return len(self._r)

    _orig_lr = cosim.load_and_reset

    async def load_and_reset(dut, *a, **kw):
        class _CoreProxy:
            def __getattr__(self, n):
                return getattr(dut.u_core, n) if n != "u_rf" else _RfProxy()

        class _RfProxy:
            def __getattr__(self, n):
                h = getattr(dut.u_core.u_rf, n)
                return _RegsProxy(h) if n == "regs" else h

        class _DutProxy:
            def __getattr__(self, n):
                return _CoreProxy() if n == "u_core" else getattr(dut, n)

            def __setattr__(self, n, v):
                setattr(dut, n, v)

        return await _orig_lr(_DutProxy(), *a, **kw)

    cosim.load_and_reset = load_and_reset

if PATCH == "gpu" and TRACE:
    # The GPU tests load code and data with backdoor writes into gpu_top's
    # instruction memory and gpu_mem, and read results back the same way.
    # Here the loads go through the host ports (imem_we / dmem_we, one word
    # a cycle, while the GPU is idle) just before each launch, and after
    # each kernel every word that changed is read through dmem_addr /
    # dmem_rdata, so the results show up at the ports the FPGA checks.
    import gpu as G
    from cocotb.triggers import FallingEdge, ReadOnly

    _pend_code: list[int] = []
    _pend_mem: dict[int, int] = {}

    def load_code(self, words):
        _pend_code[:] = list(words)

    def write_mem(self, image):
        for a in range(0, G.MEM_BYTES, 4):
            if image[a:a + 4] != self.mem_image[a:a + 4]:
                _pend_mem[a] = int.from_bytes(image[a:a + 4], "little")
        self.mem_image = bytearray(image)

    _orig_launch = G.GPU.launch

    async def launch(self, pc, grid, block, args, *a, **kw):
        d = self.dut
        for i, w in enumerate(_pend_code):
            await FallingEdge(d.clk)
            d.imem_we.value, d.imem_addr.value, d.imem_wdata.value = 1, 4 * i, w
        for addr, w in sorted(_pend_mem.items()):
            await FallingEdge(d.clk)
            d.dmem_we.value, d.dmem_addr.value, d.dmem_wdata.value = 1, addr, w
        _pend_code.clear()
        _pend_mem.clear()
        await FallingEdge(d.clk)
        d.imem_we.value = d.dmem_we.value = 0
        before = bytes(self.mem_image)
        st = await _orig_launch(self, pc, grid, block, args, *a, **kw)
        after = self.read_mem()           # backdoor read: tells us which words to fetch
        self.mem_image = bytearray(before)
        for a_ in range(0, G.MEM_BYTES, 4):
            if after[a_:a_ + 4] != before[a_:a_ + 4]:
                d.dmem_addr.value = a_
                await FallingEdge(d.clk)
        self.mem_image = bytearray(after)
        return st

    G.GPU.load_code = load_code
    G.GPU.write_mem = write_mem
    G.GPU.launch = launch
