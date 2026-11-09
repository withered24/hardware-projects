"""Everything the GPU tests share: config, assembler, golden model, driver.

- CFG: the GPU's shape, read out of rtl/gpu_pkg.sv so the two can't drift.
- assemble(): riscv-core's mini assembler, plus `mul` and the thread-ID
  CSRs by name (csrr t0, tid / ctaid / ntid / nctaid / args).
- run_model(): the golden model. Every thread of every block runs to its
  ECALL on riscv-core's ISS (tb/iss.py), one thread after another, all
  sharing one memory. For a kernel whose threads don't race, the final
  memory must be identical whatever order the GPU actually ran them in,
  and so must the total number of instructions the threads executed:
  min-PC scheduling never runs a thread through an instruction it
  wouldn't execute on its own.
- GPU: drives gpu_top from cocotb (load code and memory, launch, wait).
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO / "riscv-core" / "tb"))     # after ours: both have a runner.py

import asm  # noqa: E402
from iss import ISS, ILLEGAL_INSTR, MACHINE_ECALL, Trap  # noqa: E402


def _read_pkg() -> dict[str, int]:
    text = (REPO / "rtl" / "gpu_pkg.sv").read_text()
    vals: dict[str, int] = {}
    for name, expr in re.findall(r"localparam\s+int\s+(\w+)\s*=\s*([^;]+);", text):
        expr = expr.replace("$clog2", "clog2")
        vals[name] = int(eval(expr, {"__builtins__": {}, "clog2": lambda n: (n - 1).bit_length()}, dict(vals)))
    for name, digits in re.findall(r"localparam\s+logic\s*\[11:0\]\s*(\w+)\s*=\s*12'h(\w+);", text):
        vals[name] = int(digits, 16)
    return vals


CFG = _read_pkg()
LANES, WARPS, THREADS = CFG["LANES"], CFG["WARPS"], CFG["THREADS"]
LINE_BYTES, MEM_BYTES = CFG["LINE_BYTES"], CFG["MEM_BYTES"]
MEM_LINES = MEM_BYTES // LINE_BYTES
CSRS = {"tid": CFG["CSR_TID"], "ctaid": CFG["CSR_CTAID"], "ntid": CFG["CSR_NTID"],
        "nctaid": CFG["CSR_NCTAID"], "args": CFG["CSR_ARGS"]}

asm.CSRS.update(CSRS)
asm.R_OPS["mul"] = (0x33, 0, 0x01)


def assemble(source: str) -> list[int]:
    return asm.assemble(source)


# ---------------------------------------------------------------------------
# golden model

class Thread(ISS):
    """One GPU thread on the riscv-core ISS: code from its own instruction
    memory, data in the shared global memory, plus MUL and the ID CSRs."""

    def __init__(self, code: list[int], mem: bytearray, pc: int, ids: dict[int, int]):
        super().__init__(mem_bytes=4, reset_pc=pc)
        self.mem, self.code, self.ids = mem, code, ids
        self._fetch = False

    def read(self, addr: int, size: int) -> int:
        if self._fetch:                       # ISS.step's first read is the fetch
            self._fetch = False
            i = addr >> 2
            return self.code[i] if 0 <= i < len(self.code) else 0
        return super().read(addr, size)

    def step(self):
        self._fetch = True
        return super().step()

    def csr_read(self, num: int) -> int:
        if num in self.ids:
            return self.ids[num]
        raise Trap(ILLEGAL_INSTR)

    def _execute(self, ins: int, pc: int, ret) -> None:
        if ins & 0x7F == 0x33 and ins >> 25 == 1:            # M extension
            if (ins >> 12) & 7:
                raise Trap(ILLEGAL_INSTR, ins)               # only MUL is implemented
            rd, rs1, rs2 = (ins >> 7) & 31, (ins >> 15) & 31, (ins >> 20) & 31
            self.set_x(rd, self.x(rs1) * self.x(rs2))
            if rd:
                ret.rd, ret.rd_val = rd, self.x(rd)
            self.pc = (pc + 4) & 0xFFFF_FFFF
            return
        super()._execute(ins, pc, ret)


def run_model(code: list[int], mem: bytearray, pc: int, grid: int, block: int,
              args: int, max_steps: int = 200_000) -> int:
    """Run the kernel on `mem` in place; return the total thread instructions."""
    total = 0
    for b in range(grid):
        for t in range(block):
            ids = {CSRS["tid"]: t, CSRS["ctaid"]: b, CSRS["ntid"]: block,
                   CSRS["nctaid"]: grid, CSRS["args"]: args}
            th = Thread(code, mem, pc, ids)
            for _ in range(max_steps):
                r = th.step()
                total += 1
                if r.trap:
                    assert r.cause == MACHINE_ECALL, \
                        f"block {b} thread {t} trapped (cause {r.cause}) at pc {r.pc:#x}"
                    break
            else:
                raise AssertionError(f"block {b} thread {t} ran {max_steps} steps without exiting")
    return total


# ---------------------------------------------------------------------------
# cocotb driver

@dataclass
class Stats:
    cycles: int
    warp_instrs: int
    thread_instrs: int
    mem_lines: int

    @property
    def simd_efficiency(self) -> float:
        """Fraction of lane slots that did useful work."""
        return self.thread_instrs / max(1, self.warp_instrs * LANES)


def u(handle) -> int:
    return int(handle.value)


class GPU:
    def __init__(self, dut):
        self.dut = dut
        self.mem_image = bytearray(MEM_BYTES)   # what the DUT's memory holds

    async def reset(self):
        import cocotb
        from cocotb.clock import Clock
        from cocotb.triggers import RisingEdge
        d = self.dut
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        for sig in (d.imem_we, d.dmem_we, d.start, d.imem_addr, d.imem_wdata, d.dmem_addr,
                    d.dmem_wdata, d.kernel_pc, d.grid_dim, d.block_dim, d.args):
            sig.value = 0
        d.rst.value = 1
        for _ in range(3):
            await RisingEdge(d.clk)
        d.rst.value = 0
        await RisingEdge(d.clk)
        self.read_mem()             # earlier tests in this simulation left data behind

    def load_code(self, words: list[int]) -> None:
        for i, w in enumerate(words):
            self.dut.imem[i].value = w

    def write_mem(self, image: bytearray) -> None:
        """Make the DUT's global memory equal `image` (only lines that differ are written)."""
        for i in range(MEM_LINES):
            new = image[i * LINE_BYTES:(i + 1) * LINE_BYTES]
            if new != self.mem_image[i * LINE_BYTES:(i + 1) * LINE_BYTES]:
                self.dut.u_mem.mem[i].value = int.from_bytes(new, "little")
        self.mem_image = bytearray(image)

    def read_mem(self) -> bytearray:
        out = bytearray()
        for i in range(MEM_LINES):
            out += u(self.dut.u_mem.mem[i]).to_bytes(LINE_BYTES, "little")
        self.mem_image = bytearray(out)
        return out

    async def launch(self, pc: int, grid: int, block: int, args: int,
                     max_cycles: int = 300_000) -> Stats:
        from cocotb.triggers import First, FallingEdge, ReadOnly, RisingEdge, Timer
        d = self.dut
        await FallingEdge(d.clk)
        d.kernel_pc.value, d.grid_dim.value, d.block_dim.value, d.args.value = pc, grid, block, args
        d.start.value = 1
        await FallingEdge(d.clk)
        d.start.value = 0
        res = await First(RisingEdge(d.done), Timer(10 * max_cycles, unit="ns"))
        await ReadOnly()
        assert res is not None and u(d.done), f"kernel didn't finish in {max_cycles} cycles"
        assert not u(d.error), "a thread hit an illegal instruction / EBREAK / MRET"
        stats = Stats(u(d.cycles), u(d.warp_instrs), u(d.thread_instrs), u(d.mem_lines))
        await FallingEdge(d.clk)
        return stats


async def run_kernel(gpu: GPU, source: str, image: bytearray, grid: int, block: int,
                     args: int, log=None) -> tuple[Stats, bytearray]:
    """Run a kernel on the model and the GPU from the same starting memory;
    check the final memories and the instruction counts agree."""
    code = assemble(source)
    expect = bytearray(image)
    model_instrs = run_model(code, expect, 0, grid, block, args)

    gpu.load_code(code)
    gpu.write_mem(image)
    stats = await gpu.launch(0, grid, block, args)
    got = gpu.read_mem()

    if got != expect:
        bad = [i for i in range(0, MEM_BYTES, 4) if got[i:i + 4] != expect[i:i + 4]]
        lines = [f"  {a:#06x}: gpu {int.from_bytes(got[a:a+4], 'little'):#010x}, "
                 f"model {int.from_bytes(expect[a:a+4], 'little'):#010x}" for a in bad[:10]]
        raise AssertionError(f"{len(bad)} words differ:\n" + "\n".join(lines))
    assert stats.thread_instrs == model_instrs, \
        f"GPU executed {stats.thread_instrs} thread instructions, model {model_instrs}"
    if log:
        log(f"{stats.cycles} cycles, {stats.warp_instrs} warp instrs, {stats.thread_instrs} "
            f"thread instrs, SIMD efficiency {stats.simd_efficiency:.0%}, {stats.mem_lines} memory lines")
    return stats, got
