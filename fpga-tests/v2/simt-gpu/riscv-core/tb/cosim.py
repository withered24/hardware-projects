"""Lockstep co-simulation: run a program on the RTL core and the golden model.

Every time the core retires an instruction (trace_valid = 1), its trace_*
outputs (pc, instruction, register write, store, trap) are compared against
what the model did for the same instruction. The first disagreement fails
with the last few instructions. The single-cycle core retires one
instruction every cycle; the pipeline retires one per cycle at best, with
gaps for stalls and flushes.

Two more checks catch what a retirement trace can't see:
  - The memory bus itself is watched every cycle. On the pipeline a store
    writes memory two cycles before it retires, so a wrong-path store that
    leaked onto the bus would never show up in the trace. Every bus write
    must match the next store to retire, in order, and none may be left over.
  - At the end, the whole register file and every word the model stored to
    are compared against the model.

Used by test_core.py (hand-written programs), test_pipeline.py (hazard
programs), test_riscv_tests.py (the official suite) and test_random.py
(generated programs).
"""

import os
from collections.abc import Callable
from dataclasses import dataclass

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

from iss import ISS, MASK, Retire

MEM_WORDS = 16384                # must match sim_mem.sv's WORDS (64 KiB)

# RTL_DIR points the build at another copy of rtl/ (tb/mutate.py uses it to
# build deliberately broken cores).
RTL_DIR = os.environ.get("RTL_DIR", "rtl")

CORE_SOURCES = [f"{RTL_DIR}/{f}" for f in (
    "rv32i_pkg.sv", "regfile.sv", "immgen.sv", "alu.sv", "branch_unit.sv",
    "decoder.sv", "csr.sv", "core_single.sv", "core.sv")] + [
    "tb/sim_mem.sv",
    "tb/tb_top.sv",
]

# The two cores the regression runs on: build name suffix -> Verilog defines.
# REGRESS_CORES=pipeline (or =single) limits a run to one of them.
# "wait" is the pipeline with random data-memory wait states (tb_top MEM_WAIT).
CORES = {"pipeline": {}, "single": {"SINGLE_CYCLE": 1}, "wait": {"MEM_WAIT": 1}}


def selected_cores() -> dict[str, dict]:
    want = os.environ.get("REGRESS_CORES")
    return {k: v for k, v in CORES.items() if not want or k in want.split(",")}


@dataclass
class RunStats:
    instructions: int
    cycles: int

    @property
    def cpi(self) -> float:
        return self.cycles / max(self.instructions, 1)


# Word indices written by earlier tests in this simulation; cleared before the
# next program loads, so every test starts from all-zero memory like the model.
_dirty: set[int] = set()
_clock = None


def u(handle) -> int:
    return int(handle.value)


def dut_reg(dut, r: int) -> int:
    return 0 if r == 0 else u(dut.u_core.u_rf.regs[r])


def dut_word(dut, addr: int) -> int:
    return u(dut.u_mem.mem[(addr >> 2) % MEM_WORDS])


def byte_mask(strb: int) -> int:
    return sum(0xFF << (8 * i) for i in range(4) if strb >> i & 1)


def words_from_bytes(data: bytes) -> list[int]:
    data += b"\0" * (-len(data) % 4)
    return [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]


async def load_and_reset(dut, words: list[int], base: int = 0,
                         data: dict[int, int] | None = None) -> None:
    """Start the clock, hold reset, load `words` at address `base` (plus any
    `data` words, {byte address: word}), release reset. Can be called again
    in the same test to run another program."""
    global _clock
    if _clock is None or _clock.done():      # cocotb kills it at the end of each test
        _clock = cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1

    # Reset doesn't clear the register file (the ISA leaves registers
    # undefined after reset), but the model starts at zero, so match it.
    for r in range(1, 32):
        dut.u_core.u_rf.regs[r].value = 0

    first = (base >> 2) % MEM_WORDS
    indices = [(first + i) % MEM_WORDS for i in range(len(words))]
    if data:
        words = list(words) + list(data.values())
        indices += [(a >> 2) % MEM_WORDS for a in data]
    for i in _dirty - set(indices):        # words this program won't overwrite
        dut.u_mem.mem[i].value = 0
    _dirty.clear()
    for i, w in zip(indices, words):
        dut.u_mem.mem[i].value = w
        _dirty.add(i)

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0

    bad = [(i, w, u(dut.u_mem.mem[i])) for i, w in zip(indices, words) if u(dut.u_mem.mem[i]) != w]
    assert not bad, "program load failed (index, wanted, got): " + \
        ", ".join(f"({i}, {w:#010x}, {g:#010x})" for i, w, g in bad[:8])


async def lockstep(dut, iss: ISS, stop: Callable[[Retire], bool],
                   max_steps: int = 100_000,
                   on_trap: Callable[[Retire], None] | None = None,
                   on_retire: Callable[[Retire], None] | None = None,
                   on_cycle: Callable[[object], None] | None = None,
                   max_gap: int = 50) -> RunStats:
    """Run core and model together until stop(retire) is true.

    on_trap sees every trap, on_retire every instruction (model's view),
    on_cycle gets the dut once per cycle (for coverage of internal signals).
    Fails if the core goes max_gap cycles without retiring anything."""
    history: list[str] = []
    step = cycles = gap = 0
    bus: list[tuple[int, int, int, int]] = []   # (cycle, word addr, strobes, data) not yet retired
    stored: set[int] = set()                    # word addresses the model wrote

    def fail(what: str):
        recent = "\n".join(history[-8:])
        raise AssertionError(
            f"instruction {step} (cycle {cycles}): {what}\n"
            f"  (last instructions, newest last)\n{recent}")

    while step < max_steps:
        # Mid-cycle: everything has settled since the last rising edge.
        await FallingEdge(dut.clk)
        cycles += 1
        strb = u(dut.dmem_wstrb)
        if strb and not u(dut.dmem_wait):      # a write lands only when accepted
            m = byte_mask(strb)
            bus.append((cycles, u(dut.dmem_addr) & ~3 & MASK, strb, u(dut.dmem_wdata) & m))
            _dirty.add((u(dut.dmem_addr) >> 2) % MEM_WORDS)
        if not u(dut.trace_valid):
            if on_cycle:
                on_cycle(dut)
            gap += 1
            if gap > max_gap:
                fail(f"core stopped retiring instructions ({max_gap} cycles with none)")
            continue
        gap = 0

        hw_pc, hw_instr = u(dut.trace_pc), u(dut.trace_instr)
        exp = iss.step()
        history.append(f"  pc={exp.pc:#010x} instr={exp.instr:#010x}"
                       + (f"  TRAP cause={exp.cause}" if exp.trap else ""))

        if hw_pc != exp.pc:
            fail(f"PC mismatch: core {hw_pc:#010x}, model {exp.pc:#010x}")
        if hw_instr != exp.instr:
            fail(f"instruction mismatch: core {hw_instr:#010x}, model {exp.instr:#010x}")

        hw_trap = u(dut.trace_trap)
        if hw_trap != exp.trap:
            fail(f"trap: core {hw_trap}, model {int(exp.trap)} (model cause {exp.cause})")
        if hw_trap and u(dut.trace_cause) != exp.cause:
            fail(f"trap cause: core {u(dut.trace_cause)}, model {exp.cause}")
        if exp.trap and on_trap:
            on_trap(exp)

        hw_we = u(dut.trace_rd_we)
        if hw_we != (exp.rd is not None):
            fail(f"register write-enable: core {hw_we}, model {exp.rd is not None}")
        if hw_we:
            hw_rd, hw_val = u(dut.trace_rd), u(dut.trace_rd_wdata)
            if (hw_rd, hw_val) != (exp.rd, exp.rd_val):
                fail(f"register write: core x{hw_rd}={hw_val:#010x}, "
                     f"model x{exp.rd}={exp.rd_val:#010x}")

        hw_strb = u(dut.trace_mem_wstrb)
        if hw_strb != exp.store_strb:
            fail(f"store byte-enables: core {hw_strb:04b}, model {exp.store_strb:04b}")
        if hw_strb:
            hw_addr = u(dut.trace_mem_addr) & ~3 & MASK
            m = byte_mask(hw_strb)
            hw_data = u(dut.trace_mem_wdata) & m
            if (hw_addr, hw_data) != (exp.store_addr & MASK, exp.store_data & m):
                fail(f"store: core [{hw_addr:#010x}]={hw_data:#010x}, "
                     f"model [{exp.store_addr:#010x}]={exp.store_data & m:#010x}")
            if not bus:
                fail("a store retired but never reached the memory bus")
            _, b_addr, b_strb, b_data = bus.pop(0)
            if (b_addr, b_strb, b_data) != (hw_addr, hw_strb, hw_data):
                fail(f"memory bus wrote [{b_addr:#010x}]={b_data:#010x} (strobes {b_strb:04b}) "
                     f"but the store retiring now is [{hw_addr:#010x}]={hw_data:#010x}: "
                     f"a write from an instruction that never retired?")
            stored.add(exp.store_addr & ~3 & MASK)

        if on_retire:
            on_retire(exp)
        step += 1
        if not stop(exp):
            # On the final cycle everything in IF..MEM is younger than the
            # last instruction of the program, so its events aren't sampled.
            if on_cycle:
                on_cycle(dut)
        else:
            # Writes still on the bus may only come from this cycle (a younger
            # store in EX); anything older was a write that never retired.
            stale = [b for b in bus if b[0] != cycles]
            if stale:
                c, a, st, d = stale[0]
                fail(f"memory bus write at cycle {c} ([{a:#010x}]={d:#010x}, strobes "
                     f"{st:04b}) belongs to no retired store")
            # Let the final cycle commit (the single-cycle core's last store and
            # either core's last register write land at this edge), then
            # compare state. Words a younger, unretired store just wrote are
            # skipped: the model hasn't executed that store.
            await RisingEdge(dut.clk)
            await FallingEdge(dut.clk)
            younger = {b[1] for b in bus}
            for a in sorted(stored - younger):
                if dut_word(dut, a) != iss.read(a, 4):
                    fail(f"final memory [{a:#010x}]: core {dut_word(dut, a):#010x}, "
                         f"model {iss.read(a, 4):#010x}")
            for r in range(1, 32):
                if dut_reg(dut, r) != iss.x(r):
                    fail(f"final register x{r}: core {dut_reg(dut, r):#010x}, "
                         f"model {iss.x(r):#010x}")
            return RunStats(step, cycles)

    raise AssertionError(f"program did not finish within {max_steps} instructions")


# ---------------------------------------------------------------------------
# Assembly programs that can trap and carry on

DONE_ADDR = 0xFFF0          # a store here ends the program

# Prepended to programs run with run_asm(handler=True). The handler skips the
# instruction that trapped (mepc + 4) and returns, so a program can keep going
# after an ECALL, an illegal instruction or a misaligned access. It uses only
# t6, so programs that want to survive traps should leave t6 alone.
HANDLER_PRELUDE = """
        j    _main
    _handler:
        csrr t6, mepc
        addi t6, t6, 4
        csrw mepc, t6
        mret
    _main:
        li   t6, 4
        csrw mtvec, t6
"""

DONE = f"""
        li   t5, {DONE_ADDR}
        sw   t5, 0(t5)
"""


async def run_asm(dut, source: str, handler: bool = False, max_steps: int = 50_000,
                  **hooks) -> tuple[ISS, RunStats]:
    """Assemble and run a program in lockstep until it stores to DONE_ADDR.
    With handler=True, traps are survivable (see HANDLER_PRELUDE); otherwise
    any trap fails the test. hooks go to lockstep (on_cycle, on_retire, ...)."""
    from asm import assemble
    words = assemble((HANDLER_PRELUDE if handler else "") + source + DONE)
    iss = ISS(mem_bytes=4 * MEM_WORDS, halt_on_ecall=False)
    iss.load_words(words)
    await load_and_reset(dut, words)

    user_trap = hooks.pop("on_trap", None)

    def on_trap(r):
        if not handler:
            raise AssertionError(f"unexpected trap, cause {r.cause}, at pc {r.pc:#010x}")
        if user_trap:
            user_trap(r)

    st = await lockstep(dut, iss, stop=lambda r: r.store_addr == DONE_ADDR,
                        max_steps=max_steps, on_trap=on_trap, **hooks)
    return iss, st
