"""The official RISC-V compliance tests (riscv-tests, rv32ui-p-*) on the core.

Each test is a self-checking program: it runs its checks, then stores to a
magic address called `tohost`. A value of 1 means pass; (n << 1) | 1 means
check number n failed. On top of that self-check, every instruction is run
in lockstep against the golden model, so a failure points at the exact
instruction rather than just "test case 7".

Build the tests first (the pytest entry point below does it for you):
    python tests/build_rvtests.py
Run:
    pytest tb/test_riscv_tests.py -v
When one fails, open tests/build/<name>.dump to read the disassembly.
"""

import json
from pathlib import Path

import cocotb

from cosim import (CORE_SOURCES, MEM_WORDS, load_and_reset, lockstep, selected_cores,
                   words_from_bytes)
from iss import ILLEGAL_INSTR, ISS, MACHINE_ECALL, MISALIGNED_LOAD
from runner import REPO, run

BASE = 0x8000_0000
MANIFEST = REPO / "tests" / "build" / "manifest.json"

# Not applicable to this core, with the reason. Everything else must pass.
NOT_APPLICABLE = {
    "rv32ui-p-ma_data": "needs hardware misaligned loads/stores; this core "
                        "traps on them instead, which the spec allows "
                        "(checked by test_ma_data_traps below)",
}

_manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
# Short names ("add", "fence_i") so the results table reads well.
TESTS = sorted(n.removeprefix("rv32ui-p-") for n in _manifest if n not in NOT_APPLICABLE)


async def run_rvtest(dut, name: str, on_trap=None) -> int:
    """Run one riscv-test in lockstep; return the value it wrote to tohost."""
    info = _manifest[name if name.startswith("rv32ui-p-") else f"rv32ui-p-{name}"]
    image = Path(info["bin"]).read_bytes()
    tohost = info["tohost"]

    iss = ISS(mem_bytes=4 * MEM_WORDS, reset_pc=BASE, halt_on_ecall=False)
    iss.load_bytes(image, BASE)
    await load_and_reset(dut, words_from_bytes(image), BASE)

    result = {}

    def wrote_tohost(r) -> bool:
        if r.store_addr == tohost and r.store_data != 0:
            result["value"] = r.store_data
            return True
        return False

    st = await lockstep(dut, iss, stop=wrote_tohost, on_trap=on_trap)
    dut._log.info(f"{name}: tohost={result['value']:#x} after {st.instructions} "
                  f"instructions, {st.cycles} cycles (CPI {st.cpi:.2f})")
    return result["value"]


@cocotb.test()
@cocotb.parametrize(name=TESTS)
async def rv32ui(dut, name):
    value = await run_rvtest(dut, name)
    assert value == 1, f"{name} failed its check number {value >> 1} (tohost={value:#x})"


@cocotb.test()
async def test_ma_data_traps(dut):
    """ma_data's first check is a misaligned halfword load. This core must
    take a misaligned-load trap there, not load garbage silently.
    (Startup code traps a few times first, with cause 2 = illegal
    instruction, probing CSRs this core doesn't have; that's by design.)"""
    traps = []
    await run_rvtest(dut, "rv32ui-p-ma_data", on_trap=traps.append)
    first = next(t for t in traps if t.cause not in (ILLEGAL_INSTR, MACHINE_ECALL))
    assert first.cause == MISALIGNED_LOAD, f"first data trap had cause {first.cause}"
    assert first.instr & 0x707F == 0x1003, f"trap was not on an LH: {first.instr:#010x}"


def test_riscv_tests():
    import subprocess
    import sys
    subprocess.run([sys.executable, str(REPO / "tests" / "build_rvtests.py")], check=True)
    for core, defines in selected_cores().items():
        run("tb_top", CORE_SOURCES, "test_riscv_tests", parameters={"RESET_PC": BASE},
            defines=defines, build_name=f"tb_top_rvtests_{core}")
