"""Directed co-simulation tests, run on both cores (pipelined and single-cycle).

Every program runs on the RTL core and on the Python golden model (iss.py)
in lockstep. Each time the core retires an instruction, the testbench
compares its trace_* outputs (pc, instruction, register write, memory
write) against what the model did for the same instruction. The first disagreement fails the
test with the instruction that caused it.

On top of that, each test checks a few known answers worked out by hand, so
that a mistake shared by the RTL and the model can't slip through silently.

Run with:   pytest tb/test_core.py -v
"""

import cocotb

from asm import assemble
from cosim import (CORE_SOURCES, MEM_WORDS, dut_reg, dut_word, load_and_reset, lockstep,
                   selected_cores)
from iss import ISS, MACHINE_ECALL, MASK
from runner import run


async def run_program(dut, source: str, max_steps: int = 20_000) -> ISS:
    """Assemble `source`, run it on core + model in lockstep until ECALL.
    Any other trap fails the test: these programs should never trap.
    Returns the model so tests can inspect it (it's guaranteed to match)."""
    words = assemble(source)
    iss = ISS(mem_bytes=4 * MEM_WORDS)
    iss.load_words(words)
    await load_and_reset(dut, words)

    def no_unexpected_traps(r):
        assert r.cause == MACHINE_ECALL,             f"unexpected trap, cause {r.cause}, at pc {r.pc:#010x} (instr {r.instr:#010x})"

    st = await lockstep(dut, iss, stop=lambda r: iss.halted, max_steps=max_steps,
                        on_trap=no_unexpected_traps)
    dut._log.info(f"ECALL after {st.instructions} instructions in {st.cycles} cycles "
                  f"(CPI {st.cpi:.2f}), all matched the model")
    return iss


def check_regs(dut, expected: dict[str, int]) -> None:
    """Known-answer check straight from the core's register file."""
    from asm import REGS
    for name, want in expected.items():
        got = dut_reg(dut, REGS[name])
        assert got == want & MASK, f"{name}: core {got:#010x}, expected {want & MASK:#010x}"


# ---------------------------------------------------------------------------

@cocotb.test()
async def test_first_five(dut):
    """The five instructions from Lesson 04's build order: add addi lw sw beq."""
    await run_program(dut, """
        addi x1, x0, 5
        addi x2, x0, 7
        add  x3, x1, x2          # 12
        sw   x3, 0x100(x0)
        lw   x4, 0x100(x0)       # 12
        beq  x3, x4, skip        # taken
        addi x5, x0, 1           # must be skipped
    skip:
        beq  x1, x2, bad         # not taken
        addi x6, x0, 42
        ecall
    bad:
        addi x7, x0, -1
        ecall
    """)
    check_regs(dut, {"x3": 12, "x4": 12, "x5": 0, "x6": 42, "x7": 0})


@cocotb.test()
async def test_alu_register_ops(dut):
    """All ten R-type ops on a negative/positive pair. b = 35 also checks that
    shifts use only b[4:0] (35 -> shift by 3)."""
    await run_program(dut, """
        li   a0, -7
        li   a1, 35
        add  s0, a0, a1
        sub  s1, a0, a1
        sll  s2, a0, a1
        slt  s3, a0, a1
        sltu s4, a0, a1
        xor  s5, a0, a1
        srl  s6, a0, a1
        sra  s7, a0, a1
        or   s8, a0, a1
        and  s9, a0, a1
        ecall
    """)
    check_regs(dut, {"s0": 0x1C, "s1": 0xFFFFFFD6, "s2": 0xFFFFFFC8, "s3": 1,
                     "s4": 0, "s5": 0xFFFFFFDA, "s6": 0x1FFFFFFF, "s7": 0xFFFFFFFF,
                     "s8": 0xFFFFFFFB, "s9": 0x21})


@cocotb.test()
async def test_alu_immediate_ops(dut):
    """OP-IMM, including the trap where immediate bit 10 sits in instr[30]:
    addi with -1024 must NOT turn into a subtract."""
    await run_program(dut, """
        li    x1, 1
        slli  x2, x1, 31         # 0x80000000
        srai  x3, x2, 31         # 0xFFFFFFFF
        srli  x4, x2, 31         # 1
        li    x10, 5000
        addi  x5, x10, -1024     # 3976
        slti  x6, x3, 0          # -1 < 0
        sltiu x7, x4, -1         # 1 < 0xFFFFFFFF
        sltiu x8, x3, -1         # 0xFFFFFFFF < 0xFFFFFFFF is false
        xori  x9, x4, -1         # ~1
        andi  x11, x3, -2048
        ori   x12, x0, 0x7FF
        addi  x13, x0, -2048
        ecall
    """)
    check_regs(dut, {"x2": 0x80000000, "x3": 0xFFFFFFFF, "x4": 1, "x5": 3976,
                     "x6": 1, "x7": 1, "x8": 0, "x9": 0xFFFFFFFE,
                     "x11": 0xFFFFF800, "x12": 0x7FF, "x13": 0xFFFFF800})


@cocotb.test()
async def test_upper_immediates(dut):
    """LUI loads the top 20 bits; AUIPC adds them to its own PC."""
    await run_program(dut, """
        lui   x1, 0xDEADB        # pc 0
        auipc x2, 1              # pc 4 -> 0x1004
        auipc x3, 0xFFFFF        # pc 8 -> 8 - 0x1000
        ecall
    """)
    check_regs(dut, {"x1": 0xDEADB000, "x2": 0x1004, "x3": 0xFFFFF008})


@cocotb.test()
async def test_branches(dut):
    """Each branch type, taken and not taken, at the signed/unsigned boundary
    (-1 vs 1). s0 collects one bit per case, 1 = taken."""
    cases = [
        ("beq",  "a0", "a0", 1), ("beq",  "a0", "a1", 0),
        ("bne",  "a0", "a1", 1), ("bne",  "a0", "a0", 0),
        ("blt",  "a0", "a1", 1), ("blt",  "a1", "a0", 0),
        ("bge",  "a1", "a0", 1), ("bge",  "a0", "a1", 0), ("bge", "a0", "a0", 1),
        ("bltu", "a1", "a0", 1), ("bltu", "a0", "a1", 0),
        ("bgeu", "a0", "a1", 1), ("bgeu", "a1", "a0", 0), ("bgeu", "a1", "a1", 1),
    ]
    body = []
    for i, (op, x, y, _) in enumerate(cases):
        body += [f"slli s0, s0, 1", f"{op} {x}, {y}, t{i}", f"j n{i}",
                 f"t{i}: ori s0, s0, 1", f"n{i}:"]
    src = "li a0, -1\nli a1, 1\n" + "\n".join(body) + """
        li   t0, 10              # backward branch: count down a loop
    loop:
        addi s1, s1, 1
        addi t0, t0, -1
        bnez t0, loop
        ecall
    """
    await run_program(dut, src)
    want = int("".join(str(c[3]) for c in cases), 2)
    check_regs(dut, {"s0": want, "s1": 10})


@cocotb.test()
async def test_jumps(dut):
    """JAL links pc+4; JALR jumps to (rs1 + imm) with bit 0 cleared."""
    await run_program(dut, """
        jal   ra, func           # pc 0: ra = 4
        addi  s1, s1, 1          # pc 4: runs after the return
        auipc t0, 0              # pc 8: t0 = 8
        addi  t0, t0, 21         # t0 = 29, odd on purpose
        jalr  t1, 0(t0)          # pc 16: t1 = 20, target (29 & ~1) = 28
        addi  s2, x0, -1         # pc 20: skipped
        addi  s2, x0, -1         # pc 24: skipped
        addi  s3, x0, 7          # pc 28
        j     end
    func:
        addi  s0, x0, 99
        ret
    end:
        ecall
    """)
    check_regs(dut, {"ra": 4, "s0": 99, "s1": 1, "t1": 20, "s2": 0, "s3": 7})


@cocotb.test()
async def test_loads_and_stores(dut):
    """Every load width at every byte offset, sign vs zero extension, partial
    stores merging into an existing word, and a negative offset."""
    await run_program(dut, """
        li   t0, 0x1000
        li   t1, 0x8899AABB
        sw   t1, 0(t0)
        lb   a0, 0(t0)
        lb   a1, 1(t0)
        lb   a2, 2(t0)
        lb   a3, 3(t0)
        lbu  a4, 3(t0)
        lh   a5, 0(t0)
        lh   a6, 2(t0)
        lhu  a7, 2(t0)
        li   t2, 0x11
        sb   t2, 1(t0)           # word becomes 0x889911BB
        sh   t2, 2(t0)           # word becomes 0x001111BB
        lw   s0, 0(t0)
        sw   t1, -4(t0)
        lw   s1, -4(t0)
        lhu  s2, 0(t0)
        li   t3, 0x7F
        sb   t3, 4(t0)
        lb   s3, 4(t0)           # positive byte stays positive
        li   t4, 0x807F7F80      # halves where bit 15 != bit 7, so a sign
        sw   t4, 8(t0)           # extension from the wrong bit shows up
        lh   s4, 8(t0)           # 0x7F80 -> positive
        lh   s5, 10(t0)          # 0x807F -> negative
        lb   s6, 8(t0)           # 0x80   -> negative
        lb   s7, 9(t0)           # 0x7F   -> positive
        ecall
    """)
    check_regs(dut, {"a0": 0xFFFFFFBB, "a1": 0xFFFFFFAA, "a2": 0xFFFFFF99,
                     "a3": 0xFFFFFF88, "a4": 0x88, "a5": 0xFFFFAABB,
                     "a6": 0xFFFF8899, "a7": 0x8899, "s0": 0x001111BB,
                     "s1": 0x8899AABB, "s2": 0x11BB, "s3": 0x7F,
                     "s4": 0x00007F80, "s5": 0xFFFF807F, "s6": 0xFFFFFF80, "s7": 0x7F})
    assert dut_word(dut, 0x1000) == 0x001111BB
    assert dut_word(dut, 0x0FFC) == 0x8899AABB


@cocotb.test()
async def test_x0_stays_zero(dut):
    """Every way of writing a register, aimed at x0, then x0 used as a source."""
    await run_program(dut, """
        li   t0, 0x1234
        sw   t0, 0x200(x0)
        addi x0, x0, 5
        lui  x0, 0xFFFFF
        lw   x0, 0x200(x0)
        jal  x0, next
    next:
        add  t1, x0, x0
        addi t2, x0, 1
        fence                    # FENCE is a no-op on this core
        ecall
    """)
    check_regs(dut, {"x0": 0, "t1": 0, "t2": 1})


@cocotb.test()
async def test_fibonacci(dut):
    """A real program: fill an array with Fibonacci numbers f(0)..f(20)."""
    await run_program(dut, """
        li   t0, 0x2000
        li   a0, 0
        li   a1, 1
        sw   a0, 0(t0)
        sw   a1, 4(t0)
        addi t0, t0, 8
        li   t1, 19
    loop:
        add  a2, a0, a1
        sw   a2, 0(t0)
        mv   a0, a1
        mv   a1, a2
        addi t0, t0, 4
        addi t1, t1, -1
        bnez t1, loop
        ecall
    """)
    fib = [0, 1]
    while len(fib) < 21:
        fib.append(fib[-1] + fib[-2])
    check_regs(dut, {"a1": 6765})
    for i, f in enumerate(fib):
        assert dut_word(dut, 0x2000 + 4 * i) == f, f"f({i})"


@cocotb.test()
async def test_bubble_sort(dut):
    """Nested loops, signed compares, loads and stores: sort 16 words."""
    data = [503, -12, 77, 0, -800, 9999, 3, 3, -1, 42, 1 << 20, -(1 << 25), 15, 8, 255, -255]
    stores = "\n".join(f"li t1, {v}\nsw t1, {4 * i}(t0)" for i, v in enumerate(data))
    await run_program(dut, f"""
        li   t0, 0x3000
        {stores}
        li   s0, 16              # n
    outer:
        addi s0, s0, -1
        beqz s0, done
        li   t0, 0x3000
        mv   t2, s0              # inner count
    inner:
        lw   a0, 0(t0)
        lw   a1, 4(t0)
        bge  a1, a0, noswap
        sw   a1, 0(t0)
        sw   a0, 4(t0)
    noswap:
        addi t0, t0, 4
        addi t2, t2, -1
        bnez t2, inner
        j    outer
    done:
        ecall
    """)
    for i, v in enumerate(sorted(data)):
        assert dut_word(dut, 0x3000 + 4 * i) == v & MASK, f"sorted[{i}]"


def test_core():
    for name, defines in selected_cores().items():
        run("tb_top", CORE_SOURCES, "test_core", defines=defines, build_name=f"tb_top_{name}")
