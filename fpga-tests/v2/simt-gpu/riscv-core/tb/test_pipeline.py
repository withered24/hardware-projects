"""Pipeline-specific tests: hazards, flushes, precise traps, and timing.

Every program still runs in lockstep with the golden model (so results must
be right), and on top of that each test watches the pipeline's internal
signals to prove the mechanism it targets actually happened:

  - forwarding: results are right with NO stall, every path was used
  - load-use: exactly one bubble per dependent pair
  - taken branches/jumps: exactly two cycles lost, wrong-path work discarded
  - traps: precise (older instructions finish, younger ones vanish)

These are the "done when" checks from the course's pipelining guide, written
as tests instead of eyeballed in a waveform. Pipeline build only.

Run with:   pytest tb/test_pipeline.py -v
"""

import cocotb

from cosim import CORE_SOURCES, dut_reg, dut_word, run_asm
from iss import ILLEGAL_INSTR, MACHINE_ECALL, MISALIGNED_LOAD, MISALIGNED_STORE, MASK
from runner import run
from asm import REGS

FWD_EXMEM, FWD_MEMWB = 1, 2


def u(h) -> int:
    return int(h.value)


class Events:
    """Counts pipeline events, sampled once per cycle by lockstep's on_cycle."""

    def __init__(self):
        self.stall = self.redirect = self.trap = 0
        self.fwd = {("a", FWD_EXMEM): 0, ("a", FWD_MEMWB): 0,
                    ("b", FWD_EXMEM): 0, ("b", FWD_MEMWB): 0}
        self.bypass = 0           # WB -> ID write-through used
        self.stall_runs = []      # length of each run of consecutive stall cycles
        self._run = 0

    def __call__(self, dut):
        c = dut.u_core
        stall = u(c.stall)
        self.stall += stall
        if stall:
            self._run += 1
        elif self._run:
            self.stall_runs.append(self._run)
            self._run = 0
        self.redirect += u(c.redirect_x)
        self.trap += u(c.trap_m)
        if u(c.x_valid):
            for side, sig in (("a", c.fwd_a), ("b", c.fwd_b)):
                v = u(sig)
                if v:
                    self.fwd[(side, v)] += 1
        if u(c.d_valid) and u(c.w_rf_we):
            w_rd = u(c.w_rd)
            if w_rd and w_rd in (u(c.rs1_d), u(c.rs2_d)):
                self.bypass += 1


def check_regs(dut, expected: dict[str, int]) -> None:
    for name, want in expected.items():
        got = dut_reg(dut, REGS[name])
        assert got == want & MASK, f"{name}: core {got:#010x}, expected {want & MASK:#010x}"


async def measure(dut, source: str, handler: bool = False) -> tuple[Events, int]:
    """Run a program; return its events and its overhead = cycles - instructions
    (the cycles lost to bubbles, plus a constant pipeline fill)."""
    ev = Events()
    _, st = await run_asm(dut, source, handler=handler, on_cycle=ev)
    return ev, st.cycles - st.instructions


# ---------------------------------------------------------------------------
# Data hazards

@cocotb.test()
async def test_forwarding_no_stalls(dut):
    """Back-to-back dependent arithmetic, no nops. Every forwarding path and
    the WB->ID bypass get used, results are right, and nothing stalls: the
    pipeline isn't quietly relying on stalls to cover a missing forward."""
    ev = Events()
    await run_asm(dut, """
        li   x1, 3
        addi x2, x1, 1        # x1 at distance 1 (EX/MEM -> rs1)        x2 = 4
        add  x3, x1, x2       # x2 dist 1 (EX/MEM -> rs2), x1 dist 2 (MEM/WB -> rs1)  x3 = 7
        sub  x4, x3, x1       # x3 dist 1, x1 dist 3 (WB -> ID bypass)  x4 = 4
        add  x5, x4, x3       # x4 dist 1, x3 dist 2                    x5 = 11
        xor  x6, x2, x5       # x2 dist 4 (register file), x5 dist 1    x6 = 15
        sll  x7, x6, x2       # rs2 dist 5, rs1 dist 1                  x7 = 240
        or   x8, x1, x7       # rs1 dist 7 (rf), x7 dist 1 -> rs2       x8 = 243
        add  x9, x8, x8       # same register on both sides             x9 = 486
        sw   x9, 0x400(x0)    # store data forwarded (rs2, dist 1)
        lw   x10, 0x400(x0)   #                                         x10 = 486
    """, on_cycle=ev)
    check_regs(dut, {"x2": 4, "x3": 7, "x4": 4, "x5": 11, "x6": 15, "x7": 240,
                     "x8": 243, "x9": 486, "x10": 486})
    assert ev.stall == 0, f"{ev.stall} stall cycles in a program with no load-use pair"
    for path, n in ev.fwd.items():
        assert n > 0, f"forwarding path {path} never used"
    assert ev.bypass > 0, "WB->ID bypass never used"


@cocotb.test()
async def test_forwarding_priority(dut):
    """Two in-flight writers to the same register: the reader must get the
    NEWER one (EX/MEM beats MEM/WB). The guide's 'rare-looking,
    data-dependent' bug."""
    await run_asm(dut, """
        li   x5, 1
        li   x5, 2            # newer write
        add  x6, x5, x0       # x6 = 2, not 1
        li   x7, 10
        li   x7, 20
        li   x7, 30           # three writers in flight
        add  x8, x0, x7       # rs2 side: x8 = 30
        addi x9, x0, 7
        lw   x9, 0x404(x0)    # newer write is a load (value 0) ...
        nop
        add  x10, x9, x9      # ... from MEM/WB, beating the register file: x10 = 0
    """)
    check_regs(dut, {"x6": 2, "x8": 30, "x10": 0})


@cocotb.test()
async def test_x0_never_forwarded(dut):
    """A write to x0 is discarded, so nothing may forward it."""
    await run_asm(dut, """
        li   x1, 77
        addi x0, x1, 5        # write to x0: dropped
        add  x2, x0, x0       # EX/MEM holds rd = 0 with value 82: must read 0
        addi x0, x1, 9
        nop
        add  x3, x0, x1       # MEM/WB: x3 = 77
        lw   x0, 0x408(x0)    # load into x0, then use x0 right away:
        add  x4, x0, x1       # no stall needed, x4 = 77
    """)
    check_regs(dut, {"x2": 0, "x3": 77, "x4": 77})


@cocotb.test()
async def test_load_use_patterns(dut):
    """Every way an instruction can consume a load result one cycle later,
    each with a known answer; plus a CSR read consumed immediately."""
    ev = Events()
    await run_asm(dut, """
        li   t0, 0x2000
        li   t1, 0x1234
        sw   t1, 0(t0)
        sw   t0, 16(t0)       # mem[0x2010] = 0x2000 (an address)
        lw   a0, 0(t0)
        add  a1, a0, a0       # ALU rs1 and rs2           a1 = 0x2468
        lw   a2, 0(t0)
        sw   a2, 8(t0)        # store DATA from the load  mem[0x2008] = 0x1234
        lw   a3, 16(t0)
        sw   t1, 4(a3)        # store ADDRESS from the load (0x2004 <- 0x1234)
        lw   a4, 16(t0)
        lw   a5, 0(a4)        # load address from a load  a5 = 0x1234
        lw   a6, 0(t0)
        beq  a6, t1, ok1      # branch on a load result: taken
        li   s0, -1
    ok1:
        auipc s1, 0
        sw   s1, 12(t0)
        lw   s2, 12(t0)
        jalr ra, 20(s2)       # jump through a loaded address: to ok2
        li   s3, -1
    ok2:
        csrw mscratch, t1
        csrr s4, mscratch
        addi s5, s4, 1        # CSR read consumed immediately   s5 = 0x1235
        lw   s6, 0(t0)
        addi s7, x0, 1        # independent: no stall
        add  s8, s6, s7       # load at distance 2: no stall    s8 = 0x1235
        lw   s9, 0(t0)        # s9 = x25, then two false dependencies: fields
        csrrwi s10, mscratch, 25  # that look like rs1 = x25 but are immediates
        lw   s11, 0(t0)       # (CSR uimm, LUI imm[7:3]). Neither may stall.
        lui  t3, 216          # 216 = 27 << 3: instr[19:15] = 27 = s11
    """, on_cycle=ev)
    check_regs(dut, {"a1": 0x2468, "a5": 0x1234, "s0": 0, "s3": 0,
                     "s5": 0x1235, "s8": 0x1235})
    assert dut_word(dut, 0x2008) == 0x1234
    assert dut_word(dut, 0x2004) == 0x1234
    # one stall per pair: add, sw-data, sw-addr, lw-addr, beq, jalr, CSR use;
    # none for the distance-2 load or the two false dependencies
    assert ev.stall == 7, f"expected 7 stall cycles, saw {ev.stall}"
    assert set(ev.stall_runs) == {1}, f"every stall must last exactly 1 cycle: {ev.stall_runs}"


@cocotb.test()
async def test_load_use_costs_one_cycle(dut):
    """Timing: a dependent load-use pair costs exactly one cycle more than
    the same code with the dependency removed."""
    K = 6
    pair_dep = "lw a0, 0(t0)\nadd a1, a0, a0\n" * K
    pair_ind = "lw a0, 0(t0)\nadd a1, a2, a2\n" * K
    ev_d, over_d = await measure(dut, "li t0, 0x2000\n" + pair_dep)
    ev_i, over_i = await measure(dut, "li t0, 0x2000\n" + pair_ind)
    assert over_d - over_i == K, f"{K} load-use pairs cost {over_d - over_i} cycles"
    assert (ev_d.stall, ev_i.stall) == (K, 0)


# ---------------------------------------------------------------------------
# Control hazards

@cocotb.test()
async def test_wrong_path_is_discarded(dut):
    """After every taken branch and jump, the two instructions already
    fetched behind it must leave no trace: no register write, no store, no
    trap (an illegal instruction on the wrong path must NOT trap)."""
    ev = Events()
    await run_asm(dut, """
        li   a0, 5
        beq  x0, x0, t1
        addi s0, x0, 1        # wrong path
        sw   a0, 0x500(x0)    # wrong path store
    t1:
        jal  x0, t2
        .word 0               # wrong path: illegal instruction
        ecall                 # wrong path: ecall
    t2:
        auipc t0, 0
        jalr x0, 16(t0)
        addi s1, x0, 1        # wrong path
        lw   s2, 0x600(x0)    # wrong path
    t3:
        bne  a0, x0, t4
        csrw mscratch, a0     # wrong path CSR write
        mret                  # wrong path MRET
    t4:
        csrr s3, mscratch     # still 0
        blt  a0, x0, never    # not taken: falls through, nothing flushed
        addi s4, x0, 4
    never:
    """, on_cycle=ev)
    check_regs(dut, {"s0": 0, "s1": 0, "s2": 0, "s3": 0, "s4": 4})
    assert dut_word(dut, 0x500) == 0, "wrong-path store reached memory"
    assert ev.trap == 0, "a wrong-path instruction trapped"
    assert ev.redirect == 4, f"expected 4 redirects (beq, jal, jalr, bne), saw {ev.redirect}"


@cocotb.test()
async def test_taken_branch_costs_two_cycles(dut):
    """Timing: each taken branch, JAL and JALR loses exactly two cycles
    (EX resolution flushes two); a not-taken branch loses none."""
    K = 5
    taken = "".join(f"beq x0, x0, b{k}\naddi a0, a0, 1\nb{k}:\n" for k in range(K))
    not_taken = "".join(f"bne x0, x0, n{k}\naddi a0, a0, 1\nn{k}:\n" for k in range(K))
    jals = "".join(f"jal x0, j{k}\naddi a0, a0, 1\nj{k}:\n" for k in range(K))
    jalrs = "".join(f"auipc t0, 0\njalr x0, 12(t0)\naddi a0, a0, 1\n" for _ in range(K))
    straight = "addi a0, a0, 1\n" * K
    _, base = await measure(dut, straight)
    _, over_t = await measure(dut, taken)
    _, over_n = await measure(dut, not_taken)
    _, over_j = await measure(dut, jals)
    _, over_r = await measure(dut, jalrs)
    assert over_n - base == 0, f"not-taken branches cost {over_n - base} cycles"
    assert over_t - base == 2 * K, f"{K} taken branches cost {over_t - base} cycles"
    assert over_j - base == 2 * K, f"{K} JALs cost {over_j - base} cycles"
    assert over_r - base == 2 * K, f"{K} JALRs cost {over_r - base} cycles"


@cocotb.test()
async def test_fence_i_self_modifying_code(dut):
    """Store a new instruction over one that's already in the pipeline, then
    FENCE.I. The new instruction must run. (Without FENCE.I refetching, the
    old one was fetched before the store landed.)"""
    await run_asm(dut, """
        auipc t1, 0           # t1 = address of this instruction
        li   t0, 0x02a00513   # encoding of: addi a0, x0, 42   (lui + addi)
        sw   t0, 20(t1)       # overwrite the instruction at t1 + 20
        fence.i
        addi a0, x0, 1        # t1 + 20: replaced before it runs
    """)
    check_regs(dut, {"a0": 42})


# ---------------------------------------------------------------------------
# Precise traps

@cocotb.test()
async def test_traps_are_precise(dut):
    """Older instructions complete, younger ones vanish, and the handler
    sees the right mepc. Covers a trap with a store right behind it, a trap
    in MEM while a branch resolves in EX, back-to-back traps, and a CSR write
    to mtvec immediately followed by a trap."""
    causes = []
    ev = Events()
    await run_asm(dut, """
        li   a0, 9
        addi a1, a0, 1        # older than the trap: must complete (a1 = 10)
        ecall                 # handler skips it
        sw   a0, 0x700(x0)    # younger: runs once, AFTER the handler
        .word 0               # illegal ...
        beq  x0, x0, over     # ... while this taken branch is in EX: trap wins
        addi s0, x0, -1
    over:
        .word 0               # three traps back to back
        .word 0
        ecall
        li   t0, 0x2001
        lw   a2, 0(t0)        # misaligned load traps; a2 keeps its old value 0
        add  a3, a2, a0       # dependent on the trapped load: a3 = 9
        sh   a0, 0(t0)        # misaligned store traps; memory untouched
        auipc t2, 0
        addi t2, t2, 24       # t2 = address of new_handler below
        csrw mtvec, t2        # move the trap vector ...
        ecall                 # ... and trap on the very next instruction
        j    after
        nop
    new_handler:              # (t2 + 0)
        li   s1, 123
        csrr t6, mepc
        addi t6, t6, 4
        csrw mepc, t6
        mret
    after:
    """, handler=True, on_trap=lambda r: causes.append(r.cause), on_cycle=ev)
    check_regs(dut, {"a1": 10, "s0": 0, "a2": 0, "a3": 9, "s1": 123})
    assert dut_word(dut, 0x700) == 9
    assert dut_word(dut, 0x2000) == 0, "misaligned store wrote memory"
    assert causes == [MACHINE_ECALL, ILLEGAL_INSTR, ILLEGAL_INSTR, ILLEGAL_INSTR,
                      MACHINE_ECALL, MISALIGNED_LOAD, MISALIGNED_STORE, MACHINE_ECALL], causes
    assert ev.trap == len(causes)


def test_pipeline():
    run("tb_top", CORE_SOURCES, "test_pipeline", build_name="tb_top_pipeline_hazards")
