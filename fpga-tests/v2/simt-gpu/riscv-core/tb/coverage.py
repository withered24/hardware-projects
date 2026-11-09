"""Functional coverage for the random regression (Milestone 4).

Two kinds of bins:
  - ISA bins, sampled per retired instruction from the golden model's view,
    so they mean the same on either core: every opcode, every branch taken
    and not taken, every trap cause, read-after-write at retire distance
    1/2/3 for each producer type (ALU, load, CSR, jump link) on each
    operand, writes to x0, and a few named hazard shapes.
  - Pipeline bins, sampled every cycle from the pipelined core's internal
    signals: each forwarding path, the WB->ID bypass, each stall reason,
    each redirect kind, and the collisions (trap and branch in the same
    cycle, a store killed by an older trap, ...).
  - Wait bins (the "wait" core: random data-memory wait states), for the
    freeze logic: a load and a store waiting, waits of 2+ cycles, and what
    is stuck behind MEM while it waits (a forwarded operand that has to be
    latched, a taken branch, a load-use stall, a store in EX).

A bin that's still zero after a run is a hole: either the generator never
made that situation, or the design can't reach it. Holes are listed in the
report; docs/verification.md says what was done about each one.
"""

import json

R_NAMES = {(0, 0): "add", (0, 0x20): "sub", (1, 0): "sll", (2, 0): "slt", (3, 0): "sltu",
           (4, 0): "xor", (5, 0): "srl", (5, 0x20): "sra", (6, 0): "or", (7, 0): "and"}
I_NAMES = {0: "addi", 2: "slti", 3: "sltiu", 4: "xori", 6: "ori", 7: "andi",
           1: "slli", 5: "srli"}
LOAD_NAMES = {0: "lb", 1: "lh", 2: "lw", 4: "lbu", 5: "lhu"}
STORE_NAMES = {0: "sb", 1: "sh", 2: "sw"}
BR_NAMES = {0: "beq", 1: "bne", 4: "blt", 5: "bge", 6: "bltu", 7: "bgeu"}
CSR_NAMES = {1: "csrrw", 2: "csrrs", 3: "csrrc", 5: "csrrwi", 6: "csrrsi", 7: "csrrci"}
SYS_WORDS = {0x0000_0073: "ecall", 0x0010_0073: "ebreak", 0x3020_0073: "mret",
             0x1050_0073: "wfi"}

MNEMONICS = (list(R_NAMES.values()) + ["addi", "slti", "sltiu", "xori", "ori", "andi",
             "slli", "srli", "srai"] + list(LOAD_NAMES.values()) + list(STORE_NAMES.values())
             + list(BR_NAMES.values()) + ["lui", "auipc", "jal", "jalr", "fence", "fence.i"]
             + list(CSR_NAMES.values()) + ["ecall", "ebreak", "mret", "wfi", "illegal"])
TRAP_CAUSES = {0: "misaligned_fetch", 2: "illegal", 3: "breakpoint",
               4: "misaligned_load", 6: "misaligned_store", 11: "ecall"}
PRODUCERS = ["alu", "load", "csr", "link"]

NAMED_HAZARDS = [
    "load_to_branch",        # lw x; beq x, ...       (stall, then the branch uses it)
    "load_to_store_data",    # lw x; sw x, ...        (stall on the data operand)
    "load_to_address",       # lw p; lw/sw .., 0(p)   (stall on the address operand)
    "csr_to_use",            # csrr x; add .., x      (CSR-use stall)
    "auipc_to_jalr",         # auipc a; jalr 0(a)     (forward into the jump target)
    "mepc_write_to_mret",    # csrw mepc; mret        (MRET reads the CSR just written)
    "store_to_fence_i",      # sw; fence.i            (store must land before the refetch)
]

PIPE_BINS = [
    "fwd_exmem.rs1", "fwd_exmem.rs2", "fwd_memwb.rs1", "fwd_memwb.rs2",
    "fwd_both_operands",       # both operands forwarded in the same cycle
    "fwd_memwb_from_load",     # a load's value forwarded from MEM/WB (after the stall)
    "fwd_memwb_from_csr",
    "bypass_wb_id.rs1", "bypass_wb_id.rs2",
    "stall.load", "stall.csr",
    "redirect.branch", "redirect.jal", "redirect.jalr", "redirect.fence_i",
    "trap", "mret",
    "trap_beats_redirect",     # MEM trap and EX taken branch in the same cycle
    "mret_beats_redirect",
    "trap_during_stall",       # MEM trap while ID is stalled on a load-use
    "store_killed_by_trap",    # a valid store in EX cancelled by an older trap...
    "store_killed_by_mret",    # ...or by an older MRET (one bin for both hid a hole)
    "younger_exception_flushed",  # an exception in EX squashed by an older one in MEM
]

WAIT_BINS = [
    "load", "store",           # the access MEM is waiting on
    "multi_cycle",             # a wait of 2+ cycles in a row
    "fwd_latch",               # EX forwards from WB while frozen: must latch it
    "redirect_pending",        # a taken branch/jump in EX waits behind MEM
    "stall_pending",           # a load-use stall in ID waits behind MEM
    "store_in_ex",             # a younger store sits in EX (must not write twice)
]


def _f(ins, lo, n):
    return (ins >> lo) & ((1 << n) - 1)


def mnemonic(ins: int) -> str:
    op, f3, f7 = ins & 0x7F, _f(ins, 12, 3), ins >> 25
    if op == 0x33:
        return R_NAMES.get((f3, f7), "illegal")
    if op == 0x13:
        if f3 == 5:
            return "srai" if f7 == 0x20 else "srli"
        return I_NAMES[f3]
    if op == 0x03:
        return LOAD_NAMES.get(f3, "illegal")
    if op == 0x23:
        return STORE_NAMES.get(f3, "illegal")
    if op == 0x63:
        return BR_NAMES.get(f3, "illegal")
    if op == 0x37:
        return "lui"
    if op == 0x17:
        return "auipc"
    if op == 0x6F:
        return "jal"
    if op == 0x67:
        return "jalr" if f3 == 0 else "illegal"
    if op == 0x0F:
        return {0: "fence", 1: "fence.i"}.get(f3, "illegal")
    if op == 0x73:
        if f3 == 0:
            return SYS_WORDS.get(ins, "illegal")
        return CSR_NAMES.get(f3, "illegal")
    return "illegal"


def reads(ins: int) -> tuple[bool, bool]:
    """(reads rs1, reads rs2), the same rule as the decoder's uses_rs1/uses_rs2."""
    op, f3 = ins & 0x7F, _f(ins, 12, 3)
    if op in (0x63, 0x23, 0x33):
        return True, True
    if op in (0x67, 0x03, 0x13):
        return True, False
    if op == 0x73 and f3 not in (0, 4):
        return not f3 & 4, False
    return False, False


def producer(ins: int) -> str | None:
    op = ins & 0x7F
    if op in (0x33, 0x13, 0x37, 0x17):
        return "alu"
    if op == 0x03:
        return "load"
    if op == 0x73 and _f(ins, 12, 3) not in (0, 4):
        return "csr"
    if op in (0x6F, 0x67):
        return "link"
    return None


class Coverage:
    def __init__(self, pipeline: bool, wait: bool = False):
        self.pipeline = pipeline
        self.wait = wait
        self.prev_hold = False
        self.bins: dict[str, int] = {}
        for m in MNEMONICS:
            self.bins[f"op.{m}"] = 0
        for b in BR_NAMES.values():
            self.bins[f"branch.{b}.taken"] = 0
            self.bins[f"branch.{b}.not_taken"] = 0
        for c in TRAP_CAUSES.values():
            self.bins[f"trap.{c}"] = 0
        for p in PRODUCERS:
            for d in (1, 2, 3):
                for o in ("rs1", "rs2"):
                    self.bins[f"raw.{p}.d{d}.{o}"] = 0
            self.bins[f"x0_write.{p}"] = 0
        for h in NAMED_HAZARDS:
            self.bins[f"hazard.{h}"] = 0
        if pipeline:
            for b in PIPE_BINS:
                self.bins[f"pipe.{b}"] = 0
        if wait:
            for b in WAIT_BINS:
                self.bins[f"wait.{b}"] = 0
        self.programs = 0
        self.instructions = 0
        self.cycles = 0
        self.new_program()

    def hit(self, name: str) -> None:
        self.bins[name] += 1          # KeyError = a typo in a bin name

    def new_program(self) -> None:
        self.idx = 0
        self.writer: dict[int, tuple[int, str]] = {}   # reg -> (retire index, producer)
        self.prev = None

    # ---- per retired instruction (golden model's view) ----
    def sample_retire(self, r) -> None:
        ins, m = r.instr, mnemonic(r.instr)
        self.instructions += 1
        self.hit(f"op.{m}")
        prev = self.prev
        if prev is not None and not prev.trap:
            pm = mnemonic(prev.instr)
            if pm in BR_NAMES.values():
                self.hit(f"branch.{pm}.{'taken' if r.pc != prev.pc + 4 else 'not_taken'}")
        if r.trap:
            self.hit(f"trap.{TRAP_CAUSES[r.cause]}")

        rs1, rs2 = _f(ins, 15, 5), _f(ins, 20, 5)
        u1, u2 = reads(ins)
        for used, reg, name in ((u1, rs1, "rs1"), (u2, rs2, "rs2")):
            if used and reg and reg in self.writer:
                widx, ptype = self.writer[reg]
                d = self.idx - widx
                if d <= 3:
                    self.hit(f"raw.{ptype}.d{d}.{name}")
                if d == 1 and ptype == "load":
                    op = ins & 0x7F
                    if op == 0x63:
                        self.hit("hazard.load_to_branch")
                    elif op == 0x23 and name == "rs2":
                        self.hit("hazard.load_to_store_data")
                    if op in (0x03, 0x23) and name == "rs1":
                        self.hit("hazard.load_to_address")
                if d == 1 and ptype == "csr":
                    self.hit("hazard.csr_to_use")
        if prev is not None:
            pm = mnemonic(prev.instr)
            if pm == "auipc" and m == "jalr" and rs1 == _f(prev.instr, 7, 5):
                self.hit("hazard.auipc_to_jalr")
            if m == "mret" and prev.instr >> 20 == 0x341 and pm.startswith("csrr"):
                self.hit("hazard.mepc_write_to_mret")
            if m == "fence.i" and prev.store_strb:
                self.hit("hazard.store_to_fence_i")

        p = producer(ins)
        rd = _f(ins, 7, 5)
        if p and not r.trap:
            if rd == 0:
                self.hit(f"x0_write.{p}")
            else:
                self.writer[rd] = (self.idx, p)
        self.idx += 1
        self.prev = r

    # ---- per cycle (pipelined core's internals) ----
    def sample_cycle(self, dut) -> None:
        self.cycles += 1
        if not self.pipeline:
            return
        c = dut.u_core
        v = lambda h: int(h.value)
        x_valid, x_instr = v(c.x_valid), v(c.x_instr)
        fa, fb = v(c.fwd_a), v(c.fwd_b)
        stall, redirect, trap, mret = v(c.stall), v(c.redirect_x), v(c.trap_m), v(c.mret_m)
        if self.wait:
            hold = bool(v(c.hold))
            if hold:
                self.hit("wait.load" if v(c.m_instr) & 0x7F == 0x03 else "wait.store")
                if self.prev_hold:
                    self.hit("wait.multi_cycle")
                if x_valid:
                    u1, u2 = reads(x_instr)
                    if (u1 and fa == 2) or (u2 and fb == 2):
                        self.hit("wait.fwd_latch")
                    if x_instr & 0x7F == 0x23:
                        self.hit("wait.store_in_ex")
                if redirect:
                    self.hit("wait.redirect_pending")
                if stall:
                    self.hit("wait.stall_pending")
            self.prev_hold = hold
        if x_valid:
            u1, u2 = reads(x_instr)
            fa, fb = (fa if u1 else 0), (fb if u2 else 0)
            for f, o in ((fa, "rs1"), (fb, "rs2")):
                if f == 1:
                    self.hit(f"pipe.fwd_exmem.{o}")
                elif f == 2:
                    self.hit(f"pipe.fwd_memwb.{o}")
            if fa and fb:
                self.hit("pipe.fwd_both_operands")
            if 2 in (fa, fb):
                wp = producer(v(c.w_instr))
                if wp == "load":
                    self.hit("pipe.fwd_memwb_from_load")
                elif wp == "csr":
                    self.hit("pipe.fwd_memwb_from_csr")
            if x_instr & 0x7F == 0x23:
                if trap:
                    self.hit("pipe.store_killed_by_trap")
                if mret:
                    self.hit("pipe.store_killed_by_mret")
            if trap and (x_instr in (0x0000_0073, 0x0010_0073) or mnemonic(x_instr) == "illegal"):
                self.hit("pipe.younger_exception_flushed")
        if stall:
            self.hit("pipe.stall.load" if x_instr & 0x7F == 0x03 else "pipe.stall.csr")
        if redirect:
            kind = {0x63: "branch", 0x6F: "jal", 0x67: "jalr", 0x0F: "fence_i"}[x_instr & 0x7F]
            self.hit(f"pipe.redirect.{kind}")
        if trap:
            self.hit("pipe.trap")
            if redirect:
                self.hit("pipe.trap_beats_redirect")
            if stall:
                self.hit("pipe.trap_during_stall")
        if mret:
            self.hit("pipe.mret")
            if redirect:
                self.hit("pipe.mret_beats_redirect")
        if v(c.d_valid) and not stall and not (trap or mret or redirect) and v(c.w_rf_we):
            w_rd, d_instr = v(c.w_rd), v(c.d_instr)
            u1, u2 = reads(d_instr)
            if w_rd and u1 and w_rd == _f(d_instr, 15, 5):
                self.hit("pipe.bypass_wb_id.rs1")
            if w_rd and u2 and w_rd == _f(d_instr, 20, 5):
                self.hit("pipe.bypass_wb_id.rs2")

    # ---- report ----
    def holes(self) -> list[str]:
        return [k for k, n in self.bins.items() if n == 0]

    def report(self) -> str:
        hit = len(self.bins) - len(self.holes())
        lines = [f"coverage: {hit}/{len(self.bins)} bins hit "
                 f"({self.programs} programs, {self.instructions} instructions, {self.cycles} cycles)"]
        groups: dict[str, list[tuple[str, int]]] = {}
        for k, n in self.bins.items():
            groups.setdefault(k.split(".")[0], []).append((k, n))
        for g, items in groups.items():
            h = sum(1 for _, n in items if n)
            lo = min(items, key=lambda t: t[1])
            lines.append(f"  {g:10s} {h:3d}/{len(items):<3d}  least hit: {lo[0]} ({lo[1]})")
        if self.holes():
            lines.append("  HOLES: " + ", ".join(self.holes()))
        return "\n".join(lines)

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"programs": self.programs, "instructions": self.instructions,
                       "cycles": self.cycles, "bins": self.bins}, f, indent=1)
