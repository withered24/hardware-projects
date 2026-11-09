"""Random SIMT kernels: data-dependent if/else, loops, calls and early exits.

Each thread starts from registers seeded with its global id, so threads in a
warp take different paths through randomly nested branches and loops with
data-dependent trip counts. They gather from a shared read-only table
(uncoalesced, every lane somewhere else), scribble on their own private
scratch area, and finally dump their working registers. Nothing is shared
between threads, so the golden model's thread-by-thread run must leave
exactly the same memory as the GPU's interleaved one.

Register use:  a0 args,  s10 table base,  s11 this thread's output area,
t5 address temp,  t6/t4/t3 loop counters by nesting depth,  ra for calls,
everything in WORK is fair game for the random code.
"""

import random

from kernels import GLOBAL_ID

WORK = ["t0", "t1", "t2", "s0", "s1", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]
LOOP_REGS = ["t6", "t4", "t3"]
OUT_BYTES = 128                      # per thread: register dump, then scratch at +64
TABLE_WORDS = 64

R_OPS = ["add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and", "mul"]
I_OPS = ["addi", "slti", "sltiu", "xori", "ori", "andi"]
SHIFTS = ["slli", "srli", "srai"]
BRANCHES = ["beq", "bne", "blt", "bge", "bltu", "bgeu"]


class Gen:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.labels = 0

    def label(self) -> str:
        self.labels += 1
        return f"L{self.labels}"

    def reg(self) -> str:
        return self.rng.choice(WORK)

    def op(self) -> list[str]:
        r = self.rng
        k = r.random()
        if k < 0.45:
            return [f"{r.choice(R_OPS)} {self.reg()}, {self.reg()}, {self.reg()}"]
        if k < 0.65:
            return [f"{r.choice(I_OPS)} {self.reg()}, {self.reg()}, {r.randrange(-2048, 2048)}"]
        if k < 0.75:
            return [f"{r.choice(SHIFTS)} {self.reg()}, {self.reg()}, {r.randrange(32)}"]
        if k < 0.80:
            return [f"lui {self.reg()}, {r.randrange(1 << 20)}"]
        if k < 0.83:
            return [f"auipc {self.reg()}, {r.randrange(1 << 20)}"]
        if k < 0.93:                     # gather from the shared table
            ld, mask = r.choice([("lw", 252), ("lh", 254), ("lhu", 254), ("lb", 255), ("lbu", 255)])
            return [f"andi t5, {self.reg()}, {mask}", "add t5, t5, s10", f"{ld} {self.reg()}, 0(t5)"]
        st, mask = r.choice([("sw", 60), ("sh", 62), ("sb", 63)])   # private scratch
        return [f"andi t5, {self.reg()}, {mask}", "add t5, t5, s11", f"{st} {self.reg()}, 64(t5)"]

    def block(self, depth: int, n: int) -> list[str]:
        out: list[str] = []
        for _ in range(n):
            k = self.rng.random()
            if depth < 3 and k < 0.15:
                out += self.diamond(depth)
            elif depth < 3 and k < 0.25:
                out += self.loop(depth)
            elif k < 0.28:
                out.append("jal sub0")
            elif depth > 0 and k < 0.295:
                out.append("ecall")                         # this thread is done early
            else:
                out += self.op()
        return out

    def diamond(self, depth: int) -> list[str]:
        els, end = self.label(), self.label()
        cond = f"{self.rng.choice(BRANCHES)} {self.reg()}, {self.reg()}, {els}"
        then = self.block(depth + 1, self.rng.randrange(1, 5))
        other = self.block(depth + 1, self.rng.randrange(0, 5))
        return [cond, *then, f"j {end}", f"{els}:", *other, f"{end}:"]

    def loop(self, depth: int) -> list[str]:
        c, top = LOOP_REGS[depth], self.label()
        trips = [f"andi {c}, {self.reg()}, 3", f"addi {c}, {c}, 1"]   # 1..4 trips, per thread
        body = self.block(depth + 1, self.rng.randrange(1, 5))
        return [*trips, f"{top}:", *body, f"addi {c}, {c}, -1", f"bnez {c}, {top}"]

    def kernel(self) -> str:
        r = self.rng
        lines = [GLOBAL_ID,
                 "lw s10, 0(a0)", "lw s11, 4(a0)",
                 f"li t5, {OUT_BYTES}", "mul t5, t5, t0", "add s11, s11, t5"]
        for reg in WORK[1:]:                         # t0 (the global id) last
            lines += [f"li {reg}, {r.randrange(-2**31, 2**31)}",
                      f"{r.choice(['add', 'xor', 'mul', 'sub'])} {reg}, {reg}, t0"]
        lines += [f"mul t0, t0, {WORK[1]}"]
        lines += self.block(0, r.randrange(20, 40))
        lines += [f"sw {reg}, {4 * i}(s11)" for i, reg in enumerate(WORK)]
        lines += ["ecall",
                  "sub0:", f"xor {WORK[1]}, {WORK[1]}, {WORK[2]}", f"addi {WORK[3]}, {WORK[3]}, 7", "ret"]
        return "\n".join(lines)
