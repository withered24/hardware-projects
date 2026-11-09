"""Constrained-random RV32I program generator (Milestone 4).

Each seed gives one program: assembly text for tb/asm.py plus an initial
data image. tb/test_random.py runs it in lockstep against the golden model,
so the generator never has to know the right answers. It only has to make
programs that are legal enough to finish and nasty enough to find bugs.

Constraints (what keeps a random program runnable)
  - x28, x29, x30, x31 are reserved: x28 = pointer table base, x29 = data
    base, x30 = loop counter (and DONE), x31 = the trap handler's scratch.
    Random instructions only write x0..x27.
  - Loads and stores go through x29 (offset 0..0x7FF) or through a pointer
    loaded from the table at x28, so every access lands in the data region
    0x8000..0x87FF. Code lives at 0 and only the self-modifying-code block
    writes it.
  - Control flow only goes forward, except loops: a counted loop on x30 with
    1..4 iterations. So every program terminates.
  - Traps are welcome: the handler (cosim.HANDLER_PRELUDE) skips the
    instruction that trapped and carries on.

Bias (what makes it find bugs)
  - Each program picks 5 "hot" registers and uses them for 75% of operands,
    so read-after-write chains at distance 1, 2, 3 (the forwarding and
    stall cases) are common instead of rare.
  - Constants come from a list of edge values (0, -1, 0x80000000, ...) half
    the time.
  - Some blocks are templates for one specific hazard: pointer chasing
    (load -> address), auipc -> jalr, csrw mepc -> mret, store -> fence.i ->
    execute the stored instruction.

Run `python tb/rand_prog.py 7` to print the program for seed 7.
"""

import random
import sys

from asm import I_OPS, LOADS, R_OPS, STORES, BRANCHES, SHIFT_OPS, _li_parts, enc_i

DATA_BASE = 0x8000           # x29; random loads and stores land in DATA_BASE..+0x7FF
DATA_WORDS = 512
PTR_BASE = 0x8800            # x28; a table of pointers into the data region
PTR_WORDS = 64

POOL = range(0, 28)          # registers random instructions may write
EDGE = [0, 1, -1, 2, 31, 32, 0x7FF, -2048, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF,
        0x8000, 0xFFFF_0000, 0x5555_5555, 0xAAAA_AAAA]

# Encodings that must raise an illegal-instruction trap (both the model and
# the core): all-zeros, all-ones, custom-0, a reserved branch funct3, a
# reserved load funct3, a reserved store funct3, SYSTEM funct3 = 100, an
# unimplemented CSR (ustatus), a write to a read-only CSR (mhartid), and a
# non-zero SYSTEM funct3 = 0 word that isn't ECALL/EBREAK/MRET/WFI (uret).
ILLEGAL_WORDS = [0x0000_0000, 0xFFFF_FFFF, 0x0000_000B, 0x0000_2063, 0x0000_3003,
                 0x0000_3023, 0x0000_4073, 0x0000_2073, 0xF140_1073, 0x0020_0073]

# Top-level item weights. Loops only appear at the top level.
WEIGHTS = {
    "alu_r": 14, "alu_i": 12, "shift": 5, "lui": 2, "auipc": 2, "li": 3,
    "load": 10, "store": 8, "ptr_chase": 4, "branch": 10, "jal": 3,
    "jalr": 3, "loop": 3, "csr": 6, "trap": 4, "fence": 1, "fence_i": 1,
    "smc": 1, "mret": 1,
}


class Program:
    def __init__(self, seed: int, items: int = 120, weights: dict | None = None):
        self.seed = seed
        self.r = random.Random(seed)
        self.weights = dict(WEIGHTS, **(weights or {}))
        self.lines: list[str] = []
        self.pending: list[list] = []    # [label, items still to emit before it]
        self.nlabel = 0
        self.hot = self.r.sample(range(1, 28), 5)
        self.data = self._data_image()
        self._prologue()
        self._block(items, depth=0)
        self._flush_labels()

    # ---- output ----
    @property
    def source(self) -> str:
        return "\n".join(self.lines) + "\n"

    def emit(self, line: str) -> None:
        self.lines.append("        " + line)

    def label(self) -> str:
        self.nlabel += 1
        return f"L{self.nlabel}"

    def place(self, name: str) -> None:
        self.lines.append(f"    {name}:")

    def _forward(self, name: str, distance: int) -> None:
        """Put label `name` after `distance` more items (0 = right here)."""
        if distance == 0:
            self.place(name)
        else:
            self.pending.append([name, distance])

    def _item_done(self) -> None:
        for p in self.pending:
            p[1] -= 1
        for p in [p for p in self.pending if p[1] <= 0]:
            self.place(p[0])
            self.pending.remove(p)

    def _flush_labels(self) -> None:
        for name, _ in self.pending:
            self.place(name)
        self.pending = []

    # ---- operands ----
    def reg(self) -> int:
        """A random register, biased toward the hot set."""
        x = self.r.random()
        if x < 0.05:
            return 0
        if x < 0.80:
            return self.r.choice(self.hot)
        return self.r.choice(POOL)

    def reg_nz(self) -> int:
        while (x := self.reg()) == 0:
            pass
        return x

    def const(self) -> int:
        if self.r.random() < 0.5:
            return self.r.choice(EDGE)
        return self.r.getrandbits(32)

    def imm12(self) -> int:
        if self.r.random() < 0.3:
            return self.r.choice([0, 1, -1, 2047, -2048, 0x7FF, -0x800])
        return self.r.randint(-2048, 2047)

    # ---- data image ----
    def _data_image(self) -> dict[int, int]:
        """{byte address: word}. Data words are random; pointers point into
        the data region, mostly aligned, sometimes off by 1 or 2 (so a load
        through one can trap)."""
        r = self.r
        img = {DATA_BASE + 4 * i: r.getrandbits(32) for i in range(DATA_WORDS)}
        for i in range(PTR_WORDS):
            p = DATA_BASE + 0x40 + 4 * r.randint(0, (0x7C0 - 0x80) // 4)
            if r.random() < 0.1:
                p += r.choice([1, 2])
            img[PTR_BASE + 4 * i] = p
        return img

    # ---- program structure ----
    def _prologue(self) -> None:
        self.emit(f"li   x29, {DATA_BASE}")
        self.emit(f"li   x28, {PTR_BASE}")
        for x in self.hot:
            self.emit(f"li   x{x}, {self.const()}")

    def _block(self, n: int, depth: int) -> None:
        kinds = [k for k in self.weights if depth == 0 or k != "loop"]
        w = [self.weights[k] for k in kinds]
        for _ in range(n):
            kind = self.r.choices(kinds, w)[0]
            getattr(self, "_" + kind)()
            self._item_done()

    # ---- items ----
    def _alu_r(self):
        op = self.r.choice(list(R_OPS))
        self.emit(f"{op} x{self.reg()}, x{self.reg()}, x{self.reg()}")

    def _alu_i(self):
        op = self.r.choice(list(I_OPS))
        self.emit(f"{op} x{self.reg()}, x{self.reg()}, {self.imm12()}")

    def _shift(self):
        op = self.r.choice(list(SHIFT_OPS))
        self.emit(f"{op} x{self.reg()}, x{self.reg()}, {self.r.choice([0, 1, 31, self.r.randint(0, 31)])}")

    def _lui(self):
        self.emit(f"lui  x{self.reg()}, {self.r.getrandbits(20)}")

    def _auipc(self):
        self.emit(f"auipc x{self.reg()}, {self.r.getrandbits(20)}")

    def _li(self):
        self.emit(f"li   x{self.reg()}, {self.const()}")

    def _offset(self, size: int) -> int:
        """Offset from x29: aligned 85% of the time."""
        off = self.r.randint(0, 0x7FF - 3)
        if self.r.random() < 0.85:
            off &= ~(size - 1)
        return off

    def _load(self):
        op = self.r.choice(list(LOADS))
        size = {0: 1, 1: 2, 2: 4, 4: 1, 5: 2}[LOADS[op]]
        self.emit(f"{op} x{self.reg()}, {self._offset(size)}(x29)")

    def _store(self):
        op = self.r.choice(list(STORES))
        size = 1 << STORES[op]
        self.emit(f"{op} x{self.reg()}, {self._offset(size)}(x29)")

    def _ptr_chase(self):
        """Load a pointer, then use it as an address right away (load-use on
        the address operand), sometimes with one instruction in between."""
        p = self.reg_nz()
        self.emit(f"lw   x{p}, {4 * self.r.randrange(PTR_WORDS)}(x28)")
        if self.r.random() < 0.3:
            self.emit(self._filler(avoid=p))
        off = self.r.randint(-64, 60)
        if self.r.random() < 0.5:
            op = self.r.choice(list(LOADS))
            self.emit(f"{op} x{self.reg()}, {off & ~3}(x{p})")
        else:
            op = self.r.choice(list(STORES))
            self.emit(f"{op} x{self.reg()}, {off & ~3}(x{p})")

    def _branch(self):
        op = self.r.choice(list(BRANCHES))
        a = self.reg()
        b = a if self.r.random() < 0.15 else self.reg()
        name = self.label()
        self.emit(f"{op} x{a}, x{b}, {name}")
        self._forward(name, self.r.choice([0, 1, 1, 2, 3, 4]))

    def _jal(self):
        name = self.label()
        self.emit(f"jal  x{self.reg()}, {name}")
        self._forward(name, self.r.choice([0, 1, 2, 3]))

    def _filler(self, avoid: int) -> str:
        rd = self.reg()
        if rd == avoid:
            rd = 0
        return f"addi x{rd}, x{self.reg()}, {self.r.randint(-16, 16)}"

    def _jalr(self):
        """auipc a, 0 ; [one instruction] ; jalr rd, off(a) ; k skipped
        instructions. off lands just past the skipped ones; +1 checks that
        bit 0 is cleared, +2 is a misaligned target (trap, then carry on)."""
        a = self.reg_nz()
        mid = self.r.random() < 0.3
        skip = self.r.randint(0, 3)
        adj = self.r.choices([0, 1, 2], [70, 15, 15])[0]
        off = 4 * (2 + mid + skip) + adj
        self.emit(f"auipc x{a}, 0")
        if mid:
            self.emit(self._filler(avoid=a))
        self.emit(f"jalr x{self.reg()}, {off}(x{a})")
        for _ in range(skip):
            self.emit(self._filler(avoid=-1))

    def _loop(self):
        name = self.label()
        self.emit(f"li   x30, {self.r.randint(1, 4)}")
        self.place(name)
        outer, self.pending = self.pending, []
        self._block(self.r.randint(2, 8), depth=1)
        self._flush_labels()
        self.pending = outer
        self.emit("addi x30, x30, -1")
        self.emit(f"bnez x30, {name}")

    def _csr(self):
        r = self.r
        k = r.randrange(5)
        rw = ["mscratch", "mepc", "mtval", "mcause", "mstatus"]
        if k == 0:
            op = r.choice(["csrrw", "csrrs", "csrrc"])
            self.emit(f"{op} x{self.reg()}, {r.choice(rw)}, x{self.reg()}")
        elif k == 1:
            op = r.choice(["csrrwi", "csrrsi", "csrrci"])
            self.emit(f"{op} x{self.reg()}, {r.choice(rw)}, {r.randint(0, 31)}")
        elif k == 2:
            csr = r.choice(rw + ["misa", "mhartid", "mvendorid", "mie", "mip", "mtvec"])
            self.emit(f"csrr x{self.reg()}, {csr}")
        elif k == 3:
            op = r.choice(["csrw", "csrs", "csrc"])
            self.emit(f"{op} {r.choice(rw)}, x{self.reg()}")
        else:
            # read-modify-write chain: CSR result used right away (CSR-use stall)
            d = self.reg_nz()
            self.emit(f"csrrs x{d}, mscratch, x0")
            self.emit(f"add  x{self.reg()}, x{d}, x{self.reg()}")

    def _trap(self):
        r = self.r
        k = r.randrange(6)
        if k == 0:
            self.emit("ecall")
        elif k == 1:
            self.emit("ebreak")
        elif k == 2:
            self.emit(f".word {r.choice(ILLEGAL_WORDS):#010x}")
        elif k == 3:
            # read-only / unimplemented CSRs: illegal only if it writes
            self.emit(f"csrrs x{self.reg()}, {r.choice(['mhartid', 'satp', 'mvendorid'])}, x{self.reg()}")
        elif k == 4:
            op = r.choice(["lh", "lhu", "lw"])
            self.emit(f"{op} x{self.reg()}, {r.choice([1, 2, 3]) if op == 'lw' else 1}(x29)")
        else:
            op = r.choice(["sh", "sw"])
            self.emit(f"{op} x{self.reg()}, {r.choice([1, 2, 3]) if op == 'sw' else 3}(x29)")

    def _fence(self):
        self.emit(self.r.choice(["fence", "fence", "wfi"]))   # WFI is a no-op here

    def _fence_i(self):
        self.emit("fence.i")

    def _smc(self):
        """Self-modifying code: write an instruction into the program just
        ahead, FENCE.I, then run it."""
        a = self.reg_nz()
        b = self.r.choice([x for x in POOL if x not in (0, a)])
        op = self.r.choice(list(I_OPS))
        word = enc_i(0x13, I_OPS[op], self.reg(), self.reg(), self.r.randint(-2048, 2047))
        hi, lo = _li_parts(word)
        self.emit(f"auipc x{a}, 0")           # P
        self.emit(f"lui  x{b}, {hi}")          # P+4
        self.emit(f"addi x{b}, x{b}, {lo}")    # P+8
        self.emit(f"sw   x{b}, 20(x{a})")      # P+12
        self.emit("fence.i")                   # P+16
        self.emit("nop")                       # P+20: replaced before it runs

    def _mret(self):
        """MRET to an address computed just before it (mepc written one
        instruction ahead of the MRET that reads it), skipping one instruction."""
        a = self.reg_nz()
        self.emit(f"auipc x{a}, 0")                          # P
        self.emit(f"addi x{a}, x{a}, {20 + self.r.randint(0, 3)}")   # low bits are dropped by mepc
        self.emit(f"csrw mepc, x{a}")
        self.emit("mret")                                    # P+12
        k = self.r.random()
        if k < 0.3:
            # P+16: skipped. A store here must never reach memory (coverage
            # bin pipe.store_killed_by_mret; added after mutation testing
            # showed nothing exercised it).
            self.emit(f"sw   x{self.reg()}, {self._offset(4)}(x29)")
        elif k < 0.65:
            # P+16: skipped. Half the time it's a jump, so a taken jump sits in
            # EX while MRET is in MEM (coverage bin pipe.mret_beats_redirect;
            # the generator never made that case until this was added).
            name = self.label()
            self.emit(f"j    {name}")
            self.place(name)
        else:
            self.emit(self._filler(avoid=-1))
        # P+20: execution continues here


def generate(seed: int, **kw) -> Program:
    return Program(seed, **kw)


if __name__ == "__main__":
    print(generate(int(sys.argv[1]) if len(sys.argv) > 1 else 1).source)
