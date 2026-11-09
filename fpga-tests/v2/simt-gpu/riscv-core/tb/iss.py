"""RV32I instruction-set simulator: the golden model.

Written straight from the RISC-V ISA manual (Vol. I, "RV32I Base Integer
Instruction Set"; Vol. II, "Machine-Level ISA" for CSRs and traps) as a plain
interpreter, deliberately NOT structured like the RTL. If the two were
written the same way they could share a misunderstanding; written
differently, a disagreement means one of them is wrong, and the waveform
tells you which.

Machine mode only. Implemented CSRs: mstatus (MIE/MPIE; MPP reads as M),
misa, mie/mip (read zero), mtvec (direct mode), mscratch, mepc, mcause,
mtval, and the read-only ID registers. Any other CSR is an illegal
instruction. Misaligned loads, stores and jump targets trap.

step() executes one instruction and returns a Retire record describing it,
which the testbench compares against the core's trace_* signals.
"""

from dataclasses import dataclass

MASK = 0xFFFF_FFFF
ECALL = 0x0000_0073
EBREAK = 0x0010_0073
MRET = 0x3020_0073
WFI = 0x1050_0073

# mcause values (privileged spec, "Machine cause register")
MISALIGNED_FETCH, ILLEGAL_INSTR, BREAKPOINT = 0, 2, 3
MISALIGNED_LOAD, MISALIGNED_STORE, MACHINE_ECALL = 4, 6, 11

MSTATUS_MIE, MSTATUS_MPIE, MSTATUS_MPP = 1 << 3, 1 << 7, 3 << 11


def sext(value: int, bits: int) -> int:
    """Sign-extend the low `bits` of value to a Python int."""
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def signed(x: int) -> int:
    return sext(x, 32)


@dataclass
class Retire:
    pc: int
    instr: int
    rd: int | None = None        # register written (None if no write or rd == x0)
    rd_val: int | None = None
    store_addr: int | None = None
    store_strb: int = 0          # which bytes of the aligned word were written
    store_data: int = 0          # the word as it appears on the bus (lanes placed)
    trap: bool = False
    cause: int = 0


class Trap(Exception):
    def __init__(self, cause: int, tval: int = 0):
        self.cause, self.tval = cause, tval & MASK


class ISS:
    def __init__(self, mem_bytes: int = 64 * 1024, reset_pc: int = 0,
                 halt_on_ecall: bool = True):
        assert mem_bytes & (mem_bytes - 1) == 0, "memory size must be a power of 2"
        self.mem = bytearray(mem_bytes)
        self.regs = [0] * 32
        self.pc = reset_pc
        self.halt_on_ecall = halt_on_ecall   # directed tests end with ECALL
        self.halted = False
        self.mstatus_mie = self.mstatus_mpie = 0
        self.csr = {"mtvec": 0, "mscratch": 0, "mepc": 0, "mcause": 0, "mtval": 0}

    # ---- memory (addresses wrap modulo the memory size, like sim_mem.sv) ----
    def _a(self, addr: int) -> int:
        return addr & (len(self.mem) - 1)

    def load_words(self, words: list[int], base: int = 0) -> None:
        for i, w in enumerate(words):
            self.write(base + 4 * i, w, 4)

    def load_bytes(self, data: bytes, base: int = 0) -> None:
        a = self._a(base)
        self.mem[a:a + len(data)] = data

    def read(self, addr: int, size: int) -> int:
        a = self._a(addr)
        return int.from_bytes(self.mem[a:a + size], "little")

    def write(self, addr: int, value: int, size: int) -> None:
        a = self._a(addr)
        self.mem[a:a + size] = (value & ((1 << (8 * size)) - 1)).to_bytes(size, "little")

    # ---- registers ----
    def x(self, r: int) -> int:
        return 0 if r == 0 else self.regs[r]

    def set_x(self, r: int, value: int) -> None:
        if r != 0:
            self.regs[r] = value & MASK

    # ---- CSRs ----
    CSR_NAMES = {0x300: "mstatus", 0x301: "misa", 0x304: "mie", 0x305: "mtvec",
                 0x340: "mscratch", 0x341: "mepc", 0x342: "mcause", 0x343: "mtval",
                 0x344: "mip", 0xF11: "mvendorid", 0xF12: "marchid",
                 0xF13: "mimpid", 0xF14: "mhartid"}

    def csr_read(self, num: int) -> int:
        name = self.CSR_NAMES.get(num)
        if name is None:
            raise Trap(ILLEGAL_INSTR)
        if name == "mstatus":
            return (MSTATUS_MPP | (MSTATUS_MPIE if self.mstatus_mpie else 0)
                    | (MSTATUS_MIE if self.mstatus_mie else 0))
        if name == "misa":
            return (1 << 30) | (1 << 8)          # MXL = 32-bit, extension "I"
        return self.csr.get(name, 0)             # mie, mip, ID registers read 0

    def csr_write(self, num: int, value: int) -> None:
        name = self.CSR_NAMES[num]
        value &= MASK
        if name == "mstatus":
            self.mstatus_mie = int(bool(value & MSTATUS_MIE))
            self.mstatus_mpie = int(bool(value & MSTATUS_MPIE))
        elif name in ("mtvec", "mepc"):
            self.csr[name] = value & ~3          # direct mode only; IALIGN = 32
        elif name in self.csr:
            self.csr[name] = value
        # misa, mie, mip: writes ignored

    # ---- execute one instruction ----
    def step(self) -> Retire:
        pc = self.pc
        ins = self.read(pc, 4)
        ret = Retire(pc=pc, instr=ins)
        try:
            self._execute(ins, pc, ret)
        except Trap as t:
            # A trapping instruction has no effect except the trap itself.
            ret.rd = ret.rd_val = ret.store_addr = None
            ret.store_strb = ret.store_data = 0
            ret.trap, ret.cause = True, t.cause
            self.csr["mepc"], self.csr["mcause"], self.csr["mtval"] = pc, t.cause, t.tval
            self.mstatus_mpie, self.mstatus_mie = self.mstatus_mie, 0
            self.pc = self.csr["mtvec"]
            if t.cause == MACHINE_ECALL and self.halt_on_ecall:
                self.halted = True
        return ret

    def _execute(self, ins: int, pc: int, ret: Retire) -> None:
        opcode = ins & 0x7F
        rd = (ins >> 7) & 0x1F
        f3 = (ins >> 12) & 0x7
        rs1 = (ins >> 15) & 0x1F
        rs2 = (ins >> 20) & 0x1F
        f7 = ins >> 25
        a, b = self.x(rs1), self.x(rs2)

        imm_i = sext(ins >> 20, 12)
        imm_s = sext(((ins >> 25) << 5) | ((ins >> 7) & 0x1F), 12)
        imm_b = sext((((ins >> 31) & 1) << 12) | (((ins >> 7) & 1) << 11)
                     | (((ins >> 25) & 0x3F) << 5) | (((ins >> 8) & 0xF) << 1), 13)
        imm_u = ins & 0xFFFF_F000
        imm_j = sext((((ins >> 31) & 1) << 20) | (((ins >> 12) & 0xFF) << 12)
                     | (((ins >> 20) & 1) << 11) | (((ins >> 21) & 0x3FF) << 1), 21)

        next_pc = (pc + 4) & MASK
        result = None   # value for rd, if this instruction writes one

        def jump(target: int) -> int:
            target &= MASK
            if target & 3:
                raise Trap(MISALIGNED_FETCH, target)
            return target

        if opcode == 0x37:                                   # LUI
            result = imm_u
        elif opcode == 0x17:                                 # AUIPC
            result = pc + imm_u
        elif opcode == 0x6F:                                 # JAL
            next_pc = jump(pc + imm_j)
            result = pc + 4
        elif opcode == 0x67 and f3 == 0:                     # JALR
            next_pc = jump((a + imm_i) & ~1)
            result = pc + 4
        elif opcode == 0x63:                                 # branches
            sa, sb = signed(a), signed(b)
            cond = {0: a == b, 1: a != b, 4: sa < sb, 5: sa >= sb,
                    6: a < b, 7: a >= b}
            if f3 not in cond:
                raise Trap(ILLEGAL_INSTR, ins)
            if cond[f3]:
                next_pc = jump(pc + imm_b)
        elif opcode == 0x03:                                 # loads
            addr = (a + imm_i) & MASK
            size = {0: 1, 1: 2, 2: 4, 4: 1, 5: 2}.get(f3)
            if size is None:
                raise Trap(ILLEGAL_INSTR, ins)
            if addr % size:
                raise Trap(MISALIGNED_LOAD, addr)
            raw = self.read(addr, size)
            result = sext(raw, 8 * size) if f3 in (0, 1) else raw
        elif opcode == 0x23:                                 # stores
            addr = (a + imm_s) & MASK
            size = {0: 1, 1: 2, 2: 4}.get(f3)
            if size is None:
                raise Trap(ILLEGAL_INSTR, ins)
            if addr % size:
                raise Trap(MISALIGNED_STORE, addr)
            self.write(addr, b, size)
            lane = addr & 3
            ret.store_addr = addr & ~3
            ret.store_strb = ((1 << size) - 1) << lane
            ret.store_data = ((b & ((1 << (8 * size)) - 1)) << (8 * lane)) & MASK
        elif opcode in (0x13, 0x33):                         # OP-IMM / OP
            is_reg = opcode == 0x33
            y = b if is_reg else imm_i & MASK
            shamt = y & 0x1F
            alt = (f7 == 0x20)
            if f3 == 0:   result = (a - y) if (is_reg and alt) else (a + y)
            elif f3 == 1: result = a << shamt
            elif f3 == 2: result = int(signed(a) < signed(y))
            elif f3 == 3: result = int(a < y)
            elif f3 == 4: result = a ^ y
            elif f3 == 5: result = (signed(a) >> shamt) if alt else (a >> shamt)
            elif f3 == 6: result = a | y
            else:         result = a & y
        elif opcode == 0x0F and f3 in (0, 1):                # FENCE, FENCE.I: no-ops
            pass
        elif opcode == 0x73:                                 # SYSTEM
            if f3 == 0:
                if ins == ECALL:
                    raise Trap(MACHINE_ECALL)
                if ins == EBREAK:
                    raise Trap(BREAKPOINT, pc)
                if ins == MRET:
                    next_pc = self.csr["mepc"]
                    self.mstatus_mie, self.mstatus_mpie = self.mstatus_mpie, 1
                elif ins != WFI:                             # WFI: no-op
                    raise Trap(ILLEGAL_INSTR, ins)
            elif f3 == 4:
                raise Trap(ILLEGAL_INSTR, ins)
            else:                                            # CSRRW/S/C, CSRRWI/SI/CI
                num = ins >> 20
                src = rs1 if f3 & 4 else a                   # immediate forms: uimm = rs1 field
                kind = f3 & 3                                # 1 = RW, 2 = RS, 3 = RC
                writes = kind == 1 or rs1 != 0
                try:
                    old = self.csr_read(num)
                    if writes and (num >> 10) == 3:          # top bits 11 = read-only
                        raise Trap(ILLEGAL_INSTR)
                except Trap:
                    raise Trap(ILLEGAL_INSTR, ins) from None
                if writes:
                    new = src if kind == 1 else (old | src) if kind == 2 else (old & ~src)
                    self.csr_write(num, new)
                result = old
        else:
            raise Trap(ILLEGAL_INSTR, ins)

        if result is not None and rd != 0:
            self.set_x(rd, result)
            ret.rd, ret.rd_val = rd, result & MASK

        self.pc = next_pc
