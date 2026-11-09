"""A tiny RV32I assembler, so tests can be written as assembly text.

Written before the RISC-V GCC toolchain was installed, and kept because
tests (and the random program generator) can build programs as strings with
no toolchain round trip. It covers every RV32I instruction, Zicsr (csrrw/s/c,
the immediate forms, and csrr/csrw/csrs/csrc/csrwi), mret, wfi, fence.i, and
a handful of common pseudo-instructions (nop, li, mv, not, neg, j, jr, ret,
beqz, bnez, call-free `jal label`). CSRs can be named (mscratch) or numbered.

    words = assemble('''
        addi  t0, zero, 5
    loop:
        addi  t0, t0, -1
        bnez  t0, loop
        ecall
    ''')
"""

import re

ABI = ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1",
       "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
       "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
       "t3", "t4", "t5", "t6"]
REGS = {name: i for i, name in enumerate(ABI)} | {f"x{i}": i for i in range(32)} | {"fp": 8}

# mnemonic -> (opcode, funct3, funct7)
R_OPS = {"add": (0x33, 0, 0x00), "sub": (0x33, 0, 0x20), "sll": (0x33, 1, 0x00),
         "slt": (0x33, 2, 0x00), "sltu": (0x33, 3, 0x00), "xor": (0x33, 4, 0x00),
         "srl": (0x33, 5, 0x00), "sra": (0x33, 5, 0x20), "or": (0x33, 6, 0x00),
         "and": (0x33, 7, 0x00)}
I_OPS = {"addi": 0, "slti": 2, "sltiu": 3, "xori": 4, "ori": 6, "andi": 7}
SHIFT_OPS = {"slli": (1, 0x00), "srli": (5, 0x00), "srai": (5, 0x20)}
LOADS = {"lb": 0, "lh": 1, "lw": 2, "lbu": 4, "lhu": 5}
STORES = {"sb": 0, "sh": 1, "sw": 2}
BRANCHES = {"beq": 0, "bne": 1, "blt": 4, "bge": 5, "bltu": 6, "bgeu": 7}
CSR_OPS = {"csrrw": 1, "csrrs": 2, "csrrc": 3, "csrrwi": 5, "csrrsi": 6, "csrrci": 7}
CSRS = {"mstatus": 0x300, "misa": 0x301, "mie": 0x304, "mtvec": 0x305,
        "mscratch": 0x340, "mepc": 0x341, "mcause": 0x342, "mtval": 0x343,
        "mip": 0x344, "mvendorid": 0xF11, "marchid": 0xF12, "mimpid": 0xF13,
        "mhartid": 0xF14, "satp": 0x180}


class AsmError(Exception):
    pass


def _reg(tok: str) -> int:
    tok = tok.strip().lower()
    if tok not in REGS:
        raise AsmError(f"unknown register '{tok}'")
    return REGS[tok]


def _int(tok: str) -> int:
    return int(tok.strip(), 0)


def _check(value: int, bits: int, what: str) -> None:
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if not lo <= value <= hi:
        raise AsmError(f"{what} {value} out of range for {bits}-bit signed field")


def enc_r(op, f3, f7, rd, rs1, rs2):
    return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op


def enc_i(op, f3, rd, rs1, imm):
    _check(imm, 12, "I-immediate")
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op


def enc_s(op, f3, rs1, rs2, imm):
    _check(imm, 12, "S-immediate")
    imm &= 0xFFF
    return ((imm >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | ((imm & 0x1F) << 7) | op


def enc_b(f3, rs1, rs2, off):
    _check(off, 13, "branch offset")
    if off & 1:
        raise AsmError("branch offset must be even")
    o = off & 0x1FFF
    return (((o >> 12) & 1) << 31) | (((o >> 5) & 0x3F) << 25) | (rs2 << 20) | (rs1 << 15) \
        | (f3 << 12) | (((o >> 1) & 0xF) << 8) | (((o >> 11) & 1) << 7) | 0x63


def enc_u(op, rd, imm20):
    if not 0 <= imm20 <= 0xFFFFF:
        raise AsmError(f"U-immediate {imm20:#x} must be 0..0xFFFFF")
    return (imm20 << 12) | (rd << 7) | op


def enc_j(rd, off):
    _check(off, 21, "jump offset")
    if off & 1:
        raise AsmError("jump offset must be even")
    o = off & 0x1FFFFF
    return (((o >> 20) & 1) << 31) | (((o >> 1) & 0x3FF) << 21) | (((o >> 11) & 1) << 20) \
        | (((o >> 12) & 0xFF) << 12) | (rd << 7) | 0x6F


def _csr(tok: str) -> int:
    tok = tok.strip().lower()
    num = CSRS[tok] if tok in CSRS else _int(tok)
    if not 0 <= num <= 0xFFF:
        raise AsmError(f"CSR number {num:#x} must be 0..0xFFF")
    return num


def enc_csr(f3, rd, csr, src):
    """src is rs1 for csrrw/s/c, the 5-bit zero-extended uimm for the *i forms."""
    if not 0 <= src <= 31:
        raise AsmError("CSR source must be a register or uimm 0..31")
    return (csr << 20) | (src << 15) | (f3 << 12) | (rd << 7) | 0x73


def _li_parts(value: int) -> tuple[int, int]:
    """Split a 32-bit constant into (lui upper 20 bits, addi low 12 bits).
    addi sign-extends its immediate, so round the upper part up when bit 11 is set."""
    value &= 0xFFFF_FFFF
    lo = ((value & 0xFFF) ^ 0x800) - 0x800
    hi = ((value - lo) >> 12) & 0xFFFFF
    return hi, lo


def _size(mn: str, args: list[str]) -> int:
    """Instructions emitted by one source line (needed for label addresses)."""
    if mn == "li":
        v = _int(args[1])
        return 1 if -2048 <= v <= 2047 else 2
    return 1


def _split(line: str) -> tuple[str, list[str]]:
    parts = line.split(None, 1)
    mn = parts[0].lower()
    args = [a.strip() for a in parts[1].split(",")] if len(parts) > 1 else []
    return mn, args


def assemble(source: str, base: int = 0) -> list[int]:
    # pass 1: strip comments, collect labels
    lines, labels, pc = [], {}, base
    for raw in source.splitlines():
        line = raw.split("#")[0].strip()
        while (m := re.match(r"^([A-Za-z_.][\w.]*):\s*(.*)$", line)):
            labels[m.group(1)] = pc
            line = m.group(2).strip()
        if not line:
            continue
        mn, args = _split(line)
        lines.append((pc, mn, args, raw.strip()))
        pc += 4 * (1 if mn == ".word" else _size(mn, args))

    def target(tok: str, here: int) -> int:
        tok = tok.strip()
        return labels[tok] - here if tok in labels else _int(tok)

    # pass 2: encode
    out: list[int] = []
    for pc, mn, args, raw in lines:
        try:
            out.extend(_encode(mn, args, pc, target))
        except (AsmError, ValueError, IndexError, KeyError) as e:
            raise AsmError(f"line '{raw}': {e}") from None
    return out


def _mem_operand(tok: str) -> tuple[int, int]:
    m = re.match(r"^\s*(-?\w*)\s*\(\s*(\w+)\s*\)\s*$", tok)
    if not m:
        raise AsmError(f"expected imm(reg), got '{tok}'")
    return (_int(m.group(1)) if m.group(1) else 0), _reg(m.group(2))


def _encode(mn, a, pc, target) -> list[int]:
    if mn == ".word":
        return [_int(a[0]) & 0xFFFF_FFFF]
    if mn in R_OPS:
        op, f3, f7 = R_OPS[mn]
        return [enc_r(op, f3, f7, _reg(a[0]), _reg(a[1]), _reg(a[2]))]
    if mn in I_OPS:
        return [enc_i(0x13, I_OPS[mn], _reg(a[0]), _reg(a[1]), _int(a[2]))]
    if mn in SHIFT_OPS:
        f3, f7 = SHIFT_OPS[mn]
        sh = _int(a[2])
        if not 0 <= sh <= 31:
            raise AsmError("shift amount must be 0..31")
        return [enc_i(0x13, f3, _reg(a[0]), _reg(a[1]), (f7 << 5) | sh)]
    if mn in LOADS:
        off, rs1 = _mem_operand(a[1])
        return [enc_i(0x03, LOADS[mn], _reg(a[0]), rs1, off)]
    if mn in STORES:
        off, rs1 = _mem_operand(a[1])
        return [enc_s(0x23, STORES[mn], rs1, _reg(a[0]), off)]
    if mn in BRANCHES:
        return [enc_b(BRANCHES[mn], _reg(a[0]), _reg(a[1]), target(a[2], pc))]
    if mn == "lui":
        return [enc_u(0x37, _reg(a[0]), _int(a[1]))]
    if mn == "auipc":
        return [enc_u(0x17, _reg(a[0]), _int(a[1]))]
    if mn == "jal":
        if len(a) == 1:                       # jal label  ->  jal ra, label
            return [enc_j(1, target(a[0], pc))]
        return [enc_j(_reg(a[0]), target(a[1], pc))]
    if mn == "jalr":
        if len(a) == 1:                       # jalr rs  ->  jalr ra, 0(rs)
            return [enc_i(0x67, 0, 1, _reg(a[0]), 0)]
        if "(" in a[1]:                       # jalr rd, imm(rs1)
            off, rs1 = _mem_operand(a[1])
            return [enc_i(0x67, 0, _reg(a[0]), rs1, off)]
        return [enc_i(0x67, 0, _reg(a[0]), _reg(a[1]), _int(a[2]))]
    if mn == "ecall":
        return [0x0000_0073]
    if mn == "ebreak":
        return [0x0010_0073]
    if mn == "fence":
        return [0x0FF0_000F]
    if mn == "fence.i":
        return [0x0000_100F]
    if mn == "mret":
        return [0x3020_0073]
    if mn == "wfi":
        return [0x1050_0073]
    if mn in CSR_OPS:                         # csrrw rd, csr, rs1  /  csrrwi rd, csr, uimm
        f3 = CSR_OPS[mn]
        src = _int(a[2]) if f3 & 4 else _reg(a[2])
        return [enc_csr(f3, _reg(a[0]), _csr(a[1]), src)]
    if mn == "csrr":                          # csrrs rd, csr, x0
        return [enc_csr(2, _reg(a[0]), _csr(a[1]), 0)]
    if mn in ("csrw", "csrs", "csrc"):        # csrrw/s/c x0, csr, rs1
        return [enc_csr({"csrw": 1, "csrs": 2, "csrc": 3}[mn], 0, _csr(a[0]), _reg(a[1]))]
    if mn == "csrwi":
        return [enc_csr(5, 0, _csr(a[0]), _int(a[1]))]

    # ---- pseudo-instructions ----
    if mn == "nop":
        return [enc_i(0x13, 0, 0, 0, 0)]
    if mn == "li":
        rd, v = _reg(a[0]), _int(a[1])
        if -2048 <= v <= 2047:
            return [enc_i(0x13, 0, rd, 0, v)]
        hi, lo = _li_parts(v)
        return [enc_u(0x37, rd, hi), enc_i(0x13, 0, rd, rd, lo)]
    if mn == "mv":
        return [enc_i(0x13, 0, _reg(a[0]), _reg(a[1]), 0)]
    if mn == "not":
        return [enc_i(0x13, 4, _reg(a[0]), _reg(a[1]), -1)]
    if mn == "neg":
        return [enc_r(0x33, 0, 0x20, _reg(a[0]), 0, _reg(a[1]))]
    if mn == "j":
        return [enc_j(0, target(a[0], pc))]
    if mn == "jr":
        return [enc_i(0x67, 0, 0, _reg(a[0]), 0)]
    if mn == "ret":
        return [enc_i(0x67, 0, 0, 1, 0)]
    if mn == "beqz":
        return [enc_b(0, _reg(a[0]), 0, target(a[1], pc))]
    if mn == "bnez":
        return [enc_b(1, _reg(a[0]), 0, target(a[1], pc))]
    raise AsmError(f"unknown mnemonic '{mn}'")
