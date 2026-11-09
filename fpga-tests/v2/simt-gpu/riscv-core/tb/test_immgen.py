"""cocotb unit tests for rtl/immgen.sv.

The immediate generator is where students' cores most often go wrong: the B
and J formats scramble their bits. These tests compare the RTL against an
independent Python decode of random instruction words. The I-type example is
written; S, B, U, J are yours.

    pytest tb/test_immgen.py -v
"""

import random

import cocotb
from cocotb.triggers import Timer

from runner import run

# Must match rtl/rv32i_pkg.sv
IMM_I, IMM_S, IMM_B, IMM_U, IMM_J = 0, 1, 2, 3, 4


def sext(v, bits):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


async def imm(dut, instr, sel):
    dut.instr.value, dut.imm_sel.value = instr, sel
    await Timer(1, unit="ns")
    return int(dut.imm.value)


@cocotb.test()
async def test_i_type(dut):
    """I-immediate = sign-extended instr[31:20]."""
    for _ in range(2000):
        ins = random.getrandbits(32)
        want = sext(ins >> 20, 12) & 0xFFFF_FFFF
        assert await imm(dut, ins, IMM_I) == want, f"instr {ins:#010x}"


@cocotb.test()
async def test_s_type(dut):
    """YOUR TURN: S-immediate = sext({instr[31:25], instr[11:7]})."""
    # TODO


@cocotb.test()
async def test_b_type(dut):
    """YOUR TURN: B-immediate = sext({instr[31], instr[7], instr[30:25],
    instr[11:8], 0}), 13 bits. Bit 0 must ALWAYS be zero."""
    # TODO


@cocotb.test()
async def test_u_and_j_type(dut):
    """YOUR TURN: U = instr[31:12] << 12 (no sign extension needed, why?);
    J = sext({instr[31], instr[19:12], instr[20], instr[30:21], 0}), 21 bits."""
    # TODO


def test_immgen():
    run("immgen", ["rtl/rv32i_pkg.sv", "rtl/immgen.sv"], "test_immgen")
