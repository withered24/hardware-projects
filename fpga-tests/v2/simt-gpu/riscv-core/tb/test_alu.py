"""cocotb unit tests for rtl/alu.sv.

The core co-sim tests (test_core.py) already exercise the ALU through real
programs. These unit tests are for hammering it directly with random
operands, millions of cases cheaply, no program needed. One example is
written; the YOUR TURN ones are yours.

    pytest tb/test_alu.py -v
"""

import random

import cocotb
from cocotb.triggers import Timer

from runner import run

MASK = 0xFFFF_FFFF
# Must match rtl/rv32i_pkg.sv
ALU_ADD, ALU_SUB, ALU_SLL, ALU_SLT, ALU_SLTU = 0, 1, 2, 3, 4
ALU_XOR, ALU_SRL, ALU_SRA, ALU_OR, ALU_AND = 5, 6, 7, 8, 9


async def alu(dut, a, b, op):
    dut.a.value, dut.b.value, dut.op.value = a, b, op
    await Timer(1, unit="ns")
    return int(dut.y.value)


@cocotb.test()
async def test_add_sub_random(dut):
    """ADD and SUB on random operands wrap modulo 2^32."""
    for _ in range(2000):
        a, b = random.getrandbits(32), random.getrandbits(32)
        assert await alu(dut, a, b, ALU_ADD) == (a + b) & MASK
        assert await alu(dut, a, b, ALU_SUB) == (a - b) & MASK


@cocotb.test()
async def test_shifts(dut):
    """YOUR TURN: SLL/SRL/SRA against Python for random a and every shift
    amount 0..31. Also prove that bits above b[4:0] are ignored (b = 32 + n
    must behave like b = n). SRA needs a signed Python model."""
    # TODO


@cocotb.test()
async def test_slt_vs_sltu(dut):
    """YOUR TURN: find operand pairs where SLT and SLTU disagree (hint: one
    negative, one positive) and check both. Then random pairs."""
    # TODO


@cocotb.test()
async def test_logic_ops(dut):
    """YOUR TURN: XOR/OR/AND on random operands."""
    # TODO


def test_alu():
    run("alu", ["rtl/rv32i_pkg.sv", "rtl/alu.sv"], "test_alu")
