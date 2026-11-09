"""cocotb tests for rtl/regfile.sv.

One example test is written for you. The three marked YOUR TURN are yours:
each has a docstring that says what it must prove. Run with:

    pytest tb/test_regfile.py -v
"""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from runner import run


async def setup(dut):
    """Start a 10 ns clock and drive every input to a known idle value."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rd_we.value = 0
    dut.rd_addr.value = 0
    dut.rd_data.value = 0
    dut.rs1_addr.value = 0
    dut.rs2_addr.value = 0
    await RisingEdge(dut.clk)


async def write(dut, addr, data):
    """Drive one write, let the clock edge commit it, then deassert."""
    dut.rd_we.value = 1
    dut.rd_addr.value = addr
    dut.rd_data.value = data
    await RisingEdge(dut.clk)
    dut.rd_we.value = 0


async def read1(dut, addr):
    """Read port 1 is combinational: set the address, let it settle, sample."""
    dut.rs1_addr.value = addr
    await Timer(1, unit="ns")
    return int(dut.rs1_data.value)


@cocotb.test()
async def test_write_then_read_all(dut):
    """Write a random value to x1..x31, then read every one back on port 1."""
    await setup(dut)
    expected = {r: random.getrandbits(32) for r in range(1, 32)}

    for r, v in expected.items():
        await write(dut, r, v)

    for r, v in expected.items():
        got = await read1(dut, r)
        assert got == v, f"x{r}: wrote {v:#010x}, read {got:#010x}"


@cocotb.test()
async def test_x0_is_zero(dut):
    """YOUR TURN: write 0xDEADBEEF to x0, then prove both read ports return 0
    for address 0."""
    await setup(dut)
    # TODO


@cocotb.test()
async def test_write_enable_low(dut):
    """YOUR TURN: put a known value in x5, then drive a DIFFERENT value on
    rd_data with rd_addr=5 but rd_we=0 for a clock edge. x5 must be unchanged."""
    await setup(dut)
    # TODO


@cocotb.test()
async def test_two_ports_independent(dut):
    """YOUR TURN: load different values into two registers, point rs1 at one
    and rs2 at the other in the SAME cycle, and check both ports at once.
    (Hint: you'll want a read2 helper like read1.)"""
    await setup(dut)
    # TODO


def test_regfile():
    run("regfile", ["rtl/regfile.sv"], "test_regfile")
