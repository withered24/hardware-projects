"""bf16_mul and fp32_add against sw/golden.py, bit for bit.

The vectors are built to hit the hard parts, not just random bits:
cancellation (close exponents, opposite signs), exponent gaps right at the
guard/round/sticky edges, ties for round-to-even, the overflow and
flush-to-zero edges, and infinities/NaNs/subnormals from raw random bits.

    pytest tb/test_arith.py -v
"""

import os

import cocotb
import numpy as np
from cocotb.triggers import Timer

import golden
from runner import run

N = int(os.environ.get("ARITH_N", "40000"))


def f32(u):
    return np.array(u, np.uint32).view(np.float32)


def add_vectors(rng, n):
    """Pairs of FP32 bit patterns, mixed strategies."""
    out = []
    k = n // 6
    # 1. raw random bits: inf, NaN, subnormals, everything
    out += list(zip(rng.integers(0, 2**32, k), rng.integers(0, 2**32, k)))
    # 2. same exponent region, opposite signs: deep cancellation
    a = rng.integers(0, 2**32, k) & 0x7FFF_FFFF
    d = rng.integers(-3, 4, k)
    b = (a + (d << 23) * (rng.random(k) < 0.5) + rng.integers(-64, 64, k)) & 0x7FFF_FFFF | 0x8000_0000
    out += list(zip(a, b))
    # 3. exponent gaps 0..30 (guard/round/sticky boundaries), random signs
    e = rng.integers(30, 220, k)
    gap = rng.integers(0, 31, k)
    a = (rng.integers(0, 2, k) << 31) | (e << 23) | rng.integers(0, 2**23, k)
    b = (rng.integers(0, 2, k) << 31) | ((e - gap) << 23) | rng.integers(0, 2**23, k)
    out += list(zip(a, b))
    # 4. ties: small operand is exactly half an ulp-ish pattern
    a = (rng.integers(1, 250, k) << 23) | rng.integers(0, 2**23, k)
    b = ((a >> 23) - 24 << 23) | (rng.integers(0, 2, k) << 22)
    out += list(zip(a, b))
    # 5. near overflow and near the smallest normal
    a = (rng.integers(0, 2, k) << 31) | (rng.choice([1, 2, 253, 254], k) << 23) | rng.integers(0, 2**23, k)
    b = (rng.integers(0, 2, k) << 31) | (rng.choice([1, 2, 253, 254], k) << 23) | rng.integers(0, 2**23, k)
    out += list(zip(a, b))
    # 6. realistic: products of LLM-sized weights and activations
    w = rng.normal(0, 0.02, k).astype(np.float32)
    x = rng.normal(0, 1.0, k).astype(np.float32)
    out += list(zip(w.view(np.uint32), (w * x).view(np.uint32)))
    return [(int(p) & 0xFFFF_FFFF, int(q) & 0xFFFF_FFFF) for p, q in out]


@cocotb.test()
async def test_fp32_add(dut):
    """fp32_add == golden.add on mixed vectors, both operand orders."""
    rng = np.random.default_rng(int(os.environ.get("SEED", "1")))
    vec = add_vectors(rng, N)
    vec += [(b, a) for a, b in vec[: len(vec) // 4]]
    bad = 0
    for a, b in vec:
        dut.aa.value, dut.ab.value = a, b
        await Timer(1, unit="ns")
        got = int(dut.ay.value)
        want = golden.add(f32(a), f32(b))
        if not golden.same(f32(got), want):
            bad += 1
            if bad <= 10:
                dut._log.error(f"add {a:08x} + {b:08x}: got {got:08x} want {int(golden.f32_bits(want)):08x}")
    assert bad == 0, f"{bad} mismatches out of {len(vec)}"
    dut._log.info(f"fp32_add: {len(vec)} vectors bit-exact")


@cocotb.test()
async def test_bf16_mul(dut):
    """bf16_mul == golden.mul: every exponent pair once, plus random bits."""
    rng = np.random.default_rng(int(os.environ.get("SEED", "1")) + 7)
    vec = []
    for ea in range(256):
        for eb in range(0, 256, 3):
            vec.append(((int(rng.integers(0, 2)) << 15) | (ea << 7) | int(rng.integers(0, 128)),
                        (int(rng.integers(0, 2)) << 15) | (eb << 7) | int(rng.integers(0, 128))))
    vec += [(int(a), int(b)) for a, b in rng.integers(0, 2**16, (N // 4, 2))]
    bad = 0
    for a, b in vec:
        dut.ma.value, dut.mb.value = a, b
        await Timer(1, unit="ns")
        got = int(dut.my.value)
        want = golden.mul(np.uint16(a), np.uint16(b))
        if not golden.same(f32(got), want):
            bad += 1
            if bad <= 10:
                dut._log.error(f"mul {a:04x} * {b:04x}: got {got:08x} want {int(golden.f32_bits(want)):08x}")
    assert bad == 0, f"{bad} mismatches out of {len(vec)}"
    dut._log.info(f"bf16_mul: {len(vec)} vectors bit-exact")


def test_arith():
    run("tb_arith", ["rtl/bf16_mul.sv", "rtl/fp32_add.sv", "tb/tb_arith.sv"], "test_arith")
