"""Bit-accurate model of the GEMV datapath's arithmetic (rtl/bf16_mul.sv,
rtl/fp32_add.sv, rtl/gemv_core.sv). The RTL must match this bit for bit.

Arithmetic, chosen to be cheap in hardware and exactly reproducible here:

  * BF16 x BF16 -> FP32 product. Two 8-bit significands give a 16-bit
    product, which FP32's 24-bit significand holds exactly, so the multiply
    never rounds. Subnormal inputs read as zero (DAZ); a product below the
    smallest normal becomes a signed zero (FTZ); above the largest finite,
    infinity.
  * FP32 add: IEEE 754 round-to-nearest-even, with DAZ/FTZ the same way.
    x + (-x) = +0. inf - inf = NaN (the RTL returns 0x7FC00000; NaNs are
    compared as a class).
  * Order: each cycle the L lanes' products are summed by a pairwise tree
    (p0+p1, p2+p3, ...; then level by level), and the row accumulator, which
    starts at +0.0, adds that chunk sum: acc = acc + tree(chunk). Columns go
    left to right, L at a time.

FP32 adds below are numpy float32 ops (IEEE RNE on x86-64) on operands that
are never subnormal, followed by the flush. A sum of two normals is exact
whenever it lands in the subnormal range, so flushing after the add matches
the RTL's flush on the exact result.
"""

import numpy as np

F32_MIN_NORMAL = np.float32(2.0 ** -126)


def bf16_to_f32(w) -> np.ndarray:
    return (np.asarray(w, np.uint16).astype(np.uint32) << 16).view(np.float32)


def _ftz(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return np.where(np.abs(x) < F32_MIN_NORMAL, np.copysign(np.float32(0), x), x).astype(np.float32)


def mul(w_bf16, x_bf16) -> np.ndarray:
    """Elementwise BF16 x BF16 -> FP32, as rtl/bf16_mul.sv."""
    a = _ftz(bf16_to_f32(w_bf16)).astype(np.float64)
    b = _ftz(bf16_to_f32(x_bf16)).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        p = a * b                                   # exact in float64
        p = np.where(np.abs(p) < 2.0 ** -126, np.copysign(0.0, p), p)
        return p.astype(np.float32)                 # exact or +-inf


def add(a, b) -> np.ndarray:
    """FP32 + FP32, as rtl/fp32_add.sv."""
    with np.errstate(over="ignore", invalid="ignore"):
        return _ftz(_ftz(a) + _ftz(b))


def tree(p: np.ndarray) -> np.ndarray:
    """Pairwise reduction over the last axis (length a power of two)."""
    while p.shape[-1] > 1:
        p = add(p[..., 0::2], p[..., 1::2])
    return p[..., 0]


def gemv(w: np.ndarray, x: np.ndarray, lanes: int) -> np.ndarray:
    """y = W x with the RTL's arithmetic. w (N, K) and x (K,) are uint16
    BF16 bit patterns; returns float32 (N,)."""
    w = np.asarray(w, np.uint16)
    n, k = w.shape
    assert k % lanes == 0
    p = mul(w, np.asarray(x, np.uint16)[None, :])          # (N, K)
    chunks = tree(p.reshape(n, k // lanes, lanes))          # (N, K/L)
    acc = np.zeros(n, np.float32)
    for c in range(k // lanes):
        acc = add(acc, chunks[:, c])
    return acc


def f32_bits(y) -> np.ndarray:
    return np.asarray(y, np.float32).view(np.uint32)


def same(a, b) -> np.ndarray:
    """Bitwise equality, with every NaN equal to every other NaN."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return (f32_bits(a) == f32_bits(b)) | (np.isnan(a) & np.isnan(b))


def ulp_diff(a, b) -> np.ndarray:
    """Distance in FP32 units in the last place (finite inputs)."""
    def key(x):
        u = f32_bits(x).astype(np.int64)
        return np.where(u & 0x8000_0000, 0x8000_0000 - u, u)
    return np.abs(key(a) - key(b))


if __name__ == "__main__":
    import torch

    from ewformat import synthetic_bf16

    rng = np.random.default_rng(1)
    w = synthetic_bf16(256, 896, rng)
    x = synthetic_bf16(1, 896, rng, std=1.0)[0]
    y = gemv(w, x, lanes=8)
    wt = torch.from_numpy(w.view(np.int16)).view(torch.bfloat16)
    xt = torch.from_numpy(x.view(np.int16)).view(torch.bfloat16)
    ref32 = (wt.float() @ xt.float()).numpy()
    ref16 = (wt @ xt).float().numpy()
    print("vs torch FP32 matmul: max ULP", int(ulp_diff(y, ref32).max()),
          " max abs", float(np.abs(y - ref32).max()))
    print("vs torch BF16 matmul: max abs", float(np.abs(y - ref16).max()), "(BF16 output rounding)")
