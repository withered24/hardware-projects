"""Exponent-window weight format: encoder, decoder and DRAM layout.

A BF16 weight is sign(1) | exponent(8) | mantissa(7). The format keeps the
sign and mantissa as one raw byte and replaces the exponent with a short
code that indexes a window of 2^b - 1 consecutive exponents:

    code c in 1..2^b-1   exponent = (base + c) mod 256     base = e* - 1
    code c = 0           escape: the real exponent is the next byte of the
                         escape stream (stream order, no index needed)

Weights are taken in row-major order and grouped into 64-weight tiles
(rows must be a multiple of 64 long, so a tile never straddles two rows).
Each matrix becomes up to four byte streams, each padded to 64 bytes:

    code   per tile, b little-endian 64-bit words; word p holds bit p of the
           64 codes (bit i = weight i of the tile), so no code ever
           straddles a word
    sm     one byte per weight: sign << 7 | mantissa
    esc    one exponent byte per escaped weight, in stream order
    tbase  one base byte per tile (only with per-tile bases)

The raw-BF16 baseline stream is the matrix itself, little-endian, 2 bytes
per weight.

    python sw/ewformat.py      roundtrip self-test on synthetic weights
"""

from dataclasses import dataclass, field

import numpy as np

TILE = 64
PAD = 64        # every stream is padded to this many bytes


def fields(w: np.ndarray):
    """uint16 BF16 bit patterns -> (sign, exponent, mantissa) as uint8."""
    w = np.asarray(w, dtype=np.uint16)
    return ((w >> 15) & 1).astype(np.uint8), ((w >> 7) & 0xFF).astype(np.uint8), (w & 0x7F).astype(np.uint8)


def best_window(exp: np.ndarray, width: int) -> tuple[int, int]:
    """Lowest e* maximising how many exponents fall in [e*, e*+width-1].
    Returns (e*, covered count)."""
    hist = np.bincount(exp.reshape(-1), minlength=256).astype(np.int64)
    cs = np.concatenate([[0], np.cumsum(hist)])
    cov = cs[width:] - cs[:-width]           # cov[e] = sum hist[e:e+width]
    e = int(np.argmax(cov))
    return e, int(cov[e])


def best_window_per_tile(exp_tiles: np.ndarray, width: int) -> np.ndarray:
    """e* for each row of exp_tiles (T, 64). Same tie rule as best_window."""
    T = exp_tiles.shape[0]
    out = np.empty(T, dtype=np.int64)
    for s in range(0, T, 1 << 14):                      # chunks keep (n, 256) small
        blk = exp_tiles[s:s + (1 << 14)].astype(np.int64)
        n = blk.shape[0]
        hist = np.zeros((n, 256), dtype=np.int32)
        np.add.at(hist, (np.repeat(np.arange(n), TILE), blk.reshape(-1)), 1)
        cs = np.concatenate([np.zeros((n, 1), np.int32), np.cumsum(hist, axis=1)], axis=1)
        out[s:s + n] = np.argmax(cs[:, width:] - cs[:, :-width], axis=1)
    return out


def _pad(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.uint8).reshape(-1)
    return np.concatenate([b, np.zeros((-len(b)) % PAD, np.uint8)])


@dataclass
class Encoded:
    rows: int
    cols: int
    code_bits: int
    tile_base: bool
    base: int                     # per-matrix base (also set with tile_base, unused)
    code: np.ndarray              # bytes, padded
    sm: np.ndarray                # bytes, padded
    esc: np.ndarray               # bytes, padded
    n_esc: int
    tbase: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint8))

    @property
    def n(self) -> int:
        return self.rows * self.cols

    def bits_per_weight(self, padded: bool = False) -> float:
        """Stored bits per weight: code + sm + escape bytes (+ tile bases).
        Unpadded by default; the 64-byte stream padding and the header are
        per-matrix constants (reported separately)."""
        if padded:
            total = len(self.code) + len(self.sm) + len(self.esc) + len(self.tbase)
        else:
            n_tiles = self.n // TILE
            total = n_tiles * self.code_bits * 8 + self.n + self.n_esc + (n_tiles if self.tile_base else 0)
        return 8 * total / self.n


def encode(w: np.ndarray, code_bits: int = 3, tile_base: bool = False,
           force_e_star: int | None = None) -> Encoded:
    """w: (rows, cols) uint16 BF16 bit patterns, cols % 64 == 0.
    force_e_star pins the per-matrix window (tests use it to force escapes)."""
    w = np.asarray(w, dtype=np.uint16)
    assert w.ndim == 2 and w.shape[1] % TILE == 0, "cols must be a multiple of 64"
    rows, cols = w.shape
    width = (1 << code_bits) - 1
    sign, exp, man = fields(w.reshape(-1))
    exp_t = exp.reshape(-1, TILE).astype(np.int64)

    e_star, _ = best_window(exp, width)
    if force_e_star is not None:
        e_star = force_e_star
    if tile_base:
        e_tile = best_window_per_tile(exp_t, width)
    else:
        e_tile = np.full(exp_t.shape[0], e_star, np.int64)
    base_t = (e_tile - 1) % 256                        # 8-bit, e* = 0 wraps to 255

    off = (exp_t - e_tile[:, None]) % 256              # exponent - e*
    code = np.where(off < width, off + 1, 0).astype(np.uint8)
    esc_mask = code.reshape(-1) == 0

    planes = [np.packbits(((code >> p) & 1).astype(np.uint8), axis=1, bitorder="little")
              for p in range(code_bits)]               # each (T, 8) bytes = one LE word
    code_stream = np.stack(planes, axis=1).reshape(-1)

    return Encoded(
        rows=rows, cols=cols, code_bits=code_bits, tile_base=tile_base,
        base=int((e_star - 1) % 256),
        code=_pad(code_stream),
        sm=_pad((sign << 7) | man),
        esc=_pad(exp[esc_mask]),
        n_esc=int(esc_mask.sum()),
        tbase=_pad(base_t.astype(np.uint8)) if tile_base else np.zeros(0, np.uint8),
    )


def decode(e: Encoded) -> np.ndarray:
    """Inverse of encode: returns (rows, cols) uint16."""
    n_tiles = e.n // TILE
    planes = e.code[:n_tiles * e.code_bits * 8].reshape(n_tiles, e.code_bits, 8)
    code = np.zeros((n_tiles, TILE), np.int64)
    for p in range(e.code_bits):
        code |= np.unpackbits(planes[:, p, :], axis=1, bitorder="little").astype(np.int64) << p
    if e.tile_base:
        base = e.tbase[:n_tiles].astype(np.int64)[:, None]
    else:
        base = np.full((n_tiles, 1), e.base, np.int64)
    exp = ((base + code) % 256).reshape(-1)
    esc_mask = code.reshape(-1) == 0
    assert esc_mask.sum() == e.n_esc
    exp[esc_mask] = e.esc[:e.n_esc]
    sm = e.sm[:e.n].astype(np.int64)
    w = ((sm >> 7) << 15) | (exp << 7) | (sm & 0x7F)
    return w.astype(np.uint16).reshape(e.rows, e.cols)


def raw_stream(w: np.ndarray) -> np.ndarray:
    """Baseline layout: BF16 little-endian, padded."""
    return _pad(np.asarray(w, dtype="<u2").reshape(-1).view(np.uint8))


# ---------------------------------------------------------------- synthetic
def synthetic_bf16(rows: int, cols: int, rng: np.random.Generator, std: float = 0.02,
                   outlier_frac: float = 1e-3, zero_frac: float = 1e-4) -> np.ndarray:
    """LLM-like weights for tests that can't load a real model: Gaussian,
    a few large outliers and exact zeros. NOT a substitute for real weights
    in any reported number."""
    x = rng.normal(0.0, std, (rows, cols)).astype(np.float32)
    m = rng.random((rows, cols))
    x = np.where(m < outlier_frac, x * 50, x)
    x = np.where(m > 1 - zero_frac, 0.0, x)
    return f32_to_bf16(x)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even FP32 -> BF16 bit patterns (finite inputs)."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for code_bits in (2, 3, 4):
        for tb in (False, True):
            w = synthetic_bf16(64, 896, rng)
            e = encode(w, code_bits, tb)
            assert np.array_equal(decode(e), w)
            print(f"b={code_bits} tile_base={tb!s:5}  {e.bits_per_weight():6.3f} bits/weight  "
                  f"escapes {e.n_esc / e.n:6.2%}")
    # every exponent value, including 0 and 255, survives the roundtrip
    w = (rng.integers(0, 1 << 16, (16, 256))).astype(np.uint16)
    for code_bits in (2, 3, 4):
        assert np.array_equal(decode(encode(w, code_bits)), w)
        assert np.array_equal(decode(encode(w, code_bits, True)), w)
    print("roundtrip OK")
