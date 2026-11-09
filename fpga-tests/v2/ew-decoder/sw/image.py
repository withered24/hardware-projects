"""DRAM image layout: put matrices, x vectors and y slots at 4 KB-aligned
addresses, and produce the WSTREAM.CFG values for each GEMV job.

The same layout serves the testbench memory (tb/axi_mem.sv, as a $readmemh
hex file of 64-bit words), the RISC-V driver, and later the board.
"""

from dataclasses import dataclass

import numpy as np

from ewformat import encode, raw_stream

ALIGN = 4096


@dataclass
class Job:
    rows: int
    cols: int
    bypass: bool
    base: int
    n_esc: int
    code_addr: int
    smr_addr: int
    esc_addr: int
    tbase_addr: int
    x_addr: int
    y_addr: int
    w: np.ndarray            # the original BF16 matrix (for checking)
    x: np.ndarray
    enc: object = None

    def cfg(self) -> dict[int, int]:
        """WSTREAM.CFG field id -> value (ids from rtl/ew_pkg.sv)."""
        return {0: self.code_addr, 1: self.smr_addr, 2: self.esc_addr, 3: self.n_esc,
                4: self.tbase_addr, 5: self.x_addr, 6: self.y_addr}


class Image:
    def __init__(self, base: int = 0x1000):
        self.buf = bytearray()
        self.next = base

    def alloc(self, nbytes: int) -> int:
        addr = self.next
        self.next = (addr + max(nbytes, 1) + ALIGN - 1) // ALIGN * ALIGN
        return addr

    def put(self, data) -> int:
        data = np.asarray(data, np.uint8).tobytes()
        addr = self.alloc(len(data))
        end = addr + len(data)
        if len(self.buf) < end:
            self.buf.extend(b"\0" * (end - len(self.buf)))
        self.buf[addr:end] = data
        return addr

    def add_job(self, w: np.ndarray, x: np.ndarray, bypass: bool = False, code_bits: int = 3,
                tile_base: bool = False, force_e_star: int | None = None) -> Job:
        w = np.asarray(w, np.uint16)
        rows, cols = w.shape
        x_addr = self.put(np.asarray(x, "<u2").view(np.uint8))
        y_addr = self.alloc(rows * 4)
        if bypass:
            smr = self.put(raw_stream(w))
            return Job(rows, cols, True, 0, 0, 0, smr, 0, 0, x_addr, y_addr, w, x)
        e = encode(w, code_bits, tile_base, force_e_star)
        code = self.put(e.code)
        smr = self.put(e.sm)
        esc = self.put(e.esc) if e.n_esc else 0
        tb = self.put(e.tbase) if tile_base else 0
        return Job(rows, cols, False, e.base, e.n_esc, code, smr, esc, tb, x_addr, y_addr, w, x, e)

    def words(self) -> np.ndarray:
        b = bytes(self.buf) + b"\0" * ((-len(self.buf)) % 8)
        return np.frombuffer(b, "<u8")

    def write_hex(self, path) -> None:
        with open(path, "w") as f:
            f.write("\n".join(f"{int(v):016x}" for v in self.words()))
            f.write("\n")


def read_hex(path) -> np.ndarray:
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//") and not line.startswith("@"):
                vals.append(int(line, 16))
    return np.array(vals, np.uint64)


def y_from_words(words: np.ndarray, rows: int) -> np.ndarray:
    return np.asarray(words, "<u8").view("<u4")[:rows].view(np.float32)
