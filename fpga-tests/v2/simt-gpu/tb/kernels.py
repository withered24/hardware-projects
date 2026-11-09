"""GPU kernels, written in RV32I assembly (+ mul and the ID CSRs).

Every kernel gets a pointer to its arguments with `csrr a0, args` and
loads them from there, the way CUDA passes kernel parameters in a constant
buffer. Each thread works out its global index the CUDA way:

    i = blockIdx.x * blockDim.x + threadIdx.x  =  ctaid * ntid + tid

and finishes with ECALL. `Image` lays out the arguments and arrays in a
global-memory image, aligned to memory lines.
"""

import struct

from gpu import LINE_BYTES, MEM_BYTES

# i = ctaid * ntid + tid  ->  t0; leaves a0 = args
GLOBAL_ID = """
        csrr  t0, ctaid
        csrr  t1, ntid
        mul   t0, t0, t1
        csrr  t1, tid
        add   t0, t0, t1
        csrr  a0, args
"""

# c[i] = a[i] + b[i]        args: a, b, c, n
VECADD = GLOBAL_ID + """
        lw    t2, 12(a0)
        bge   t0, t2, done          # the last block's spare threads drop out
        slli  t0, t0, 2
        lw    a1, 0(a0)
        lw    a2, 4(a0)
        lw    a3, 8(a0)
        add   a1, a1, t0
        add   a2, a2, t0
        add   a3, a3, t0
        lw    t3, 0(a1)
        lw    t4, 0(a2)
        add   t3, t3, t4
        sw    t3, 0(a3)
done:
        ecall
"""

# y[i] = alpha * x[i] + y[i]     args: x, y, alpha, n
SAXPY = GLOBAL_ID + """
        lw    t2, 12(a0)
        bge   t0, t2, done
        slli  t0, t0, 2
        lw    a1, 0(a0)
        lw    a2, 4(a0)
        lw    a3, 8(a0)
        add   a1, a1, t0
        add   a2, a2, t0
        lw    t3, 0(a1)
        lw    t4, 0(a2)
        mul   t3, t3, a3
        add   t3, t3, t4
        sw    t3, 0(a2)
done:
        ecall
"""

# C = A @ B, one thread per element of C: block = row, thread = column.
# args: A (MxK), B (KxN), C (MxN), K, N       (grid = M, block = N)
MATMUL = """
        csrr  t0, ctaid             # row
        csrr  t1, tid               # column
        csrr  a0, args
        lw    a1, 0(a0)
        lw    a2, 4(a0)
        lw    a3, 8(a0)
        lw    a4, 12(a0)            # K
        lw    a5, 16(a0)            # N
        mul   t2, t0, a4            # &A[row][0]
        slli  t2, t2, 2
        add   a1, a1, t2
        slli  t3, t1, 2             # &B[0][col]
        add   a2, a2, t3
        slli  a6, a5, 2             # bytes per row of B
        li    s0, 0                 # sum
        mv    t4, a4
loop:
        lw    t5, 0(a1)             # A[row][k]: the same word for every lane (broadcast)
        lw    t6, 0(a2)             # B[k][col]: consecutive words across lanes (coalesced)
        mul   t5, t5, t6
        add   s0, s0, t5
        addi  a1, a1, 4
        add   a2, a2, a6
        addi  t4, t4, -1
        bnez  t4, loop
        mul   t2, t0, a5            # &C[row][col]
        add   t2, t2, t1
        slli  t2, t2, 2
        add   a3, a3, t2
        sw    s0, 0(a3)
        ecall
"""

# steps[i] = number of Collatz steps from start[i] down to 1.
# Every thread loops a different number of times and takes the odd/even
# branch in its own pattern: the worst case for SIMT.   args: start, steps, n
COLLATZ = GLOBAL_ID + """
        lw    t2, 8(a0)
        bge   t0, t2, done
        slli  t0, t0, 2
        lw    a1, 0(a0)
        lw    a2, 4(a0)
        add   a1, a1, t0
        add   a2, a2, t0
        lw    t3, 0(a1)             # n
        li    t4, 0                 # steps
        li    t5, 1
loop:
        beq   t3, t5, out
        andi  t6, t3, 1
        beqz  t6, even
        slli  t6, t3, 1             # odd: n = 3n + 1
        add   t3, t3, t6
        addi  t3, t3, 1
        j     next
even:
        srli  t3, t3, 1             # even: n = n / 2
next:
        addi  t4, t4, 1
        j     loop
out:
        sw    t4, 0(a2)
done:
        ecall
"""

# out[i] = in[i * stride]: stride 1 is one memory line per warp, stride 8
# is eight.   args: in, out, stride, n
STRIDED = GLOBAL_ID + """
        lw    t2, 12(a0)
        bge   t0, t2, done
        lw    a1, 0(a0)
        lw    a2, 4(a0)
        lw    a3, 8(a0)
        mul   t3, t0, a3
        slli  t3, t3, 2
        add   a1, a1, t3
        slli  t0, t0, 2
        add   a2, a2, t0
        lw    t4, 0(a1)
        sw    t4, 0(a2)
done:
        ecall
"""

# Byte and halfword access, signed and unsigned. Per thread i:
#   out[4i..4i+3] = lb(b8[i]), lbu(b8[i]), lh(h16[i]), lhu(h16[i])
#   b8[i] += 1 (sb), h16[i] += 3 (sh)          args: b8, h16, out, n
BYTES = GLOBAL_ID + """
        lw    t2, 12(a0)
        bge   t0, t2, done
        lw    a1, 0(a0)
        lw    a2, 4(a0)
        lw    a3, 8(a0)
        add   a1, a1, t0            # &b8[i]
        slli  t3, t0, 1
        add   a2, a2, t3            # &h16[i]
        slli  t3, t0, 4
        add   a3, a3, t3            # &out[4i]
        lb    t4, 0(a1)
        lbu   t5, 0(a1)
        lh    t6, 0(a2)
        lhu   s0, 0(a2)
        sw    t4, 0(a3)
        sw    t5, 4(a3)
        sw    t6, 8(a3)
        sw    s0, 12(a3)
        addi  t4, t4, 1
        sb    t4, 0(a1)
        addi  t6, t6, 3
        sh    t6, 0(a2)
done:
        ecall
"""

# y[i] = clamp(|x[i]|, 0, LIMIT) through a subroutine (JAL/JALR), with
# nested branches inside it.   args: x, y, limit, n
CALLS = GLOBAL_ID + """
        lw    t2, 12(a0)
        bge   t0, t2, done
        slli  t0, t0, 2
        lw    a1, 0(a0)
        lw    a2, 4(a0)
        lw    a3, 8(a0)
        add   a1, a1, t0
        add   a2, a2, t0
        lw    a4, 0(a1)
        jal   clamp_abs
        sw    a4, 0(a2)
done:
        ecall

clamp_abs:                          # a4 = min(|a4|, a3)
        bge   a4, zero, pos
        sub   a4, zero, a4
pos:
        blt   a4, a3, ok
        mv    a4, a3
ok:
        ret
"""


class Image:
    """A global-memory image with a bump allocator."""

    def __init__(self):
        self.mem = bytearray(MEM_BYTES)
        self.top = LINE_BYTES          # keep address 0 unused

    def alloc(self, nbytes: int) -> int:
        addr = self.top
        self.top += (nbytes + LINE_BYTES - 1) // LINE_BYTES * LINE_BYTES
        assert self.top <= MEM_BYTES, "out of global memory"
        return addr

    def words(self, values, signed: bool = True) -> int:
        values = list(values)
        addr = self.alloc(4 * len(values))
        for i, v in enumerate(values):
            struct.pack_into("<I", self.mem, addr + 4 * i, v & 0xFFFF_FFFF)
        return addr

    def raw(self, data: bytes) -> int:
        addr = self.alloc(len(data))
        self.mem[addr:addr + len(data)] = data
        return addr

    def read_words(self, mem: bytearray, addr: int, n: int, signed: bool = True) -> list[int]:
        fmt = "<%d%s" % (n, "i" if signed else "I")
        return list(struct.unpack_from(fmt, mem, addr))
