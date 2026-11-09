"""cocotb tests for the GPU (rtl/gpu_top.sv), against the golden model.

    pytest tb/test_gpu.py -v -s       (-s shows cycle counts and SIMD efficiency)

Every kernel runs twice from the same memory image: on the GPU, and thread
by thread on riscv-core's ISS (gpu.run_model). The final 64 KiB of memory
must match byte for byte, and the GPU's count of executed thread
instructions must equal the model's. On top of that, each test checks the
result against plain Python, so a bug shared by model and kernel can't hide.
"""

import random

import cocotb

import kernels as K
from gpu import GPU, LANES, THREADS, run_kernel, u
from runner import GPU_SOURCES, SRAM_MODELS, run


async def setup(dut) -> GPU:
    gpu = GPU(dut)
    await gpu.reset()
    return gpu


def s32(x: int) -> int:
    x &= 0xFFFF_FFFF
    return x - (1 << 32) if x >> 31 else x


@cocotb.test()
async def test_vecadd(dut):
    """1000 elements = 31 full blocks + one block where only 8 threads have work."""
    gpu = await setup(dut)
    n, rng = 1000, random.Random(1)
    a = [rng.randrange(-2**31, 2**31) for _ in range(n)]
    b = [rng.randrange(-2**31, 2**31) for _ in range(n)]
    img = K.Image()
    pa, pb, pc = img.words(a), img.words(b), img.words([0] * n)
    args = img.words([pa, pb, pc, n])
    _, mem = await run_kernel(gpu, K.VECADD, img.mem, (n + THREADS - 1) // THREADS, THREADS, args, dut._log.info)
    assert img.read_words(mem, pc, n) == [s32(x + y) for x, y in zip(a, b)]


@cocotb.test()
async def test_saxpy(dut):
    gpu = await setup(dut)
    n, alpha, rng = 640, -7, random.Random(2)
    x = [rng.randrange(-1000, 1000) for _ in range(n)]
    y = [rng.randrange(-1000, 1000) for _ in range(n)]
    img = K.Image()
    px, py = img.words(x), img.words(y)
    args = img.words([px, py, alpha, n])
    _, mem = await run_kernel(gpu, K.SAXPY, img.mem, n // THREADS, THREADS, args, dut._log.info)
    assert img.read_words(mem, py, n) == [alpha * xi + yi for xi, yi in zip(x, y)]


@cocotb.test()
async def test_matmul(dut):
    """16x12 @ 12x32: one block per row of C, one thread per column."""
    gpu = await setup(dut)
    M, Kd, N, rng = 16, 12, THREADS, random.Random(3)
    A = [[rng.randrange(-128, 128) for _ in range(Kd)] for _ in range(M)]
    B = [[rng.randrange(-128, 128) for _ in range(N)] for _ in range(Kd)]
    img = K.Image()
    pA = img.words([v for row in A for v in row])
    pB = img.words([v for row in B for v in row])
    pC = img.words([0] * (M * N))
    args = img.words([pA, pB, pC, Kd, N])
    _, mem = await run_kernel(gpu, K.MATMUL, img.mem, M, N, args, dut._log.info)
    C = [sum(A[i][k] * B[k][j] for k in range(Kd)) for i in range(M) for j in range(N)]
    assert img.read_words(mem, pC, M * N) == C


def collatz_steps(n: int) -> int:
    s = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        s += 1
    return s


@cocotb.test()
async def test_collatz_divergence(dut):
    """Data-dependent loops and if/else in every thread: the warp splits and
    must reconverge. SIMD efficiency is well below 100% here, by design."""
    gpu = await setup(dut)
    n, rng = 256, random.Random(4)
    start = [rng.randrange(1, 1000) for _ in range(n)]
    img = K.Image()
    ps, pout = img.words(start), img.words([0] * n)
    args = img.words([ps, pout, n])
    st, mem = await run_kernel(gpu, K.COLLATZ, img.mem, n // THREADS, THREADS, args, dut._log.info)
    assert img.read_words(mem, pout, n) == [collatz_steps(v) for v in start]
    assert st.simd_efficiency < 0.9, "Collatz should diverge"


@cocotb.test()
async def test_coalescing(dut):
    """mem_lines counts exactly what the coalescer should send. Per warp:
    4 argument loads (one broadcast line each), the gather (stride lines
    for stride <= 8), and the store (one line)."""
    gpu = await setup(dut)
    n = 256
    warps = n // LANES
    for stride in (1, 2, 4, 8):
        img = K.Image()
        src = list(range(1000, 1000 + n * stride))
        pin, pout = img.words(src), img.words([0] * n)
        args = img.words([pin, pout, stride, n])
        st, mem = await run_kernel(gpu, K.STRIDED, img.mem, n // THREADS, THREADS, args)
        assert img.read_words(mem, pout, n) == src[::stride]
        dut._log.info(f"stride {stride}: {st.mem_lines} memory lines, {st.cycles} cycles")
        assert st.mem_lines == warps * (4 + stride + 1), f"stride {stride}: {st.mem_lines} lines"


@cocotb.test()
async def test_bytes_and_halfwords(dut):
    gpu = await setup(dut)
    n, rng = 96, random.Random(5)
    b8 = bytes(rng.randrange(256) for _ in range(n))
    h16 = [rng.randrange(65536) for _ in range(n)]
    img = K.Image()
    pb = img.raw(b8)
    ph = img.raw(b"".join(v.to_bytes(2, "little") for v in h16))
    pout = img.words([0] * (4 * n))
    args = img.words([pb, ph, pout, n])
    _, mem = await run_kernel(gpu, K.BYTES, img.mem, n // THREADS, THREADS, args, dut._log.info)
    out = img.read_words(mem, pout, 4 * n)
    for i in range(n):
        sb, sh = b8[i] - 256 * (b8[i] >> 7), h16[i] - 65536 * (h16[i] >> 15)
        assert out[4 * i:4 * i + 4] == [sb, b8[i], sh, h16[i]], f"thread {i}"
        assert mem[pb + i] == (b8[i] + 1) & 0xFF
        assert int.from_bytes(mem[ph + 2 * i:ph + 2 * i + 2], "little") == (h16[i] + 3) & 0xFFFF


@cocotb.test()
async def test_subroutine_calls(dut):
    gpu = await setup(dut)
    n, limit, rng = 160, 500, random.Random(6)
    x = [rng.randrange(-1000, 1000) for _ in range(n)]
    img = K.Image()
    px, py = img.words(x), img.words([0] * n)
    args = img.words([px, py, limit, n])
    _, mem = await run_kernel(gpu, K.CALLS, img.mem, n // THREADS, THREADS, args, dut._log.info)
    assert img.read_words(mem, py, n) == [min(abs(v), limit) for v in x]


@cocotb.test()
async def test_odd_block_sizes(dut):
    """Blocks smaller than a core, and not a multiple of a warp: some warps
    start with lanes switched off, some warps don't exist at all."""
    gpu = await setup(dut)
    for block in (1, 5, 13, 24):
        n = 3 * block
        a, b = list(range(n)), list(range(100, 100 + n))
        img = K.Image()
        pa, pb, pc = img.words(a), img.words(b), img.words([0] * n)
        args = img.words([pa, pb, pc, n])
        _, mem = await run_kernel(gpu, K.VECADD, img.mem, 3, block, args)
        assert img.read_words(mem, pc, n) == [x + y for x, y in zip(a, b)], f"block {block}"


@cocotb.test()
async def test_warps_hide_latency(dut):
    """The same 512 threads, as 64 blocks of one warp or 16 blocks of four
    warps. With one warp per core, every load stalls the core for the whole
    memory latency; with four, the other warps fill those cycles."""
    gpu = await setup(dut)
    n = 512
    cycles = {}
    for block in (LANES, THREADS):
        img = K.Image()
        pa, pb, pc = img.words(range(n)), img.words(range(n)), img.words([0] * n)
        args = img.words([pa, pb, pc, n])
        st, _ = await run_kernel(gpu, K.VECADD, img.mem, n // block, block, args)
        cycles[block] = st.cycles
        dut._log.info(f"{block // LANES} warp(s) per block: {st.cycles} cycles")
    assert cycles[LANES] > 1.5 * cycles[THREADS], cycles


@cocotb.test()
async def test_host_port(dut):
    """The word-wide host port into global memory (for a host CPU)."""
    from cocotb.triggers import FallingEdge, ReadOnly
    gpu = await setup(dut)
    rng = random.Random(7)
    vals = {4 * rng.randrange(16384): rng.randrange(2**32) for _ in range(64)}
    for addr, v in vals.items():
        await FallingEdge(dut.clk)
        dut.dmem_we.value, dut.dmem_addr.value, dut.dmem_wdata.value = 1, addr, v
    await FallingEdge(dut.clk)
    dut.dmem_we.value = 0
    for addr, v in vals.items():
        dut.dmem_addr.value = addr
        await ReadOnly()
        assert u(dut.dmem_rdata) == v, f"{addr:#x}"
        await FallingEdge(dut.clk)


@cocotb.test()
async def test_random_kernels(dut):
    """Random nested branches, loops, calls, gathers and early exits
    (tb/randkernel.py), random grid and block sizes. RAND_KERNELS sets how
    many, RAND_SEED the first seed."""
    import os
    from randkernel import OUT_BYTES, TABLE_WORDS, Gen
    gpu = await setup(dut)
    count = int(os.environ.get("RAND_KERNELS", "25"))
    first = int(os.environ.get("RAND_SEED", "1"))
    eff = []
    for seed in range(first, first + count):
        rng = random.Random(seed)
        grid, block = rng.randrange(1, 7), rng.randrange(1, THREADS + 1)
        img = K.Image()
        table = img.words(rng.randrange(2**32) for _ in range(TABLE_WORDS))
        out = img.alloc(grid * block * OUT_BYTES)
        args = img.words([table, out])
        try:
            st, _ = await run_kernel(gpu, Gen(seed).kernel(), img.mem, grid, block, args)
        except AssertionError as e:
            raise AssertionError(f"seed {seed} (grid {grid}, block {block}): {e}") from None
        eff.append(st.simd_efficiency)
    dut._log.info(f"{count} random kernels passed; SIMD efficiency {min(eff):.0%}..{max(eff):.0%}")


def test_gpu():
    run("gpu_top", GPU_SOURCES, "test_gpu")


def test_gpu_sram_macros():
    """Every test again with the register files on OpenRAM's model of the
    SkyWater SRAM macro, as the layout builds them."""
    run("gpu_top", SRAM_MODELS + GPU_SOURCES, "test_gpu", build_name="gpu_top_macro",
        defines={"SKY130_SRAM": 1}, extra_env={"RAND_KERNELS": "10"})
