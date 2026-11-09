"""Whole-engine tests: ew_top + testbench DRAM, driven through the command
port exactly as the CPU's custom instructions would.

Two checks on every job:
  * decode: a monitor on the decoder's output compares every chunk it hands
    to the GEMV core with the original BF16 matrix, bit for bit. This is the
    lossless-storage claim, checked directly (y alone could hide a decode
    error behind flush-to-zero).
  * y: the result in memory == sw/golden.py gemv(), bit for bit.

Tests: directed corners, randomized jobs (RAND_SEEDS, default 100; the
brief's 10,000 is `RAND_SEEDS=10000 pytest tb/test_top.py -k random`),
and a simulated decoded-vs-raw cycle count.

    pytest tb/test_top.py -v
"""

import os
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge, Timer

import golden
from ewformat import TILE, synthetic_bf16
from image import Image, read_hex, y_from_words
from coverage import Coverage
from runner import SRAM_MODELS, TB_TOP, run

F3_EXP_BASE, F3_WSTREAM_CFG, F3_GEMV_RUN, F3_GEMV_WAIT, F3_DEC_BYPASS, F3_EW_STAT = range(6)
ST_CYCLES, ST_CHUNKS, ST_STALL_ESC, ST_STALL_DATA, ST_ESCAPES = range(5)
MASK = 0xFFFF_FFFF

L = int(os.environ.get("EW_L", "8"))
CODE_BITS = int(os.environ.get("EW_CODE_BITS", "3"))
TILE_BASE = os.environ.get("EW_TILE_BASE", "0") == "1"
MEMFILE = os.environ.get("EW_MEMFILE", "mem.hex")
DUMPFILE = os.environ.get("EW_DUMPFILE", "dump.hex")


# ------------------------------------------------------------------ helpers
def chunks(w):
    """The decoder's output words for matrix w: L weights per chunk, lane i in
    bits 16i+15..16i (so little-endian bytes of the flat matrix)."""
    b = np.ascontiguousarray(w, "<u2").tobytes()
    return [int.from_bytes(b[i:i + 2 * L], "little") for i in range(0, len(b), 2 * L)]


class Engine:
    def __init__(self, dut, dec=None):
        self.dut = dut
        self.dec = dec if dec is not None else dut.u_dut.u_dec    # the decoder the monitor watches
        self.expect = []          # decoded chunks the monitor should see, in order
        self.seen = 0
        self.errors = []
        self.cov = Coverage(L, CODE_BITS, TILE_BASE)

    async def reset(self):
        d = self.dut
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        d.rst_n.value = 0
        d.cmd_valid.value = 0
        d.cmd_funct3.value = 0
        d.cmd_rs1.value = 0
        d.cmd_rs2.value = 0
        d.load.value = 0
        d.dump.value = 0
        d.dump_lo.value = 0
        d.dump_hi.value = 0
        for _ in range(5):
            await RisingEdge(d.clk)
        d.rst_n.value = 1
        cocotb.start_soon(self._monitor())

    async def cmd(self, f3, rs1=0, rs2=0, limit=2_000_000):
        d = self.dut
        await FallingEdge(d.clk)
        d.cmd_funct3.value = f3
        d.cmd_rs1.value = rs1 & MASK
        d.cmd_rs2.value = rs2 & MASK
        d.cmd_valid.value = 1
        for _ in range(limit):
            await Timer(1, unit="ns")
            if int(d.cmd_ready.value):
                r = int(d.cmd_rdata.value)
                await RisingEdge(d.clk)
                await FallingEdge(d.clk)
                d.cmd_valid.value = 0
                return r
            await RisingEdge(d.clk)
            await FallingEdge(d.clk)
        raise TimeoutError(f"command f3={f3} never completed")

    async def _monitor(self):
        """Every chunk the decoder hands over must be the next expected one."""
        d, dec = self.dut, self.dec
        while True:
            await FallingEdge(d.clk)
            await ReadOnly()
            self.cov.cycle_sample(int(dec.stall_esc.value), int(dec.stall_data.value))
            if int(dec.out_valid.value) and int(dec.out_ready.value):
                got = int(dec.out_w.value)
                self.cov.chunk(got, int(dec.out_nesc.value), int(dec.out_j.value), int(dec.base.value),
                               int(dec.bypass.value))
                if self.seen >= len(self.expect):
                    self.errors.append(f"unexpected chunk #{self.seen}")
                elif got != self.expect[self.seen]:
                    want = self.expect[self.seen]
                    lanes = [i for i in range(L) if (got >> 16 * i) & 0xFFFF != (want >> 16 * i) & 0xFFFF]
                    self.errors.append(f"chunk #{self.seen}: lanes {lanes} got {got:0{4*L}x} want {want:0{4*L}x}")
                self.seen += 1

    async def load(self, img: Image):
        img.write_hex(MEMFILE)
        d = self.dut
        await FallingEdge(d.clk)
        d.load.value = 1
        await Timer(1, unit="ns")
        d.load.value = 0

    async def read_words(self, addr, nwords):
        d = self.dut
        await FallingEdge(d.clk)
        d.dump_lo.value = addr >> 3
        d.dump_hi.value = (addr >> 3) + nwords - 1
        await Timer(1, unit="ns")
        d.dump.value = 1
        await Timer(1, unit="ns")
        d.dump.value = 0
        await Timer(1, unit="ns")
        return read_hex(DUMPFILE)

    async def run_job(self, job) -> int:
        """Configure, run and wait. Returns the job's cycle count."""
        for fid, val in job.cfg().items():
            await self.cmd(F3_WSTREAM_CFG, fid, val)
        await self.cmd(F3_EXP_BASE, job.base)
        await self.cmd(F3_DEC_BYPASS, int(job.bypass))
        self.expect.extend(chunks(job.w))
        h = await self.cmd(F3_GEMV_RUN, job.rows, job.cols)
        return await self.cmd(F3_GEMV_WAIT, h, limit=20_000_000)   # big matrices at low bandwidth

    async def check_y(self, job, tag=""):
        words = await self.read_words(job.y_addr, (job.rows + 1) // 2)
        y = y_from_words(words, job.rows)
        want = golden.gemv(job.w, job.x, L)
        ok = golden.same(y, want)
        if not ok.all():
            i = int(np.argmin(ok))
            self.errors.append(f"{tag} y[{i}] got {golden.f32_bits(y[i]):08x} "
                               f"want {golden.f32_bits(want[i]):08x} ({int((~ok).sum())} rows wrong)")

    def finish(self, tag=""):
        self.cov.save(f"L{L}_b{CODE_BITS}_tb{int(TILE_BASE)}")
        if self.seen != len(self.expect):
            self.errors.append(f"{tag} saw {self.seen} chunks, expected {len(self.expect)}")
        assert not self.errors, "\n".join(self.errors[:20])


def weights_in_window(rng, rows, cols, e_lo, width):
    """Every exponent inside [e_lo, e_lo+width-1]: zero escapes."""
    e = rng.integers(e_lo, e_lo + width, (rows, cols))
    return ((rng.integers(0, 2, (rows, cols)) << 15) | (e << 7) | rng.integers(0, 128, (rows, cols))).astype(np.uint16)


def x_vec(rng, cols):
    return synthetic_bf16(1, cols, rng, std=1.0)[0]


async def go(dut, jobs_spec, tag):
    """jobs_spec: list of (w, x, kwargs). Lays them out, runs them in order."""
    eng = Engine(dut)
    await eng.reset()
    img = Image()
    jobs = [img.add_job(w, x, code_bits=CODE_BITS, tile_base=TILE_BASE, **kw) for w, x, kw in jobs_spec]
    await eng.load(img)
    for k, job in enumerate(jobs):
        await eng.run_job(job)
        await eng.check_y(job, f"{tag} job {k}")
    await RisingEdge(dut.clk)
    eng.finish(tag)
    return eng


# ------------------------------------------------------------------ directed
@cocotb.test()
async def test_zero_escapes(dut):
    rng = np.random.default_rng(10)
    w = weights_in_window(rng, 4, 256, 118, (1 << CODE_BITS) - 1)
    eng = await go(dut, [(w, x_vec(rng, 256), {})], "zero escapes")
    assert await eng.cmd(F3_EW_STAT, ST_ESCAPES) == 0


@cocotb.test()
async def test_all_escaped(dut):
    """Window pinned where no weight lives: every lane escapes every cycle."""
    rng = np.random.default_rng(11)
    w = weights_in_window(rng, 4, 256, 118, 4)
    eng = await go(dut, [(w, x_vec(rng, 256), {"force_e_star": 20})], "all escaped")
    assert await eng.cmd(F3_EW_STAT, ST_ESCAPES) == w.size


@cocotb.test()
async def test_escape_boundaries(dut):
    """Escapes right at tile edges (63/64), smr burst edges (every 256
    weights) and enough of them to cross several escape-stream bursts."""
    rng = np.random.default_rng(12)
    w = weights_in_window(rng, 8, 512, 118, (1 << CODE_BITS) - 1)
    flat = w.reshape(-1)
    for pos in list(range(TILE - 2, flat.size, TILE)) + list(range(TILE, flat.size, TILE)) + list(range(255, flat.size, 256)):
        flat[pos] = (flat[pos] & 0x807F) | (200 << 7)
    await go(dut, [(w, x_vec(rng, 512), {})], "escape boundaries")


@cocotb.test()
async def test_edge_exponents(dut):
    """Window at exponent 0 (base wraps to 255), exponent 255 weights
    (inf/NaN) as escapes and as window members."""
    rng = np.random.default_rng(13)
    w = weights_in_window(rng, 4, 128, 0, (1 << CODE_BITS) - 1)
    w.reshape(-1)[::37] |= 0x7F80                                        # exponent 255
    w2 = weights_in_window(rng, 4, 128, 256 - (1 << CODE_BITS) + 1, (1 << CODE_BITS) - 1)
    await go(dut, [(w, x_vec(rng, 128), {}), (w2, x_vec(rng, 128), {})], "edge exponents")


@cocotb.test()
async def test_back_to_back_and_bypass(dut):
    """Different bases job to job, and decoded <-> raw switching, no reset."""
    rng = np.random.default_rng(14)
    a = synthetic_bf16(6, 192, rng)
    b = synthetic_bf16(6, 192, rng, std=0.5)
    c = synthetic_bf16(2, 64, rng, std=1e-3)
    spec = [(a, x_vec(rng, 192), {}), (b, x_vec(rng, 192), {}), (a, x_vec(rng, 192), {"bypass": True}),
            (c, x_vec(rng, 64), {}), (b, x_vec(rng, 192), {"bypass": True}), (b, x_vec(rng, 192), {})]
    await go(dut, spec, "back to back")


# ---------------------------------------------------------------- randomized
@cocotb.test()
async def test_random(dut):
    seeds = int(os.environ.get("RAND_SEEDS", "100"))
    first = int(os.environ.get("RAND_FIRST", "0"))
    eng = Engine(dut)
    await eng.reset()
    for s in range(first, first + seeds):
        rng = np.random.default_rng(1000 + s)
        rows = 2 * int(rng.integers(1, 9))
        cols = TILE * int(rng.integers(1, 7))
        kind = rng.integers(0, 4)
        if kind == 0:
            w = synthetic_bf16(rows, cols, rng, std=float(10 ** rng.uniform(-4, 1)))
        elif kind == 1:                                   # random escape density
            width = (1 << CODE_BITS) - 1
            w = weights_in_window(rng, rows, cols, int(rng.integers(1, 250 - width)), width)
            m = rng.random(w.shape) < rng.choice([0.0, 0.01, 0.1, 0.5, 1.0])
            w = np.where(m, (w & 0x807F) | (rng.integers(0, 256, w.shape) << 7).astype(np.uint16), w).astype(np.uint16)
        elif kind == 2:
            w = rng.integers(0, 1 << 16, (rows, cols)).astype(np.uint16)    # anything, incl. inf/NaN
        else:
            w = synthetic_bf16(rows, cols, rng)
        kw = {"bypass": bool(rng.random() < 0.25)}
        img = Image()
        job = img.add_job(w, x_vec(rng, cols), code_bits=CODE_BITS, tile_base=TILE_BASE, **kw)
        await eng.load(img)
        await eng.run_job(job)
        await eng.check_y(job, f"seed {s}")
        if eng.errors:
            break
    await RisingEdge(dut.clk)
    eng.finish("random")
    dut._log.info(f"random: {seeds} seeds, {eng.seen} chunks, all bit-exact")


# ------------------------------------------------------------- performance
@cocotb.test()
async def test_cycles(dut):
    """Same matrix, decoded vs raw, against the memory model's bandwidth
    budget. SIMULATED cycle counts on synthetic weights; not a result."""
    rng = np.random.default_rng(20)
    w = synthetic_bf16(64, 896, rng)
    x = x_vec(rng, 896)
    eng = Engine(dut)
    await eng.reset()
    img = Image()
    jd = img.add_job(w, x, code_bits=CODE_BITS, tile_base=TILE_BASE)
    jr = img.add_job(w, x, bypass=True)
    await eng.load(img)
    cd = await eng.run_job(jd)
    st = {k: await eng.cmd(F3_EW_STAT, k) for k in range(5)}
    cr = await eng.run_job(jr)
    await eng.check_y(jd, "cycles dec")
    await eng.check_y(jr, "cycles raw")
    eng.finish("cycles")
    bits = jd.enc.bits_per_weight()
    dut._log.info(f"SIMULATED (synthetic weights, BW budget per tb params): decoded {cd} cycles, raw {cr} cycles, "
                  f"speedup {cr / cd:.3f} (ceiling 16/{bits:.2f} = {16 / bits:.3f}); "
                  f"esc stalls {st[ST_STALL_ESC]}, data stalls {st[ST_STALL_DATA]}, escapes {st[ST_ESCAPES]}")
    Path(os.environ.get("EW_PERF_OUT", "perf.txt")).write_text(
        f"decoded_cycles={cd}\nraw_cycles={cr}\nbits={bits:.4f}\nstall_esc={st[ST_STALL_ESC]}\n"
        f"stall_data={st[ST_STALL_DATA]}\nescapes={st[ST_ESCAPES]}\n")


# ------------------------------------------------------------------ pytest
def _run(build, L_=8, code_bits=3, tile_base=0, bw=(16, 1), testcase=None, env=None, module="test_top",
         mem_words=1 << 18, sram=None):
    """sram: None = flip-flop buffers, "model" = sram_2p's behavioural model,
    "macro" = OpenRAM's models of the SkyWater macros the ASIC uses"""
    from runner import REPO
    bdir = REPO / "sim_build" / build
    bdir.mkdir(parents=True, exist_ok=True)
    e = {"EW_L": str(L_), "EW_CODE_BITS": str(code_bits), "EW_TILE_BASE": str(tile_base),
         "EW_MEMFILE": str(bdir / "mem.hex"), "EW_DUMPFILE": str(bdir / "dump.hex"),
         "EW_PERF_OUT": str(bdir / "perf.txt"), **(env or {})}
    run("tb_top", SRAM_MODELS + TB_TOP if sram == "macro" else TB_TOP, module, build_name=build,
        parameters={"L": L_, "CODE_BITS": code_bits, "PER_TILE_BASE": tile_base,
                    "BW_NUM": bw[0], "BW_DEN": bw[1], "MEM_WORDS": mem_words, "SRAM": int(sram is not None)},
        defines={"SKY130_SRAM": 1} if sram == "macro" else None,
        plusargs=[f"+MEMFILE={e['EW_MEMFILE']}", f"+DUMPFILE={e['EW_DUMPFILE']}"],
        extra_env=e, testcase=testcase)


def test_top_default():
    """L=8, 3-bit codes, per-matrix base: every test."""
    _run("top_l8_b3")


def test_top_tile_base():
    _run("top_l8_b3_tile", tile_base=1, testcase="test_zero_escapes,test_escape_boundaries,test_back_to_back_and_bypass,test_random",
         env={"RAND_SEEDS": "40"})
    _run("top_l8_b3_tile_bw8", tile_base=1, bw=(8, 1), testcase="test_cycles")   # data stalls


def test_top_code_bits():
    for b in (2, 4):
        _run(f"top_l8_b{b}", code_bits=b, testcase="test_all_escaped,test_edge_exponents,test_random",
             env={"RAND_SEEDS": "30"})
        _run(f"top_l8_b{b}_bw8", code_bits=b, bw=(8, 1), testcase="test_cycles")


def test_top_bw_limited():
    """Cycle comparison with the memory at 8 bytes/cycle: the raw baseline
    needs 16 (L=8 x 2 bytes), so both modes are bandwidth-bound here."""
    _run("top_l8_b3_bw8", bw=(8, 1), testcase="test_cycles")


def test_top_l16():
    _run("top_l16_b3", L_=16, testcase="test_escape_boundaries,test_edge_exponents,test_back_to_back_and_bypass,test_random",
         env={"RAND_SEEDS": "30"})


def test_top_sram():
    """The ASIC build's buffers in SRAM: sram_2p's model at two widths, then
    every test on OpenRAM's models of the SkyWater macros."""
    _run("top_l8_b3_sram", sram="model")
    _run("top_l4_b3_sram", L_=4, sram="model", testcase="test_random,test_back_to_back_and_bypass")
    _run("top_l8_b3_macro", sram="macro")
    _run("top_l8_b3_macro_bw8", sram="macro", bw=(8, 1), testcase="test_cycles")


def test_top_l4():
    """L=4: one 64-bit beat is the whole x word (gemv_core's g_x1 branch)."""
    _run("top_l4_b3", L_=4, env={"RAND_SEEDS": "30"})
    _run("top_l4_b3_bw4", L_=4, bw=(4, 1), testcase="test_cycles")    # data stalls
