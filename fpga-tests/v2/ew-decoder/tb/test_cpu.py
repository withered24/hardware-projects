"""System test: riscv-core runs sw/driver/layer.c, which drives the engine
with the custom-0 instructions. Nothing in the testbench touches the
command port; the CPU does all of it.

The host's part is what it will be on a board: lay the matrices out in DRAM
(sw/image.py), write the job table into the CPU's memory, release reset,
wait for the driver's done flag. Then the checks:
  * the decode monitor (as in test_top.py): every chunk bit-exact
  * y in DRAM == sw/golden.py, bit for bit, for every job
  * the escape count the driver read back (EW.STAT) == the encoder's
  * no instruction retired illegal or trapped

    pytest tb/test_cpu.py -v
"""

import os
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge

from ewformat import synthetic_bf16
from image import Image
from runner import TB_SOC, run
from test_top import Engine, chunks

L = int(os.environ.get("EW_L", "8"))
PROGRAM = os.environ.get("EW_PROGRAM", "")
JOBS, RES, DONE = 0x8000, 0xC000, 0xFFFC
J_WORDS = 11
MAX_JOBS = (RES - JOBS) // 4 // J_WORDS


def job_record(job) -> list[int]:
    """One job-table entry, in sw/driver/layer.c's J_* order."""
    return [job.rows, job.cols, int(job.bypass), job.base, job.n_esc, job.code_addr,
            job.smr_addr, job.esc_addr, job.tbase_addr, job.x_addr, job.y_addr]


class Soc:
    def __init__(self, dut):
        self.dut = dut
        self.mem = dut.u_mem.mem
        self.eng = Engine(dut, dec=dut.u_eng.u_dut.u_dec)
        self.bad_retire = []

    def poke(self, addr, words):
        for i, w in enumerate(words):
            self.mem[(addr >> 2) + i].value = int(w) & 0xFFFF_FFFF

    def peek(self, addr, n=1):
        return [int(self.mem[(addr >> 2) + i].value) for i in range(n)]

    async def _retire_monitor(self):
        d = self.dut.u_core
        while True:
            await RisingEdge(self.dut.clk)
            await ReadOnly()
            if int(d.trace_valid.value) and (int(d.trace_illegal.value) or int(d.trace_trap.value)):
                self.bad_retire.append(f"pc {int(d.trace_pc.value):08x} instr {int(d.trace_instr.value):08x}")

    async def run(self, img: Image, jobs, limit=400_000):
        d = self.dut
        assert len(jobs) <= MAX_JOBS
        cocotb.start_soon(Clock(d.clk, 10, unit="ns").start())
        d.rst_n.value = 0
        d.load.value = 0
        d.dump.value = 0
        d.dump_lo.value = 0
        d.dump_hi.value = 0
        await RisingEdge(d.clk)

        prog = Path(PROGRAM).read_bytes()
        prog += b"\0" * (-len(prog) % 4)
        self.poke(0, np.frombuffer(prog, "<u4"))
        table = [len(jobs)] + [v for j in jobs for v in job_record(j)]
        self.poke(JOBS, table)
        self.poke(DONE, [0])
        await self.eng.load(img)
        for j in jobs:                       # what the monitor should see, in order
            self.eng.expect.extend(chunks(j.w))

        for _ in range(5):
            await RisingEdge(d.clk)
        await FallingEdge(d.clk)
        d.rst_n.value = 1
        cocotb.start_soon(self.eng._monitor())
        cocotb.start_soon(self._retire_monitor())

        for cyc in range(0, limit, 1000):
            await self._wait(1000)
            if self.peek(DONE)[0] == 1:
                break
        else:
            raise TimeoutError("driver never set the done flag")
        await self._wait(10)
        res = self.peek(RES, 4 * len(jobs))
        return [res[4 * i: 4 * i + 4] for i in range(len(jobs))], cyc + 1000

    async def _wait(self, n):
        for _ in range(n):
            await RisingEdge(self.dut.clk)

    async def check(self, jobs, res, tag):
        for k, (job, r) in enumerate(zip(jobs, res)):
            await self.eng.check_y(job, f"{tag} job {k}")
            cycles, escapes, _, _ = r
            if cycles == 0:
                self.eng.errors.append(f"{tag} job {k}: GEMV.WAIT returned 0 cycles")
            if not job.bypass and escapes != job.n_esc:
                self.eng.errors.append(f"{tag} job {k}: EW.STAT escapes {escapes}, encoder wrote {job.n_esc}")
        if self.bad_retire:
            self.eng.errors.append(f"{tag}: illegal/trap at {self.bad_retire[:4]}")
        self.eng.finish(tag)


def layer(rng, d, kv, ffn):
    """One decoder layer's seven projections, synthetic BF16 weights (rows, cols)."""
    shapes = {"q": (d, d), "k": (kv, d), "v": (kv, d), "o": (d, d),
              "gate": (ffn, d), "up": (ffn, d), "down": (d, ffn)}
    return {n: synthetic_bf16(r, c, rng) for n, (r, c) in shapes.items()}


@cocotb.test()
async def test_layer(dut):
    """A scaled-down layer through the CPU, then `up` again as raw BF16."""
    rng = np.random.default_rng(7)
    ws = layer(rng, d=128, kv=32, ffn=384)
    img = Image()
    jobs = []
    for name, w in ws.items():
        jobs.append(img.add_job(w, synthetic_bf16(1, w.shape[1], rng, std=1.0)[0]))
    jobs.append(img.add_job(ws["up"], jobs[5].x, bypass=True))
    soc = Soc(dut)
    res, total = await soc.run(img, jobs)
    await soc.check(jobs, res, "layer")
    for (name, _), r in zip(list(ws.items()) + [("up (raw)", None)], res):
        dut._log.info(f"{name:9} cycles {r[0]:6d}  escapes {r[1]:4d}  esc stalls {r[2]:4d}  data stalls {r[3]:5d}")
    dut._log.info(f"whole program: {total} cycles (simulated, synthetic weights)")


@cocotb.test()
async def test_random_jobs(dut):
    """Random shapes and modes, many jobs back to back from one program run."""
    rng = np.random.default_rng(int(os.environ.get("RAND_FIRST", "11")))
    img = Image()
    jobs = []
    for _ in range(int(os.environ.get("CPU_JOBS", "24"))):
        rows = int(rng.integers(1, 9)) * 8
        cols = int(rng.integers(1, 5)) * 64
        w = synthetic_bf16(rows, cols, rng, std=float(rng.choice([0.002, 0.02, 0.3])),
                           outlier_frac=float(rng.choice([0, 0.01, 0.2])))
        jobs.append(img.add_job(w, synthetic_bf16(1, cols, rng, std=1.0)[0], bypass=bool(rng.random() < 0.25)))
    soc = Soc(dut)
    res, _ = await soc.run(img, jobs)
    await soc.check(jobs, res, "random")


# ------------------------------------------------------------------ pytest
def test_cpu():
    import sys
    from runner import REPO
    sys.path.insert(0, str(REPO / "sw" / "driver"))
    from build import build
    binf = build()
    bdir = REPO / "sim_build" / "soc_l8_b3"
    bdir.mkdir(parents=True, exist_ok=True)
    e = {"EW_L": "8", "EW_PROGRAM": str(binf),
         "EW_MEMFILE": str(bdir / "mem.hex"), "EW_DUMPFILE": str(bdir / "dump.hex")}
    run("tb_soc", TB_SOC, "test_cpu", build_name="soc_l8_b3",
        parameters={"L": 8, "CODE_BITS": 3},
        plusargs=[f"+MEMFILE={e['EW_MEMFILE']}", f"+DUMPFILE={e['EW_DUMPFILE']}"],
        extra_env=e)
