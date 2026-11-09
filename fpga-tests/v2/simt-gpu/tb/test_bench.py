"""CPU vs GPU: the same work on riscv-core's 5-stage pipeline and on the GPU.

    pytest tb/test_bench.py -s

Two workloads, same data, same algorithm:
  vecadd   c[i] = a[i] + b[i] for 1024 elements: perfectly parallel, no divergence
  collatz  Collatz step counts for 256 starting values: every thread loops a
           different number of times (the GPU's worst case)

The CPU runs a hand-scheduled loop (no load-use stalls) in lockstep with
riscv-core's golden model, so its result is checked too. Its memory answers
in one cycle. The GPU runs twice: with 2-cycle memory (the fastest gpu_mem
can be, close to the CPU's), and with its default 8-cycle memory (a real
GPU's DRAM is far slower still). The
numbers go to docs/bench.json and a table is printed at the end.

Matmul isn't compared: riscv-core is RV32I and has no multiply instruction.
"""

import json
import random

import cocotb

import kernels as K
from gpu import GPU, REPO, THREADS, run_kernel
from runner import GPU_SOURCES, run

RESULTS = REPO / "docs" / "bench.json"
N_ADD, N_COLLATZ = 1024, 256
A_BASE, B_BASE, C_BASE = 0x1000, 0x3000, 0x5000      # CPU data addresses


def workload():
    rng = random.Random(42)
    a = [rng.randrange(-2**31, 2**31) for _ in range(N_ADD)]
    b = [rng.randrange(-2**31, 2**31) for _ in range(N_ADD)]
    start = [rng.randrange(1, 1000) for _ in range(N_COLLATZ)]
    return a, b, start


def record(key: str, value) -> None:
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(data, indent=2) + "\n")


def collatz_steps(n: int) -> int:
    s = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        s += 1
    return s


# ---------------------------------------------------------------------------
# CPU side (top level: riscv-core's tb_top, pipelined core)

CPU_VECADD = f"""
        li    a1, {A_BASE}
        li    a2, {B_BASE}
        li    a3, {C_BASE}
        li    t2, {N_ADD}
loop:
        lw    t3, 0(a1)
        lw    t4, 0(a2)
        addi  a1, a1, 4
        addi  a2, a2, 4
        add   t3, t3, t4
        sw    t3, 0(a3)
        addi  a3, a3, 4
        addi  t2, t2, -1
        bnez  t2, loop
"""

CPU_COLLATZ = f"""
        li    a1, {A_BASE}
        li    a2, {C_BASE}
        li    t2, {N_COLLATZ}
        li    t5, 1
outer:
        lw    t3, 0(a1)
        li    t4, 0
inner:
        beq   t3, t5, out
        andi  t6, t3, 1
        beqz  t6, even
        slli  t6, t3, 1
        add   t3, t3, t6
        addi  t3, t3, 1
        j     next
even:
        srli  t3, t3, 1
next:
        addi  t4, t4, 1
        j     inner
out:
        sw    t4, 0(a2)
        addi  a1, a1, 4
        addi  a2, a2, 4
        addi  t2, t2, -1
        bnez  t2, outer
"""


async def run_cpu(dut, source: str, data: dict[int, int]):
    from asm import assemble
    from cosim import DONE, DONE_ADDR, MEM_WORDS, load_and_reset, lockstep
    from iss import ISS
    words = assemble(source + DONE)
    iss = ISS(mem_bytes=4 * MEM_WORDS, halt_on_ecall=False)
    iss.load_words(words)
    for addr, w in data.items():
        iss.write(addr, w, 4)
    await load_and_reset(dut, words, data=data)
    st = await lockstep(dut, iss, stop=lambda r: r.store_addr == DONE_ADDR, max_steps=1_000_000)
    return st, iss


@cocotb.test()
async def cpu_vecadd(dut):
    a, b, _ = workload()
    data = {A_BASE + 4 * i: v & 0xFFFF_FFFF for i, v in enumerate(a)}
    data |= {B_BASE + 4 * i: v & 0xFFFF_FFFF for i, v in enumerate(b)}
    st, iss = await run_cpu(dut, CPU_VECADD, data)
    for i in range(N_ADD):
        assert iss.read(C_BASE + 4 * i, 4) == (a[i] + b[i]) & 0xFFFF_FFFF
    record("cpu_vecadd", {"cycles": st.cycles, "instructions": st.instructions})


@cocotb.test()
async def cpu_collatz(dut):
    _, _, start = workload()
    st, iss = await run_cpu(dut, CPU_COLLATZ, {A_BASE + 4 * i: v for i, v in enumerate(start)})
    for i, v in enumerate(start):
        assert iss.read(C_BASE + 4 * i, 4) == collatz_steps(v)
    record("cpu_collatz", {"cycles": st.cycles, "instructions": st.instructions})


# ---------------------------------------------------------------------------
# GPU side (top level: gpu_top)

async def gpu_run(dut, source, img, grid, args):
    import os
    gpu = GPU(dut)
    await gpu.reset()
    st, mem = await run_kernel(gpu, source, img.mem, grid, THREADS, args)
    tag = f"gpu{os.environ['MEM_LATENCY']}"
    return st, mem, tag


@cocotb.test()
async def gpu_vecadd(dut):
    a, b, _ = workload()
    img = K.Image()
    pa, pb, pc = img.words(a), img.words(b), img.words([0] * N_ADD)
    args = img.words([pa, pb, pc, N_ADD])
    st, mem, tag = await gpu_run(dut, K.VECADD, img, N_ADD // THREADS, args)
    assert img.read_words(mem, pc, N_ADD, signed=False) == [(x + y) & 0xFFFF_FFFF for x, y in zip(a, b)]
    record(f"{tag}_vecadd", st.__dict__ | {"simd_efficiency": round(st.simd_efficiency, 3)})


@cocotb.test()
async def gpu_collatz(dut):
    _, _, start = workload()
    img = K.Image()
    ps, pout = img.words(start), img.words([0] * N_COLLATZ)
    args = img.words([ps, pout, N_COLLATZ])
    st, mem, tag = await gpu_run(dut, K.COLLATZ, img, N_COLLATZ // THREADS, args)
    assert img.read_words(mem, pout, N_COLLATZ) == [collatz_steps(v) for v in start]
    record(f"{tag}_collatz", st.__dict__ | {"simd_efficiency": round(st.simd_efficiency, 3)})


# ---------------------------------------------------------------------------

CPU_SOURCES = [f"riscv-core/{s}" for s in (
    "rtl/rv32i_pkg.sv", "rtl/regfile.sv", "rtl/immgen.sv", "rtl/alu.sv", "rtl/branch_unit.sv",
    "rtl/decoder.sv", "rtl/csr.sv", "rtl/core_single.sv", "rtl/core.sv",
    "tb/sim_mem.sv", "tb/tb_top.sv")]


def test_bench():
    RESULTS.unlink(missing_ok=True)
    run("tb_top", CPU_SOURCES, "test_bench", build_name="bench_cpu", testcase=["cpu_vecadd", "cpu_collatz"])
    for lat in (2, 8):
        run("gpu_top", GPU_SOURCES, "test_bench", build_name=f"bench_gpu{lat}",
            parameters={"MEM_LATENCY": lat}, extra_env={"MEM_LATENCY": str(lat)},
            testcase=["gpu_vecadd", "gpu_collatz"])
    r = json.loads(RESULTS.read_text())
    print("\n workload   CPU cycles   GPU cycles (2-cycle mem)   GPU cycles (8-cycle mem)   SIMD eff.")
    for w in ("vecadd", "collatz"):
        cpu = r[f"cpu_{w}"]["cycles"]
        g1, g8 = r[f"gpu2_{w}"]["cycles"], r[f"gpu8_{w}"]["cycles"]
        print(f" {w:9} {cpu:11,} {g1:12,} ({cpu / g1:4.1f}x) {g8:17,} ({cpu / g8:4.1f}x) "
              f"{r[f'gpu8_{w}']['simd_efficiency']:13.0%}")
