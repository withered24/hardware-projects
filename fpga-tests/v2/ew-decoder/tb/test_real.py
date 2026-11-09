"""Real weights through the engine (brief §5.4): every linear matrix of one
Qwen2.5-0.5B decoder layer, decoded and then on the raw-BF16 bypass, with
the decode monitor and y == golden.py bit for bit on every job.

Needs the model in analysis/weights (python analysis/measure.py downloads
it). EW_LAYER picks the layer (default 12).

    pytest tb/test_real.py -s
"""

import os
import sys
from pathlib import Path

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge

from ewformat import synthetic_bf16
from image import Image
from test_top import Engine

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = REPO / "analysis" / "weights"
PROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def layer(n):
    sys.path.insert(0, str(REPO / "analysis"))
    from measure import bf16_tensors
    want = [f"model.layers.{n}.{p}.weight" for p in PROJ]
    got = {}
    for f in sorted(WEIGHTS.glob("*.safetensors")):
        for name, w in bf16_tensors(f):
            if name in want:
                got[name] = np.array(w)
    return [(p.split(".")[-1], got[w]) for p, w in zip(PROJ, want)]


@cocotb.test()
async def test_real_layer(dut):
    n = int(os.environ.get("EW_LAYER", "12"))
    mats = layer(n)
    rng = np.random.default_rng(n)
    img = Image()
    jobs = []
    for bypass in (False, True):
        for name, w in mats:
            x = synthetic_bf16(1, w.shape[1], rng, std=1.0)[0]
            jobs.append((name, img.add_job(w, x, bypass=bypass)))
    assert img.next <= 8 * int(dut.u_mem.WORDS.value), "layer doesn't fit the testbench DRAM"
    eng = Engine(dut)
    await eng.reset()
    await eng.load(img)
    for name, job in jobs:
        cycles = await eng.run_job(job)
        await eng.check_y(job, f"layer {n} {name}{' (bypass)' if job.bypass else ''}")
        dut._log.info(f"{name:8} {'raw' if job.bypass else 'ew ':3} {job.rows:5}x{job.cols:<5} "
                      f"cycles {cycles:7d}  escapes {job.n_esc:6d}  errors so far {len(eng.errors)}")
    await RisingEdge(dut.clk)
    eng.finish(f"layer {n}")


@cocotb.test()
async def test_real_point(dut):
    """One real matrix, decoded with this build's (L, code bits, tile base)
    and on the raw bypass, against the memory model's bandwidth budget
    (Phase 7 sweep). EW_MAT picks the projection of layer EW_LAYER."""
    from test_top import CODE_BITS, F3_EW_STAT, L, ST_ESCAPES, ST_STALL_DATA, ST_STALL_ESC, TILE_BASE
    n = int(os.environ.get("EW_LAYER", "21"))
    want = os.environ.get("EW_MAT", "q_proj")
    w = dict(layer(n))[want]
    rng = np.random.default_rng(n)
    x = synthetic_bf16(1, w.shape[1], rng, std=1.0)[0]
    img = Image()
    jd = img.add_job(w, x, code_bits=CODE_BITS, tile_base=TILE_BASE)
    jr = img.add_job(w, x, bypass=True)
    assert img.next <= 8 * int(dut.u_mem.WORDS.value), "matrix doesn't fit the testbench DRAM"
    eng = Engine(dut)
    await eng.reset()
    await eng.load(img)
    cd = await eng.run_job(jd)
    st = {k: await eng.cmd(F3_EW_STAT, k) for k in (ST_STALL_ESC, ST_STALL_DATA, ST_ESCAPES)}
    await eng.check_y(jd, f"{want} decoded")
    cr = await eng.run_job(jr)
    await eng.check_y(jr, f"{want} raw")
    eng.finish(f"layer {n} {want}")
    bits = jd.enc.bits_per_weight()
    line = (f"layer={n} mat={want} L={L} code_bits={CODE_BITS} tile_base={int(TILE_BASE)} "
            f"bits={bits:.4f} decoded_cycles={cd} raw_cycles={cr} speedup={cr / cd:.4f} "
            f"ceiling={16 / bits:.4f} stall_esc={st[ST_STALL_ESC]} stall_data={st[ST_STALL_DATA]} "
            f"escapes={st[ST_ESCAPES]}")
    dut._log.info("SIMULATED, real weights: " + line)
    Path(os.environ.get("EW_PERF_OUT", "perf.txt")).write_text(line + "\n")


@pytest.mark.parametrize("L_,code_bits,tile_base", [(8, 2, 0), (8, 3, 0), (8, 4, 0), (8, 3, 1), (4, 3, 0), (16, 3, 0)])
@pytest.mark.parametrize("mat", ["q_proj", "up_proj"])
def test_sweep(L_, code_bits, tile_base, mat):
    """Phase 7 points. Bandwidth budget L bytes/cycle: half of what raw BF16
    needs at that L, so both modes are memory-bound at every point."""
    if not list(WEIGHTS.glob("*.safetensors")):
        pytest.skip("no model in analysis/weights (run analysis/measure.py)")
    from test_top import _run
    _run(f"sweep_L{L_}_b{code_bits}_tb{tile_base}_{mat}", L_=L_, code_bits=code_bits, tile_base=tile_base,
         bw=(L_, 1), module="test_real", testcase="test_real_point", env={"EW_MAT": mat},
         mem_words=(1 << 22) if mat == "up_proj" else (1 << 20))


def test_real():
    if not list(WEIGHTS.glob("*.safetensors")):
        pytest.skip("no model in analysis/weights (run analysis/measure.py)")
    from test_top import _run
    _run("real_l8_b3", module="test_real", mem_words=1 << 23)   # 64 MB: a layer twice


def test_real_sram():
    """The real layer again with the ASIC build's buffers on OpenRAM's models
    of the SkyWater SRAM macros, KMAX at its full 4864."""
    if not list(WEIGHTS.glob("*.safetensors")):
        pytest.skip("no model in analysis/weights (run analysis/measure.py)")
    from test_top import _run
    _run("real_l8_b3_macro", module="test_real", mem_words=1 << 23, sram="macro")


def test_real_bw8():
    """Same jobs with DRAM as the bottleneck: 8 bytes/cycle shared by all ports,
    half what raw BF16 needs at L = 8 (the setting of the synthetic run in
    LEDGER.md), so both modes are bandwidth-bound and cycles follow bytes."""
    if not list(WEIGHTS.glob("*.safetensors")):
        pytest.skip("no model in analysis/weights (run analysis/measure.py)")
    from test_top import _run
    _run("real_l8_b3_bw8", module="test_real", bw=(8, 1), mem_words=1 << 23)
