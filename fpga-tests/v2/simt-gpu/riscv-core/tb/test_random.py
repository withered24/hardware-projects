"""Constrained-random regression (Milestone 4), run on both cores.

Generates RAND_PROGRAMS programs (default 200) from seeds RAND_SEED,
RAND_SEED + 1, ... (default 1) with tb/rand_prog.py and runs each in
lockstep against the golden model. Functional coverage (tb/coverage.py) is
collected over the whole run, printed, and saved to
sim_build/coverage_<core>.json. With the default seeds the run must close
every coverage bin; a hole fails the test, because it means the generator
or the design changed in a way that stopped exercising something.

A failing program is saved to sim_build/rand_fail_<seed>.S (with its data
image) and the seed is in the error, so it can be rerun alone:

    RAND_SEED=1234 RAND_PROGRAMS=1 pytest tb/test_random.py
    python tb/rand_prog.py 1234          # print the program

Long soak:   RAND_PROGRAMS=5000 RAND_SEED=100000 pytest tb/test_random.py
"""

import os
from pathlib import Path

import cocotb

from asm import assemble
from cosim import (CORE_SOURCES, DONE, DONE_ADDR, HANDLER_PRELUDE, MEM_WORDS,
                   load_and_reset, lockstep, selected_cores)
from coverage import Coverage
from iss import ISS
from rand_prog import generate
from runner import REPO, run


@cocotb.test()
async def test_random_programs(dut):
    seed0 = int(os.environ.get("RAND_SEED", "1"))
    count = int(os.environ.get("RAND_PROGRAMS", "200"))
    core = os.environ.get("CORE", "pipeline")
    out = Path(os.environ.get("SIM_BUILD_ROOT", REPO / "sim_build"))
    cov = Coverage(pipeline=(core in ("pipeline", "wait")), wait=(core == "wait"))
    total_i = total_c = 0

    for seed in range(seed0, seed0 + count):
        prog = generate(seed)
        words = assemble(HANDLER_PRELUDE + prog.source + DONE)
        iss = ISS(mem_bytes=4 * MEM_WORDS, halt_on_ecall=False)
        iss.load_words(words)
        for a, w in prog.data.items():
            iss.write(a, w, 4)
        await load_and_reset(dut, words, data=prog.data)
        cov.new_program()
        try:
            st = await lockstep(dut, iss, stop=lambda r: r.store_addr == DONE_ADDR,
                                max_steps=20_000, on_retire=cov.sample_retire,
                                on_cycle=cov.sample_cycle)
        except AssertionError as e:
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"rand_fail_{seed}.S"
            path.write_text(f"# seed {seed}, core {core}\n{prog.source}\n# data image\n"
                            + "".join(f"# {a:#06x}: {w:#010x}\n" for a, w in prog.data.items()),
                            encoding="utf-8")
            raise AssertionError(f"random program seed {seed} failed (saved to {path}):\n{e}") from None
        cov.programs += 1
        total_i += st.instructions
        total_c += st.cycles

    dut._log.info(f"{count} random programs (seeds {seed0}..{seed0 + count - 1}) on the "
                  f"{core} core: {total_i} instructions in {total_c} cycles "
                  f"(CPI {total_c / max(total_i, 1):.2f}), all matched the model")
    dut._log.info(cov.report())
    out.mkdir(parents=True, exist_ok=True)
    cov.save(out / f"coverage_{core}.json")
    if os.environ.get("RAND_REQUIRE_COVERAGE", "1") != "0" and count >= 200:
        assert not cov.holes(), f"coverage holes: {', '.join(cov.holes())}"


def test_random():
    for name, defines in selected_cores().items():
        run("tb_top", CORE_SOURCES, "test_random", build_name=f"tb_top_random_{name}",
            defines=defines, waves=False, extra_env={
                "CORE": name,
                **{k: os.environ[k] for k in ("RAND_SEED", "RAND_PROGRAMS", "RAND_REQUIRE_COVERAGE",
                                               "SIM_BUILD_ROOT") if k in os.environ}})
