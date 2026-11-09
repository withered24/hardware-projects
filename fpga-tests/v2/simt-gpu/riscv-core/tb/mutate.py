"""Mutation testing for the pipelined core: does the regression notice bugs?

Each mutant is one deliberate, realistic bug (a missing forward, a missing
flush, a wrong priority...). For each one this script copies rtl/ to a
scratch directory, plants the bug, and runs every suite against that copy
(RTL_DIR / SIM_BUILD_ROOT point the normal tests at it), plus the formal
proof (formal/core.sby, needs `sby` on PATH: source env.sh). A mutant is
"killed" if at least one suite fails. A surviving mutant is either a hole
in the tests or an equivalent mutant (a change no program can observe),
and each survivor needs an argument for which one it is.

    python tb/mutate.py                  # all mutants, all suites
    python tb/mutate.py -j 4 -k stall    # only mutants whose name contains "stall"

Writes a Markdown table to sim_build/mutation.md.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "sim_build" / "mutants"

SUITES = {                       # column name -> test file
    "directed": "tb/test_core.py",
    "hazard": "tb/test_pipeline.py",
    "riscv-tests": "tb/test_riscv_tests.py",
    "random": "tb/test_random.py",
    "formal": "formal/core.sby",     # k-induction proof of the SVA properties in core.sv
}

# (name, file in rtl/, original text, mutated text[, cores to run on])
# Mutants of the memory-wait freeze only do anything when the data port
# waits, so they run on the "wait" core (random wait states) instead.
MUTANTS = [
    ("no EX/MEM forward (rs1)", "core.sv",
     "if (m_ctrl.reg_write && m_rd != 5'd0 && m_rd == x_rs1)  fwd_a = FWD_EXMEM;",
     "if (1'b0)  fwd_a = FWD_EXMEM;"),
    ("no MEM/WB forward (rs2)", "core.sv",
     "else if (w_rf_we && w_rd != 5'd0 && w_rd == x_rs2)      fwd_b = FWD_MEMWB;",
     "else if (1'b0)      fwd_b = FWD_MEMWB;"),
    ("forward priority reversed (rs1)", "core.sv",
     "if (m_ctrl.reg_write && m_rd != 5'd0 && m_rd == x_rs1)  fwd_a = FWD_EXMEM;\n"
     "        else if (w_rf_we && w_rd != 5'd0 && w_rd == x_rs1)      fwd_a = FWD_MEMWB;",
     "if (w_rf_we && w_rd != 5'd0 && w_rd == x_rs1)      fwd_a = FWD_MEMWB;\n"
     "        else if (m_ctrl.reg_write && m_rd != 5'd0 && m_rd == x_rs1)  fwd_a = FWD_EXMEM;"),
    ("x0 forwarded from EX/MEM (rs1)", "core.sv",
     "m_ctrl.reg_write && m_rd != 5'd0 && m_rd == x_rs1",
     "m_ctrl.reg_write && m_rd == x_rs1"),
    ("no WB->ID bypass (rs2)", "core.sv",
     "assign rs2_val_d = (w_rf_we && w_rd != 5'd0 && w_rd == rs2_d) ? w_wdata : rf_rs2;",
     "assign rs2_val_d = rf_rs2;"),
    ("store data not forwarded (SW)", "core.sv",
     "dmem_wdata = rs2_x;\n                    dmem_wstrb = 4'b1111;",
     "dmem_wdata = x_rs2_val;\n                    dmem_wstrb = 4'b1111;"),
    ("branch compares unforwarded rs1", "core.sv",
     "branch_unit u_br (.a(rs1_x)", "branch_unit u_br (.a(x_rs1_val)"),
    ("CSR write source not forwarded", "core.sv",
     "m_csr_wsrc  <= x_funct3[2] ? {27'd0, x_rs1} : rs1_x;",
     "m_csr_wsrc  <= x_funct3[2] ? {27'd0, x_rs1} : x_rs1_val;"),
    ("no load-use stall", "core.sv",
     "assign stall  = d_valid && x_late", "assign stall  = 1'b0 && x_late"),
    ("stall ignores CSR reads", "core.sv",
     "assign x_late = x_ctrl.mem_read || x_ctrl.csr_en;", "assign x_late = x_ctrl.mem_read;"),
    ("stall ignores rs2", "core.sv",
     "(uses_rs2_d && rs2_d == x_rd)", "(1'b0)"),
    ("stall inserts no bubble", "core.sv",
     "assign flush_x = trap_m || mret_m || redirect_x || stall;",
     "assign flush_x = trap_m || mret_m || redirect_x;"),
    ("IF/ID not held on stall", "core.sv",
     "end else if (!stall) begin\n            d_valid <= 1'b1;",
     "end else begin\n            d_valid <= 1'b1;"),
    ("PC not held on stall", "core.sv",
     "else if (stall)      pc_next = pc_f;", "else if (1'b0)      pc_next = pc_f;"),
    ("taken branch doesn't flush ID/EX", "core.sv",
     "assign flush_x = trap_m || mret_m || redirect_x || stall;",
     "assign flush_x = trap_m || mret_m || stall;"),
    ("taken branch doesn't flush IF/ID", "core.sv",
     "assign flush_d = trap_m || mret_m || redirect_x;",
     "assign flush_d = trap_m || mret_m;"),
    ("branch beats trap in the PC mux", "core.sv",
     "else if (trap_m)     pc_next = mtvec_base;\n"
     "        else if (mret_m)     pc_next = mepc_q;\n"
     "        else if (redirect_x) pc_next = redirect_pc;",
     "else if (redirect_x) pc_next = redirect_pc;\n"
     "        else if (trap_m)     pc_next = mtvec_base;\n"
     "        else if (mret_m)     pc_next = mepc_q;"),
    ("store not cancelled by an older trap", "core.sv",
     "!exc_x && !trap_m && !mret_m && !rst", "!exc_x && !mret_m && !rst"),
    ("store not cancelled by an older MRET", "core.sv",
     "!exc_x && !trap_m && !mret_m && !rst", "!exc_x && !trap_m && !rst"),
    ("faulting store still writes", "core.sv",
     "x_ctrl.mem_write && !exc_x && !trap_m", "x_ctrl.mem_write && !trap_m"),
    ("trap doesn't cancel the register write", "core.sv",
     "w_rd_we   <= m_ctrl.reg_write && !trap_m;", "w_rd_we   <= m_ctrl.reg_write;"),
    ("MRET doesn't flush EX/MEM", "core.sv",
     "assign flush_m = trap_m || mret_m;", "assign flush_m = trap_m;"),
    ("FENCE.I doesn't refetch", "core.sv",
     "(taken_x || x_ctrl.fence_i)", "(taken_x)"),
    ("stores don't mark rs1 as read", "decoder.sv",
     "mem_write = 1'b1;\n                uses_rs1  = 1'b1;",
     "mem_write = 1'b1;\n                uses_rs1  = 1'b0;"),
    ("branches don't mark rs2 as read", "decoder.sv",
     "uses_rs2  = 1'b1;\n                if (funct3[2:1] == 2'b01)",
     "uses_rs2  = 1'b0;\n                if (funct3[2:1] == 2'b01)"),
    ("CSR-immediate forms mark rs1 as read", "decoder.sv",
     "uses_rs1  = !funct3[2];", "uses_rs1  = 1'b1;"),
    # Expected to survive (equivalent), kept to show why:
    ("faulting jump also redirects", "core.sv",
     "assign redirect_x  = x_valid && !exc_x &&", "assign redirect_x  = x_valid &&"),
    ("bubbles decode real controls", "core.sv",
     "if (d_valid) begin\n            ctrl_d.reg_write", "if (1'b1) begin\n            ctrl_d.reg_write"),
    ("frozen EX doesn't latch forwarded operands", "core.sv",
     "            x_rs1_val <= rs1_x;\n            x_rs2_val <= rs2_x;\n", "", "wait"),
    ("frozen MEM doesn't bubble WB", "core.sv",
     "            w_valid   <= 1'b0;\n            w_rd_we   <= 1'b0;\n            w_trap    <= 1'b0;\n",
     "", "wait"),
    ("fetch address moves while frozen", "core.sv",
     "        else if (hold)       pc_next = pc_f;\n", "", "wait"),
    ("EX/MEM advances while frozen", "core.sv",
     "        end else if (!hold) begin\n            m_valid     <= x_valid;",
     "        end else begin\n            m_valid     <= x_valid;", "wait"),
    ("IF/ID flushed while frozen", "core.sv",
     "if (rst || (flush_d && !hold)) begin", "if (rst || flush_d) begin", "wait"),
]


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")[:48]


def run_mutant(m) -> tuple[str, dict[str, bool], float]:
    name, fname, old, new = m[:4]
    cores = m[4] if len(m) > 4 else "pipeline"
    d = WORK / slug(name)
    rtl = d / "rtl"
    shutil.rmtree(rtl, ignore_errors=True)
    shutil.copytree(REPO / "rtl", rtl, dirs_exist_ok=True)   # builds are redone anyway
    src = (rtl / fname).read_text(encoding="utf-8")
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"mutant '{name}': original text found {n} times in {fname}")
    (rtl / fname).write_text(src.replace(old, new), encoding="utf-8")

    env = dict(os.environ, RTL_DIR=str(rtl), SIM_BUILD_ROOT=str(d / "build"),
               REGRESS_CORES=cores, RAND_REQUIRE_COVERAGE="0")
    t0 = time.time()
    failed = {}
    for suite, path in SUITES.items():
        if path.endswith(".sby"):
            # core.sby reads ../rtl/*.sv, so give it a formal/ next to the mutant rtl/
            (d / "formal").mkdir(exist_ok=True)
            shutil.copy(REPO / path, d / "formal")
            cmd, cwd = ["sby", "-f", "core.sby", "prove"], d / "formal"
        else:
            cmd, cwd = [sys.executable, "-m", "pytest", path, "-q"], REPO
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=1800)
        failed[suite] = r.returncode != 0
    return name, failed, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=4, help="mutants run in parallel")
    ap.add_argument("-k", default="", help="only mutants whose name contains this")
    ap.add_argument("--suites", default=",".join(SUITES), help="comma-separated columns to run")
    args = ap.parse_args()
    todo = [m for m in MUTANTS if args.k.lower() in m[0].lower()]
    for s in set(SUITES) - set(args.suites.split(",")):
        del SUITES[s]

    rows = []
    with ThreadPoolExecutor(args.j) as pool:
        for name, failed, secs in pool.map(run_mutant, todo):
            killed = any(failed.values())
            marks = " ".join(f"{s}={'FAIL' if f else 'pass'}" for s, f in failed.items())
            print(f"{'KILLED ' if killed else 'SURVIVED'} {name:42s} {marks}  ({secs:.0f}s)", flush=True)
            rows.append((name, failed))

    killed = sum(any(f.values()) for _, f in rows)
    head = "| Planted bug | " + " | ".join(SUITES) + " |\n|---|" + "---|" * len(SUITES) + "\n"
    body = "".join(f"| {n} | " + " | ".join("❌ caught" if f[s] else "·" for s in SUITES) + " |\n"
                   for n, f in rows)
    table = head + body + f"\n{killed}/{len(rows)} mutants killed.\n"
    out = REPO / "sim_build" / "mutation.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(table, encoding="utf-8")
    print(f"\n{killed}/{len(rows)} killed; table in {out}")


if __name__ == "__main__":
    main()
