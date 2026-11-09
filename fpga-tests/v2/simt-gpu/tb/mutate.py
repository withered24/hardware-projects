"""Mutation testing: plant a known bug, check the test suite catches it.

    python tb/mutate.py            (all of them, about a minute each)
    python tb/mutate.py 3 5        (just mutants 3 and 5)

A passing suite only means something if it fails on a broken GPU. Each
mutant below is a realistic mistake in one place of the RTL. The script
applies it, runs tb/test_gpu.py (with 10 random kernels), and restores
the file. Every mutant must be killed (the suite fails).
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MUTANTS = [
    ("rtl/simt_core.sv", "f_mask[l] = pick_alive[l] && pc[pick*LANES + l] == f_pc;",
     "f_mask[l] = pick_alive[l];",
     "no divergence: every live lane runs, whatever its PC"),
    ("rtl/simt_core.sv", "alive[x_warp*LANES +: LANES] <= alive[x_warp*LANES +: LANES] & ~x_mask;",
     "alive[x_warp*LANES +: LANES] <= '0;",
     "an ECALL kills the whole warp, not just the lanes that ran it"),
    ("rtl/simt_core.sv", "if (pick_valid)                 busy[pick]          <= 1'b1;",
     "",
     "a warp can issue again before its last instruction finished"),
    ("rtl/simt_core.sv", "CSR_TID:    x_csrv[l] = 32'(x_warp) * LANES + l;",
     "CSR_TID:    x_csrv[l] = l;",
     "thread id forgets the warp number"),
    # (Making x0 writable is an equivalent mutant: reads of x0 are hardwired to
    # 0, so nothing can observe the write. Not hardwiring the read is the real bug.)
    ("rtl/simt_lane.sv", "rs1v <= r_z1 ? 32'd0 : q1;",
     "rs1v <= q1;",
     "x0 reads the register file instead of a hardwired 0"),
    ("rtl/simt_core.sv", "assign lq_apply = lq_count != 0 && !(lq_port && x_wr);",
     "assign lq_apply = lq_count != 0;",
     "load data and X's result both take the one register-file write port"),
    ("rtl/lsu.sv", "group[l] = pending[l] && la[31:OFF_BITS] == line;",
     "group[l] = pending[l];",
     "coalescer puts every lane in the first lane's line"),
    ("rtl/lsu.sv", "req_be[byte_at +: 2]       = 2'b11;",
     "req_be[byte_at +: 2]       = 2'b01;",
     "SH writes only its low byte"),
    ("rtl/lsu.sv", "3'b000:  wb_data[l] = {{24{word[7]}},  word[7:0]};    // LB",
     "3'b000:  wb_data[l] = {24'd0, word[7:0]};    // LB",
     "LB zero-extends"),
    ("rtl/lsu.sv", "done_mask  = t_mask[t_head];",
     "done_mask  = t_group[t_head];",
     "only the last request's lanes move past a load"),
    ("rtl/gpu_top.sv", "if (busy && next_block < k_grid)",
     "if (busy && next_block <= k_grid)",
     "dispatcher launches one block too many"),
]


def main() -> int:
    env = dict(os.environ, RAND_KERNELS="10", SIM_BUILD_ROOT=str(REPO / "sim_build" / "mutants"))
    survivors = []
    pick = {int(a) for a in sys.argv[1:]} or set(range(1, len(MUTANTS) + 1))
    for i, (path, old, new, what) in enumerate(MUTANTS, 1):
        if i not in pick:
            continue
        f = REPO / path
        text = f.read_text()
        assert text.count(old) == 1, f"mutant {i}: pattern not found once in {path}"
        f.write_text(text.replace(old, new), newline="\n")
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "tb/test_gpu.py"],
                               cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
            killed = r.returncode != 0
        except subprocess.TimeoutExpired:
            killed = True                    # a hang is a failure too
        finally:
            f.write_text(text, newline="\n")
        print(f"{'killed  ' if killed else 'SURVIVED'}  {i:2}. {what}", flush=True)
        if not killed:
            survivors.append(what)
    print(f"\n{len(pick) - len(survivors)}/{len(pick)} mutants killed")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
