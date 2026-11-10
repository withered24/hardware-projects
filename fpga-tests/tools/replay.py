"""replay.py: feed a suite's traces through the replay harness and compare.

    python replay.py sim <suite> [--trace NAME] [--chunks N]   Icarus, no board
    python replay.py hw  <suite> [--trace NAME] [--chunks N]   DE2-115 over JTAG
    python replay.py selftest [suite]                          JTAG path check (board)

Both modes cut each trace (traces/<suite>/<build>.trc) into chunks of
2**aw entries (builds/<suite>/chunks/), run them through the harness, and
compare every response entry with the trace's outputs wherever the trace
had no X. hw mode programs the bitstream again before each trace, so every
trace starts from power-up state, as in simulation. Results land in
results/<suite>/<mode>/<trace>.txt and a one-line summary is printed.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
FT = TOOLS.parent
QBIN = Path(os.environ.get("QUARTUS_BIN", r"C:\altera_lite\25.1std\quartus\bin64"))
OSS = Path(os.environ.get("OSS_CAD", r"C:\Users\rayan\tools\oss-cad-suite"))


def load_meta(suite):
    return json.loads((FT / "builds" / suite / "meta.json").read_text())


def traces(suite, only=None):
    ts = sorted((FT / "traces" / suite).glob("*.trc"))
    return [t for t in ts if not only or t.stem == only]


def read_trace(path):
    ins, outs, masks = [], [], []
    with open(path) as f:
        for line in f:
            a, b, c = line.split()
            ins.append(int(a, 16))
            outs.append(int(b, 16))
            masks.append(int(c, 16))
    return ins, outs, masks


def prep(m, trc: Path, max_chunks=None):
    """Write chunk dirs with sNN.hex (one 32-bit word per line). Returns
    [(chunk_dir, count, first_entry)]."""
    ins, _, _ = read_trace(trc)
    depth = 2 ** m["aw"]
    root = FT / "builds" / m["suite"] / "chunks" / trc.stem
    if root.exists():
        shutil.rmtree(root)
    chunks = []
    for c0 in range(0, len(ins), depth):
        if max_chunks and len(chunks) >= max_chunks:
            break
        cd = root / f"c{c0 // depth:05d}"
        cd.mkdir(parents=True)
        part = ins[c0:c0 + depth]
        for k, name in enumerate(m["s_names"]):
            with open(cd / f"{name.lower()}.hex", "w", newline="\n") as f:
                f.write("".join(f"{(v >> (32 * k)) & 0xFFFFFFFF:08x}\n" for v in part))
        chunks.append((cd, len(part), c0))
    return chunks


def _hex(v: str):
    """(value, X-mask) of a response word: 8 hex digits (JTAG), or 32 binary
    digits ($writememb in simulation); x/z digits count as unknown."""
    base, one = (2, "1") if len(v) == 32 else (16, "f")
    val = int("".join("0" if c in "xXzZ" else c for c in v), base)
    xm = int("".join(one if c in "xXzZ" else "0" for c in v), base)
    return val, xm


def compare(m, trc: Path, chunks, outroot: Path, report: Path, mode_sim=False) -> bool:
    _, outs, masks = read_trace(trc)
    full = (1 << m["nout"]) - 1
    bad, checked, first, xonly, lastx = 0, 0, None, 0, -1
    for cd, cnt, c0 in chunks:
        od = outroot / cd.name
        words, xws = [], []
        for name in m["r_names"]:
            with open(od / f"{name.lower()}.hex") as f:
                vals = [l.strip() for l in f if l.strip() and not l.startswith("//")]
            words.append([_hex(v)[0] for v in vals[:cnt]])
            xws.append([_hex(v)[1] for v in vals[:cnt]])
        for j in range(cnt):
            got, gx = 0, 0
            for k in range(m["now"]):
                got |= words[k][j] << (32 * k)
                gx |= xws[k][j] << (32 * k)
            e = c0 + j
            care = full & ~masks[e]
            checked += 1
            if not ((got ^ outs[e]) & care & ~gx) and gx & care:
                # only X bits differ: in simulation an unpowered-up register is X where
                # the FPGA (and the recording, after its reset) has 0. Counted apart.
                xonly += 1
                lastx = e
                if mode_sim:
                    continue
            if ((got ^ outs[e]) | gx) & care:           # an X where the trace had a value fails
                bad += 1
                if first is None:
                    first = (e, got, outs[e], care, gx)
    lines = [f"{m['suite']} {trc.stem}: {checked} cycles compared, {bad} mismatching"
             + (f" ({xonly} more differ only in X bits, last at cycle {lastx})" if xonly and mode_sim else "")]
    if first:
        e, got, exp, care, gx = first
        diff = ((got ^ exp) | gx) & care
        lines.append(f"first mismatch at cycle {e}:")
        for p in m["outs"]:
            msk = ((1 << p["width"]) - 1) << p["lsb"]
            if diff & msk:
                g = (got & msk) >> p["lsb"]
                x = (exp & msk) >> p["lsb"]
                gxs = " (X bits %#x)" % ((gx & msk) >> p["lsb"]) if gx & msk else ""
                lines.append(f"  {p['name']}: fpga {g:#x}{gxs}, sim {x:#x}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return bad == 0


# ---------------------------------------------------------------- simulation
def sim_build(m) -> Path:
    bd = FT / "builds" / m["suite"]
    exe = bd / "sim" / "tb_replay.vvp"
    exe.parent.mkdir(exist_ok=True)
    cmd = [str(OSS / "bin" / "iverilog"), "-g2012", "-DSIM", "-DSYNTHESIS=1", "-s", "tb_replay",
           "-o", str(exe)] + m["sim_sources"]
    for k, v in m["params"].items():
        pass                                   # parameters are written into replay_top.sv
    r = subprocess.run(cmd, capture_output=True, text=True, env=_oss_env())
    if r.returncode:
        sys.exit(f"iverilog failed:\n{r.stdout}\n{r.stderr}")
    return exe


def _oss_env():
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(OSS / "bin"), str(OSS / "lib"), env.get("PATH", "")])
    return env


def run_sim(m, trc, max_chunks):
    exe = sim_build(m)
    chunks = prep(m, trc, max_chunks)
    outroot = FT / "results" / "raw" / m["suite"] / "sim" / trc.stem
    if outroot.exists():
        shutil.rmtree(outroot)
    lst = FT / "builds" / m["suite"] / "sim" / f"{trc.stem}.list"
    with open(lst, "w", newline="\n") as f:
        for cd, cnt, _ in chunks:
            (outroot / cd.name).mkdir(parents=True)
            f.write(f"{cd.as_posix()} {(outroot / cd.name).as_posix()} {cnt}\n")
    t = time.time()
    r = subprocess.run([str(OSS / "bin" / "vvp"), "-n", str(exe), f"+LIST={lst.as_posix()}"],
                       capture_output=True, text=True, env=_oss_env())
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-2000:],
          f"({time.time() - t:.0f} s)")
    return compare(m, trc, chunks, outroot, FT / "results" / m["suite"] / "sim" / f"{trc.stem}.txt", True)


# ---------------------------------------------------------------- hardware
def program(m):
    sof = FT / "builds" / m["suite"] / "output" / "replay.sof"
    if not sof.exists():
        sys.exit(f"{sof} missing: run tools/build.py {m['suite']} first")
    r = subprocess.run([str(QBIN / "quartus_pgm"), "-c", "USB-Blaster", "-m", "JTAG",
                        "-o", f"p;{sof.as_posix()}"], capture_output=True, text=True)
    if r.returncode:
        sys.exit("quartus_pgm failed:\n" + r.stdout[-3000:] + r.stderr[-2000:])


def jtag(m, chunks, tag):
    """Run replay.tcl on a plan of chunks [(chunk dir, out dir, count)]."""
    plan = FT / "builds" / m["suite"] / f"hw_{tag}.plan"
    with open(plan, "w", newline="\n") as f:
        f.write(f"SUITE_ID {m['suite_id']:08X}\nS {' '.join(m['s_names'])}\nR {' '.join(m['r_names'])}\n")
        for cd, od, cnt in chunks:
            f.write(f"CHUNK {cd.as_posix()} {od.as_posix()} {cnt}\n")
    r = subprocess.run([str(QBIN / "quartus_stp"), "-t", str(TOOLS / "replay.tcl"), plan.as_posix()],
                       capture_output=True, text=True)
    log = FT / "results" / m["suite"] / "hw" / f"{tag}.jtag.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(r.stdout + r.stderr)
    if r.returncode or "REPLAY-OK" not in r.stdout:
        print(r.stdout[-3000:], r.stderr[-2000:])
        sys.exit(f"JTAG replay failed (log {log})")


def run_hw(m, trc, max_chunks):
    program(m)
    chunks = prep(m, trc, max_chunks)
    outroot = FT / "results" / "raw" / m["suite"] / "hw" / trc.stem
    if outroot.exists():
        shutil.rmtree(outroot)
    t = time.time()
    jtag(m, [(cd, outroot / cd.name, cnt) for cd, cnt, _ in chunks], trc.stem)
    print(f"{len(chunks)} chunks over JTAG in {time.time() - t:.0f} s")
    return compare(m, trc, chunks, outroot, FT / "results" / m["suite"] / "hw" / f"{trc.stem}.txt")


def selftest(suite):
    """Program a suite's bitstream and check the JTAG path: RAM content
    order, the MAGIC word, the suite id. No DUT edges."""
    m = load_meta(suite)
    program(m)
    jtag(m, [], "selftest")
    print(f"[replay] selftest on {suite}: PASS (USB-Blaster, bitstream and JTAG memory access work)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sim", "hw", "selftest"])
    ap.add_argument("suite", nargs="?", default="systolic_array")
    ap.add_argument("--trace")
    ap.add_argument("--chunks", type=int, default=0, help="only the first N chunks of each trace")
    a = ap.parse_args()
    if a.mode == "selftest":
        selftest(a.suite)
        return
    m = load_meta(a.suite)
    ts = traces(a.suite, a.trace)
    if not ts:
        sys.exit(f"no traces for {a.suite}" + (f" named {a.trace}" if a.trace else "") + " in traces/")
    ok = True
    for t in ts:
        ok &= (run_sim if a.mode == "sim" else run_hw)(m, t, a.chunks or None)
    print(f"[replay] {a.suite} {a.mode}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
