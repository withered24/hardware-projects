"""build.py: compile replay bitstreams with Quartus and summarise fit and timing.

    python build.py [suite ...]          all suites when none given
    python build.py --report [suite ...] only re-read the reports

For each suite: tools/gen.py's project in builds/<suite>/ is compiled with
`quartus_sh --flow compile replay` (log: builds/<suite>/compile.log), giving
builds/<suite>/output/replay.sof. The summary (also written to
builds/<suite>/build.json and printed) has the resources used and the worst
setup/hold slack per clock over all timing corners. Timing must close (no
negative slack) for the bitstream to be trusted. (The DUT runs at 50/8 =
6.25 MHz of edges here; the slack says nothing direct about its own fmax.)
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
FT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
from suites import SUITES  # noqa: E402
import gen  # noqa: E402

QBIN = Path(os.environ.get("QUARTUS_BIN", r"C:\altera_lite\25.1std\quartus\bin64"))
TIMEOUT_S = int(os.environ.get("BUILD_TIMEOUT_S", "7200"))


def fit_summary(out: Path) -> dict:
    res = {}
    f = out / "replay.fit.summary"
    if not f.exists():
        return res
    for line in f.read_text(errors="replace").splitlines():
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if k in ("Fitter Status", "Total logic elements", "Total registers", "Total pins",
                 "Total memory bits", "Embedded Multiplier 9-bit elements", "Total PLLs"):
            res[k] = v
    return res


def sta_summary(out: Path) -> dict:
    """Worst slack per (analysis, clock) over all corners from replay.sta.summary."""
    f = out / "replay.sta.summary"
    worst = {}
    if not f.exists():
        return worst
    typ = None
    for line in f.read_text(errors="replace").splitlines():
        m = re.match(r"Type\s*:\s*(.*)", line)
        if m:
            typ = m.group(1)
            continue
        m = re.match(r"Slack\s*:\s*(-?[\d.]+)", line)
        if m and typ:
            t = re.search(r"\b(Setup|Hold|Recovery|Removal|Minimum Pulse Width)\b\s*'([^']+)'", typ)
            if t:
                key = f"{t.group(1).lower()} {t.group(2)}"
                worst[key] = min(worst.get(key, 1e9), float(m.group(1)))
            typ = None
    return worst


def report(suite: str, seconds=None) -> dict:
    bd = FT / "builds" / suite
    out = bd / "output"
    fit, sta = fit_summary(out), sta_summary(out)
    sof = out / "replay.sof"
    timing_ok = bool(sta) and all(v >= 0 for v in sta.values())
    r = dict(suite=suite, sof=sof.exists(), timing_ok=timing_ok, fit=fit, slack=sta,
             compile_s=seconds)
    (bd / "build.json").write_text(json.dumps(r, indent=1))
    status = "OK" if r["sof"] and timing_ok else ("TIMING FAIL" if r["sof"] else "NO BITSTREAM")
    print(f"[build] {suite}: {status}; LEs {fit.get('Total logic elements', '?')}, "
          f"memory bits {fit.get('Total memory bits', '?')}, mults {fit.get('Embedded Multiplier 9-bit elements', '?')}"
          + (f", {seconds:.0f} s" if seconds else ""))
    for k, v in sorted(sta.items()):
        print(f"          {k:28s} worst slack {v:9.3f} ns" + ("   <-- NEGATIVE" if v < 0 else ""))
    return r


def build(suite: str) -> dict:
    gen.generate(suite)
    bd = FT / "builds" / suite
    t = time.time()
    with open(bd / "compile.log", "w") as log:
        try:
            p = subprocess.run([str(QBIN / "quartus_sh"), "--flow", "compile", "replay"], cwd=bd,
                               stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT_S)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    secs = time.time() - t
    if rc != 0:
        tail = (bd / "compile.log").read_text(errors="replace").splitlines()
        errs = [l for l in tail if l.startswith("Error")][:15]
        print(f"[build] {suite}: quartus_sh failed ({rc}) after {secs:.0f} s; see {bd / 'compile.log'}")
        print("\n".join(errs or tail[-15:]))
    return report(suite, secs)


def main():
    args = sys.argv[1:]
    only_report = "--report" in args
    names = [a for a in args if not a.startswith("--")] or list(SUITES)
    res = [report(n) if only_report else build(n) for n in names]
    bad = [r["suite"] for r in res if not (r["sof"] and r["timing_ok"])]
    print(f"[build] {len(res) - len(bad)}/{len(res)} bitstreams ready" + (f"; not ready: {', '.join(bad)}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
