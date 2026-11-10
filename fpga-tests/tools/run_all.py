"""run_all.py: every suite on the DE2-115, one command.

    python tools/run_all.py              all suites, every trace in full
    python tools/run_all.py --quick      first 2 chunks of every trace (a few minutes)
    python tools/run_all.py gpu_v1 ...   only these suites

Needs the board on USB-Blaster and powered, and the bitstreams from
tools/build.py. Checks the JTAG chain, runs the selftest, then each suite
with replay.py hw (fast suites first, the long random CPU traces last), and
writes results/summary.json and results/summary.html. A suite fails if any
trace has a cycle where the FPGA's outputs differ from the simulator's.
"""
import html
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
FT = TOOLS.parent
QBIN = Path(os.environ.get("QUARTUS_BIN", r"C:\altera_lite\25.1std\quartus\bin64"))
ORDER = ["systolic_array", "accel_axil", "riscv_rvtests", "gpu_v1", "gpu_v2", "ew_v1", "ew_v2", "riscv"]


def jtag_ok() -> bool:
    r = subprocess.run([str(QBIN / "jtagconfig")], capture_output=True, text=True)
    print(r.stdout.strip())
    return "USB-Blaster" in r.stdout and "EP4CE115" in r.stdout.replace("EP3C120/", "")


def main():
    args = sys.argv[1:]
    quick = "--quick" in args
    suites = [a for a in args if not a.startswith("--")] or ORDER
    missing = [s for s in suites if not (FT / "builds" / s / "output" / "replay.sof").exists()]
    if missing:
        sys.exit(f"no bitstream for {', '.join(missing)}: run python tools/build.py {' '.join(missing)}")
    if not jtag_ok():
        sys.exit("jtagconfig doesn't show a USB-Blaster with the EP4CE115: plug the board's USB-Blaster "
                 "port in, power it on, and check Device Manager for the Altera USB-Blaster driver")
    r = subprocess.run([sys.executable, str(TOOLS / "replay.py"), "selftest", suites[0]])
    if r.returncode:
        sys.exit("selftest failed: JTAG memory access to the replay bitstream doesn't work (see above)")

    rows = []
    for s in suites:
        t = time.time()
        cmd = [sys.executable, str(TOOLS / "replay.py"), "hw", s] + (["--chunks", "2"] if quick else [])
        print(f"\n=== {s} {'(quick)' if quick else ''}", flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        print(p.stdout[-4000:], p.stderr[-2000:])
        traces = []
        for f in sorted((FT / "results" / s / "hw").glob("*.txt")):
            traces.append(f.read_text().strip())
        rows.append(dict(suite=s, ok=p.returncode == 0, seconds=round(time.time() - t), quick=quick,
                         traces=traces, error="" if p.returncode == 0 else (p.stdout + p.stderr)[-1500:]))
        (FT / "results").mkdir(exist_ok=True)
        (FT / "results" / "summary.json").write_text(json.dumps(rows, indent=1))
        write_html(rows)

    print("\n=== summary")
    for r in rows:
        print(f"  {r['suite']:16s} {'PASS' if r['ok'] else 'FAIL'}  {r['seconds']:6d} s")
    print(f"results/summary.html; {sum(r['ok'] for r in rows)}/{len(rows)} suites pass")
    sys.exit(0 if all(r["ok"] for r in rows) else 1)


def write_html(rows):
    body = []
    for r in rows:
        tr = "<br>".join(html.escape(t.splitlines()[0]) for t in r["traces"]) or "&nbsp;"
        det = "".join(f"<pre>{html.escape(t)}</pre>" for t in r["traces"] if "mismatching" in t and " 0 mismatching" not in t)
        if r["error"]:
            det += f"<pre>{html.escape(r['error'])}</pre>"
        body.append(f"<tr><td>{r['suite']}</td><td class={'ok' if r['ok'] else 'bad'}>{'PASS' if r['ok'] else 'FAIL'}"
                    f"</td><td>{r['seconds']} s</td><td>{tr}{det}</td></tr>")
    page = f"""<!doctype html><meta charset="utf-8"><title>DE2-115 replay results</title>
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#d2d2d7;--ok:#1a7f37;--bad:#c62828}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--mut:#999;--line:#333;--ok:#4ac26b;--bad:#ff6b6b}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}}
.ok{{color:var(--ok);font-weight:600}}.bad{{color:var(--bad);font-weight:600}}pre{{white-space:pre-wrap;color:var(--mut);font-size:13px}}
</style>
<h1>DE2-115 replay results</h1>
<p>Each suite's recorded simulation inputs were replayed into the design on the FPGA one clock edge at a time,
and the FPGA's outputs compared with the simulator's at every edge.{' Quick run: first 2 chunks of each trace.' if rows and rows[0]['quick'] else ''}
Generated {time.strftime('%Y-%m-%d %H:%M')}.</p>
<table><tr><th>Suite</th><th>Result</th><th>Time</th><th>Traces</th></tr>{''.join(body)}</table>"""
    (FT / "results" / "summary.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
