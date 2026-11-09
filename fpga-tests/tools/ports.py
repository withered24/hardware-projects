"""ports.py: a module's ports (name, direction, width) after parameters, via Yosys.

    python ports.py <repo_dir> <top> <params k=v,...> <src1> <src2> ...

Prints JSON [{name, dir, width}] in declaration order. The FPGA wrapper and
the trace recorder are both generated from this list, so they agree.
Uses yosys-slang, the SystemVerilog front end LibreLane ran (USE_SLANG).
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

YOSYS = os.environ.get("YOSYS", r"C:\Users\rayan\tools\oss-cad-suite\bin\yosys.exe")


def ports(repo: Path, top: str, params: dict, sources: list[str], defines: dict | None = None) -> list[dict]:
    g = " ".join(f"-G {k}={v}" for k, v in params.items())
    d = " ".join(f"-D {k}={v}" for k, v in {"SYNTHESIS": 1, **(defines or {})}.items())
    files = " ".join((Path(repo) / s).as_posix() for s in sources)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "p.json"
        lines = ["plugin -i slang",
                 f"read_slang --top {top} {g} {d} {files}",
                 f"hierarchy -top {top}",
                 "proc",
                 f"write_json {out.as_posix()}"]
        (Path(td) / "s.ys").write_text("\n".join(lines) + "\n")
        r = subprocess.run([YOSYS, "-q", "-s", str(Path(td) / "s.ys")], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"yosys failed for {top}:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
        mods = json.loads(out.read_text())["modules"]
    m = next(v for v in mods.values() if v.get("attributes", {}).get("top"))
    return [{"name": n, "dir": p["direction"], "width": len(p["bits"])} for n, p in m["ports"].items()]


if __name__ == "__main__":
    repo, top, ps, *srcs = sys.argv[1:]
    params = dict(kv.split("=", 1) for kv in ps.split(",") if kv)
    print(json.dumps(ports(Path(repo), top, params, srcs), indent=1))
