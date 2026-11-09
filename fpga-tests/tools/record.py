"""record.py: run a repo's own cocotb regression and log DUT port traces.

    python record.py <suite> [pytest selectors...]

Suites are defined in suites.py. For each simulator build whose name
matches the suite's `builds` pattern, the run gets fpga_rec as an extra
cocotb test module and writes traces/<suite>/<build>.trc. The tests
themselves run unchanged and must still pass; a failing test means the
trace is not a passing regression and record.py exits non-zero.
"""
import json
import os
import re
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent
FT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
from ports import ports  # noqa: E402
from suites import SUITES  # noqa: E402

STARTED: set = set()


def main():
    name, *sel = sys.argv[1:]
    s = SUITES[name]
    repo = Path(s["repo"])
    out = FT / "traces" / name
    out.mkdir(parents=True, exist_ok=True)
    pj = out / "ports.json"
    pj.write_text(json.dumps(ports(repo, s["top"], s.get("params", {}), s["sources"], s.get("defines")), indent=1))

    import cocotb_tools.runner as R
    real_get = R.get_runner

    class Wrap:
        def __init__(self, sim):
            self.r = real_get(sim)
            self.build_name = None

        def build(self, **kw):
            self.build_name = Path(kw["build_dir"]).name
            return self.r.build(**kw)

        def test(self, **kw):
            if not re.fullmatch(s["builds"], self.build_name or ""):
                print(f"[record] {self.build_name}: not this FPGA build's config, running without a trace")
                return self.r.test(**kw)
            trc = out / f"{self.build_name}.trc"
            if trc not in STARTED:              # first run of this build: start a fresh file
                trc.unlink(missing_ok=True)
                STARTED.add(trc)
            mods = kw["test_module"]
            mods = [mods] if isinstance(mods, str) else list(mods)
            env = dict(kw.get("extra_env") or {})
            pp = env.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
            env.update({
                "PYTHONPATH": os.pathsep.join(filter(None, [str(TOOLS), pp])),
                "FPGA_TRACE": str(trc), "FPGA_PORTS": str(pj), "FPGA_DUT": s.get("dut", ""),
                "FPGA_PATCH": s.get("patch", ""), "FPGA_MAX_CYCLES": str(s.get("max_cycles", 0)),
            })
            kw["extra_env"] = env
            kw["test_module"] = ["fpga_rec"] + mods
            print(f"[record] {self.build_name} -> {trc}")
            return self.r.test(**kw)

    R.get_runner = Wrap
    os.environ.update(s.get("env", {}))
    os.environ["SIM_BUILD_ROOT"] = str(FT / "sim_build" / name)
    os.chdir(repo)
    rc = pytest.main(["-x", "-q", "-p", "no:cacheprovider", *(sel or s["tests"])])
    for t in sorted(out.glob("*.trc")):
        n = sum(1 for _ in open(t))
        print(f"[record] {t.name}: {n} cycles")
    sys.exit(rc)


if __name__ == "__main__":
    main()
