"""qprobe.py: Quartus Analysis & Synthesis of a suite's DUT on its own.

    python qprobe.py <suite> [--patched]

Makes builds/probe_<suite>/ with the DUT as top level (every port a virtual
pin, so the DE2-115's pin count doesn't matter) and runs quartus_map. It
shows what Quartus rejects and roughly what the DUT costs (LEs, registers,
memory bits) before any harness is built around it.
"""
import os
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from suites import SUITES, sources_for  # noqa: E402

QBIN = Path(os.environ.get("QUARTUS_BIN", r"C:\altera_lite\25.1std\quartus\bin64"))
DEVICE = "EP4CE115F29C7"


def qsf_sources(files: list[Path]) -> str:
    out = []
    for f in files:
        kind = "SYSTEMVERILOG_FILE" if f.suffix == ".sv" else "VERILOG_FILE"
        out.append(f'set_global_assignment -name {kind} "{f.as_posix()}"')
    return "\n".join(out)


def probe(name: str, patched: bool) -> int:
    s = SUITES[name]
    bd = TOOLS.parent / "builds" / f"probe_{name}{'_p' if patched else ''}"
    bd.mkdir(parents=True, exist_ok=True)
    from qcompat import convert
    src = bd / "src"
    src.mkdir(exist_ok=True)
    files = []
    for f in sources_for(name, patched):             # Quartus-compatible copies
        out = src / f.name
        out.write_text(convert(f.read_text()), newline=chr(10))
        files.append(out)
    files.insert(0, TOOLS.parent / "fpga" / "rtl" / "qcompat_pkg.sv")
    params = "\n".join(f'set_parameter -name {k} "{v}"' for k, v in s.get("params", {}).items())
    defines = {"SYNTHESIS": 1, **s.get("defines", {})}
    macros = "\n".join(f'set_global_assignment -name VERILOG_MACRO "{k}={v}"' for k, v in defines.items())
    (bd / "probe.qpf").write_text('PROJECT_REVISION = "probe"\n')
    (bd / "probe.qsf").write_text(f"""set_global_assignment -name FAMILY "Cyclone IV E"
set_global_assignment -name DEVICE {DEVICE}
set_global_assignment -name TOP_LEVEL_ENTITY {s['top']}
set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output
{macros}
{params}
{qsf_sources(files)}
set_instance_assignment -name VIRTUAL_PIN ON -to *
""")
    r = subprocess.run([str(QBIN / "quartus_map"), "probe", "-c", "probe"], cwd=bd,
                       capture_output=True, text=True)
    (bd / "map.log").write_text(r.stdout + r.stderr)
    errs = [l for l in r.stdout.splitlines() if l.startswith("Error")]
    print(f"[qprobe] {name}: quartus_map exit {r.returncode}, {len(errs)} errors")
    for l in errs[:25]:
        print("   ", l[:300])
    summ = bd / "output" / "probe.map.summary"
    if summ.exists():
        print(summ.read_text())
    return r.returncode


if __name__ == "__main__":
    sys.exit(probe(sys.argv[1], "--patched" in sys.argv))
