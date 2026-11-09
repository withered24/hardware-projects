"""Build the driver into a flat binary for the testbench's memory.

    python sw/driver/build.py      -> sw/driver/build/layer.bin (+ .dump)
"""
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "build"


def build() -> Path:
    OUT.mkdir(exist_ok=True)
    gcc = shutil.which("riscv-none-elf-gcc") or shutil.which("riscv64-unknown-elf-gcc")
    assert gcc, "no RISC-V gcc on PATH (source ../riscv-core/env.sh)"
    g = Path(gcc)
    pre = str(g.with_name(g.stem[: -len("gcc")]))          # .../riscv-none-elf-
    elf, binf = OUT / "layer.elf", OUT / "layer.bin"
    subprocess.run([gcc, "-march=rv32i_zicsr", "-mabi=ilp32", "-O2", "-nostdlib", "-ffreestanding",
                    "-T", str(HERE / "link.ld"), str(HERE / "crt0.S"), str(HERE / "layer.c"), "-o", str(elf)],
                   check=True)
    subprocess.run([pre + "objcopy", "-O", "binary", str(elf), str(binf)], check=True)
    dump = subprocess.run([pre + "objdump", "-d", str(elf)], check=True, capture_output=True, text=True).stdout
    (OUT / "layer.dump").write_text(dump)
    return binf


if __name__ == "__main__":
    b = build()
    print(b, b.stat().st_size, "bytes")
