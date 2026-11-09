"""Functional coverage of the weight decoder, sampled by test_top's monitor.

Each bin is a situation the decoder must handle; the regression should hit
every one. Counts merge across tests and builds into sim_build/coverage/
(one JSON per build); `python tb/coverage.py` prints the report and writes
tb/coverage_report.txt.

Bins (L lanes, b code bits, CHUNKS = 64 / L chunks per tile):
  nesc=k          a chunk with k escaped lanes, k = 0..L
  lane i code c   lane i decoded code c (c = 0 is an escape), per-matrix base builds
  j=t esc         chunk t of its tile went out with at least one escape
  stall_esc       the decoder waited for escape bytes
  stall_data      the decoder waited for codes / sign-mantissa bytes
  bypass          a raw-BF16 chunk went out
  wrap            base + code passed 255 (8-bit wrap of the window)
  esc exp 0/255   an escaped weight with exponent 0 / 255 (zero/denormal, inf/NaN)
  tile b2b        last chunk of a tile and first of the next on consecutive cycles
  all-esc x2      two chunks in a row with every lane escaped
"""

import json
import os
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "sim_build" / "coverage"


def bins(L, b, tile_base):
    chunks = 64 // L
    names = [f"nesc={k}" for k in range(L + 1)]
    if not tile_base:
        names += [f"lane {i} code {c}" for i in range(L) for c in range(1 << b)]
    names += [f"j={t} esc" for t in range(chunks)]
    names += ["stall_esc", "stall_data", "bypass", "esc exp 0", "esc exp 255", "tile b2b", "all-esc x2"]
    if not tile_base:
        names += ["wrap"]
    return names


class Coverage:
    def __init__(self, L, b, tile_base):
        self.L, self.b, self.tile_base = L, b, tile_base
        self.hits = Counter()
        self.prev_fire = None            # (cycle, j, nesc) of the last chunk
        self.cycle = 0

    def cycle_sample(self, stall_esc, stall_data):
        self.cycle += 1
        if stall_esc:
            self.hits["stall_esc"] += 1
        if stall_data:
            self.hits["stall_data"] += 1

    def chunk(self, w, nesc, j, base, bypass):
        if bypass:
            self.hits["bypass"] += 1
            self.prev_fire = None
            return
        L, width = self.L, (1 << self.b) - 1
        self.hits[f"nesc={nesc}"] += 1
        if nesc:
            self.hits[f"j={j} esc"] += 1
        for i in range(L):
            e = (w >> (16 * i + 7)) & 0xFF
            off = (e - base) & 0xFF
            coded = 1 <= off <= width
            if not self.tile_base:
                self.hits[f"lane {i} code {off if coded else 0}"] += 1
                if coded and base + off > 255:
                    self.hits["wrap"] += 1
            if not coded and e in (0, 255):
                self.hits[f"esc exp {e}"] += 1
        p = self.prev_fire
        if p and p[0] == self.cycle - 1 and p[1] == 64 // L - 1 and j == 0:
            self.hits["tile b2b"] += 1
        if p and p[2] == L and nesc == L:
            self.hits["all-esc x2"] += 1
        self.prev_fire = (self.cycle, j, nesc)

    def save(self, build):
        OUT.mkdir(parents=True, exist_ok=True)
        f = OUT / f"{build}.json"
        old = json.loads(f.read_text()) if f.exists() else {"hits": {}}
        hits = Counter(old["hits"])
        hits.update(self.hits)
        f.write_text(json.dumps({"L": self.L, "b": self.b, "tile_base": self.tile_base,
                                 "hits": dict(hits)}, indent=1))


def report():
    lines = ["Weight-decoder functional coverage (merged over every test of each build)", ""]
    for f in sorted(OUT.glob("*.json")):
        d = json.loads(f.read_text())
        names = bins(d["L"], d["b"], d["tile_base"])
        hit = [n for n in names if d["hits"].get(n, 0) > 0]
        miss = [n for n in names if n not in hit]
        lines.append(f"{f.stem}: L={d['L']} b={d['b']} tile_base={d['tile_base']}: "
                     f"{len(hit)}/{len(names)} bins hit ({100 * len(hit) / len(names):.1f}%)")
        if miss:
            lines.append("   missed: " + ", ".join(miss))
    txt = "\n".join(lines) + "\n"
    (REPO / "tb" / "coverage_report.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    report()
