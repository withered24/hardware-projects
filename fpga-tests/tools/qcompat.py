"""qcompat.py: rewrite legal SystemVerilog that Quartus Standard (Lite) rejects.

    python qcompat.py <in.sv> <out.sv>

Quartus Prime Standard 25.1's parser does not accept two IEEE 1800 forms
the RTL uses:
  * generate constructs (for / if / case) directly at module level, without
    a generate ... endgenerate region around them
  * a genvar declared inside the loop header: for (genvar i = 0; ...)

The rewrite is purely syntactic: each module-level generate construct gets
`generate` / `endgenerate` around it, and every `for (genvar x` becomes
`for (x` with `genvar x;` declared once at the top of the module. $countones(x) becomes
qcompat_pkg::qc_popcount(x) (fpga/rtl/qcompat_pkg.sv). Nothing else changes, and line numbers stay the same (insertions go on existing
lines), so Quartus messages still point at the original source lines.
"""
import re
import sys

PROC = {"always", "always_ff", "always_comb", "always_latch", "initial", "final"}
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*|`[A-Za-z_]\w*|\d+'[sS]?[bBoOdDhH][0-9a-fA-FxXzZ_?]+|'\{|.",
                   re.S)


def mask_comments(src: str) -> str:
    """Same length as src, comments and strings blanked (newlines kept)."""
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        elif src[i] == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            for k in range(i + 1, min(j, n)):
                out[k] = " "
            i = j + 1
        else:
            i += 1
    return "".join(out)


def tokens(masked: str):
    return [(m.group(0), m.start()) for m in TOKEN.finditer(masked) if not m.group(0).isspace()]


class Rewriter:
    def __init__(self, src: str):
        self.src = src
        self.toks = tokens(mask_comments(src))
        self.ins = []          # (pos, text) insertions
        self.dels = []         # (start, end) deletions

    # ---- helpers on the token stream
    def skip_parens(self, i):
        """toks[i] is '('; return index after the matching ')'."""
        d = 0
        while i < len(self.toks):
            t = self.toks[i][0]
            if t == "(":
                d += 1
            elif t == ")":
                d -= 1
                if d == 0:
                    return i + 1
            i += 1
        return i

    def skip_statement(self, i):
        """Skip one statement / generate item starting at toks[i]; return the
        index after it (including any else branches)."""
        t = self.toks[i][0]
        if t == "begin":
            i = self.skip_block(i, "begin", "end")
            return self.after_label(i)
        if t in ("case", "casez", "casex"):
            return self.skip_block(i, None, "endcase")
        if t in ("unique", "priority", "unique0"):
            return self.skip_statement(i + 1)
        if t in ("if", "for", "while", "foreach", "repeat"):
            i = self.skip_parens(i + 1)
            i = self.skip_statement(i)
            if t == "if" and i < len(self.toks) and self.toks[i][0] == "else":
                i = self.skip_statement(i + 1)
            return i
        if t == "forever":
            return self.skip_statement(i + 1)
        if t == "@":
            i += 1
            if self.toks[i][0] == "(":
                i = self.skip_parens(i)
            else:
                i += 1
            return self.skip_statement(i)
        if t == "#":
            i += 1
            if self.toks[i][0] == "(":
                i = self.skip_parens(i)
            else:
                i += 1
            return self.skip_statement(i)
        if t == "fork":
            return self.skip_block(i, "fork", ("join", "join_any", "join_none"))
        # simple statement: up to ';' (skip nested parens / braces)
        d = 0
        while i < len(self.toks):
            t = self.toks[i][0]
            if t in ("(", "{", "'{", "["):
                d += 1
            elif t in (")", "}", "]"):
                d -= 1
            elif t == ";" and d == 0:
                return i + 1
            i += 1
        return i

    def after_label(self, i):
        if i < len(self.toks) and self.toks[i][0] == ":" :
            return i + 2
        return i

    def skip_block(self, i, open_kw, close_kw):
        closes = close_kw if isinstance(close_kw, tuple) else (close_kw,)
        opens = {"begin": "end", "case": "endcase", "casez": "endcase", "casex": "endcase",
                 "fork": "join", "generate": "endgenerate", "function": "endfunction",
                 "task": "endtask"}
        stack = []
        while i < len(self.toks):
            t = self.toks[i][0]
            if t in opens:
                stack.append(opens[t])
            elif t in ("end", "endcase", "join", "join_any", "join_none", "endgenerate",
                       "endfunction", "endtask"):
                if stack:
                    stack.pop()
                if not stack:
                    return i + 1
            i += 1
        return i

    # ---- module-level walk
    def run(self) -> str:
        i = 0
        while i < len(self.toks):
            t, pos = self.toks[i]
            if t in ("module", "interface"):
                i = self.module(i)
            else:
                i += 1
        return self.apply()

    def module(self, i):
        # header ends at the first ';' outside parentheses
        j = i + 1
        while j < len(self.toks) and self.toks[j][0] != ";":
            j = self.skip_parens(j) if self.toks[j][0] == "(" else j + 1
        header_end = self.toks[j][1] + 1
        genvars = []
        j += 1
        j = self.items(j, ("endmodule", "endinterface"), genvars, top=True)
        if genvars:
            self.ins.append((header_end, " genvar " + ", ".join(dict.fromkeys(genvars)) + ";"))
        return j + 1

    def items(self, i, ends, genvars, top):
        """Module items until a token in `ends`; wrap generate constructs if top."""
        while i < len(self.toks):
            t, pos = self.toks[i]
            if t in ends:
                return i
            if t in PROC:
                i = self.skip_statement(i + 1)
            elif t in ("function", "task"):
                i = self.skip_block(i, t, "end" + t)
            elif t == "generate":
                end = self.skip_block(i, "generate", "endgenerate")
                self.collect_genvars(i, end, genvars)
                i = end
            elif t in ("for", "if", "case"):
                end = self.skip_statement(i)
                self.collect_genvars(i, end, genvars)
                if top:
                    self.ins.append((pos, "generate "))
                    last = self.toks[end - 1]
                    self.ins.append((last[1] + len(last[0]), " endgenerate"))
                i = end
            elif t == "begin":                   # stray block (shouldn't happen at module level)
                i = self.skip_block(i, "begin", "end")
            else:
                i += 1
        return i

    def collect_genvars(self, a, b, genvars):
        for k in range(a, b - 2):
            if self.toks[k][0] == "for" and self.toks[k + 1][0] == "(" and self.toks[k + 2][0] == "genvar":
                gpos = self.toks[k + 2][1]
                self.dels.append((gpos, gpos + len("genvar")))
                genvars.append(self.toks[k + 3][0])

    def apply(self) -> str:
        edits = [(p, p, s) for p, s in self.ins] + [(a, b, "") for a, b in self.dels]
        out = self.src
        for a, b, s in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
            out = out[:a] + s + out[b:]
        return out


def convert(src: str) -> str:
    out = Rewriter(src).run()
    # $countones is not synthesizable in Quartus Standard (fpga/rtl/qcompat_pkg.sv)
    return out.replace("$countones(", "qcompat_pkg::qc_popcount(")


if __name__ == "__main__":
    a, b = sys.argv[1:3]
    open(b, "w", newline="\n").write(convert(open(a).read()))
