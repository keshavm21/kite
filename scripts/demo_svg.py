#!/usr/bin/env python3
"""Renders docs/demo.svg: real kitec output replayed in an animated terminal window, for the README.

    python3 scripts/demo_svg.py [--kitec build/kitec] [-o docs/demo.svg] [--static]

The animation is pure CSS inside the SVG (no recorder, no JavaScript): lines appear one after another and the
loop restarts. Viewers that disable animation (prefers-reduced-motion) or --static show the finished frame.

Every line in the image is produced by running the commands below, so regenerate it whenever the output
changes. Standard library only.
"""
import argparse
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
WIDTH_CHARS = 84  # longer lines are cut with an ellipsis

# (command shown, kitec arguments, keep only lines from the first match of this text (None = all), max lines)
DEMO = [
    ("kitec examples/row_sum.kite -O1 --emit=ir", ["examples/row_sum.kite", "-O1", "--emit=ir"], None, 40),
    ("kitec run examples/row_sum.kite -O0", ["run", "examples/row_sum.kite", "-O0"], None, 1),
    ("kitec run examples/row_sum.kite -O1", ["run", "examples/row_sum.kite", "-O1"], None, 1),
]
# Lines containing this text are highlighted, with a note.
HIGHLIGHT = ("%14 = mul i32 %8, %cols.1", "   ; hoisted out of the loop by licm")

BG, FG, DIM, PROMPT, ACCENT, TITLE = "#0d1117", "#c9d1d9", "#8b949e", "#3fb950", "#f2cc60", "#161b22"
CHAR_W, LINE_H, PAD = 9.0, 18, 16  # generous: monospace fonts differ in width


def lines_for(kitec: str):
    out = []
    for shown, args, start, limit in DEMO:
        out.append(("prompt", f"$ {shown}"))
        text = subprocess.run([kitec, *args], cwd=ROOT, capture_output=True, text=True, check=True,
                              stdin=subprocess.DEVNULL).stdout.splitlines()
        if start:
            text = text[next(i for i, l in enumerate(text) if start in l):]
        for l in text[:limit]:
            kind = "out"
            if HIGHLIGHT[0] in l:
                l, kind = l + HIGHLIGHT[1], "hl"
            out.append((kind, l if len(l) <= WIDTH_CHARS else l[: WIDTH_CHARS - 1] + "…"))
        out.append(("out", ""))
    return out[:-1]


def schedule(lines):
    """Appearance time of every line in seconds, and the loop length."""
    t, times = 0.6, []
    for i, (kind, _) in enumerate(lines):
        if kind == "prompt":
            if i:
                t += 2.4  # let the previous output be read
            times.append(t)
            t += 0.9
        else:
            times.append(t)
            t += 0.07
    return times, t + 4.0


def render(lines, animate=True) -> str:
    w = int(PAD * 2 + CHAR_W * WIDTH_CHARS)
    top = 34
    h = top + PAD + LINE_H * len(lines) + PAD
    color = {"prompt": PROMPT, "out": FG, "hl": ACCENT}
    times, total = schedule(lines)
    css = []
    if animate:
        css.append("@media (prefers-reduced-motion: no-preference) {")
        for i, at in enumerate(times):
            p = 100 * at / total
            css.append(f".a{i}{{opacity:0;animation:k{i} {total:.2f}s linear infinite}}"
                       f"@keyframes k{i}{{0%,{p:.3f}%{{opacity:0}}{p + 0.01:.3f}%,96%{{opacity:1}}100%{{opacity:0}}}}")
        css.append("}")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="kitec output for row_sum: optimized IR with the hoisted multiply, and identical -O0 and -O1 results">',
        f"<style>{''.join(css)}</style>",
        f'<rect width="{w}" height="{h}" rx="8" fill="{BG}"/>',
        f'<rect width="{w}" height="{top - 6}" rx="8" fill="{TITLE}"/><rect y="{top - 14}" width="{w}" height="8" fill="{TITLE}"/>',
        *(f'<circle cx="{18 + 18 * i}" cy="14" r="6" fill="{c}"/>' for i, c in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"])),
        f'<text x="{w / 2}" y="18" fill="{DIM}" font-family="monospace" font-size="12" text-anchor="middle">kitec</text>',
        '<g font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="14" xml:space="preserve">',
    ]
    for i, (kind, text) in enumerate(lines):
        y = top + PAD + LINE_H * (i + 1) - 4
        cls = f' class="a{i}"' if animate else ""
        parts.append(f"<g{cls}>")
        if kind == "hl":
            parts.append(f'<rect x="{PAD - 4}" y="{y - 13}" width="{w - 2 * PAD + 8}" height="{LINE_H}" fill="{ACCENT}" fill-opacity="0.12"/>')
        # non-breaking spaces: renderers collapse ordinary runs of spaces, which would lose the IR's indentation
        parts.append(f'<text x="{PAD}" y="{y}" fill="{color[kind]}">{escape(text).replace(" ", "&#160;")}</text></g>')
    parts += ["</g>", "</svg>", ""]
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitec", default=str(ROOT / "build" / "kitec"))
    ap.add_argument("-o", "--out", default=str(ROOT / "docs" / "demo.svg"))
    ap.add_argument("--static", action="store_true", help="no animation: the finished frame only")
    opts = ap.parse_args()
    Path(opts.out).write_text(render(lines_for(opts.kitec), animate=not opts.static))
    print(f"wrote {opts.out}")


if __name__ == "__main__":
    main()
