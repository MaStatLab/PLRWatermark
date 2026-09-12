"""Assert exact node recovery and plate membership for all three model figures.

The six panels distinguish document-level state S and hyperparameters from
token-specific quantities.  Clean, contaminated, and partially pooled models
have separate declared node inventories; deleting a node must fail as surely
as moving it to the wrong plate.

Rather than trust a visual reading, this compiles the figure with a
\\pgfpointanchor probe appended to each panel and checks every node's
bounding box against the plate rectangle TikZ actually emitted.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

TEX = Path(__file__).resolve().parents[1] / "manuscript" / "bayesian_pivot_watermark_gumbel.tex"

# panel label -> (plate node, nodes required INSIDE, nodes required OUTSIDE).
# Panel labels are global: they tag the \typeout probe lines.
# The contaminated panels moved to the supplement when the contamination
# development did; Figure 1 now carries the two clean hierarchies only.
HIERARCHY = {
    # (a) clean shared: one (Delta, alpha, J) for the whole document.
    "A": ("plateU", {"q", "p", "y"},
          {"b", "j", "alpha", "delta", "pis", "pij", "pia", "pid"}),
    # (b) clean tokenwise: the same three latents move INSIDE the plate while
    # their priors stay outside it.  That move is the entire difference
    # between the two hierarchies, so a panel drawn the other way round would
    # misstate the comparison Table 2 reports.
    "B": ("plateUT", {"j", "alpha", "delta", "q", "p", "y"},
          {"b", "pis", "pij", "pia", "pid"}),
}

# The same two hierarchies with the contamination layer, now a supplementary
# figure.  rho is document level in (a) and position specific in (b); C_t is
# per token in both.
CONTAMINATION = {
    "C": ("plateS", {"q", "p", "y", "c"},
          {"b", "j", "alpha", "delta", "rho",
           "pis", "pij", "pia", "pid", "pir"}),
    "D": ("plateT", {"j", "alpha", "delta", "rho", "q", "p", "y", "c"},
          {"b", "pis", "pij", "pia", "pid", "pir"}),
}

# The partial-pooling layers.  Each panel's claim is that the plate holds
# exactly what the layer redraws and the hyperparameter stays outside it; a
# figure that put psi or phi inside the plate would draw a different model
# from the one the code implements.
POOLING = {
    # (a) pooled deficit: Figure 1(a) with the Delta column modified.  The
    # hyperparameter psi stays document level and Delta_t moves inside.
    "E": ("platePD", {"q", "delta", "p", "y"},
          {"b", "j", "alpha", "psi", "pis", "pij", "pia", "pipsi"}),
    # (b) pooled tail width: the same move on the J column.  Delta must stay
    # OUTSIDE -- that is the whole difference from the tokenwise union rule,
    # which redraws the deficit too.
    "F": ("platePW", {"j", "q", "p", "y"},
          {"b", "phi", "alpha", "delta", "pis", "piphi", "pia", "pid"}),
}

FIGURES = (("fig:hierarchical-model", HIERARCHY),
           ("fig:contamination-model", CONTAMINATION),
           ("fig:pooling-model", POOLING))
EXPECTED = {**HIERARCHY, **CONTAMINATION, **POOLING}
PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{amsmath,mathtools}
\usepackage{bm}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,backgrounds}
\usepackage{xcolor}
\newcommand{\bP}{\bm p}
\newcommand{\Prb}{\mathbb P}
\newcommand{\ind}{\mathbf 1}
"""
PROBE = (r"\pgfextra{\pgfpointanchor{%s}{south west}"
         r"\edef\gxa{\the\pgf@x}\edef\gya{\the\pgf@y}"
         r"\pgfpointanchor{%s}{north east}"
         r"\typeout{GEO %s %s SW \gxa\space \gya\space NE "
         r"\the\pgf@x\space \the\pgf@y}}")
LINE = re.compile(r"^GEO (\w+) (\w+) SW ([-\d.]+)pt ([-\d.]+)pt "
                  r"NE ([-\d.]+)pt ([-\d.]+)pt")


def panels(source: str, label: str, want: int) -> list[str]:
    """The tikzpicture bodies of one figure, in order."""
    end = source.index("\\label{" + label + "}")
    start = source.rindex(r"\begin{figure}", 0, end)
    found = re.findall(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
                       source[start:end], re.S)
    if len(found) != want:
        raise SystemExit(
            f"expected {want} tikzpictures in {label}, found {len(found)}"
        )
    return found


def instrument(body: str, label: str, plate: str) -> str:
    styles = {"latentnode", "obsnode", "detnode", "priornode"}
    names = [name for options, name in re.findall(
        r"\\node\[([^\]]*)\]\s*\((\w+)\)", body)
        if styles.intersection(option.strip() for option in options.split(","))]
    if len(names) != len(set(names)):
        raise ValueError(f"panel {label}: duplicate node names")
    block = "  \\path " + "".join(
        PROBE % (n, n, label, n) + "%\n  " for n in names + [plate]) + ";\n"
    return body.replace(r"\end{tikzpicture}", block + r"\end{tikzpicture}")


def parse_geometry(log: str) -> dict[str, dict[str, tuple[float, ...]]]:
    boxes: dict[str, dict[str, tuple[float, ...]]] = {}
    for line in log.splitlines():
        m = LINE.match(line.strip())
        if m:
            if m.group(2) in boxes.get(m.group(1), {}):
                raise ValueError(f"duplicate geometry for panel {m.group(1)} node {m.group(2)}")
            boxes.setdefault(m.group(1), {})[m.group(2)] = tuple(
                float(v) for v in m.groups()[2:])
    return boxes


def validate_geometry(boxes: dict[str, dict[str, tuple[float, ...]]]) -> list[str]:
    failures: list[str] = []
    for label in boxes.keys() - EXPECTED.keys():
        failures.append(f"unexpected panel {label}")
    for label, (plate, want_in, want_out) in EXPECTED.items():
        nodes = boxes.get(label)
        if not nodes or plate not in nodes:
            failures.append(f"panel {label}: no geometry recovered")
            continue
        expected = want_in | want_out | {plate}
        missing = expected - nodes.keys()
        extra = nodes.keys() - expected
        if missing:
            failures.append(f"panel {label}: missing nodes {sorted(missing)}")
        if extra:
            failures.append(f"panel {label}: unexpected nodes {sorted(extra)}")
        px0, py0, px1, py1 = nodes[plate]
        for name, (x0, y0, x1, y1) in nodes.items():
            if name == plate:
                continue
            inside = x0 >= px0 and x1 <= px1 and y0 >= py0 and y1 <= py1
            overlaps = not (x1 < px0 or x0 > px1 or y1 < py0 or y0 > py1)
            if name in want_in and not inside:
                failures.append(f"panel {label}: {name} must sit INSIDE {plate}")
            elif name in want_out and overlaps:
                failures.append(f"panel {label}: {name} must sit OUTSIDE {plate}")
    return failures


def compile_geometry(document: str) -> str:
    with tempfile.TemporaryDirectory(prefix="figgeo-") as directory:
        work = Path(directory)
        (work / "geo.tex").write_text(document, encoding="utf-8")
        result = subprocess.run(
            ["pdflatex", "-halt-on-error", "-interaction=nonstopmode", "geo.tex"],
            cwd=work, capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"pdflatex failed with exit {result.returncode}:\n"
                + (result.stdout + result.stderr)[-3000:]
            )
        return (work / "geo.log").read_text(encoding="utf-8", errors="replace")


def main() -> int:
    try:
        source = TEX.read_text(encoding="utf-8")
        styles = source[source.index(r"\definecolor{wmnavy}"):
                        source.index(r"\hypersetup{")]
        doc = [PREAMBLE, styles, r"\makeatletter", r"\begin{document}"]
        for figure, spec in FIGURES:
            for body, (label, (plate, _, _)) in zip(
                panels(source, figure, len(spec)), spec.items()
            ):
                doc.append(instrument(body, label, plate))
        doc += [r"\makeatother", r"\end{document}"]
        boxes = parse_geometry(compile_geometry("\n".join(doc)))
        failures = validate_geometry(boxes)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"figure geometry check FAILED: {exc}")
        return 1

    if failures:
        print("figure geometry check FAILED:")
        for f in failures:
            print("  " + f)
        return 1
    total = sum(len(v) - 1 for v in boxes.values())
    print(f"figure geometry check passed: {total} nodes in the expected plates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
