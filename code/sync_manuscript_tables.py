#!/usr/bin/env python3
"""Rewrite manuscript table rows from the generated artifacts.

:mod:`check_manuscript_tables` already knows, for every row the manuscript
displays, which artifact cell it must equal.  Rather than restate that mapping
-- and risk the two drifting apart -- this command imports the checker and
replaces its ``assert_row`` with a recorder, so a row can only be rewritten
where the checker would have looked.  Anything the checker does not cover is
left untouched, and a row whose manuscript text already matches is not edited.

One pass records numeric-cell corrections; any unrelated failed assertion is
reported immediately.  The complete candidate is checked before publication.

``--dry-run`` reports cell and emphasis corrections without writing.  A valid
candidate replaces the original file atomically, preserving its permissions.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

import check_manuscript_tables as chk

ROW = re.compile(r"\\(best)?mcse\{(" + chk.CELL_NUMBER + r")\}\{(" + chk.CELL_NUMBER + r")\}")

# Which end of a column the boldface marks.  Every table carrying \bestmcse
# must appear here.  Silently defaulting to "min" is what bolded the worst
# weight in all eight columns of tab:union-weight, whose entries are areas
# under the curve rather than error rates.
BOLD_DIRECTION = {f"tab:{name}": direction
                  for name, direction in chk.BOLD_DIRECTION.items()}


def collect_edits(tex: str) -> list[tuple[int, int, str]]:
    """Return (start, stop, replacement) spans, using the checker's own mapping."""
    edits: list[tuple[int, int, str]] = []
    sources: dict[int, tuple[str, int]] = {}

    original_block = chk.table_block
    original_body = chk.table_body
    original_assert_row = chk.assert_row

    def table_block(source: str, label: str) -> str:
        block = original_block(source, label)
        raw = chk.raw_table_block(tex, label)
        sources[id(block)] = (raw, tex.index(raw))
        return block

    def table_body(source: str, label: str) -> str:
        block = original_block(source, label)
        body = block[block.index(r"\midrule"):]
        raw = chk.raw_table_block(tex, label)
        start = tex.index(raw)
        body_start = raw.index(r"\midrule")
        sources[id(body)] = (raw[body_start:], start + body_start)
        return body

    def assert_row(block, marker, expected, expected_mc_se, *,
                   occurrence=0, continuation=False, signed=False):
        # `continuation` mirrors the checker: a second prior arm that does not
        # repeat its label is reached by stepping whole rows, not by finding
        # the marker again.  Omitting it here made the synchroniser crash on a
        # table the checker was already handling.
        source = sources.get(id(block))
        if source is None:                    # a block we did not intercept
            return
        raw, base = source
        if continuation:
            position = block.find(marker)
            if position < 0:
                raise AssertionError(f"missing table row marker: {marker!r}")
            row_end = block.index(r"\\", position)
            first = block.index("&", position)
            for _ in range(occurrence):
                first = block.index("&", row_end)
                row_end = block.index(r"\\", first)
        else:
            position = -1
            for _ in range(occurrence + 1):
                position = block.find(marker, position + 1)
                if position < 0:
                    raise AssertionError(f"missing table row marker: {marker!r}")
            row_end = block.index(r"\\", position)
            first = block.index("&", position)
        expected, expected_mc_se = list(expected), list(expected_mc_se)
        if len(expected) != len(expected_mc_se):
            raise AssertionError(f"row {marker!r}: estimate/MCSE counts differ")
        wanted = [(chk.formatted(abs(value) if signed else value), chk.formatted(error))
                  for value, error in zip(expected, expected_mc_se)]
        segment = block[first + 1 : row_end]
        found = list(ROW.finditer(segment))
        if len(found) != len(wanted):
            raise AssertionError(
                f"row {marker!r}: manuscript has {len(found)} cells, "
                f"artifacts have {len(wanted)}"
            )
        for index, (match, (value, error)) in enumerate(zip(found, wanted)):
            sign = "-" if expected[index] < 0 else "+"
            previous_sign = (re.search(r"\$([+-])\$\s*$", segment[:match.start()])
                             if signed else None)
            sign_matches = not signed or (previous_sign and previous_sign.group(1) == sign)
            if match.group(2) == value and match.group(3) == error and sign_matches:
                continue
            star = match.group(1) or ""
            token_start = previous_sign.start() if previous_sign else match.start()
            start, stop = chk.source_cell_span(
                block, raw, first + 1 + token_start, first + 1 + match.end()
            )
            edits.append((
                base + start,
                base + stop,
                (f"${sign}$" if signed else "") + f"\\{star}mcse{{{value}}}{{{error}}}",
            ))

    chk.table_block = table_block
    chk.table_body = table_body
    chk.assert_row = assert_row
    # The boldface check is about markers, which renormalise_bold fixes AFTER
    # this collection pass.  Leaving it live deadlocks the tool: it aborts the
    # checker before any cell is recorded, so the markers it would fix never
    # get fixed.
    original_boldface = chk.check_boldface
    chk.check_boldface = lambda tex: 0
    try:
        chk.check_manuscript(tex)
    finally:
        chk.check_boldface = original_boldface
        chk.table_block = original_block
        chk.table_body = original_body
        chk.assert_row = original_assert_row

    # The same source row can be checked by more than one assertion.
    unique: dict[tuple[int, int], str] = {}
    for start, stop, text in edits:
        unique[(start, stop)] = text
    return [(a, b, t) for (a, b), t in sorted(unique.items())]


def bold_panels(lines: list[str]) -> list[list[int]]:
    """Group body line indices into the blocks a reader compares within.

    A panel ends at a rule or at a sub-header (``\\midrule`` or a
    ``\\multicolumn`` line carrying ``\\textit``).  ``\\addlinespace`` used to
    end one too, which is wrong: the captions say "column minima" over a whole
    horizon block, and treating the visual gap as a boundary left a tie in the
    maximum-regret column of tab:regret with only one of its two cells bold.
    """
    panels: list[list[int]] = []
    current: list[int] = []
    for index, line in enumerate(lines):
        boundary = (
            r"\midrule" in line
            or ("\\multicolumn" in line
                and ("\\textit" in line or "\\emph" in line))
        )
        if boundary:
            if current:
                panels.append(current)
            current = []
            continue
        if ROW.search(line):
            current.append(index)
    if current:
        panels.append(current)
    return panels


def renormalise_bold(tex: str) -> tuple[str, int]:
    """Put ``\\bestmcse`` on each panel column's best value, ties included.

    Rewriting an estimate can move which row is best, so the emphasis has to be
    recomputed or the table will bold a value that is no longer the extremum.
    Which extremum is per table: see :data:`BOLD_DIRECTION`.
    """
    changes = 0
    for match in reversed(list(re.finditer(r"\\label\{tab:([a-z-]+)\}", tex))):
        start = tex.rfind(r"\begin{table}", 0, match.start())
        stop = tex.index(r"\end{table}", match.start())
        block = tex[start:stop]
        name = match.group(1)
        required = chk.GUMBEL_ONLY and name in chk.BOLD_DIRECTION
        if "bestmcse" not in block and not required:
            continue
        label = "tab:" + match.group(1)
        if label not in BOLD_DIRECTION:
            raise KeyError(
                f"{label} carries \\bestmcse but declares no direction; add it "
                "to BOLD_DIRECTION rather than inheriting a minimum"
            )
        targets, _ = chk.boldface_plan(block, name)
        for cell, emphasized, _ in sorted(targets, key=lambda item: item[0].start(), reverse=True):
            want = "best" if emphasized else ""
            if (cell.group(1) or "") == want:
                continue
            replacement = f"\\{want}mcse{{{cell.group(2)}}}{{{cell.group(3)}}}"
            block = block[:cell.start()] + replacement + block[cell.end():]
            changes += 1
        tex = tex[:start] + block + tex[stop:]
    return tex, changes


def atomic_write(path: Path, candidate: str, original: str) -> None:
    """Publish a validated candidate without overwriting concurrent edits."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(candidate)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        if path.read_text(encoding="utf-8") != original:
            raise RuntimeError("manuscript changed during validation; no write performed")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    path = Path(chk.TEX_PATH)
    tex = path.read_text(encoding="utf-8")
    original = tex
    edits = collect_edits(tex)
    for start, stop, text in edits:
        print(f"  {tex[start:stop]}  ->  {text}")
    print(f"{len(edits)} cell(s) to update")
    for start, stop, text in reversed(edits):   # right to left keeps offsets valid
        tex = tex[:start] + text + tex[stop:]
    # Renormalise the boldface even when no cell moved.  Deleting a row changes
    # which value is a column minimum without changing any surviving value, and
    # returning early on "no drift" left those tables permanently mismarked.
    tex, moved = renormalise_bold(tex)
    if moved:
        print(f"updated {moved} \\bestmcse marker(s) according to table policies")
    checked = chk.check_manuscript(tex)
    print(f"candidate validates: {checked} manuscript rows/checks")
    if args.dry_run:
        print("dry run: no files written")
        return 0
    if not edits and not moved:
        print("no drift: manuscript already matches the artifacts")
        return 0
    atomic_write(path, tex, original)
    print(f"wrote {path}")

    print("published the validated manuscript atomically")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
