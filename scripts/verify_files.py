#!/usr/bin/env python3
"""Check the exported files before opening archives or running analyses.

This standard-library check performs no network access or deserialization.
Additional local files (for example a virtual environment) are not inspected.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


def verify(root: Path) -> int:
    root = root.resolve()
    checksums = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not re.fullmatch(r"[a-f0-9]{64}  [A-Za-z0-9_+.,/\-]+", line):
            raise ValueError("Malformed checksum entry")
        expected, name = line.split("  ", 1)
        relative = Path(name)
        if (relative.is_absolute() or any(part in {"..", ".git"} for part in relative.parts)
                or str(relative) != name or name in checksums):
            raise ValueError(f"Unsafe or duplicate checksum path: {name}")
        path = root / relative
        if not path.resolve().is_relative_to(root) or path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or unsafe file: {name}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {name}")
        checksums[name] = expected
    if not checksums or "FILE_INDEX.json" not in checksums or "SHA256SUMS" in checksums:
        raise ValueError("Incomplete checksum inventory")
    index = json.loads((root / "FILE_INDEX.json").read_text(encoding="utf-8"))
    if set(index["files"]) != set(checksums) - {"FILE_INDEX.json"}:
        raise ValueError("File index and checksum inventory differ")
    for name, record in index["files"].items():
        if record["sha256"] != checksums[name] or record["bytes"] != (root / name).stat().st_size:
            raise ValueError(f"File-index metadata mismatch: {name}")
    print(f"Verified {len(checksums)} files against SHA256SUMS and FILE_INDEX.json.")
    return len(checksums)


if __name__ == "__main__":
    verify(Path(__file__).resolve().parents[1])
