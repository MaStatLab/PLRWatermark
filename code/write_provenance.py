#!/usr/bin/env python3
"""Fingerprint the current source/artifact snapshot and manifest-writer runtime.

This is not a replacement for immutable generation-time run records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "bayesian_paper_benchmark" / "provenance.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def tracked_inputs() -> list[Path]:
    paths = sorted((ROOT / "code").glob("*.py"))
    paths.extend(
        path
        for path in (
            ROOT / "requirements.txt",
            ROOT / "code" / "requirements.txt",
            ROOT / "code" / "requirements-real-data.txt",
            ROOT / "manuscript" / "bayesian_pivot_watermark_gumbel.tex",
            ROOT / "manuscript" / "references.bib",
            ROOT / "README.md",
            ROOT / "HANDOVER.md",
            ROOT / "code" / "README.md",
        )
        if path.exists()
    )
    return sorted(set(paths))


def tracked_artifacts() -> list[Path]:
    """Generated outputs linked to the audited input snapshot.

    ``provenance.json`` is excluded to avoid a self-referential hash.  The
    manifest intentionally covers machine-readable results, reports, figures,
    and the definitive distributed manuscript.
    """

    results = ROOT / "results" / "bayesian_paper_benchmark"
    # .npy was missing, which left the extension prompt tables -- the input
    # that decides which 2500 documents the matched experiment uses -- outside
    # the manifest entirely.
    suffixes = {".csv", ".json", ".md", ".npy", ".npz", ".pdf", ".png"}
    paths = [
        path
        for path in results.rglob("*")
        if path.is_file()
        and path.name != "provenance.json"
        and path.suffix.lower() in suffixes
        and "quick" not in path.relative_to(results).parts
        and "upstream_assets" not in path.relative_to(results).parts
    ] if results.exists() else []
    definitive_pdf = ROOT / "output" / "pdf" / "bayesian_pivot_watermark_gumbel.pdf"
    if definitive_pdf.exists():
        paths.append(definitive_pdf)
    return sorted(set(paths))


# HEAD alone does not identify uncommitted inputs and is not a generation
# revision. The per-file hashes identify the actual snapshot at write time.
def aggregate_hash(hashes: dict[str, str]) -> str:
    aggregate = hashlib.sha256()
    for relative, digest in sorted(hashes.items()):
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return aggregate.hexdigest()


def build_manifest() -> dict[str, object]:
    inputs = tracked_inputs()
    artifacts = tracked_artifacts()
    input_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in inputs}
    artifact_hashes = {
        str(path.relative_to(ROOT)): sha256(path) for path in artifacts
    }
    return {
        "manifest_kind": "current_source_and_artifact_snapshot",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(ROOT),
        "git_revision": git_revision(),
        "input_snapshot_sha256": aggregate_hash(input_hashes),
        "inputs": input_hashes,
        "artifact_snapshot_sha256": aggregate_hash(artifact_hashes),
        "artifacts": artifact_hashes,
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": version("numpy"),
            "matplotlib": version("matplotlib"),
            "scipy": version("scipy"),
            "torch": version("torch"),
            "transformers": version("transformers"),
            "ipython": version("IPython"),
        },
        "runtime_scope": (
            "Environment that wrote this snapshot manifest, not necessarily "
            "the environment that generated any stored experiment."
        ),
        "generation_provenance_scope": (
            "Use each experiment's original run records for generation-time "
            "code, RNG, model, and runtime provenance. Historical matched "
            "archives without run_metadata remain historical inputs; this "
            "manifest does not certify their original settings or raw-stream "
            "independence."
        ),
        "note": (
            "A null git_revision denotes an unversioned snapshot. In that case, "
            "input_snapshot_sha256 and the per-file hashes identify the audited "
            "inputs. These hashes also include uncommitted changes when HEAD "
            "is recorded. artifact_snapshot_sha256 identifies the stored "
            "outputs; it does not establish that current code generated them. "
            "Regenerating any artifact or editing any input requires a fresh "
            "snapshot manifest without rewriting historical run records."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
