"""Compare complete JSON/CSV inventories against a historical snapshot.

Under the restored prior the refactor must be a no-op, so every number should
reproduce exactly.  Anything that moves is another generator/prior wiring bug.
The comparison recurses and fails for missing or extra JSON/CSV files, changed
row/column counts, malformed files, or an empty inventory.  NPZ and figures are
not compared: use the manifest and array checks for those.  Provenance files
and volatile fields (timestamps, runtimes, paths) are explicitly excluded.
"""
import argparse
import json, sys, csv, math
from pathlib import Path

VOLATILE = {"generated_utc", "runtime", "repository_root", "git_revision",
            "elapsed_seconds", "wall_clock_seconds", "hostname", "output_dir",
            "runtime_seconds", "seconds", "generated", "python_version",
            "results_dir", "timestamp", "duration_seconds"}

def walk(a, b, path="", out=None):
    if out is None: out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in VOLATILE: continue
            if k not in a: out.append((f"{path}.{k}", "<missing in old>", b[k]))
            elif k not in b: out.append((f"{path}.{k}", a[k], "<missing in new>"))
            else: walk(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((f"{path}[]", f"len {len(a)}", f"len {len(b)}"))
        else:
            for i, (x, y) in enumerate(zip(a, b)): walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) and isinstance(b, float):
        if not (math.isnan(a) and math.isnan(b)) and a != b:
            out.append((path, a, b))
    elif a != b:
        out.append((path, a, b))
    return out

def cmp_csv(p_old, p_new):
    with p_old.open(newline="", encoding="utf-8") as stream:
        ro = list(csv.reader(stream))
    with p_new.open(newline="", encoding="utf-8") as stream:
        rn = list(csv.reader(stream))
    if len(ro) != len(rn): return [("<rows>", f"{len(ro)}", f"{len(rn)}")]
    diffs = []
    for i, (a, b) in enumerate(zip(ro, rn)):
        if len(a) != len(b):
            diffs.append((f"row{i}.<columns>", f"{len(a)}", f"{len(b)}"))
        for j, (x, y) in enumerate(zip(a, b)):
            if x != y:
                col = ro[0][j] if i and j < len(ro[0]) else f"col{j}"
                diffs.append((f"row{i}.{col}", x, y))
    return diffs

def inventory(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*")
            if path.is_file() and path.suffix in (".json", ".csv")
            and path.name != "provenance.json"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("regenerated", type=Path)
    args = parser.parse_args(argv)
    snap, new = args.snapshot, args.regenerated
    if not snap.is_dir() or not new.is_dir():
        parser.error("both arguments must name existing directories")
    old_files, new_files = inventory(snap), inventory(new)
    if not old_files:
        print("ERROR: snapshot contains no comparable JSON/CSV artifacts")
        return 1
    total_files = total_diffs = 0
    for relative in sorted(old_files | new_files):
        if relative not in new_files:
            print(f"  MISSING  {relative} (not regenerated)")
            total_diffs += 1
            continue
        if relative not in old_files:
            print(f"  EXTRA    {relative} (absent from snapshot)")
            total_diffs += 1
            continue
        f, g = snap / relative, new / relative
        total_files += 1
        try:
            d = (walk(json.loads(f.read_text(encoding="utf-8")),
                      json.loads(g.read_text(encoding="utf-8")))
                 if f.suffix == ".json" else cmp_csv(f, g))
        except (OSError, ValueError, csv.Error) as exc:
            print(f"  ERROR    {relative}: {exc}")
            total_diffs += 1
            continue
        if not d:
            print(f"  IDENTICAL {relative}")
        else:
            total_diffs += len(d)
            print(f"  DIFFERS   {relative}  ({len(d)} field(s))")
            for path, a, b in d[:12]:
                print(f"      {path}\n        snapshot: {a}\n        new:      {b}")
            if len(d) > 12:
                print(f"      ... {len(d)-12} more")
    print(f"\n{total_files} file(s) compared, {total_diffs} differing field(s)")
    return 1 if total_diffs else 0


if __name__ == "__main__":
    sys.exit(main())
