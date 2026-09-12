"""Report EVERY manuscript/artifact drift at once (a worklist), not just the first.

``check_manuscript_tables.main`` stops at the first mismatch, which is the right
behaviour for a gate but useless when reconciling a table after a rerun.  This
wraps ``assert_row`` so every row is checked and the failures are collected.

Keep the wrapper signature in step with ``check_manuscript_tables.assert_row``.
It previously omitted ``expected_mc_se`` and therefore raised ``TypeError``
inside ``main``; the exception was swallowed and the script still printed
"no drift", so a broken run was indistinguishable from a clean one.
"""

import inspect
import contextlib
import io
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "code"))
import check_manuscript_tables as C  # noqa: E402

def main() -> int:
    failures: list[str] = []
    original = C.assert_row

    def collecting(block, marker, expected, expected_mc_se, *,
                   occurrence=0, continuation=False, signed=False):
        try:
            original(block, marker, expected, expected_mc_se,
                     occurrence=occurrence, continuation=continuation, signed=signed)
        except AssertionError as exc:
            failures.append(str(exc))

    # Fail loudly on another signature change instead of swallowing a broken
    # row wrapper and accidentally claiming that no drift was found.
    wanted = list(inspect.signature(collecting).parameters)
    actual = list(inspect.signature(original).parameters)
    if actual != wanted:
        print("drift_report is out of step with check_manuscript_tables.assert_row: "
              f"expected parameters {wanted}, found {actual}")
        return 1

    C.assert_row = collecting
    crashed = False
    try:
        # A captured row failure must not coexist with the checker's ordinary
        # success banner.  Print only this wrapper's final summary.
        with contextlib.redirect_stdout(io.StringIO()):
            C.main()
    except (Exception, SystemExit):
        crashed = True
        traceback.print_exc()
    finally:
        C.assert_row = original

    if crashed:
        print("\nINCOMPLETE: a check raised before every row was compared; "
              "the list below is partial. Fix the error above and rerun.")
    if failures:
        print(f"\n{len(failures)} DRIFTED ROWS:\n")
        for item in failures:
            print(" -", item)
    elif not crashed:
        print("\nno drift: every checked row matches the artifacts")
    return 1 if (failures or crashed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
