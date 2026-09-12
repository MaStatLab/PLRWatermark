#!/usr/bin/env python3
"""Rebuild the C4 prompt table the temperature-matched runs are conditioned on.

The released assets supply 500 prompts.  Raising the simulation size past that
needs more prompts drawn from the *same* population by the *same* filter, and
until now the script that built them lived outside the repository: the prompt
tables in ``.../temperature_matched/extension/`` were data with no producer, so
a reader could not check that rows past 500 extend the released population
rather than some other one.  This is that producer.

The construction is upstream's.  Records are read from the C4 stream in file
order with no shuffle, tokenized with truncation at ``2028`` tokens, dropped if
shorter than ``PROMPT_TOKENS + CONTINUATION_TOKENS``, and the prompt is the
50-token window ``tokens[-250:-200]`` that immediately precedes the 200 tokens
upstream scored.

Two things make the output checkable rather than asserted:

* the C4 slice is fetched from a pinned repository revision by URL and its
  SHA-256 is recorded, so "which shard" is a hash and not a filename; and
* the first 500 rows are compared against the released prompt tensor and the
  script fails unless they agree exactly.  That equality is the evidence that
  the filter reproduced here is the filter upstream ran -- 500 rows of 50
  tokens over a 50k vocabulary do not coincide by accident.

Reading the shard directly rather than through ``datasets`` is deliberate: a
streaming split iterates its shards in file order, so the two agree on the
records, but the raw read pins the bytes and drops a heavy dependency that
would otherwise sit between the paper and its prompts.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Upstream's realnewslike train split, first shard, at a pinned repository
# revision.  ``main`` would follow the head and silently change the prompts.
C4_REPO = "allenai/c4"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
C4_FILE = "realnewslike/c4-train.00000-of-00512.json.gz"
C4_URL = f"https://huggingface.co/datasets/{C4_REPO}/resolve/{C4_REVISION}/{C4_FILE}"
C4_SHA256 = "6666a680b0a34eb8756dcb5fd2b12f0078237f3502e8a513bd3e5b71bb92be00"

# Upstream's filter.  The prompt is the window before the scored continuation,
# so a document must carry both.
PROMPT_TOKENS = 50
CONTINUATION_TOKENS = 200
# Upstream truncates at the context window less its ``buffer_tokens`` default,
# not at the context window itself.  The 20-token difference is invisible on
# short documents and moves the window by exactly 20 positions on every
# document long enough to truncate, which is what the released prompts show:
# for those, the accepted window sits at absolute offset 1778 = 2028 - 250
# whatever the document's full length.
BUFFER_TOKENS = 20
TRUNCATE_AT = 2048 - BUFFER_TOKENS

# Tokenizer repository and revision.  The tokenizer ships in the model
# repository, so an unpinned load follows the branch head exactly as an
# unpinned model load would, and a tokenizer change moves every prompt.  These
# are the same commits generate_temperature_matched.py pins for the weights.
TOKENIZERS = {
    "1p3B": ("facebook/opt-1.3b", "3f5c25d0bc631cb57ac65913f76e22c2dfb61d62"),
    "2p7B": ("princeton-nlp/Sheared-LLaMA-2.7B",
             "2f157a0306b75d37694ae05f6a4067220254d540"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_shard(cache: Path) -> Path:
    """Download the pinned shard once, and verify its digest every time.

    The digest is checked on the cached copy too.  A cache that silently went
    stale is exactly the failure this script exists to rule out.
    """
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / Path(C4_FILE).name
    if not local.exists():
        print(f"  fetching {C4_URL}", flush=True)
        with urllib.request.urlopen(C4_URL) as src, local.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    digest = sha256(local)
    if digest != C4_SHA256:
        raise SystemExit(
            f"C4 shard digest mismatch\n  expected {C4_SHA256}\n  got      {digest}\n"
            f"  file     {local}"
        )
    return local


def build(model: str, count: int, shard: Path) -> tuple[np.ndarray, int]:
    from transformers import AutoTokenizer

    name, revision = TOKENIZERS[model]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    need = PROMPT_TOKENS + CONTINUATION_TOKENS
    rows: list[np.ndarray] = []
    scanned = 0
    with gzip.open(shard, "rt") as fh:
        for line in fh:
            scanned += 1
            tokens = tokenizer.encode(
                json.loads(line)["text"],
                return_tensors="pt",
                truncation=True,
                max_length=TRUNCATE_AT,
            )[0]
            if len(tokens) < need:
                continue
            rows.append(tokens[-need:-CONTINUATION_TOKENS].numpy())
            if len(rows) == count:
                break
            if len(rows) % 500 == 0:
                print(f"  {len(rows)}/{count} prompts ({scanned} records read)",
                      flush=True)
    if len(rows) < count:
        raise SystemExit(
            f"shard exhausted after {scanned} records with only {len(rows)} of "
            f"{count} prompts; a second shard would be needed"
        )
    return np.stack(rows).astype(np.int64), scanned


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=sorted(TOKENIZERS), required=True)
    ap.add_argument("--count", type=int, default=2500)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--cache", type=Path, default=REPO / "results/cache/c4")
    args = ap.parse_args()

    out = args.out or (
        REPO / "results/bayesian_paper_benchmark/real_model/temperature_matched"
        / "extension" / f"prompts_{args.model}.npy"
    )

    started = time.time()
    shard = fetch_shard(args.cache)
    table, scanned = build(args.model, args.count, shard)

    # The released prompts are the ground truth for the construction.  Rows
    # past 500 are only an extension of the same population if the first 500
    # are the released ones exactly.
    import real_data_experiment as real
    data_dir = REPO / "results/bayesian_paper_benchmark/real_model/upstream_assets"
    released, _ = real.load_released_model_data(args.model, data_dir)
    reference = np.asarray(released.prompts)
    n = reference.shape[0]
    if not (table[:n] == reference).all():
        bad = int((table[:n] != reference).any(axis=1).sum())
        raise SystemExit(
            f"rebuilt prompts do not reproduce the released tensor: {bad} of "
            f"{n} rows differ; the filter reconstructed here is not the one "
            f"upstream ran"
        )
    print(f"  first {n} rows reproduce the released prompt tensor exactly")

    np.save(out, table)
    meta = {
        "script": "code/build_prompts.py",
        "model": args.model,
        "tokenizer": TOKENIZERS[args.model][0],
        "tokenizer_revision": TOKENIZERS[args.model][1],
        "c4_repo": C4_REPO,
        "c4_revision": C4_REVISION,
        "c4_file": C4_FILE,
        "c4_sha256": C4_SHA256,
        "prompt_tokens": PROMPT_TOKENS,
        "continuation_tokens": CONTINUATION_TOKENS,
        "buffer_tokens": BUFFER_TOKENS,
        "truncate_at": TRUNCATE_AT,
        "records_scanned": scanned,
        "prompts": int(table.shape[0]),
        "released_rows_verified": n,
        "output_sha256": sha256(out),
        "seconds": round(time.time() - started, 1),
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {out} and {meta_path.name} in {meta['seconds']}s")


if __name__ == "__main__":
    main()
