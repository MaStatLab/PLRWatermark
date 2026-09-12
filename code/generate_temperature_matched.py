"""Temperature-matched regeneration of the released open-model experiment.

The release generates watermarked text at ``--temp`` (0.1) but its unwatermarked
generator never receives that argument and samples from unscaled logits, i.e.
temperature one; the upstream script says so at
``generating_samples.py:199 ## We don't adjust the temperature parameter here``.
The two released samples therefore differ in entropy as well as in watermarking,
and are separable by pivot repetition alone.

This regenerates BOTH arms ourselves at a common temperature, sweeping it, so
the comparison is matched and the version drift falls on both arms equally.
The Gumbel key/sampler/pivot functions are imported unchanged. The raw arm
divides logits by the common temperature and uses reproducible per-global-prompt
random streams; the inverse implementation separately corrects upstream batching.

Prompts are the released ``prompts`` tensors or a verified extension built by
``build_prompts.py``.  Resume requires the same generation configuration and
prompt bytes; additional temperatures and methods may be appended.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import os
import tempfile

# Portable defaults; environment variables may point to an existing checkout
# and working directory. No author's machine-specific paths are required.
REPO = Path(
    os.environ.get("WATERMARK_REPO", Path(__file__).resolve().parents[1])
)
UPSTREAM = Path(os.environ.get(
    "WATERMARK_FRAMEWORK_DIR", REPO / "third_party" / "WatermarkFramework")) / "real data"
SCRATCH = Path(os.environ.get("WM_SCRATCH", REPO / "tmp" / "generation"))

import numpy as np
import torch

sys.path.insert(0, str(REPO / "code"))

KEY = 15_485_863
CONTEXT = 4
# Every arm array is keyed "<tag>__<field>"; the run record is the one entry
# that is not an arm, so consumers recognise it by the missing separator.
ARM_SEP = "__"
METADATA_KEY = "run_metadata"
METADATA_VERSION = 3
RAW_RNG_SCHEME = "sha256-arm-global-index-prompt-row-v1"
EXPECTED_UPSTREAM_COMMIT = "05b7ffda9279fc9e645f38807e4a0e2dbcff4330"
SEEDING = "skipgram_prf"
# Model revisions are pinned so a regeneration reads the same weights.  Without
# a revision, from_pretrained follows the branch head and a silent upstream
# update changes the text without changing this file.
SPECS = {
    # Commit SHAs, not "main": a branch name follows the head and is not a pin.
    # These pins define future regeneration. Historical arrays lack a run
    # record and do not establish which revisions were used to generate them.
    "1p3B": ("facebook/opt-1.3b", 50272,
             "3f5c25d0bc631cb57ac65913f76e22c2dfb61d62"),
    "2p7B": ("princeton-nlp/Sheared-LLaMA-2.7B", 32000,
             "2f157a0306b75d37694ae05f6a4067220254d540"),
}
# The C4 slice the prompts come from.  This used to name the ``en`` config and
# a shard from its 1024-file layout; the prompts are from ``realnewslike``,
# whose first shard ``code/build_prompts.py`` fetches by digest and whose
# output reproduces the released prompt tensor exactly.  Kept as a pinned
# (repo, revision, file) triple rather than a bare filename.
C4_DATASET = ("allenai/c4", "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
              "realnewslike/c4-train.00000-of-00512.json.gz")


def run_metadata(args, name, revision, device, prompt_record) -> dict:
    """Immutable run identity, plus a manifest of completed arms.

    The arm manifest may grow on resume. All other fields must match, so an
    existing arm can never be relabeled with another invocation's settings.
    Package versions are read without importing transformers or loading a model.
    """
    import platform
    from importlib.metadata import version
    record = {
        "schema_version": METADATA_VERSION,
        "script": "code/generate_temperature_matched.py",
        "code_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("inverse_batched_sampling.py"))
        },
        "model_key": args.model,
        "model": name,
        "model_revision": revision,
        "tokenizer": name,
        "tokenizer_revision": revision,
        "vocab_size": SPECS[args.model][1],
        "c4_repo": C4_DATASET[0],
        "c4_revision": C4_DATASET[1],
        "c4_file": C4_DATASET[2],
        **prompt_record,
        "prompt_offset": args.prompt_offset,
        "documents": args.documents,
        "tokens": args.tokens,
        "batch": args.batch,
        "seed": args.seed,
        "raw_rng_scheme": RAW_RNG_SCHEME,
        "context": CONTEXT,
        "seeding": SEEDING,
        "key_seed": KEY,
        "upstream_commit": upstream_commit(),
        "device": device,
        "dtype": "float32",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "platform": platform.platform(),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "arms": {},
        "array_sha256": {},
    }
    if device == "cuda":
        record["device_name"] = torch.cuda.get_device_name()
    return record


def upstream_commit() -> str:
    """Require the exact clean clone whose algorithms define this experiment."""
    import subprocess
    try:
        sha = subprocess.run(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if sha.returncode != 0:
            raise ValueError("WATERMARK_FRAMEWORK_DIR must name the pinned git checkout")
        dirty = subprocess.run(
            ["git", "-C", str(UPSTREAM), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        if sha.stdout.strip() != EXPECTED_UPSTREAM_COMMIT:
            raise ValueError(f"upstream must be at commit {EXPECTED_UPSTREAM_COMMIT}; "
                             f"found {sha.stdout.strip()}")
        if dirty.returncode != 0 or dirty.stdout.strip():
            raise ValueError("upstream checkout must be clean (including untracked files)")
        return sha.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"cannot verify upstream checkout: {exc}") from exc


def load_upstream():
    """Import the pinned sampler only when a watermarked arm is generated."""
    import importlib
    upstream_commit()
    sys.path.insert(0, str(UPSTREAM))
    # Importing must not dirty an otherwise clean upstream checkout with an
    # untracked __pycache__, which would make its next resume fail validation.
    previous_bytecode_setting = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sampling = importlib.import_module("sampling")
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
    if Path(sampling.__file__).resolve() != (UPSTREAM / "sampling.py").resolve():
        raise ValueError("sampling was imported from a different upstream checkout")
    import inverse_batched_sampling as inverse
    return sampling, inverse


def select_device(requested):
    available = {"cuda": torch.cuda.is_available(),
                 "mps": torch.backends.mps.is_available(), "cpu": True}
    if requested == "auto":
        return next(device for device in ("cuda", "mps", "cpu") if available[device])
    if not available.get(requested, False):
        raise ValueError(f"requested device {requested!r} is unavailable")
    return requested


def prompt_sha256(table):
    """Digest token values and shape, independent of tensor/file container."""
    values = np.ascontiguousarray(table, dtype="<i8")
    h = hashlib.sha256(json.dumps(list(values.shape)).encode())
    h.update(values.tobytes())
    return h.hexdigest()


def load_prompts(args):
    import real_data_experiment as real
    from build_prompts import (C4_SHA256, PROMPT_TOKENS, CONTINUATION_TOKENS,
                               BUFFER_TOKENS, TRUNCATE_AT)
    data_dir = REPO / "results/bayesian_paper_benchmark/real_model/upstream_assets"
    released, _ = real.load_released_model_data(args.model, data_dir)
    reference = np.asarray(released.prompts)
    table = reference
    source = "released tensor"
    digest = prompt_sha256(table)
    if args.prompts is not None:
        sidecar = args.prompts.with_suffix(".json")
        if not sidecar.is_file():
            raise ValueError(f"required prompt sidecar is missing: {sidecar}")
        meta = json.loads(sidecar.read_text())
        table = np.load(args.prompts, allow_pickle=False)
        if table.ndim != 2:
            raise ValueError("prompt table must have shape (N, 50)")
        digest = hashlib.sha256(args.prompts.read_bytes()).hexdigest()
        source = args.prompts.name
        expected = {
            "model": args.model, "tokenizer": SPECS[args.model][0],
            "tokenizer_revision": SPECS[args.model][2],
            "c4_repo": C4_DATASET[0], "c4_revision": C4_DATASET[1],
            "c4_file": C4_DATASET[2], "c4_sha256": C4_SHA256,
            "prompt_tokens": PROMPT_TOKENS,
            "continuation_tokens": CONTINUATION_TOKENS,
            "buffer_tokens": BUFFER_TOKENS, "truncate_at": TRUNCATE_AT,
            "prompts": len(table), "released_rows_verified": len(reference),
            "output_sha256": digest,
        }
        if not isinstance(meta, dict):
            raise ValueError("prompt sidecar must contain a JSON object")
        mismatches = [key for key, value in expected.items() if meta.get(key) != value]
        if mismatches:
            raise ValueError("prompt sidecar mismatch: " + ", ".join(mismatches))
    if (table.ndim != 2 or table.shape[1] != PROMPT_TOKENS or
            not np.issubdtype(table.dtype, np.integer) or
            np.any(table < 0) or np.any(table >= SPECS[args.model][1])):
        raise ValueError("prompt table must be an integer (N, 50) array of valid token IDs")
    if args.prompts is not None and not np.array_equal(table[:len(reference)], reference):
        raise ValueError("prompt table does not reproduce all released prompts in its first rows")
    stop = args.prompt_offset + args.documents
    if args.prompt_offset < 0 or args.documents <= 0 or stop > len(table):
        raise ValueError(f"requested {args.documents} prompts at offset {args.prompt_offset}, "
                         f"but the table contains {len(table)} rows")
    selected = table[args.prompt_offset:stop]
    return torch.as_tensor(selected, dtype=torch.long), {
        "prompt_source": source, "prompt_table_sha256": digest,
        "selected_prompt_sha256": prompt_sha256(selected),
    }


def arm_spec(args, temperature, method):
    seed = int.from_bytes(hashlib.sha256(
        f"{args.model}|{temperature:g}|{method}|{args.seed}".encode()
    ).digest()[:4], "big")
    return {"temperature": temperature, "method": method, "raw_seed": seed}


def prompt_generators(raw_seed, prompts, prompt_offset):
    """Independent raw streams indexed by global prompt identity, not batching.

    The same row at the same global index has the same stream when regenerated
    in another chunk. Different global indices do not restart an arm's stream.
    This fixes sampling-RNG invariance, not device/batch effects in model logits.
    """
    if prompt_offset < 0:
        raise ValueError("prompt offset must be nonnegative")
    values = np.asarray(prompts.cpu() if isinstance(prompts, torch.Tensor) else prompts)
    streams = []
    for index, row in enumerate(values, start=prompt_offset):
        identity = f"{RAW_RNG_SCHEME}|{raw_seed}|{index}|{prompt_sha256(row)}"
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
        streams.append(torch.Generator().manual_seed(seed))
    return streams


def array_sha256(values):
    """Hash the stored dtype, shape, and contiguous value bytes."""
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256(json.dumps([values.dtype.str, list(values.shape)]).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def validate_payload(payload, metadata):
    """Require complete companion arrays for every arm, not just tokens."""
    arms = metadata.get("arms")
    if not isinstance(arms, dict):
        raise ValueError("run metadata has no completed-arm manifest")
    expected_keys = {METADATA_KEY}
    shape = (metadata["documents"], metadata["tokens"])
    for tag, spec in arms.items():
        if not isinstance(spec, dict) or spec.get("method") not in {"raw", "gumbel", "transform"}:
            raise ValueError(f"invalid arm manifest for {tag}")
        method = spec["method"]
        temperature = spec.get("temperature")
        if (not isinstance(temperature, (int, float)) or not np.isfinite(temperature)
                or temperature <= 0 or tag != f"t{temperature:g}_{method}"):
            raise ValueError(f"invalid temperature/tag for {tag}")
        expected_seed = int.from_bytes(hashlib.sha256(
            f"{metadata['model_key']}|{temperature:g}|{method}|{metadata['seed']}".encode()
        ).digest()[:4], "big")
        if spec.get("raw_seed") != expected_seed:
            raise ValueError(f"invalid seed in the manifest for {tag}")
        fields = {"tokens", "top_probs"}
        if method != "raw":
            fields.add("Y")
        if method == "transform":
            fields.update({"U", "eta"})
        for field in fields:
            key = f"{tag}{ARM_SEP}{field}"
            expected_keys.add(key)
            if key not in payload or payload[key].shape != shape:
                raise ValueError(f"arm {tag} requires {field} with shape {shape}")
            values = payload[key]
            if field == "tokens":
                valid = (np.issubdtype(values.dtype, np.integer) and
                         np.all((values >= 0) & (values < metadata["vocab_size"])))
            else:
                valid = (np.issubdtype(values.dtype, np.number) and
                         not np.issubdtype(values.dtype, np.complexfloating) and
                         np.isfinite(values).all())
                if valid:
                    tolerance = 0.0
                    if method == "transform" and field in {"Y", "eta"}:
                        # The released inverse convention is shifted by one rank:
                        # eta=(rank-1)/(V-1).  Rank zero therefore gives a small
                        # negative eta, and Y=-|U-eta| can be correspondingly
                        # smaller than -1.  Allow a few storage-dtype epsilons at
                        # these computed endpoints, but not a material excursion.
                        offset = 1.0 / (metadata["vocab_size"] - 1)
                        if field == "eta":
                            lower, upper = -offset, 1.0 - offset
                        else:
                            lower, upper = -1.0 - offset, 0.0
                        if np.issubdtype(values.dtype, np.floating):
                            tolerance = 4 * np.finfo(values.dtype).eps
                    else:
                        lower, upper = 0.0, 1.0
                    valid = np.all((values >= lower - tolerance) &
                                   (values <= upper + tolerance))
            if not valid:
                raise ValueError(f"invalid values in {key}")
    if set(payload) != expected_keys:
        raise ValueError("NPZ arrays do not match the completed-arm manifest")
    if metadata.get("schema_version") == METADATA_VERSION:
        hashes = {key: array_sha256(value) for key, value in payload.items() if key != METADATA_KEY}
        if metadata.get("array_sha256") != hashes:
            raise ValueError("array digests do not match the generation manifest")


def load_resume(path, current):
    if not path.exists():
        return {}, current
    try:
        with np.load(path, allow_pickle=False) as stored:
            payload = {key: stored[key] for key in stored.files}
        if METADATA_KEY not in payload:
            raise ValueError("historical NPZ has no trustworthy run metadata")
        record = json.loads(payload[METADATA_KEY].item())
        if not isinstance(record, dict) or record.get("schema_version") != METADATA_VERSION:
            raise ValueError("unsupported or legacy run metadata")
        # Filenames are descriptive; actual table/selected-prompt hashes govern
        # identity. Keep the original filename in the record after a valid move.
        identity = lambda value: {key: item for key, item in value.items()
                                  if key not in {"arms", "array_sha256", "prompt_source"}}
        old, new = identity(record), identity(current)
        changes = [key for key in sorted(old.keys() | new.keys())
                   if key not in old or key not in new or old[key] != new[key]]
        if changes:
            raise ValueError("incompatible resume configuration: " + ", ".join(changes))
        validate_payload(payload, record)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise ValueError(f"Cannot resume {path}: {exc}. Use a new --out path; "
                         "the existing file has not been changed.") from exc
    return payload, record


def atomic_checkpoint(path, payload):
    """Replace the previous checkpoint only after the complete new ZIP is written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as fh:
            temporary = Path(fh.name)
            np.savez_compressed(fh, **payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _step(model, inputs, attn, past):
    with torch.no_grad():
        out = model(inputs[:, -1:], past_key_values=past, attention_mask=attn) if past \
              else model(inputs)
    return out


def generate(model, prompts, *, vocab_size, m, temperature, method, device,
             raw_generator=None, raw_generators=None, watermark_functions=None):
    """One arm.  method in {raw, gumbel, transform}.  Mirrors generation.py.

    The watermarked arms are deterministic given the key and the context, but
    the raw arm is ordinary multinomial sampling and needs its own seed: the
    earlier version called torch.multinomial without a generator, so it drew
    from the global RNG and the unwatermarked arm could not be reproduced.

    The command supplies one ``raw_generators`` stream per global prompt, so
    splitting or batching an arm does not restart or reorder its random draws.
    ``raw_generator`` remains an explicitly supplied legacy single-stream API;
    it is not used by the command. Batch size remains recorded because batched
    model evaluation itself can change floating-point logits.
    """

    generator = torch.Generator()
    if method == "raw":
        if raw_generators is not None:
            if raw_generator is not None or len(raw_generators) != len(prompts):
                raise ValueError("supply exactly one raw generator per prompt, or one legacy generator")
            if len({id(stream) for stream in raw_generators}) != len(raw_generators):
                raise ValueError("each prompt requires its own raw generator")
        elif raw_generator is None:
            raise ValueError("raw generation requires explicitly supplied random streams")
    inputs = prompts.to(device)
    attn = torch.ones_like(inputs)
    past = None
    Ys, tops, Us, Etas = [], [], [], []
    if method not in {"raw", "gumbel", "transform"}:
        raise ValueError(f"unknown generation method: {method}")
    if method != "raw":
        sampling, inverse = watermark_functions or load_upstream()
    for _ in range(m):
        out = _step(model, inputs, attn, past)
        # Softmax in float32 regardless of model dtype: at temperature .1 the
        # deficit 1 - max p is the quantity of interest and it underflows fast.
        probs = torch.nn.functional.softmax(out.logits[:, -1].float() / temperature, dim=-1).cpu()
        tops.append(torch.max(probs, axis=1)[0].unsqueeze(0))
        if method == "raw":
            # Sampling uses the temperature-adjusted probabilities computed above.
            tokens = (torch.cat([
                torch.multinomial(row, 1, generator=stream).reshape(1, 1)
                for row, stream in zip(probs, raw_generators)
            ]) if raw_generators is not None else
                torch.multinomial(probs, 1, generator=raw_generator)).to(device)
        elif method == "transform":
            # The released transform_key_func asserts a batch of one because its
            # loop seeds on the whole batch; inverse_batched_sampling fixes that
            # and is proved element-wise equal to the released code at batch one.
            xi, pi = inverse.transform_key_func_batched(
                generator, inputs, vocab_size, KEY, CONTEXT, SEEDING
            )
            tokens = inverse.transform_sampling_batched(probs, pi, xi).to(device)
            y, u, eta = inverse.transform_Y_batched(tokens, pi, xi)
            Ys.append(y)
            Us.append(u)
            Etas.append(eta)
        else:
            xi, pi = sampling.gumbel_key_func(generator, inputs, vocab_size, KEY, CONTEXT, SEEDING)
            tokens = sampling.gumbel_sampling(probs, pi, xi).to(device)
            Ys.append(sampling.gumbel_Y(tokens, pi, xi).unsqueeze(0))
        inputs = torch.cat([inputs, tokens.reshape(-1, 1)], dim=-1)
        past = out.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
    out_tokens = inputs[:, prompts.shape[1]:].detach().cpu().numpy()
    out_top = torch.vstack(tops).T.detach().cpu().numpy()
    if method == "transform":
        # Already (batch, 1) per step, so concatenate along the token axis.
        cat = lambda chunks: torch.cat(chunks, dim=1).detach().cpu().numpy()
        return out_tokens, cat(Ys), out_top, cat(Us), cat(Etas)
    out_Y = torch.vstack(Ys).T.detach().cpu().numpy() if Ys else None
    return out_tokens, out_Y, out_top, None, None


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(SPECS))
    ap.add_argument("--temps", default="0.1,0.5,1.0")
    ap.add_argument("--documents", type=int, default=500)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
                    help="auto selects CUDA, then MPS, then CPU; part of run identity")
    # "transform" is the inverse arm.  It runs through inverse_batched_sampling
    # rather than the released transform_key_func, whose batch-size-1 assertion
    # guards a seeding defect; the per-row key work is ~0.18 ms, so batching
    # makes it cost about what the Gumbel arm costs.
    ap.add_argument("--methods", default="raw,gumbel")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260902,
                    help="base seed for the unwatermarked arm; the watermarked "
                         "arms are keyed and do not use it")
    # An .npy of shape (N, 50) built by the same C4 stream and filter the
    # release uses; its first 500 rows are bit-identical to the released prompt
    # tensor, so rows past 500 extend the same population.  Used to raise the
    # simulation size beyond the 500 released prompts.
    ap.add_argument("--prompts", type=Path, default=None)
    ap.add_argument("--prompt-offset", type=int, default=0,
                    help="skip this many prompts, so an extension run does not "
                         "regenerate documents that already exist")
    args = ap.parse_args(argv)
    if min(args.documents, args.tokens, args.batch) <= 0 or args.prompt_offset < 0:
        ap.error("documents, tokens and batch must be positive; prompt-offset must be nonnegative")
    try:
        args.temps = [float(value) for value in args.temps.split(",")]
    except ValueError:
        ap.error("temps must be a comma-separated list of positive finite numbers")
    if (any(not np.isfinite(value) or value <= 0 for value in args.temps) or
            len({f"{value:g}" for value in args.temps}) != len(args.temps)):
        ap.error("temperatures must be positive, finite and have distinct arm tags")
    args.methods = [value.strip() for value in args.methods.split(",")]
    if (not set(args.methods) <= {"raw", "gumbel", "transform"} or
            len(set(args.methods)) != len(args.methods)):
        ap.error("methods must be distinct choices from raw,gumbel,transform")
    return args


def load_model(name, revision, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32, revision=revision
    ).to(device).eval()


def main(argv=None) -> None:
    args = parse_args(argv)
    name, vocab, revision = SPECS[args.model]
    prompts, prompt_record = load_prompts(args)
    device = select_device(args.device)
    metadata = run_metadata(args, name, revision, device, prompt_record)
    payload, metadata = load_resume(args.out, metadata)
    pending = []
    for temp in args.temps:
        for method in args.methods:
            tag = f"t{temp:g}_{method}"
            spec = arm_spec(args, temp, method)
            if tag in metadata["arms"]:
                if metadata["arms"][tag] != spec:
                    raise ValueError(f"existing arm {tag} has a different specification; use a new --out path")
                print(f"  {args.model} {tag}: already present, skipping", flush=True)
            else:
                pending.append((tag, spec))
    if pending:
        watermark_functions = load_upstream() if any(spec["method"] != "raw" for _, spec in pending) else None
        model = load_model(name, revision, device)
        for tag, spec in pending:
            temp, method, raw_seed = spec["temperature"], spec["method"], spec["raw_seed"]
            if method == "raw":
                print(f"  {tag}: per-prompt raw streams anchored at {raw_seed}", flush=True)
            t0 = time.perf_counter()
            toks, ys, tops, us, etas = [], [], [], [], []
            for s in range(0, prompts.shape[0], args.batch):
                chunk = prompts[s : s + args.batch]
                streams = (prompt_generators(raw_seed, chunk, args.prompt_offset + s)
                           if method == "raw" else None)
                a, b, c, u, e = generate(model, chunk, vocab_size=vocab, m=args.tokens,
                                         raw_generators=streams,
                                         watermark_functions=watermark_functions,
                                         temperature=temp, method=method, device=device)
                toks.append(a); tops.append(c)
                if b is not None: ys.append(b)
                if u is not None: us.append(u)
                if e is not None: etas.append(e)
            payload[f"{tag}__tokens"] = np.concatenate(toks)
            payload[f"{tag}__top_probs"] = np.concatenate(tops)
            if ys: payload[f"{tag}__Y"] = np.concatenate(ys)
            # The inverse arm also stores its two components, so any pivot
            # convention (|U - eta|, or the upstream shifted eta) can be formed
            # downstream without regenerating.
            if us: payload[f"{tag}__U"] = np.concatenate(us)
            if etas: payload[f"{tag}__eta"] = np.concatenate(etas)
            print(f"  {args.model} {tag}: {time.perf_counter()-t0:.0f}s", flush=True)
            metadata["arms"][tag] = spec
            metadata["array_sha256"] = {key: array_sha256(value) for key, value in payload.items()
                                        if key != METADATA_KEY}
            payload[METADATA_KEY] = np.array(json.dumps(metadata, indent=2, sort_keys=True))
            validate_payload(payload, metadata)
            atomic_checkpoint(args.out, payload)
    arms = sum(1 for k in payload if k.endswith(ARM_SEP + "tokens"))
    print(json.dumps({"output": str(args.out), "arms": arms}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
