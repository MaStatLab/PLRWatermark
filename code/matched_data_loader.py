"""Fail-closed loading and prompt pairing for temperature-matched archives.

Schema-v3 generation records are verified against the actual prompt rows.
Eight historical files can be read only with explicit legacy opt-in and an
exact byte digest. Their layout is known; their missing generation history is
not reconstructed or upgraded to a manifest.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import generate_temperature_matched as gen


# name: (SHA-256, model, global offset, documents, token budget)
LEGACY_ARCHIVES = {
    "matched_1p3B_hi.npz": ("ef51c2d0d03e6afe51f204b49901c25daf2aab27a0f382dd1bafb2171aa3230d", "1p3B", 0, 500, 100),
    "matched_1p3B_lo.npz": ("24b3e6c3031e60ece47ec428d16d6e68b689b8fa204ad271a653ffdaef3b9172", "1p3B", 0, 500, 200),
    "matched_2p7B_hi.npz": ("99ed5cf30ffaa36ff03bed40a7a9f9f4d653fb6515bd3f0adea5e89cc0bb0961", "2p7B", 0, 500, 100),
    "matched_2p7B_lo.npz": ("335433658d779a7a52642bb306aeb0fee4a8bf23cd46974fc1757815d8b3cafa", "2p7B", 0, 500, 200),
    "ext_1p3B_hi.npz": ("62eb715bb3696a3cf462fcd8f5da82af141fac39cf12a47165523c33e24b1cc8", "1p3B", 500, 2000, 100),
    "ext_1p3B_lo.npz": ("a18bb7b7d7df261fbf51a925936256e54828c221c123423e076ee4f135880d69", "1p3B", 500, 2000, 200),
    "ext_2p7B_hi.npz": ("821f44100760f3c144fc545cc24b0251a8d95e884432f3edca10d8420fff767f", "2p7B", 500, 2000, 100),
    "ext_2p7B_lo.npz": ("56700de6cb8e3cec8f44e3cf1812a63686913a5c3f7748685a85e39aef6b0822", "2p7B", 500, 2000, 200),
}

# Bind the SHA-pinned historical arrays to the prompt rows audited with them,
# not merely to any valid C4 sidecar sharing the released leading 500 rows.
# Digests use gen.prompt_sha256: shape plus contiguous little-endian int64
# token values, independent of the NPY container and input integer dtype.
# These pins describe the audited artifacts, not recovered generation history.
LEGACY_PROMPT_BLOCK_SHA256 = {
    ("1p3B", 0, 500): "6b08bae0447951dc3d8b09ee787d473bbc1a6663917cc66e56eae9fcc553b5f0",
    ("1p3B", 500, 2000): "d87fbea74a78dec829b2825a1cfc896ba02f6e5baf09a47cd96314be5f2afbbd",
    ("2p7B", 0, 500): "c4a6e146879059f34a671407883542be1983a20158749079341e7848369ea315",
    ("2p7B", 500, 2000): "fac10bf0731b3e41fd60ea4bb460b17a873272f890adf48e1a0e90d127b5b4ab",
}


@dataclass
class Archive:
    arrays: dict[str, np.ndarray]
    model: str
    offset: int
    documents: int
    tokens: int
    provenance: dict
    metadata: dict | None


@dataclass
class PairedCell:
    temperature: str
    raw: dict[str, np.ndarray]
    watermarked: dict[str, np.ndarray]
    prompts: np.ndarray
    prompt_indices: np.ndarray
    provenance: list[dict]


def read_archive(path: Path, *, allow_legacy: bool = False) -> Archive:
    """Validate complete arrays and retain, rather than discard, provenance."""
    path = Path(path)
    contents = path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    with np.load(io.BytesIO(contents), allow_pickle=False) as stored:
        if len(set(stored.files)) != len(stored.files):
            raise ValueError(f"{path}: duplicate NPZ array keys")
        payload = {key: stored[key] for key in stored.files}
    metadata = None
    if gen.METADATA_KEY not in payload:
        known = LEGACY_ARCHIVES.get(path.name)
        if not allow_legacy:
            raise ValueError(f"{path}: no run manifest; use --allow-legacy-archives only for the eight SHA-allowlisted historical files")
        if known is None or digest != known[0]:
            raise ValueError(f"{path}: unmanifested archive is not SHA-allowlisted")
        _, model, offset, documents, tokens = known
        provenance = {"mode": "sha256_allowlisted_legacy", "path": str(path),
                      "sha256": digest, "generation_manifest": None,
                      "historical_rng_and_model_revision": "unknown"}
        # Digest pins the complete historical payload. Check the known layout
        # explicitly without manufacturing a generation record for these data.
        if not payload or any("__" not in key or value.shape != (documents, tokens)
                              for key, value in payload.items()):
            raise ValueError(f"{path}: historical layout mismatch")
    else:
        try:
            metadata = json.loads(payload[gen.METADATA_KEY].item())
            if not isinstance(metadata, dict) or metadata.get("schema_version") != gen.METADATA_VERSION:
                raise ValueError("unsupported run manifest; schema v3 is required")
            model = metadata["model_key"]
            name, vocab, revision = gen.SPECS[model]
            expected = {"model": name, "model_revision": revision,
                        "tokenizer": name, "tokenizer_revision": revision,
                        "vocab_size": vocab, "key_seed": gen.KEY,
                        "context": gen.CONTEXT, "seeding": gen.SEEDING,
                        "upstream_commit": gen.EXPECTED_UPSTREAM_COMMIT,
                        "raw_rng_scheme": gen.RAW_RNG_SCHEME,
                        "c4_repo": gen.C4_DATASET[0], "c4_revision": gen.C4_DATASET[1],
                        "c4_file": gen.C4_DATASET[2], "dtype": "float32"}
            for field, value in expected.items():
                if metadata.get(field) != value:
                    raise ValueError(f"incompatible {field}")
            for field in ("documents", "tokens", "batch"):
                if type(metadata[field]) is not int or metadata[field] <= 0:
                    raise ValueError(f"invalid {field}")
            if type(metadata["prompt_offset"]) is not int or metadata["prompt_offset"] < 0:
                raise ValueError("invalid prompt_offset")
            if type(metadata["seed"]) is not int:
                raise ValueError("invalid seed")
            for field in ("prompt_table_sha256", "selected_prompt_sha256"):
                value = metadata[field]
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError(f"invalid {field}")
            if not isinstance(metadata.get("versions"), dict) or not metadata["versions"]:
                raise ValueError("missing versions")
            hashes = metadata.get("code_sha256")
            if not isinstance(hashes, dict) or not all(
                isinstance(hashes.get(key), str) and len(hashes[key]) == 64
                for key in ("generate_temperature_matched.py", "inverse_batched_sampling.py")
            ):
                raise ValueError("missing code_sha256")
            if metadata.get("device") not in {"cpu", "cuda", "mps"}:
                raise ValueError("invalid device")
            gen.validate_payload(payload, metadata)
            offset, documents, tokens = (metadata[k] for k in ("prompt_offset", "documents", "tokens"))
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid generation manifest: {exc}") from exc
        provenance = {"mode": "verified_manifest_v3", "path": str(path),
                      "sha256": digest, "generation_manifest": metadata}
    arrays = {key: value for key, value in payload.items() if "__" in key}
    return Archive(arrays, model, offset, documents, tokens, provenance, metadata)


def discover_archives(scratch: Path, model: str, scheme: str = "gumbel") -> list[Path]:
    """Find conventional archive names; ambiguous duplicate locations fail."""
    scratch = Path(scratch)
    paths = []
    kinds = ("", "_inv") if scheme == "inverse" else ("",)
    for kind in kinds:
        for band in ("lo", "hi"):
            base = scratch / f"matched_{model}{kind}_{band}.npz"
            if base.exists():
                paths.append(base)
            options = [scratch / "extension" / f"ext_{model}{kind}_{band}.npz",
                       scratch / f"ext_{model}{kind}_{band}.npz"]
            existing = [path for path in options if path.exists()]
            if len(existing) > 1:
                raise ValueError(f"duplicate extension locations: {existing}")
            paths.extend(existing)
    if not paths:
        raise ValueError(f"{model}: no generated arrays in {scratch}")
    return paths


def load_prompt_table(scratch: Path, model: str) -> np.ndarray:
    options = [Path(scratch) / "extension" / f"prompts_{model}.npy",
               Path(scratch) / f"prompts_{model}.npy"]
    existing = [path for path in options if path.exists()]
    if len(existing) > 1:
        raise ValueError(f"ambiguous prompt tables: {existing}")
    path = existing[0] if existing else None
    # load_prompts checks the sidecar/digest, tokenizer/C4 pins and released
    # leading rows; the selected per-block digest is checked again below.
    if path is None:
        import real_data_experiment as real
        released, _ = real.load_released_model_data(model, real.DEFAULT_DATA_DIR)
        return np.asarray(released.prompts)
    with path.open("rb") as handle:
        table = np.load(handle, allow_pickle=False)
    args = SimpleNamespace(model=model, prompts=path, prompt_offset=0, documents=len(table))
    selected, _ = gen.load_prompts(args)
    return selected.numpy()


def paired_cells(paths, prompts, model, *, scheme="gumbel", allow_legacy=False):
    """Join contiguous blocks only after verifying each arm's global row IDs."""
    if scheme not in {"gumbel", "inverse"}:
        raise ValueError("scheme must be gumbel or inverse")
    paths = [Path(path) for path in paths]
    if not paths or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("archive paths must be nonempty and distinct")
    prompts = np.asarray(prompts)
    vocab = gen.SPECS[model][1]
    if (prompts.ndim != 2 or prompts.shape[1] != 50 or
            not np.issubdtype(prompts.dtype, np.integer) or
            np.any((prompts < 0) | (prompts >= vocab))):
        raise ValueError("invalid prompt table")
    archives = [read_archive(path, allow_legacy=allow_legacy) for path in paths]
    if len({archive.metadata is None for archive in archives}) != 1:
        raise ValueError("cannot mix historical legacy archives with manifested generation")
    fragments = {}
    identity = None
    varying = {"arms", "array_sha256", "prompt_source", "prompt_table_sha256", "selected_prompt_sha256",
               "prompt_offset", "documents", "tokens", "batch"}
    for archive in archives:
        if archive.model != model:
            raise ValueError("archive model does not match requested model")
        stop = archive.offset + archive.documents
        if stop > len(prompts):
            raise ValueError("archive prompt offsets exceed available table")
        selected_digest = gen.prompt_sha256(prompts[archive.offset:stop])
        if archive.metadata is not None:
            meta = archive.metadata
            if selected_digest != meta["selected_prompt_sha256"]:
                raise ValueError("selected prompt digest/offset mismatch")
            current = {key: value for key, value in meta.items() if key not in varying}
            if identity is not None and current != identity:
                raise ValueError("incompatible generation identities across archives")
            identity = current
        else:
            expected = LEGACY_PROMPT_BLOCK_SHA256[(model, archive.offset, archive.documents)]
            if selected_digest != expected:
                raise ValueError("historical prompt block digest/offset mismatch")
            archive.provenance["audited_prompt_block_sha256"] = expected
        tags = {key.split("__")[0] for key in archive.arrays}
        for tag in tags:
            fields = {key.split("__", 1)[1]: value for key, value in archive.arrays.items()
                      if key.startswith(tag + "__")}
            temperature = (archive.metadata["arms"][tag]["temperature"] if archive.metadata is not None
                           else float(tag.split("_")[0][1:]))
            fragments.setdefault(tag, []).append((archive.offset, stop, fields, archive.provenance, temperature))
    assembled = {}
    for tag, blocks in fragments.items():
        blocks.sort(key=lambda block: block[0])
        end = 0
        fields = set(blocks[0][2])
        width = next(iter(blocks[0][2].values())).shape[1]
        actual_temperature = blocks[0][4]
        for start, stop, values, _, temperature in blocks:
            if start != end:
                raise ValueError(f"{tag}: duplicate/overlapping blocks or gap in global prompt offsets")
            if set(values) != fields or any(value.shape != (stop - start, width) for value in values.values()):
                raise ValueError(f"{tag}: incompatible companion arrays or token budgets")
            if temperature != actual_temperature:
                raise ValueError(f"{tag}: unequal temperatures share a rounded arm tag")
            end = stop
        assembled[tag] = ({field: np.concatenate([block[2][field] for block in blocks]) for field in fields},
                          np.arange(end), [block[3] for block in blocks], actual_temperature)
    method = "gumbel" if scheme == "gumbel" else "transform"
    raw_temps = {tag.split("_")[0][1:] for tag in assembled if tag.endswith("_raw")}
    wm_temps = {tag.split("_")[0][1:] for tag in assembled if tag.endswith("_" + method)}
    if not raw_temps or raw_temps != wm_temps:
        raise ValueError("raw and watermarked temperature cells are incomplete or unpaired")
    cells = []
    for temperature in sorted(raw_temps, key=float):
        raw, raw_ids, raw_prov, raw_temperature = assembled[f"t{temperature}_raw"]
        wm, wm_ids, wm_prov, wm_temperature = assembled[f"t{temperature}_{method}"]
        if raw_temperature != wm_temperature:
            raise ValueError(f"t{temperature}: raw/watermarked sampling temperatures differ")
        if not np.array_equal(raw_ids, wm_ids) or raw["tokens"].shape != wm["tokens"].shape:
            raise ValueError(f"t{temperature}: raw/watermarked prompt pairing or token budgets differ")
        provenance = list({item["path"]: item for item in raw_prov + wm_prov}.values())
        cells.append(PairedCell(temperature, raw, wm, prompts[raw_ids], raw_ids, provenance))
    return cells


def load_cells(scratch, model, *, scheme="gumbel", allow_legacy=False):
    return paired_cells(discover_archives(scratch, model, scheme),
                        load_prompt_table(scratch, model), model,
                        scheme=scheme, allow_legacy=allow_legacy)
