#!/usr/bin/env python3
"""Reanalyse the released real-model watermark outputs without an LM runtime.

The commit-pinned release contains the watermarked pivots and the raw-model
token continuations needed for detection.  This script therefore does *not*
load model weights, a tokenizer, or C4.  PyTorch is used only after each pickle
has passed a SHA256 allowlist check, and then only to deserialize tensors and to
replay the release's CPU key/hash/random-number operations for the raw tokens.

The primary inverse-transform analysis uses the formal rank convention
``eta(j)=j/(V-1)``.  The released inverse pickles instead store
``(j-1)/(V-1)``; their watermarked pivots are corrected by adding ``1/(V-1)``
to the stored eta.  A clearly labelled sensitivity analysis retains the
released shifted convention.

Fixed-horizon thresholds are calibrated on one independent exact-pivot-null
sample per model, scheme, and inverse convention.  The 500 released raw-model
continuations are a separate empirical Type-I evaluation, not calibration
data.  The 500 watermarked continuations provide the power evaluation.

By default, the Bayesian rows use the same ``Uniform(0.001, 0.5)`` deficit
prior as the synthetic experiments.  Command-line bounds permit isolated
real-output sensitivity analyses without changing that default.

The rule menu also carries Tr-GoF at ``s = 2`` for both schemes.  It is the one
competitor here that is not a token sum, so it is recomputed at every reported
horizon rather than accumulated.  Its Gumbel p-value ``p = 1 - Y`` is the
authors'; its inverse p-value ``p = F_0(d)`` at the released vocabulary is our
extension, since their code covers Gumbel only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import shutil
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiment_metadata import union_tail_metadata

import benchmark_paper_experiment as benchmark
import dirichlet_detector as dirichlet
import trgof


OFFICIAL_COMMIT = "05b7ffda9279fc9e645f38807e4a0e2dbcff4330"
OFFICIAL_TREE_URL = (
    "https://github.com/lx10077/WatermarkFramework/tree/"
    f"{OFFICIAL_COMMIT}/real%20data"
)
OFFICIAL_RAW_BASE = (
    "https://raw.githubusercontent.com/lx10077/WatermarkFramework/"
    f"{OFFICIAL_COMMIT}/real%20data/results_data"
)

# Pickles can execute code while loading.  Never expand this mapping from a
# remote manifest at runtime: the fixed filename/digest pairs are the trust
# boundary for the six released artifacts used here.
ASSET_SHA256: dict[str, str] = {
    "1p3B-gumbel-c4-m200-T500-skipgram_prf-15485863-temp0.1.pkl":
        "760f50b4fa185f17679144dd2d51418865f4a5cf925557bf4978ba0dca6ebcb7",
    "1p3B-raw-m200-T500.pkl":
        "d70b40fe31b053e7bf7eb73efe29795125d6c8075a00144ba3017cb3b8ba862a",
    "1p3B-transform-c4-m200-T500-skipgram_prf-15485863-temp0.1.pkl":
        "6bcb8fa45365f6fae6ff5521efd41faeb3c73204c0d00996505d1bec9cccfa13",
    "2p7B-gumbel-c4-m200-T500-skipgram_prf-15485863-temp0.1.pkl":
        "0b33c4eeca1b3a1e6f41c001113896dadeedb238799bc8922d70883551f7fb67",
    "2p7B-raw-m200-T500.pkl":
        "d04f82b584d973cc4346e65e7a6d5529741d57754ee161ab35b3a78e90962d82",
    "2p7B-transform-c4-m200-T500-skipgram_prf-15485863-temp0.1.pkl":
        "80fa7a6d81ebfdf2d37efc077917cdca660e62ced6b527cb92ec66f8bf1a5136",
}

EXPECTED_REPLAY_SHA256: dict[str, dict[str, str]] = {
    "1p3B": {
        "gumbel": "411df3d79b2f4771ac40a4c592cc14c0f9aafdde1534d434f3c714e6a1174aed",
        "inverse_formal": "b8b9f079d8050336858441baf11e35ffa8aaa4d8e6d94d9adb541e533c3ebba6",
        # Filled from the literal torch-float32 transform_Y arithmetic.
        "inverse_shifted": "4553e3107a7ce8a97b3cf0a555d39f49033f52cb5202b2cc2980e695791527b1",
    },
    "2p7B": {
        "gumbel": "5ef407b091b098181a6f58e5e6a0a0d686fbf999c40f281823f7ad51e6893e68",
        "inverse_formal": "33d76c6984f9b6ac9eb1ea1330a6937f656d75d2e5b035e46ccb05bb843f7153",
        "inverse_shifted": "7933a06cdf11a6480ab5b4a579a64cfd2779c5dd5386102b70a60e1a1370ebaf",
    },
}

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "1p3B": {
        "model_id": "facebook/opt-1.3b",
        "vocabulary_size": 50_272,
    },
    "2p7B": {
        "model_id": "princeton-nlp/Sheared-LLaMA-2.7B",
        "vocabulary_size": 32_000,
    },
}
MODEL_CODES = {name: index for index, name in enumerate(MODEL_SPECS)}

# The enlarged Gumbel layer places a prior on the number of live tail
# coordinates J instead of fixing the tail to be equal over all V-1 of them.
# J = V-1 is exactly the equal-tail spike that every other Gumbel rule assumes,
# so the family contains them.  The component is closed form, so unlike the
# Dirichlet tail-shape layer it needs no transform table and no interpolation.
# Shared hierarchy only: the tokenwise rule averages over J before multiplying
# and so cannot learn a document-level tail width.  Gumbel only, exactly like
# the Dirichlet layer: the inverse limiting alternative depends on Delta alone,
# so a prior on the tail is inert for it by construction.
# Tr-GoF of \citet{li2024robust}: the one competitor here that is not a token
# sum.  It sorts the p-values of the entire prefix and maximises a phi-
# divergence over their order statistics, so it is recomputed at every reported
# horizon instead of accumulated.  ``trgof`` owns the statistic; this file only
# supplies pivots, p-values, and the shared calibration.
TRGOF_S = trgof.DEFAULT_S
TRGOF_METHOD = "trgof_s2"
if TRGOF_S != 2.0:  # pragma: no cover - guards the frozen method name
    raise RuntimeError(
        "trgof.DEFAULT_S is no longer 2, but the rule name and label say s=2"
    )

GUMBEL_METHODS = benchmark.GUMBEL_METHODS + (
    "h_gum_star_0.1",
    "bayes_shared",
    benchmark.DIRICHLET_SHARED_METHOD,
    benchmark.UNIONTAIL_SHARED_METHOD,
    TRGOF_METHOD,
)
INVERSE_METHODS = benchmark.INVERSE_METHODS + ("bayes_shared", TRGOF_METHOD)

METHOD_LABELS = {
    "h_ars": "h_ars",
    "h_log": "h_log",
    "h_ind_1_over_e": "h_ind,1/e",
    "h_gum_star_0.01": "h*_gum,.01",
    "h_gum_star_0.005": "h*_gum,.005",
    "h_gum_star_0.1": "h*_gum,.1",
    "h_spike_0.01": "h^sp_.01",
    "h_spike_0.05": "h^sp_.05",
    "h_neg": "h_neg",
    "h_dif_star_0.1": "h*_dif,.1",
    "h_dif_star_0.01": "h*_dif,.01",
    "h_dif_star_0.001": "h*_dif,.001",
    "bayes_tokenwise": "Bayes, tokenwise Delta",
    "bayes_shared": "Bayes, shared Delta",
    benchmark.DIRICHLET_TOKENWISE_METHOD:
        "Bayes, tokenwise Delta + Dirichlet alpha",
    benchmark.DIRICHLET_SHARED_METHOD:
        "Bayes, shared Delta + Dirichlet alpha",
    benchmark.UNIONTAIL_SHARED_METHOD:
        "Bayes, shared Delta + union tail",
    TRGOF_METHOD: "Tr-GoF, s=2 (HC)",
}

METHOD_ORIGINS = {
    "h_ars": "reference_score",
    "h_log": "reference_score",
    "h_ind_1_over_e": "reference_score",
    "h_gum_star_0.01": "reference_score",
    "h_gum_star_0.005": "reference_score",
    "h_gum_star_0.1": "reference_score",
    "h_neg": "reference_score",
    "h_dif_star_0.1": "reference_score",
    "h_dif_star_0.01": "reference_score",
    "h_dif_star_0.001": "reference_score",
    "h_spike_0.01": "diagnostic_fixed_spike_likelihood",
    "h_spike_0.05": "diagnostic_fixed_spike_likelihood",
    "bayes_tokenwise": "bayesian",
    "bayes_shared": "bayesian",
    benchmark.DIRICHLET_TOKENWISE_METHOD: "bayesian_dirichlet_tail",
    benchmark.DIRICHLET_SHARED_METHOD: "bayesian_dirichlet_tail",
    benchmark.UNIONTAIL_SHARED_METHOD: "bayesian_union_tail",
    # Its own origin tag: Tr-GoF is neither one of the Li et al. (2025)
    # sum-based reference scores nor a Bayesian rule of this study.
    TRGOF_METHOD: "reference_score_trgof",
}

PRIMARY_INVERSE_CONVENTION = "formal_eta_j_over_v_minus_1"
SHIFTED_INVERSE_CONVENTION = "upstream_shifted_eta_j_minus_1_over_v_minus_1"
GUMBEL_CONVENTION = "stored_gumbel_pivot"

DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "bayesian_paper_benchmark"
    / "real_model"
)
DEFAULT_DATA_DIR = DEFAULT_RESULTS_DIR / "upstream_assets"

PRF_TABLE_SIZE = 1_000_003
PRF_TABLE_SEED = 2_971_215_073
KEY_SEED = 15_485_863
CONTEXT_WIDTH = 4
MAX_HORIZON = 200

UPSTREAM_GENERATION_DESIGN = {
    "dataset": "C4 realnewslike streaming train split",
    "document_selection": (
        "first 500 stream records with sufficient tokenized length; no shuffle"
    ),
    "prompt_tokens": 50,
    "scored_continuation_tokens": MAX_HORIZON,
    "raw_continuation_tokens_stored": 220,
    "watermarked_temperature": 0.1,
    "raw_temperature": 1.0,
    "raw_temperature_note": (
        "the released raw-generation function does not apply the command-line "
        "temperature, so it samples from unscaled logits"
    ),
    "n_documents_per_model": 500,
    "one_generation_per_prompt": True,
    "model_revisions_pinned_by_upstream": False,
    "dataset_revision_pinned_by_upstream": False,
}


@dataclass(frozen=True)
class RealDataConfig:
    alpha: float = 0.05
    horizons: tuple[int, ...] = (50, 100, 200)
    n_calibration: int = 10_000
    calibration_seed: int = 240_401_253
    delta_prior: str = "uniform"
    # Prior support only; the released outputs are fixed data, so there is no
    # generating law to keep separate here.
    delta_low: float = 0.001
    delta_high: float = 0.5
    bayes_quadrature_nodes: int = 96
    gumbel_lookup_size: int = 80_001
    gumbel_lookup_logit_limit: float = 30.0
    dirichlet_c_nodes: int = dirichlet.DEFAULT_C_NODES
    score_batch_size: int = 500

    def validate(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0,1)")
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be strictly increasing and distinct")
        if self.horizons[0] < 1 or self.horizons[-1] > MAX_HORIZON:
            raise ValueError(f"horizons must lie in 1..{MAX_HORIZON}")
        if self.n_calibration < 1:
            raise ValueError("n_calibration must be positive")
        if self.delta_prior != "uniform":
            raise ValueError("the released-output analysis requires uniform Delta")
        if not 0.0 <= self.delta_low < self.delta_high < 1.0:
            raise ValueError("require 0 <= delta_low < delta_high < 1")
        if self.bayes_quadrature_nodes < 8:
            raise ValueError("bayes_quadrature_nodes must be at least 8")
        if self.gumbel_lookup_size < 1_001:
            raise ValueError("gumbel_lookup_size must be at least 1001")
        if self.dirichlet_c_nodes < 16:
            raise ValueError("dirichlet_c_nodes must be at least 16")
        if self.score_batch_size < 1:
            raise ValueError("score_batch_size must be positive")


@dataclass(frozen=True)
class ReleasedModelData:
    model_name: str
    vocabulary_size: int
    prompts: np.ndarray
    raw_tokens: np.ndarray
    gumbel_watermarked: np.ndarray
    inverse_watermarked_formal: np.ndarray
    inverse_watermarked_shifted: np.ndarray
    top_probabilities_gumbel: np.ndarray
    top_probabilities_inverse: np.ndarray
    # The realized watermarked tokens.  Needed to recover the PRF address of
    # each position -- the released skipgram seeds from the token four back --
    # which is the grouping any cluster-aware analysis of the watermarked arm
    # has to use.
    gumbel_tokens: np.ndarray


def asset_names(model_name: str) -> dict[str, str]:
    if model_name not in MODEL_SPECS:
        raise KeyError(model_name)
    prefix = f"{model_name}"
    common = "c4-m200-T500-skipgram_prf-15485863-temp0.1.pkl"
    return {
        "gumbel": f"{prefix}-gumbel-{common}",
        "raw": f"{prefix}-raw-m200-T500.pkl",
        "inverse": f"{prefix}-transform-{common}",
    }


def file_sha256(path: Path, *, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_asset(
    path: Path, *, allowlisted_name: str | None = None
) -> dict[str, Any]:
    filename = path.name if allowlisted_name is None else allowlisted_name
    expected = ASSET_SHA256.get(filename)
    if expected is None:
        raise ValueError(f"{filename!r} is not one of the six allowlisted pickles")
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = file_sha256(path)
    if observed != expected:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {expected}, observed {observed}; "
            "the pickle was not loaded"
        )
    return {
        "filename": filename,
        "sha256": observed,
        "size_bytes": path.stat().st_size,
        "upstream_url": f"{OFFICIAL_RAW_BASE}/{filename}",
    }


def ensure_assets(
    data_dir: Path,
    *,
    download_missing: bool,
    filenames: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    provenance: list[dict[str, Any]] = []
    requested = tuple(ASSET_SHA256) if filenames is None else tuple(filenames)
    if set(requested) - set(ASSET_SHA256):
        raise ValueError("requested filenames include a non-allowlisted asset")
    for filename in requested:
        path = data_dir / filename
        if not path.exists():
            if not download_missing:
                raise FileNotFoundError(
                    f"missing {path}; omit --no-download to fetch the commit-pinned asset"
                )
            temporary = path.with_name(f".{path.name}.part")
            try:
                with urllib.request.urlopen(
                    f"{OFFICIAL_RAW_BASE}/{filename}", timeout=120
                ) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
                # Verify before making the download visible under an allowlisted name.
                verify_asset(temporary, allowlisted_name=filename)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            os.replace(temporary, path)
        provenance.append(verify_asset(path))
    return provenance


def _torch_module():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required only for the verified pickle load and exact "
            "CPU PRF replay. Install a CPU build of torch; transformers, datasets, "
            "model weights, C4, and a GPU are not required."
        ) from exc
    return torch


def load_verified_pickle(path: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    provenance = verify_asset(path)
    # Import torch only after the exact bytes have passed the fixed digest check;
    # pickle.load needs torch's tensor reconstruction functions to be importable.
    _torch_module()
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"unexpected top-level object in {path.name}")
    return value, provenance


def _as_numpy(value: Any, *, dtype: np.dtype[Any] | type | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=dtype)
    return np.ascontiguousarray(array)


def _require_shape(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} has shape {value.shape}, expected {shape}")


def _namespace_value(container: Mapping[str, Any], name: str) -> Any:
    args = container.get("args")
    if args is None or not hasattr(args, name):
        raise ValueError(f"released pickle is missing args.{name}")
    return getattr(args, name)


def extract_released_model_data(
    model_name: str,
    raw: Mapping[str, Any],
    gumbel: Mapping[str, Any],
    inverse: Mapping[str, Any],
) -> ReleasedModelData:
    spec = MODEL_SPECS[model_name]
    vocabulary_size = int(spec["vocabulary_size"])
    expected_model_id = spec["model_id"]
    for label, value in (("raw", raw), ("gumbel", gumbel), ("inverse", inverse)):
        if _namespace_value(value, "model") != expected_model_id:
            raise ValueError(f"{label} pickle has an unexpected model identifier")
        if int(_namespace_value(value, "m")) != MAX_HORIZON:
            raise ValueError(f"{label} pickle does not contain m={MAX_HORIZON}")
        if int(_namespace_value(value, "T")) != 500:
            raise ValueError(f"{label} pickle does not contain T=500")

    prompts_raw = _as_numpy(raw["prompts"], dtype=np.int64)
    prompts_gumbel = _as_numpy(gumbel["prompts"], dtype=np.int64)
    prompts_inverse = _as_numpy(inverse["prompts"], dtype=np.int64)
    _require_shape("raw prompts", prompts_raw, (500, 50))
    if not np.array_equal(prompts_raw, prompts_gumbel):
        raise ValueError("raw and Gumbel prompt tensors differ")
    if not np.array_equal(prompts_raw, prompts_inverse):
        raise ValueError("raw and inverse prompt tensors differ")

    raw_tokens = _as_numpy(raw["null"]["tokens"], dtype=np.int64)
    _require_shape("raw continuations", raw_tokens, (500, 220))
    gumbel_y = _as_numpy(gumbel["watermark"]["Ys"], dtype=float)
    _require_shape("Gumbel watermarked pivots", gumbel_y, (500, MAX_HORIZON))
    gumbel_tokens = _as_numpy(gumbel["watermark"]["tokens"], dtype=np.int64)
    _require_shape("Gumbel watermarked tokens", gumbel_tokens, (500, MAX_HORIZON))

    inverse_watermark = inverse["watermark"]
    inverse_u_float32 = _as_numpy(inverse_watermark["Us"], dtype=np.float32)
    inverse_eta_shifted_float32 = _as_numpy(
        inverse_watermark["etas"], dtype=np.float32
    )
    inverse_u = inverse_u_float32.astype(float)
    inverse_eta_shifted = inverse_eta_shifted_float32.astype(float)
    _require_shape("inverse U", inverse_u, (500, MAX_HORIZON))
    _require_shape("inverse shifted eta", inverse_eta_shifted, (500, MAX_HORIZON))
    reconstructed_rank_float = inverse_eta_shifted * (vocabulary_size - 1.0) + 1.0
    reconstructed_rank = np.rint(reconstructed_rank_float).astype(np.int64)
    reconstruction_error = np.abs(reconstructed_rank_float - reconstructed_rank)
    if float(reconstruction_error.max()) > 0.01:
        raise ValueError("stored inverse eta does not reconstruct integer ranks")
    if np.any((reconstructed_rank < 0) | (reconstructed_rank >= vocabulary_size)):
        raise ValueError("reconstructed inverse ranks fall outside 0..V-1")
    inverse_eta_formal = reconstructed_rank / (vocabulary_size - 1.0)
    inverse_formal = np.abs(inverse_u - inverse_eta_formal)
    # Preserve the float32 subtraction used by the released scoring script.
    inverse_shifted = np.abs(
        inverse_u_float32 - inverse_eta_shifted_float32
    ).astype(float)

    top_gumbel = _as_numpy(gumbel["watermark"]["top_probs"], dtype=float)
    top_inverse = _as_numpy(inverse_watermark["top_probs"], dtype=float)
    _require_shape("Gumbel top probabilities", top_gumbel, (500, MAX_HORIZON))
    _require_shape("inverse top probabilities", top_inverse, (500, MAX_HORIZON))
    for name, values in (
        ("Gumbel top probabilities", top_gumbel),
        ("inverse top probabilities", top_inverse),
    ):
        if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{name} fall outside [0,1]")

    for name, values in (
        ("raw tokens", raw_tokens),
        ("Gumbel watermarked tokens", gumbel_tokens),
        ("prompts", prompts_raw),
    ):
        if np.any(values < 0) or np.any(values >= vocabulary_size):
            raise ValueError(f"{name} contain ids outside the declared vocabulary")
    if np.any((gumbel_y < 0.0) | (gumbel_y > 1.0)):
        raise ValueError("stored Gumbel pivots fall outside [0,1]")
    for name, values in (
        ("formal inverse pivots", inverse_formal),
        ("shifted inverse pivots", inverse_shifted),
    ):
        if np.any((values < 0.0) | (values >= 1.0)):
            raise ValueError(f"{name} fall outside [0,1)")

    return ReleasedModelData(
        model_name=model_name,
        vocabulary_size=vocabulary_size,
        prompts=prompts_raw,
        raw_tokens=raw_tokens,
        gumbel_watermarked=gumbel_y,
        inverse_watermarked_formal=inverse_formal,
        inverse_watermarked_shifted=inverse_shifted,
        top_probabilities_gumbel=top_gumbel,
        top_probabilities_inverse=top_inverse,
        gumbel_tokens=gumbel_tokens,
    )


def load_released_model_data(
    model_name: str, data_dir: Path
) -> tuple[ReleasedModelData, list[dict[str, Any]]]:
    names = asset_names(model_name)
    loaded: dict[str, Mapping[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    for role, filename in names.items():
        loaded[role], record = load_verified_pickle(data_dir / filename)
        record["role"] = role
        record["model_name"] = model_name
        provenance.append(record)
    return (
        extract_released_model_data(
            model_name, loaded["raw"], loaded["gumbel"], loaded["inverse"]
        ),
        provenance,
    )


def prf_addresses(
    prompts: np.ndarray, tokens: np.ndarray, *, horizon: int = MAX_HORIZON
) -> np.ndarray:
    """The token that addresses the keyed draw at each position.

    The released skipgram function hashes the OLDEST token of the four-token
    window, so position ``t`` is seeded by the token four back.  Positions
    sharing this value share a pseudorandom vector and are not independent; any
    analysis that treats released positions as an i.i.d. sample has to cluster
    on it.  Returns an array shaped like the first ``horizon`` columns of
    ``tokens``.
    """

    prompts_array = np.asarray(prompts)
    tokens_array = np.asarray(tokens)
    if prompts_array.shape[1] < CONTEXT_WIDTH:
        raise ValueError("prompts are shorter than the PRF context width")
    return np.concatenate(
        (prompts_array[:, -CONTEXT_WIDTH:], tokens_array[:, :horizon]), axis=1
    )[:, :horizon]


def _official_fixed_hash_table(torch: Any):
    generator = torch.Generator(device=torch.device("cpu"))
    generator.manual_seed(PRF_TABLE_SEED)
    return torch.randperm(PRF_TABLE_SIZE, device="cpu", generator=generator)


def replay_raw_pivots(
    prompts: np.ndarray,
    raw_tokens: np.ndarray,
    *,
    vocabulary_size: int,
    horizon: int = MAX_HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay official skipgram PRF semantics on released raw continuations.

    Returns Gumbel pivots, formal inverse pivots, and upstream-shifted inverse
    pivots.  The same prompt/raw-token tensors are used in all three replays.
    """

    torch = _torch_module()
    prompts_t = torch.as_tensor(prompts, dtype=torch.long, device="cpu")
    tokens_t = torch.as_tensor(raw_tokens, dtype=torch.long, device="cpu")
    if prompts_t.ndim != 2 or tokens_t.ndim != 2:
        raise ValueError("prompts and raw_tokens must be two-dimensional")
    if prompts_t.shape[0] != tokens_t.shape[0]:
        raise ValueError("prompts and raw_tokens must have the same row count")
    if prompts_t.shape[1] < CONTEXT_WIDTH or tokens_t.shape[1] < horizon:
        raise ValueError("insufficient context or continuation length")
    if vocabulary_size < 2:
        raise ValueError("vocabulary_size must be at least two")

    rows = int(tokens_t.shape[0])
    # At time j, official seed_rng selects the oldest token in the last c-token
    # context because skipgram_prf hashes input_ids[0].  These are therefore the
    # first `horizon` entries of prompt[-c:] concatenated with the continuation.
    seed_context = torch.cat(
        (prompts_t[:, -CONTEXT_WIDTH:], tokens_t[:, :horizon]), dim=1
    )[:, :horizon]
    fixed_table = _official_fixed_hash_table(torch)
    hashed_seeds = fixed_table[(KEY_SEED * seed_context) % PRF_TABLE_SIZE] + 1

    gumbel = np.empty((rows, horizon), dtype=float)
    inverse_formal = np.empty_like(gumbel)
    inverse_shifted = np.empty_like(gumbel)
    generator = torch.Generator(device=torch.device("cpu"))
    denominator = vocabulary_size - 1
    for row in range(rows):
        for column in range(horizon):
            seed = int(hashed_seeds[row, column].item())
            token = int(tokens_t[row, column].item())

            generator.manual_seed(seed)
            xi = torch.rand((vocabulary_size,), generator=generator)
            gumbel[row, column] = float(xi[token].item())

            generator.manual_seed(seed)
            u = torch.rand((1,), generator=generator)[0]
            permutation = torch.randperm(vocabulary_size, generator=generator)
            rank = permutation[token]
            # Keep the shifted sensitivity arithmetic in torch as in the
            # released scorer.
            eta_shifted = (rank - 1) / denominator
            # Promote the stored float32 U value to Python/NumPy float64 and
            # use the exact integer rank for the primary formal convention.
            inverse_formal[row, column] = abs(
                float(u.item()) - int(rank.item()) / denominator
            )
            inverse_shifted[row, column] = float(torch.abs(u - eta_shifted).item())

    return gumbel, inverse_formal, inverse_shifted


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def validate_replay_hashes(
    model_name: str,
    gumbel: np.ndarray,
    inverse_formal: np.ndarray,
    inverse_shifted: np.ndarray,
) -> dict[str, str]:
    observed = {
        "gumbel": _array_sha256(gumbel),
        "inverse_formal": _array_sha256(inverse_formal),
        "inverse_shifted": _array_sha256(inverse_shifted),
    }
    expected = EXPECTED_REPLAY_SHA256[model_name]
    for role, digest in observed.items():
        if digest != expected[role]:
            raise RuntimeError(
                f"fail-closed PRF replay mismatch for {model_name} {role}: "
                f"expected {expected[role]}, observed {digest}"
            )
    return observed


def _row_logsumexp(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=1)
    output = np.full(maximum.shape, -np.inf, dtype=float)
    finite = np.isfinite(maximum)
    if np.any(finite):
        output[finite] = maximum[finite] + np.log(
            np.exp(values[finite] - maximum[finite, None]).sum(axis=1)
        )
    return output


def shared_spike_paths(
    pivots: np.ndarray,
    *,
    scheme: str,
    vocabulary_size: int,
    deltas: np.ndarray,
    delta_weights: np.ndarray,
    horizons: Sequence[int],
) -> np.ndarray:
    values = np.asarray(pivots, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    delta_weights = np.asarray(delta_weights, dtype=float)
    wanted = tuple(int(h) for h in horizons)
    if values.ndim != 2 or not wanted or wanted[-1] > values.shape[1]:
        raise ValueError("invalid pivot array or horizons")
    if deltas.ndim != 1 or delta_weights.shape != deltas.shape:
        raise ValueError("deltas and delta_weights must be matching vectors")
    if np.any((deltas <= 0.0) | (deltas >= 1.0)):
        raise ValueError("deltas must lie in (0,1)")
    if np.any(delta_weights <= 0.0) or not np.isclose(delta_weights.sum(), 1.0):
        raise ValueError("delta_weights must be positive and sum to one")
    log_weights = np.log(delta_weights)
    accumulator = np.zeros((values.shape[0], deltas.size), dtype=float)
    output = np.empty((values.shape[0], len(wanted)), dtype=float)
    record = {h - 1: index for index, h in enumerate(wanted)}

    if scheme == "gumbel":
        top_exponents = deltas / (1.0 - deltas)
        tail_exponents = (vocabulary_size - 1.0) / deltas - 1.0
        log_tail_count = math.log(vocabulary_size - 1.0)
    elif scheme == "inverse":
        inverse_scales = 1.0 - deltas
        inverse_constants = np.log(2.0 / inverse_scales)
    else:
        raise ValueError("scheme must be 'gumbel' or 'inverse'")

    for time_index in range(wanted[-1]):
        observation = values[:, time_index, None]
        if scheme == "gumbel":
            log_r = np.log(observation)
            component = np.logaddexp(
                log_r * top_exponents[None, :],
                log_tail_count + log_r * tail_exponents[None, :],
            )
        else:
            inside = 1.0 - observation / inverse_scales[None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                log_alt = np.where(
                    inside > 0.0,
                    inverse_constants[None, :] + np.log(inside),
                    -np.inf,
                )
            log_null = np.log(
                benchmark.inverse_exact_null_density(
                    observation[:, 0], vocabulary_size
                )
            )
            component = log_alt - log_null[:, None]
        accumulator += component
        if time_index in record:
            output[:, record[time_index]] = _row_logsumexp(
                accumulator + log_weights[None, :]
            )
    return output


def trgof_paths(
    p_values: np.ndarray,
    *,
    horizons: Sequence[int],
    s: float = TRGOF_S,
) -> np.ndarray:
    """Tr-GoF evaluated separately on each reported prefix.

    The statistic sorts the p-values of the whole prefix, so unlike every
    sum-based score it cannot be accumulated with ``cumsum``; each horizon is a
    fresh maximisation over that prefix's own order statistics.  Like every
    other collector here it reads its argument and touches no generator, so
    adding this column cannot perturb the streams that drive the other rules.
    """

    values = np.asarray(p_values, dtype=float)
    wanted = tuple(int(h) for h in horizons)
    if values.ndim != 2 or not wanted or wanted[-1] > values.shape[1]:
        raise ValueError("invalid p-value array or horizons")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Tr-GoF p-values must be finite and lie in [0,1]")
    output = np.empty((values.shape[0], len(wanted)), dtype=float)
    for index, horizon in enumerate(wanted):
        output[:, index] = trgof.statistic(values[:, :horizon], s=s)
    return output


def _benchmark_config(
    config: RealDataConfig, vocabulary_size: int
) -> benchmark.BenchmarkConfig:
    return benchmark.BenchmarkConfig(
        vocabulary_size=vocabulary_size,
        max_horizon=MAX_HORIZON,
        alpha=config.alpha,
        # The released outputs are fixed data, so this config's bounds are the
        # detector's prior; there is no generating law here to keep separate.
        prior_low=config.delta_low,
        prior_high=config.delta_high,
        n_calibration=config.n_calibration,
        n_evaluation_null=500,
        n_evaluation_alternative=500,
        batch_size=config.score_batch_size,
        seed=config.calibration_seed,
        bayes_quadrature_nodes=config.bayes_quadrature_nodes,
        gumbel_lookup_size=config.gumbel_lookup_size,
        gumbel_lookup_logit_limit=config.gumbel_lookup_logit_limit,
        dirichlet_c_nodes=config.dirichlet_c_nodes,
    )


class ModelScorers:
    def __init__(self, config: RealDataConfig, vocabulary_size: int) -> None:
        self.config = config
        self.vocabulary_size = vocabulary_size
        self.benchmark_config = _benchmark_config(config, vocabulary_size)
        # Every spike-family rule must integrate the same prior the Gumbel
        # lookup does, i.e. the support clamped to 1 - 1/V.
        self.deltas, self.delta_weights = dirichlet.gauss_legendre_delta_grid(
            *benchmark.effective_prior_support(self.benchmark_config),
            config.bayes_quadrature_nodes,
        )
        # NumPy 2.0/Accelerate can emit spurious floating-point warnings from
        # this finite matrix product at V=50,272.  Check the completed table
        # explicitly and fail if a non-finite value was actually produced.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            self.gumbel_lookup = benchmark.GumbelBayesLookup(
                self.benchmark_config
            )
        if not np.all(np.isfinite(self.gumbel_lookup.log_density)):
            raise FloatingPointError("non-finite Gumbel Bayes lookup")
        self.dirichlet_grid = dirichlet.DirichletBayesGrid(
            delta_grid=self.deltas,
            delta_weights=self.delta_weights,
            alpha_grid=benchmark.DIRICHLET_ALPHA_GRID,
            tail_size=vocabulary_size - 1,
            c_nodes=config.dirichlet_c_nodes,
            allow_large_delta=False,
        )
        self.dirichlet_lookup = self.dirichlet_grid.tokenwise_lookup(
            logit_limit=config.gumbel_lookup_logit_limit,
            size=config.gumbel_lookup_size,
        )
        # The union reuses the same Gauss-Legendre deficit nodes and combines
        # the full-width Dirichlet shape branch with a restricted-width branch.
        # ``tail_width_grid=None`` selects 1, 4, 16, ... strictly below V-1 for
        # this model's released vocabulary. The full-width equal-tail atom
        # belongs to the shape branch (alpha=inf), not the width ladder.
        # This experiment evaluates the shared hierarchy, without a tokenwise
        # union lookup.
        self.uniontail_grid = dirichlet.UnionTailBayesGrid(
            delta_grid=self.deltas,
            delta_weights=self.delta_weights,
            tail_size=vocabulary_size - 1,
            tail_width_grid=None,
            c_nodes=config.dirichlet_c_nodes,
            allow_large_delta=False,
        )

    def score_gumbel(self, pivots: np.ndarray) -> dict[str, np.ndarray]:
        values = np.asarray(pivots, dtype=float)
        indices = np.asarray(self.config.horizons, dtype=int) - 1
        output: dict[str, np.ndarray] = {}
        for method in benchmark.GUMBEL_METHODS:
            increments = benchmark.gumbel_score(
                values,
                method,
                self.gumbel_lookup,
                vocabulary_size=self.vocabulary_size,
                dirichlet_lookup=self.dirichlet_lookup,
            )
            output[method] = np.cumsum(increments, axis=1)[:, indices]
        output["h_gum_star_0.1"] = np.cumsum(
            benchmark.paper_gumbel_optimal_score(values, 0.1), axis=1
        )[:, indices]
        output["bayes_shared"] = shared_spike_paths(
            values,
            scheme="gumbel",
            vocabulary_size=self.vocabulary_size,
            deltas=self.deltas,
            delta_weights=self.delta_weights,
            horizons=self.config.horizons,
        )
        # Every shared-latent grid is scored on the *same* pivot array, so all
        # detectors see bit-for-bit identical inputs and no rule consumes any
        # randomness: `values` is supplied by the caller and none of these
        # collectors touch a generator.  Adding a grid here therefore cannot
        # perturb the streams that drive the existing rules.
        for method, grid in (
            (benchmark.DIRICHLET_SHARED_METHOD, self.dirichlet_grid),
            (benchmark.UNIONTAIL_SHARED_METHOD, self.uniontail_grid),
        ):
            shared_chunks: list[np.ndarray] = []
            for start in range(0, values.shape[0], self.config.score_batch_size):
                stop = min(start + self.config.score_batch_size, values.shape[0])
                paths, _ = grid.shared_paths(
                    values[start:stop], horizons=self.config.horizons
                )
                shared_chunks.append(paths)
            output[method] = np.vstack(shared_chunks)
        # Tr-GoF on the authors' own Gumbel p-value, p = 1 - Y, whose null is
        # exactly uniform.  Recomputed per horizon; see ``trgof_paths``.
        output[TRGOF_METHOD] = trgof_paths(
            trgof.gumbel_p_values(values), horizons=self.config.horizons
        )
        return output

    def score_inverse(self, pivots: np.ndarray) -> dict[str, np.ndarray]:
        values = np.asarray(pivots, dtype=float)
        indices = np.asarray(self.config.horizons, dtype=int) - 1
        output: dict[str, np.ndarray] = {}
        for method in INVERSE_METHODS:
            if method in ("bayes_shared", TRGOF_METHOD):
                continue
            increments = real_data_inverse_score(
                values, method, self.benchmark_config
            )
            output[method] = np.cumsum(increments, axis=1)[:, indices]
        output["bayes_shared"] = shared_spike_paths(
            values,
            scheme="inverse",
            vocabulary_size=self.vocabulary_size,
            deltas=self.deltas,
            delta_weights=self.delta_weights,
            horizons=self.config.horizons,
        )
        # OUR extension, not the authors': their released code applies Tr-GoF
        # to Gumbel only and defines no inverse p-value.  ``p = F_0(d)`` is the
        # exact finite-vocabulary null CDF, evaluated at *this model's* released
        # vocabulary, which reaches here through ReleasedModelData and is never
        # a literal in the Tr-GoF path.
        output[TRGOF_METHOD] = trgof_paths(
            trgof.inverse_p_values(values, self.vocabulary_size),
            horizons=self.config.horizons,
        )
        return output


def real_data_inverse_score(
    pivots: np.ndarray, method: str, config: benchmark.BenchmarkConfig
) -> np.ndarray:
    """Inverse reference scores using the released real-data implementation.

    Unlike the pinned synthetic script, the real-data script uses a ``1e-6``
    triangular-factor floor and includes ``1/(1-Delta)``.  The latter is an
    additive per-token constant on the log scale and hence cannot alter a
    separately calibrated fixed-horizon ordering, but it is retained here for
    an exact specification-level match.
    """

    values = np.asarray(pivots, dtype=float)
    if method == "h_neg":
        return -values
    delta_by_method = {
        "h_dif_star_0.1": 0.1,
        "h_dif_star_0.01": 0.01,
        "h_dif_star_0.001": 0.001,
    }
    if method in delta_by_method:
        delta = delta_by_method[method]
        numerator = np.maximum(1.0 - values / (1.0 - delta), 1e-6)
        denominator = np.maximum(1.0 - values, np.finfo(float).tiny)
        return np.log(numerator / denominator / (1.0 - delta))
    if method == "bayes_tokenwise":
        return benchmark.inverse_bayes_log_ratio(values, config)
    raise KeyError(method)


def simulate_exact_null(
    rng: np.random.Generator,
    *,
    scheme: str,
    rows: int,
    horizon: int,
    vocabulary_size: int,
    inverse_convention: str = PRIMARY_INVERSE_CONVENTION,
) -> np.ndarray:
    uniforms = rng.random((rows, horizon))
    if scheme == "gumbel":
        return uniforms
    if scheme != "inverse":
        raise ValueError("scheme must be gumbel or inverse")
    ranks = rng.integers(0, vocabulary_size, size=(rows, horizon))
    if inverse_convention == PRIMARY_INVERSE_CONVENTION:
        eta = ranks / (vocabulary_size - 1.0)
    elif inverse_convention == SHIFTED_INVERSE_CONVENTION:
        eta = (ranks - 1.0) / (vocabulary_size - 1.0)
    else:
        raise ValueError("unknown inverse convention")
    pivots = np.abs(uniforms - eta)
    if np.any(pivots >= 1.0):
        # The shifted convention has support just above one.  This event is
        # vanishingly rare at the released vocabulary sizes; the frozen
        # inverse likelihood scores are defined only below one.
        raise RuntimeError(
            "shifted inverse calibration produced d>=1; increase the seed audit "
            "or explicitly define an endpoint policy before using this sample"
        )
    return pivots


def _rng(seed: int, *stream_codes: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *stream_codes]))


def _calibrate(
    scores: dict[str, np.ndarray], alpha: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return benchmark.calibrate_randomized_boundary(scores, alpha)


def _decision_uniforms(
    seed: int,
    *,
    model_code: int,
    scheme_code: int,
    convention_code: int,
    sample_code: int,
    horizon_code: int,
    rows: int,
) -> np.ndarray:
    return _rng(
        seed,
        91_337,
        model_code,
        scheme_code,
        convention_code,
        sample_code,
        horizon_code,
    ).random(rows)


def _decision(
    values: np.ndarray, cutoff: float, gamma: float, uniforms: np.ndarray
) -> np.ndarray:
    return np.asarray((values > cutoff) | ((values == cutoff) & (uniforms < gamma)))


def _rao_blackwellized_summary(
    scores: np.ndarray, cutoff: float, gamma: float
) -> tuple[float, float, np.ndarray]:
    """Rate and MCSE after integrating out boundary randomization.

    Per-document contributions are ``1{S>c} + gamma*1{S=c}``.  Their sample
    standard deviation divided by ``sqrt(n)`` is the appropriate Monte Carlo
    standard error, including for a score with an atom at the cutoff.
    """

    values = np.asarray(scores, dtype=float)
    contributions = (values > cutoff).astype(float)
    contributions += gamma * (values == cutoff)
    rate = float(contributions.mean())
    if contributions.size < 2:
        mcse = 0.0
    else:
        mcse = float(contributions.std(ddof=1) / math.sqrt(contributions.size))
    return rate, mcse, contributions


def _top_probability_summary(
    top_probabilities: np.ndarray, config: RealDataConfig
) -> dict[str, Any]:
    delta = 1.0 - np.asarray(top_probabilities, dtype=float)
    return {
        "n_token_positions": int(delta.size),
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "delta_quantiles": {
            str(q): float(np.quantile(delta, q))
            for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        },
        "fraction_delta_below_prior_support": float(np.mean(delta < config.delta_low)),
        "fraction_delta_above_prior_support": float(np.mean(delta > config.delta_high)),
        "fraction_delta_recorded_zero": float(np.mean(delta == 0.0)),
        "stored_top_probability_precision": "torch.float32",
        "delta_prior_family": config.delta_prior,
        "prior_support": [config.delta_low, config.delta_high],
    }


def _union_tail_summary(
    grid: dirichlet.UnionTailBayesGrid, config: RealDataConfig
) -> dict[str, Any]:
    """Per-model record of the frozen shape/width union.

    The ladder depends on the released vocabulary, so this is recorded once per
    model rather than once per run.  The probe uniforms come from a dedicated
    throwaway generator so the containment diagnostic cannot advance any stream
    that a scored rule draws from.
    """

    probe = np.random.default_rng(config.calibration_seed).uniform(size=20_000)
    return {
        **union_tail_metadata(grid, methods=[benchmark.UNIONTAIL_SHARED_METHOD]),
        "tail_size": int(grid.tail_size),
        "tail_width_grid": [int(j) for j in grid.tail_widths],
        "n_components": int(grid.n_components),
        "prior_fingerprint": grid.prior_fingerprint(),
        "containment_check": {
            "probe": "20000 Uniform(0,1) pivots from a dedicated generator",
            "shape_equal_tail_atom_vs_closed_form_spike_max_abs_log_density_difference": (
                grid.spike_agreement(probe)
            ),
        },
        "analytic_component_mass_max_abs_deviation": (
            grid.analytic_normalisation()
        ),
    }


def _seed_context_summary(
    prompts: np.ndarray, raw_tokens: np.ndarray, *, horizon: int = MAX_HORIZON
) -> dict[str, Any]:
    """Summarize repetition in the token contexts that address the PRF.

    The released ``skipgram_prf`` hashes the oldest token in the four-token
    context.  Repeated values therefore reuse the same keyed pseudorandom draw;
    this matters for sequential claims even though fixed-horizon empirical size
    is reported directly on the released raw continuations.
    """

    prompts_array = np.asarray(prompts)
    tokens_array = np.asarray(raw_tokens)
    seed_context = np.concatenate(
        (prompts_array[:, -CONTEXT_WIDTH:], tokens_array[:, :horizon]), axis=1
    )[:, :horizon]
    total = int(seed_context.size)
    distinct = int(np.unique(seed_context).size)
    return {
        "n_positions": total,
        "n_distinct_context_tokens": distinct,
        "fraction_positions_beyond_first_occurrence": float(1.0 - distinct / total),
        "interpretation": (
            "repeated context tokens address repeated PRF seeds under the "
            "released skipgram implementation"
        ),
    }


def _result_rows(
    *,
    model_name: str,
    scheme: str,
    convention: str,
    calibration_scores: dict[str, np.ndarray],
    empirical_null_scores: dict[str, np.ndarray],
    alternative_scores: dict[str, np.ndarray],
    cutoffs: dict[str, np.ndarray],
    gammas: dict[str, np.ndarray],
    config: RealDataConfig,
    indicators: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    model_code = MODEL_CODES[model_name]
    scheme_code = 0 if scheme == "gumbel" else 1
    convention_code = 0 if convention in (GUMBEL_CONVENTION, PRIMARY_INVERSE_CONVENTION) else 1
    rows: list[dict[str, Any]] = []
    for method in calibration_scores:
        bayesian = METHOD_ORIGINS[method].startswith("bayesian")
        for index, horizon in enumerate(config.horizons):
            cutoff = float(cutoffs[method][index])
            gamma = float(gammas[method][index])
            samples = (
                ("calibration_exact_null", calibration_scores[method][:, index], 0),
                ("released_raw_empirical_null", empirical_null_scores[method][:, index], 1),
                ("released_watermarked", alternative_scores[method][:, index], 2),
            )
            for sample, scores, sample_code in samples:
                uniforms = _decision_uniforms(
                    config.calibration_seed,
                    model_code=model_code,
                    scheme_code=scheme_code,
                    convention_code=convention_code,
                    sample_code=sample_code,
                    horizon_code=index,
                    rows=scores.size,
                )
                decisions = _decision(scores, cutoff, gamma, uniforms)
                key = "|".join(
                    (model_name, scheme, convention, sample, str(horizon), method)
                )
                indicators[key] = decisions.astype(np.bool_)
                rate, mcse, _ = _rao_blackwellized_summary(
                    scores, cutoff, gamma
                )
                rows.append(
                    {
                        "model": model_name,
                        "model_id": MODEL_SPECS[model_name]["model_id"],
                        "vocabulary_size": MODEL_SPECS[model_name]["vocabulary_size"],
                        "scheme": scheme,
                        "pivot_convention": convention,
                        "sample": sample,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "method_origin": METHOD_ORIGINS[method],
                        "bayesian_delta_prior": (
                            config.delta_prior if bayesian else "not_applicable"
                        ),
                        "bayesian_delta_prior_low": (
                            config.delta_low if bayesian else ""
                        ),
                        "bayesian_delta_prior_high": (
                            config.delta_high if bayesian else ""
                        ),
                        "horizon": horizon,
                        "n_documents": int(scores.size),
                        "rejection_rate": rate,
                        "mcse": mcse,
                        "cutoff": cutoff,
                        "boundary_probability": gamma,
                        "n_at_boundary": int(np.count_nonzero(scores == cutoff)),
                        "n_rejected_randomized_realization": int(decisions.sum()),
                        "rejection_rate_randomized_realization": float(decisions.mean()),
                        "decision_rule": "score_gt_c_or_score_eq_c_and_u_lt_gamma",
                    }
                )
    return rows


def run_real_data_experiment(
    config: RealDataConfig,
    *,
    data_dir: Path,
    models: Sequence[str] = ("1p3B", "2p7B"),
    download_missing: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    config.validate()
    unknown = set(models) - set(MODEL_SPECS)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    if len(set(models)) != len(models):
        raise ValueError("models must not contain duplicates")
    started = time.time()
    required_assets = tuple(
        filename for model_name in models for filename in asset_names(model_name).values()
    )
    ensure_assets(
        data_dir,
        download_missing=download_missing,
        filenames=required_assets,
    )
    all_rows: list[dict[str, Any]] = []
    indicators: dict[str, np.ndarray] = {}
    input_records: list[dict[str, Any]] = []
    replay_records: list[dict[str, Any]] = []
    top_probability_records: dict[str, Any] = {}
    union_tail_records: dict[str, Any] = {}
    trgof_vocabulary_records: dict[str, int] = {}
    torch = _torch_module()

    for model_name in models:
        model_index = MODEL_CODES[model_name]
        released, provenance = load_released_model_data(model_name, data_dir)
        input_records.extend(provenance)
        gumbel_null, inverse_null_formal, inverse_null_shifted = replay_raw_pivots(
            released.prompts,
            released.raw_tokens,
            vocabulary_size=released.vocabulary_size,
        )
        replay_hashes = validate_replay_hashes(
            model_name,
            gumbel_null,
            inverse_null_formal,
            inverse_null_shifted,
        )
        replay_records.append(
            {
                "model": model_name,
                "n_documents": int(gumbel_null.shape[0]),
                "horizon": int(gumbel_null.shape[1]),
                "gumbel_sha256": replay_hashes["gumbel"],
                "inverse_formal_sha256": replay_hashes["inverse_formal"],
                "inverse_shifted_sha256": replay_hashes["inverse_shifted"],
                "fail_closed_hash_validation": True,
                "prompt_sha256": _array_sha256(released.prompts),
                "raw_token_sha256": _array_sha256(released.raw_tokens),
                "seed_context_reuse": _seed_context_summary(
                    released.prompts, released.raw_tokens
                ),
                "prf": {
                    "name": "skipgram_prf",
                    "context_width": CONTEXT_WIDTH,
                    "key": KEY_SEED,
                    "fixed_table_size": PRF_TABLE_SIZE,
                    "fixed_table_seed": PRF_TABLE_SEED,
                    "torch_device": "cpu",
                },
            }
        )
        top_probability_records[model_name] = {
            "gumbel": _top_probability_summary(
                released.top_probabilities_gumbel, config
            ),
            "inverse": _top_probability_summary(
                released.top_probabilities_inverse, config
            ),
        }

        scorers = ModelScorers(config, released.vocabulary_size)
        union_tail_records[model_name] = _union_tail_summary(
            scorers.uniontail_grid, config
        )
        # The exact inverse null used by the Tr-GoF p-value is a function of
        # this model's released vocabulary; record which one was used.
        trgof_vocabulary_records[model_name] = int(released.vocabulary_size)

        gumbel_calibration = simulate_exact_null(
            _rng(config.calibration_seed, model_index, 0, 0),
            scheme="gumbel",
            rows=config.n_calibration,
            horizon=MAX_HORIZON,
            vocabulary_size=released.vocabulary_size,
        )
        gumbel_calibration_scores = scorers.score_gumbel(gumbel_calibration)
        gumbel_cutoffs, gumbel_gammas = _calibrate(
            gumbel_calibration_scores, config.alpha
        )
        all_rows.extend(
            _result_rows(
                model_name=model_name,
                scheme="gumbel",
                convention=GUMBEL_CONVENTION,
                calibration_scores=gumbel_calibration_scores,
                empirical_null_scores=scorers.score_gumbel(gumbel_null),
                alternative_scores=scorers.score_gumbel(
                    released.gumbel_watermarked
                ),
                cutoffs=gumbel_cutoffs,
                gammas=gumbel_gammas,
                config=config,
                indicators=indicators,
            )
        )
        del gumbel_calibration, gumbel_calibration_scores

        # Use common U/rank draws for the formal and shifted inverse calibration
        # streams so the convention sensitivity is paired.
        inverse_seed_codes = (model_index, 1, 0)
        formal_calibration = simulate_exact_null(
            _rng(config.calibration_seed, *inverse_seed_codes),
            scheme="inverse",
            rows=config.n_calibration,
            horizon=MAX_HORIZON,
            vocabulary_size=released.vocabulary_size,
            inverse_convention=PRIMARY_INVERSE_CONVENTION,
        )
        shifted_calibration = simulate_exact_null(
            _rng(config.calibration_seed, *inverse_seed_codes),
            scheme="inverse",
            rows=config.n_calibration,
            horizon=MAX_HORIZON,
            vocabulary_size=released.vocabulary_size,
            inverse_convention=SHIFTED_INVERSE_CONVENTION,
        )
        for convention, calibration_pivots, empirical_pivots, alternative_pivots in (
            (
                PRIMARY_INVERSE_CONVENTION,
                formal_calibration,
                inverse_null_formal,
                released.inverse_watermarked_formal,
            ),
            (
                SHIFTED_INVERSE_CONVENTION,
                shifted_calibration,
                inverse_null_shifted,
                released.inverse_watermarked_shifted,
            ),
        ):
            calibration_scores = scorers.score_inverse(calibration_pivots)
            cutoffs, gammas = _calibrate(calibration_scores, config.alpha)
            all_rows.extend(
                _result_rows(
                    model_name=model_name,
                    scheme="inverse",
                    convention=convention,
                    calibration_scores=calibration_scores,
                    empirical_null_scores=scorers.score_inverse(empirical_pivots),
                    alternative_scores=scorers.score_inverse(alternative_pivots),
                    cutoffs=cutoffs,
                    gammas=gammas,
                    config=config,
                    indicators=indicators,
                )
            )

    metadata: dict[str, Any] = {
        "experiment": "released_real_model_output_reanalysis",
        "upstream": {
            "commit": OFFICIAL_COMMIT,
            "tree_url": OFFICIAL_TREE_URL,
            "assets": input_records,
            "generation_design": UPSTREAM_GENERATION_DESIGN,
            "provenance_limit": (
                "the released tensors are commit-pinned, but the model and C4 "
                "revisions used to generate them were not recorded upstream"
            ),
        },
        "resource_scope": {
            "model_weights_loaded": False,
            "tokenizer_loaded": False,
            "c4_downloaded": False,
            "gpu_used": False,
            "torch_uses": [
                "deserialize SHA256-verified released tensor pickles",
                "replay the released CPU fixed-table hash and seeded PRF draws",
            ],
        },
        "configuration": asdict(config),
        "alternative_prior": {
            "scope": "Bayesian methods in the released-output analysis",
            "delta": {
                "family": "uniform",
                "statement": (
                    f"Delta ~ Uniform({config.delta_low:g}, "
                    f"{config.delta_high:g})"
                ),
                "density": (
                    f"1 / ({config.delta_high:g} - {config.delta_low:g}) "
                    f"on [{config.delta_low:g}, {config.delta_high:g}]"
                ),
                "support": [config.delta_low, config.delta_high],
                "median": 0.5 * (config.delta_low + config.delta_high),
                "mean": 0.5 * (config.delta_low + config.delta_high),
                "quadrature": {
                    "variable": "Delta",
                    "rule": "Gauss-Legendre",
                    "nodes": config.bayes_quadrature_nodes,
                    "normalised_weights": True,
                },
                "synthetic_experiments_changed": False,
                "matches_synthetic_default": bool(
                    config.delta_low == 0.001 and config.delta_high == 0.5
                ),
            },
            "alpha": {
                "scope": "Gumbel Dirichlet-tail rows only",
                "family": "uniform_discrete",
                "atoms": [
                    "0.1", "1", "10", "100", "1000", "inf"
                ],
                "independent_of_delta": True,
            },
            "tail_width": {
                "scope": "width branch of the Gumbel union-tail rows",
                "family": "uniform_discrete",
                "atoms": (
                    "dyadic ladder 1, 4, 16, ... strictly below V-1 on the "
                    "released vocabulary; see bayes_uniontail.by_model"
                ),
                "independent_of_delta": True,
            },
            "rho": {
                "family": "point_mass",
                "value": 0.0,
            },
            "hierarchies": {
                "shared": "one (Delta, alpha) component per document",
                "tokenwise": "an independent (Delta_t, alpha_t) component per token",
            },
        },
        "models": list(models),
        "method_sets": {
            "gumbel": list(GUMBEL_METHODS),
            "inverse": list(INVERSE_METHODS),
        },
        "calibration": {
            "source": "independent exact pivot-null Monte Carlo",
            "n_paths": config.n_calibration,
            "common_pivots_within_each_model_scheme_convention": True,
            "released_raw_paths_used_for_calibration": False,
            "boundary_rule": "score>c plus randomized score=c boundary",
            "primary_inverse_uses_exact_finite_vocabulary_null": True,
            "shifted_sensitivity_recalibrated_from_simulated_shifted_null": True,
            "stream_construction": "numpy SeedSequence([base_seed, integer stream codes])",
        },
        "inverse_conventions": {
            "primary": {
                "name": PRIMARY_INVERSE_CONVENTION,
                "watermarked_reconstruction": (
                    "recover rank = round(eta_stored*(V-1)+1), validate its "
                    "integer residual, then abs(float64(U_float32)-rank/(V-1))"
                ),
                "raw_reconstruction": "abs(U - rank/(V-1))",
            },
            "sensitivity": {
                "name": SHIFTED_INVERSE_CONVENTION,
                "watermarked_reconstruction":
                    "float32 abs(U_float32 - eta_stored_float32)",
                "raw_reconstruction": "abs(U - (rank-1)/(V-1))",
                "score_definition": (
                    "apply the primary score formulas to shifted pivots and "
                    "recalibrate each score on simulated shifted-null paths"
                ),
                "status": (
                    "coordinate sensitivity only; not interpreted as a Bayes "
                    "factor under the shifted pivot law"
                ),
            },
        },
        "replay": replay_records,
        "released_top_probability_diagnostics": top_probability_records,
        "bayes_uniontail": {
            **union_tail_metadata(methods=[benchmark.UNIONTAIL_SHARED_METHOD], tokenwise=False),
            'delta_prior_shared_with_every_other_spike_rule': True,
            'by_model': union_tail_records,
        },
        "trgof": {
            "method": TRGOF_METHOD,
            "label": METHOD_LABELS[TRGOF_METHOD],
            "origin": METHOD_ORIGINS[TRGOF_METHOD],
            "schemes": ["gumbel", "inverse"],
            "reference": (
                "Tr-GoF of Li, Ruan, Wang, Long and Su (2026; preprint 2024); released "
                "implementation github.com/lx10077/TrGoF, compute_score"
            ),
            "implementation": (
                "code/trgof.py, the single owner of the statistic in this "
                "study, asserted term for term against the released code"
            ),
            "statistic": (
                "S_n(s) = n * max_t K_s(t/n, p_(t)) over the truncated "
                "one-sided index set p_(t) >= 1/n and t/n >= p_(t)"
            ),
            "cressie_read_index": TRGOF_S,
            "is_higher_criticism": bool(TRGOF_S == 2.0),
            "sensitivity_indices_available_but_not_reported_as_rules": [
                float(s) for s in trgof.DEFAULT_S_VALUES
            ],
            "sensitivity_note": (
                "the authors' simulation code sweeps s over "
                f"{[float(s) for s in trgof.DEFAULT_S_VALUES]}; only "
                f"s = {TRGOF_S:g}, the Higher Criticism member their figures "
                "lead with, is carried as a rule, because the released-output "
                "tables are already large.  The other indices are available "
                "from trgof.DEFAULT_S_VALUES and change no other rule."
            ),
            "not_a_token_sum": True,
            "accumulation": (
                "the statistic sorts the p-values of the whole prefix, so it "
                "is recomputed from scratch at every reported horizon; no "
                "cumulative sum exists for it"
            ),
            "horizons_evaluated": list(config.horizons),
            "gumbel_p_value": {
                "definition": "p = 1 - Y",
                "ours": False,
                "status": "the authors' definition, exactly as released",
            },
            "inverse_p_value": {
                "definition": (
                    "p = F_0(d), the exact finite-vocabulary null CDF of "
                    "d = |U - eta(I)|"
                ),
                "ours": True,
                "status": (
                    "OUR EXTENSION, not the authors'.  Their released code "
                    "applies Tr-GoF to Gumbel-max pivots only and defines no "
                    "p-value for the inverse-transform pivot.  Small d is the "
                    "evidence direction there, so the one-sided p-value is the "
                    "null CDF rather than its complement; the exact "
                    "finite-vocabulary CDF keeps it uniform under the null."
                ),
                "vocabulary_size_used_for_the_exact_null": (
                    trgof_vocabulary_records
                ),
                "vocabulary_source": (
                    "the released vocabulary of each model, carried through "
                    "ReleasedModelData and cross-checked against the released "
                    "eta tensors by the integer-rank reconstruction; no "
                    "vocabulary size is hardcoded in the Tr-GoF path"
                ),
                "shifted_convention_treatment": (
                    "the shifted sensitivity applies the same p-value map to "
                    "shifted pivots and recalibrates on the simulated "
                    "shifted-null sample, exactly as every other inverse "
                    "score is treated under that convention"
                ),
            },
            "calibration": (
                "routed through the shared calibration path: the same common "
                "exact pivot-null sample, the same nominal level, and the same "
                "randomized score=c boundary rule as every other rule"
            ),
            "atom_at_zero": (
                "rows whose truncated index set is empty score exactly zero, "
                "so the statistic has an atom; the randomized-boundary "
                "treatment already applied to every rule covers it, and "
                "n_at_boundary is reported per row"
            ),
            "randomness_drawn": 0,
            "randomness_note": (
                "the rule reads the pivot arrays already scored by every other "
                "detector and touches no generator, so all pre-existing "
                "records are bit-identical with and without it"
            ),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "output_schema": {
            "rows": "one sample x method x horizon record",
            "indicator_key":
                "model|scheme|pivot_convention|sample|horizon|method",
        },
        "runtime_seconds": time.time() - started,
    }
    return all_rows, metadata, indicators


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty results table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "results": list(rows)}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_indicators(indicators: Mapping[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dict(indicators))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_SPECS), default=tuple(MODEL_SPECS)
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--n-calibration", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=240_401_253)
    parser.add_argument("--bayes-quadrature-nodes", type=int, default=96)
    parser.add_argument("--delta-low", type=float, default=0.001)
    parser.add_argument("--delta-high", type=float, default=0.5)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.quick:
        config = RealDataConfig(
            n_calibration=500,
            calibration_seed=args.seed,
            delta_low=args.delta_low,
            delta_high=args.delta_high,
            bayes_quadrature_nodes=32,
            gumbel_lookup_size=8_001,
            dirichlet_c_nodes=20_001,
            score_batch_size=250,
        )
    else:
        config = RealDataConfig(
            n_calibration=args.n_calibration,
            calibration_seed=args.seed,
            delta_low=args.delta_low,
            delta_high=args.delta_high,
            bayes_quadrature_nodes=args.bayes_quadrature_nodes,
        )
    rows, metadata, indicators = run_real_data_experiment(
        config,
        data_dir=args.data_dir,
        models=args.models,
        download_missing=not args.no_download,
    )
    output_dir = args.results_dir / "quick" if args.quick else args.results_dir
    write_csv(rows, output_dir / "real_data_results.csv")
    write_json(rows, metadata, output_dir / "real_data_summary.json")
    write_indicators(indicators, output_dir / "real_data_indicators.npz")
    print(f"Wrote {len(rows)} result rows to {output_dir}")
    print(f"Runtime: {metadata['runtime_seconds']:.1f} seconds")


if __name__ == "__main__":
    main()
