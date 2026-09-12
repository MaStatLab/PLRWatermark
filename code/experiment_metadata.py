"""Shared descriptions of the numerical union-tail implementation.

These records describe the evaluated model; they are not generation manifests
or certificates of one-sided numerical normalization error.
"""

from __future__ import annotations

from typing import Any


def union_tail_metadata(
    grid: Any | None = None, *, methods: list[str], tokenwise: bool = False
) -> dict[str, object]:
    """Describe both union branches without confusing analytic and fast laws.

    A missing grid is used only for an aggregate real-model description; its
    model-specific priors and component counts are recorded under ``by_model``.
    Numerical mass diagnostics must be supplied by the caller, not invented.
    """
    result: dict[str, object] = {
        "methods": list(methods),
        "scheme": "gumbel only",
        "hierarchy": (
            "shared, and tokenwise within each branch with the tail state S "
            "held at document level"
            if tokenwise else "shared only in this experiment"
        ),
        "hierarchy_scope": (
            "This records the hierarchies evaluated, not a general claim that "
            "tokenwise mixtures cannot improve power."
        ),
        "why_gumbel_only": (
            "The working inverse limiting alternative depends on Delta alone."
        ),
        "component": (
            "A union of a full-width Dirichlet tail-shape branch and an "
            "equal-weight live-tail-width branch restricted to J < V-1. "
            "The width component is f_{Delta,J}(r) = "
            "r**(Delta/(1-Delta)) + J*r**(J/Delta-1). The shape component "
            "uses the tabulated psi_alpha transform."
        ),
        "tail_width_prior": {
            "family": "uniform on the configured ladder strictly below V-1",
            "note": (
                "The equal-tail atom (alpha=inf, J=V-1) belongs to the shape "
                "branch. It is excluded from the width branch to avoid "
                "counting the full-width atom twice."
            ),
        },
        "normalisation": {
            "closed_form": False,
            "closed_form_scope": "width branch only",
            "interpolated": True,
            "interpolation_scope": (
                "shape transform and branch-specific tokenwise outer lookups"
                if tokenwise else "shape transform"
            ),
            "diagnostic_only": True,
            "certified_one_sided_bound": False,
            "note": (
                "Both branches integrate to one analytically. The width "
                "formula is closed form; the shape implementation interpolates "
                "a transform table. Tokenwise scoring, when requested, also "
                "uses outer lookups. Numerical mass and node-sensitivity "
                "checks are not certificates that the evaluated densities "
                "have mass at most one. The exact test-martingale result "
                "requires conditional null validity and analytic or certified "
                "normalized evaluation; it is not asserted for these "
                "interpolated implementations."
            ),
        },
    }
    if grid is not None:
        result["prior_fingerprint"] = grid.prior_fingerprint()
        result["branch_prior"] = {
            "shape_weight": float(grid.dirichlet_weight),
            "width_weight": float(1.0 - grid.dirichlet_weight),
            "component_counts": [int(block.n_components) for block in grid.blocks],
        }
        result["tail_width_prior"].update({
            "grid": [int(j) for j in grid.tail_widths],
            "weights": [float(w) for w in grid.tail_width_weights],
        })
        result["normalisation"]["analytic_component_mass_max_abs_deviation"] = (
            grid.analytic_normalisation()
        )
    else:
        result["branch_prior"] = "See by_model prior fingerprints and component counts."
    return result
