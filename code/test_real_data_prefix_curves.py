"""Focused, non-canonical tests for full-prefix released-output curves."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import real_data_experiment as real
import real_data_prefix_curves as prefix


def _upstream_payload(scheme: str, rate: float) -> dict[str, object]:
    payload: dict[str, object] = {
        key: [rate] * prefix.MAX_HORIZON
        for key in prefix.UPSTREAM_METHODS[scheme]
    }
    payload["top_probs"] = [[1.0]]
    return payload


def _record(name: str) -> dict[str, str]:
    return {"filename": name, "sha256": "a" * 64}


def _curve_row(
    *,
    analysis: str,
    convention: str,
    error_rate: float,
    horizon: int = 200,
) -> dict[str, object]:
    return {
        "analysis": analysis,
        "model": "1p3B",
        "model_id": real.MODEL_SPECS["1p3B"]["model_id"],
        "scheme": "inverse",
        "pivot_convention": convention,
        "sample": "released_watermarked",
        "error_type": "type_ii_error",
        "method": "h_neg",
        "method_label": "h_neg",
        "method_origin": "reference_score",
        "horizon": horizon,
        "n_documents": 500,
        "rejection_rate": 1.0 - error_rate,
        "error_rate": error_rate,
        "mcse": 0.02,
        "cutoff": "",
        "boundary_probability": "",
        "source_file": "fixture",
        "source_sha256": "fixture",
        "calibration": "fixture",
        "bayesian_delta_prior": "not_applicable",
        "bayesian_delta_prior_low": "",
        "bayesian_delta_prior_high": "",
    }


class PrefixCurveTests(unittest.TestCase):
    def test_upstream_allowlist_is_complete(self) -> None:
        expected = {
            filename
            for model in real.MODEL_SPECS
            for filename in prefix.upstream_json_names(model).values()
        }
        self.assertEqual(set(prefix.UPSTREAM_JSON_SHA256), expected)
        self.assertEqual(len(expected), 8)
        self.assertTrue(
            all(len(digest) == 64 for digest in prefix.UPSTREAM_JSON_SHA256.values())
        )

    def test_parse_upstream_gumbel_converts_power_to_type_ii(self) -> None:
        self._check_upstream_conversion("gumbel", 12)

    def test_parse_upstream_inverse_converts_power_to_type_ii(self) -> None:
        self._check_upstream_conversion("inverse", 10)

    def _check_upstream_conversion(self, scheme: str, method_count: int) -> None:
        rows = prefix.parse_upstream_payloads(
            model="1p3B",
            scheme=scheme,
            null_payload=_upstream_payload(scheme, 0.05),
            result_payload=_upstream_payload(scheme, 0.8),
            null_record=_record("null.json"),
            result_record=_record("result.json"),
        )
        self.assertEqual(len(rows), method_count * 2 * prefix.MAX_HORIZON)
        null = next(row for row in rows if str(row["sample"]).endswith("null"))
        alternative = next(
            row for row in rows if row["sample"] == "released_watermarked"
        )
        self.assertAlmostEqual(float(null["rejection_rate"]), 0.05)
        self.assertAlmostEqual(float(null["error_rate"]), 0.05)
        self.assertAlmostEqual(float(alternative["rejection_rate"]), 0.8)
        self.assertAlmostEqual(float(alternative["error_rate"]), 0.2)
        self.assertAlmostEqual(
            float(alternative["mcse"]), math.sqrt(0.8 * 0.2 / 499)
        )
        expected_convention = (
            real.GUMBEL_CONVENTION
            if scheme == "gumbel"
            else real.SHIFTED_INVERSE_CONVENTION
        )
        self.assertEqual(
            {row["pivot_convention"] for row in rows}, {expected_convention}
        )

    def test_bad_upstream_curve_length_is_rejected(self) -> None:
        null_payload = _upstream_payload("gumbel", 0.05)
        null_payload["ars"] = [0.05] * 199
        with self.assertRaisesRegex(ValueError, "shape"):
            prefix.parse_upstream_payloads(
                model="1p3B",
                scheme="gumbel",
                null_payload=null_payload,
                result_payload=_upstream_payload("gumbel", 0.8),
                null_record=_record("null.json"),
                result_record=_record("result.json"),
            )

    def test_upstream_rate_not_from_500_documents_is_rejected(self) -> None:
        null_payload = _upstream_payload("gumbel", 0.05)
        null_payload["ars"] = [0.051] * 200
        with self.assertRaisesRegex(ValueError, "rates from 500 documents"):
            prefix.parse_upstream_payloads(
                model="1p3B",
                scheme="gumbel",
                null_payload=null_payload,
                result_payload=_upstream_payload("gumbel", 0.8),
                null_record=_record("null.json"),
                result_record=_record("result.json"),
            )

    def test_common_rows_keep_both_inverse_conventions(self) -> None:
        source_rows = []
        for convention, rate in (
            (real.PRIMARY_INVERSE_CONVENTION, 0.4),
            (real.SHIFTED_INVERSE_CONVENTION, 0.42),
        ):
            for sample in (
                "calibration_exact_null",
                "released_raw_empirical_null",
                "released_watermarked",
            ):
                source_rows.append(
                    {
                        "model": "1p3B",
                        "model_id": real.MODEL_SPECS["1p3B"]["model_id"],
                        "scheme": "inverse",
                        "pivot_convention": convention,
                        "sample": sample,
                        "method": "bayes_tokenwise",
                        "method_label": "Bayes, tokenwise Delta",
                        "method_origin": "bayesian",
                        "bayesian_delta_prior": "uniform",
                        "bayesian_delta_prior_low": 0.001,
                        "bayesian_delta_prior_high": 0.5,
                        "horizon": 200,
                        "n_documents": 500,
                        "rejection_rate": rate,
                        "mcse": 0.02,
                        "cutoff": 1.25,
                        "boundary_probability": 0.0,
                    }
                )
        rows = prefix.common_rows_from_experiment(source_rows)
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {row["pivot_convention"] for row in rows},
            {
                real.PRIMARY_INVERSE_CONVENTION,
                real.SHIFTED_INVERSE_CONVENTION,
            },
        )
        alternatives = [
            row for row in rows if row["error_type"] == "type_ii_error"
        ]
        np.testing.assert_allclose(
            sorted(float(row["error_rate"]) for row in alternatives),
            [0.58, 0.60],
        )
        self.assertTrue(
            all(row["bayesian_delta_prior"] == "uniform" for row in rows)
        )

    def test_comparison_has_primary_and_matched_shifted_differences(self) -> None:
        rows = [
            _curve_row(
                analysis="upstream_recorded",
                convention=real.SHIFTED_INVERSE_CONVENTION,
                error_rate=0.60,
            ),
            _curve_row(
                analysis="common_calibration",
                convention=real.PRIMARY_INVERSE_CONVENTION,
                error_rate=0.65,
            ),
            _curve_row(
                analysis="common_calibration",
                convention=real.SHIFTED_INVERSE_CONVENTION,
                error_rate=0.63,
            ),
        ]
        summary = prefix.comparison_summary(rows, horizons=(200,))
        comparison = summary["comparisons"][0]
        self.assertAlmostEqual(
            comparison["common_primary_minus_upstream"], 0.05
        )
        self.assertAlmostEqual(
            comparison["common_matched_shifted_minus_upstream"], 0.03
        )
        self.assertIn("paired MCSE", summary["mcse_warning"])

    def test_duplicate_curve_key_fails_closed(self) -> None:
        row = _curve_row(
            analysis="upstream_recorded",
            convention=real.SHIFTED_INVERSE_CONVENTION,
            error_rate=0.6,
        )
        with self.assertRaisesRegex(ValueError, "duplicate curve row key"):
            prefix._index_rows([row, dict(row)])

    def test_curve_csv_schema_and_uniform_full_prefix_config(self) -> None:
        row = _curve_row(
            analysis="upstream_recorded",
            convention=real.SHIFTED_INVERSE_CONVENTION,
            error_rate=0.6,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curves.csv"
            prefix.write_curve_csv([row], path)
            with path.open(newline="", encoding="utf-8") as handle:
                loaded = list(csv.DictReader(handle))
        self.assertEqual(tuple(loaded[0]), prefix.CURVE_FIELDS)

        config = prefix._uniform_real_config(prefix.parse_args(["--skip-plots"]))
        self.assertEqual(config.horizons, tuple(range(1, 201)))
        self.assertEqual(config.delta_prior, "uniform")
        self.assertEqual((config.delta_low, config.delta_high), (0.001, 0.5))

    def test_verify_json_rejects_unlisted_and_tampered_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unlisted = root / "unlisted.json"
            unlisted.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not an allowlisted"):
                prefix.verify_upstream_json(unlisted)

            filename = next(iter(prefix.UPSTREAM_JSON_SHA256))
            tampered = root / filename
            tampered.write_text(json.dumps({"tampered": True}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                prefix.verify_upstream_json(tampered)

    @unittest.skipUnless(
        importlib.util.find_spec("matplotlib"), "matplotlib is not installed"
    )
    def test_tiny_vector_and_raster_plot(self) -> None:
        rows = []
        for analysis in ("upstream_recorded", "common_calibration"):
            for error_type in ("type_i_error", "type_ii_error"):
                for horizon in range(1, 5):
                    row = _curve_row(
                        analysis=analysis,
                        convention=real.GUMBEL_CONVENTION,
                        error_rate=(0.05 if error_type == "type_i_error" else 0.8),
                        horizon=horizon,
                    )
                    row.update(
                        {
                            "scheme": "gumbel",
                            "method": "h_ars",
                            "method_label": "h_ars",
                            "error_type": error_type,
                            "sample": (
                                "released_raw_empirical_null"
                                if error_type == "type_i_error"
                                else "released_watermarked"
                            ),
                        }
                    )
                    rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "tiny.pdf"
            png = Path(directory) / "tiny.png"
            prefix._plot_grid(
                rows,
                analysis="common_calibration",
                methods_by_scheme={"gumbel": ("h_ars",), "inverse": ("h_neg",)},
                output_pdf=pdf,
                output_png=png,
                title="Tiny fixture",
                subtitle="No released assets are used.",
            )
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
            self.assertTrue(png.read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
