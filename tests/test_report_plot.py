from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from doc_opt.report_plot import (
    PlotStage,
    normalize_report_checkpoints,
    normalize_report_stages,
    validate_metric_families,
)


def _old_schema_report() -> dict[str, object]:
    return {
        "dataset": {
            "dataset_root": "data/DS1000Retrieval",
        },
        "stages": {
            "baseline": {
                "metrics": {
                    "ndcg@1": 0.20,
                    "ndcg@5": 0.40,
                    "ndcg@10": 0.45,
                    "recall@1": 0.30,
                    "recall@5": 0.50,
                    "recall@10": 0.60,
                }
            },
            "doc-tranform": {
                "metrics": {
                    "ndcg@1": 0.22,
                    "ndcg@5": 0.44,
                    "ndcg@10": 0.48,
                    "recall@1": 0.34,
                    "recall@5": 0.56,
                    "recall@10": 0.65,
                }
            },
            "refresh": [
                {
                    "completed_steps": 50,
                    "metrics": {
                        "ndcg@1": 0.25,
                        "ndcg@5": 0.49,
                        "ndcg@10": 0.52,
                        "recall@1": 0.36,
                        "recall@5": 0.61,
                        "recall@10": 0.71,
                    },
                }
            ],
            "optimized": None,
        }
    }


def _new_schema_report() -> dict[str, object]:
    return {
        "stages": {
            "direct retrieval": {
                "metrics": {
                    "ndcg@1": 0.21,
                    "ndcg@5": 0.41,
                    "ndcg@10": 0.46,
                    "recall@1": 0.31,
                    "recall@5": 0.51,
                    "recall@10": 0.61,
                }
            },
            "direct transformation": {
                "metrics": {
                    "ndcg@1": 0.23,
                    "ndcg@5": 0.45,
                    "ndcg@10": 0.49,
                    "recall@1": 0.35,
                    "recall@5": 0.57,
                    "recall@10": 0.66,
                }
            },
            "refresh": [
                {
                    "completed_steps": 50,
                    "metrics": {
                        "ndcg@1": 0.26,
                        "ndcg@5": 0.50,
                        "ndcg@10": 0.54,
                        "recall@1": 0.37,
                        "recall@5": 0.63,
                        "recall@10": 0.72,
                    },
                },
                {
                    "completed_steps": 100,
                    "metrics": {
                        "ndcg@1": 0.28,
                        "ndcg@5": 0.53,
                        "ndcg@10": 0.57,
                        "recall@1": 0.40,
                        "recall@5": 0.67,
                        "recall@10": 0.75,
                    },
                },
            ],
            "document optimization": {
                "metrics": {
                    "ndcg@1": 0.30,
                    "ndcg@5": 0.55,
                    "ndcg@10": 0.60,
                    "recall@1": 0.42,
                    "recall@5": 0.70,
                    "recall@10": 0.78,
                }
            },
        }
    }


class NormalizeReportStagesTest(unittest.TestCase):
    def test_normalizes_old_schema(self) -> None:
        stages = normalize_report_stages(_old_schema_report())

        self.assertEqual(
            [stage.label for stage in stages],
            ["Direct Retrieval", "Document Transformation", "Document Optimization"],
        )
        self.assertEqual(stages[0].style_key, "baseline")
        self.assertEqual(stages[1].style_key, "direct")
        self.assertEqual(stages[2].style_key, "refresh")

    def test_normalizes_new_schema(self) -> None:
        stages = normalize_report_stages(_new_schema_report())

        self.assertEqual(
            [stage.label for stage in stages],
            [
                "Direct Retrieval",
                "Document Transformation",
                "Document Optimization",
            ],
        )

    def test_multiple_refresh_checkpoints_are_sorted(self) -> None:
        report = _new_schema_report()
        report["stages"]["refresh"] = list(reversed(report["stages"]["refresh"]))

        stages = normalize_report_stages(report)

        self.assertEqual(stages[2].label, "Document Optimization")
        self.assertAlmostEqual(stages[2].metrics["ndcg@5"], 0.55)

    def test_null_optimized_stage_is_ignored(self) -> None:
        report = _new_schema_report()
        report["stages"]["document optimization"] = None

        stages = normalize_report_stages(report)

        self.assertEqual(
            [stage.label for stage in stages],
            ["Direct Retrieval", "Document Transformation", "Document Optimization"],
        )
        self.assertAlmostEqual(stages[2].metrics["ndcg@5"], 0.53)


class NormalizeReportCheckpointsTest(unittest.TestCase):
    def test_uses_direct_transformation_as_step_zero_and_sorts_refreshes(self) -> None:
        report = _new_schema_report()
        report["stages"]["refresh"] = list(reversed(report["stages"]["refresh"]))

        checkpoints = normalize_report_checkpoints(report)

        self.assertEqual([checkpoint.step for checkpoint in checkpoints], [0, 50, 100])
        self.assertEqual(checkpoints[0].label, "Document Transformation")
        self.assertAlmostEqual(checkpoints[0].metrics["ndcg@5"], 0.45)
        self.assertAlmostEqual(checkpoints[-1].metrics["recall@10"], 0.75)


class ValidateMetricFamiliesTest(unittest.TestCase):
    def test_raises_when_metric_family_is_missing_everywhere(self) -> None:
        stages = [
            PlotStage(
                key="baseline",
                label="Direct Retrieval",
                metrics={"ndcg@1": 0.1, "ndcg@5": 0.2},
                style_key="baseline",
            )
        ]

        with self.assertRaisesRegex(ValueError, "No Recall metrics are available to plot."):
            validate_metric_families(stages)


class PlotScriptSmokeTest(unittest.TestCase):
    def test_script_writes_svg(self) -> None:
        with TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            report_path = temp_path / "run_report.json"
            output_path = temp_path / "run_report_plot.svg"
            report_path.write_text(json.dumps(_old_schema_report()), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    (REPO_ROOT / "scripts/plot_run_report.py").as_posix(),
                    "--report-path",
                    report_path.as_posix(),
                    "--output-path",
                    output_path.as_posix(),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            svg = output_path.read_text(encoding="utf-8")
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertIn("Direct Retrieval", svg)
            self.assertIn("Document Transformation", svg)
            self.assertIn("Document Optimization", svg)
            self.assertIn("Dataset: DS1000Retrieval", svg)
            self.assertIn("Ranking quality", svg)
            self.assertIn("Document coverage", svg)
            self.assertIn("@1", svg)
            self.assertIn("40.00", svg)
            self.assertNotIn("50 policy optimization steps", svg)

    def test_steps_script_writes_svg(self) -> None:
        with TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            report_path = temp_path / "run_report.json"
            output_path = temp_path / "run_report_steps_plot.svg"
            report_path.write_text(json.dumps(_new_schema_report()), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    (REPO_ROOT / "scripts/plot_run_report_steps.py").as_posix(),
                    "--report-path",
                    report_path.as_posix(),
                    "--output-path",
                    output_path.as_posix(),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            svg = output_path.read_text(encoding="utf-8")
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertIn("Retrieval Improves through Training", svg)
            self.assertIn("NDCG", svg)
            self.assertIn("Recall", svg)
            self.assertIn("Ranking quality", svg)
            self.assertIn("Document coverage", svg)
            self.assertIn("Stage (ordered)", svg)
            self.assertIn("Performance (%)", svg)
            self.assertIn("NDCG@5", svg)
            self.assertIn("NDCG@10", svg)
            self.assertIn("Recall@5", svg)
            self.assertIn("Recall@10", svg)
            self.assertNotIn(">NDCG@1</text>", svg)
            self.assertNotIn(">Recall@1</text>", svg)
            self.assertIn("X-axis shows ordered stages", svg)
            self.assertIn("Code retrieval uses text-embedding-3-small.", svg)
            self.assertNotIn("refresh", svg)
