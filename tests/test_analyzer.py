from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from agent.analyzer import AnalysisConfig, analyze_csv


class AnalyzeCsvTest(unittest.TestCase):
    def test_detects_core_quality_issues(self) -> None:
        path = self._write_csv(
            [
                {"text": "Same text", "label": "a", "score": "0.1"},
                {"text": "Same text", "label": "a", "score": "0.1"},
                {"text": "same text", "label": "b", "score": "0.2"},
                {"text": "Different sample", "label": "a", "score": "99"},
                {"text": "", "label": "a", "score": "0.3"},
            ]
        )

        report = analyze_csv(path, AnalysisConfig(label_column="label", text_column="text"))

        self.assertEqual(report["summary"]["rows"], 5)
        self.assertEqual(report["checks"]["duplicate_rows"]["duplicate_rows"], 1)
        self.assertEqual(report["checks"]["missing_values"]["total_missing_cells"], 1)
        self.assertEqual(report["checks"]["label_conflicts"]["conflict_count"], 1)
        self.assertLess(report["quality_score"], 100)
        self.assertTrue(report["actions"])

    def test_empty_dataset_returns_zero_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            path.write_text("text,label\n", encoding="utf-8")

            report = analyze_csv(path)

        self.assertEqual(report["summary"]["rows"], 0)
        self.assertEqual(report["quality_score"], 0)

    def _write_csv(self, rows: list[dict[str, str]]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "dataset.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path


if __name__ == "__main__":
    unittest.main()
