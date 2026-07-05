from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnalysisConfig:
    label_column: str | None = None
    text_column: str | None = None
    max_examples: int = 10


def analyze_csv(path: str | Path, config: AnalysisConfig | None = None) -> dict[str, Any]:
    config = config or AnalysisConfig()
    rows = _read_csv(path)

    if not rows:
        return {
            "summary": {"rows": 0, "columns": 0, "column_names": []},
            "quality_score": 0,
            "checks": {},
            "actions": ["Add data before running quality analysis."],
        }

    columns = list(rows[0].keys())
    label_column = _pick_column(config.label_column, columns, ["label", "class", "category", "target"])
    text_column = _pick_column(config.text_column, columns, ["text", "content", "description", "prompt"])

    checks = {
        "missing_values": _missing_values(rows, columns),
        "duplicate_rows": _duplicate_rows(rows, config.max_examples),
        "numeric_outliers": _numeric_outliers(rows, columns, config.max_examples),
        "label_distribution": _label_distribution(rows, label_column),
        "label_conflicts": _label_conflicts(rows, label_column, text_column, config.max_examples),
        "near_duplicate_text": _near_duplicate_text(rows, text_column, config.max_examples),
    }

    score = _quality_score(len(rows), checks)

    return {
        "summary": {
            "rows": len(rows),
            "columns": len(columns),
            "column_names": columns,
            "label_column": label_column,
            "text_column": text_column,
        },
        "quality_score": score,
        "checks": checks,
        "actions": _recommended_actions(checks),
    }


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def _pick_column(requested: str | None, columns: list[str], candidates: list[str]) -> str | None:
    if requested:
        if requested not in columns:
            raise ValueError(f"Column '{requested}' was not found in the dataset.")
        return requested

    normalized = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _missing_values(rows: list[dict[str, str]], columns: list[str]) -> dict[str, Any]:
    counts = {column: sum(1 for row in rows if row[column] == "") for column in columns}
    rates = {column: round(count / len(rows), 4) for column, count in counts.items() if count}
    return {
        "total_missing_cells": sum(counts.values()),
        "columns": rates,
    }


def _duplicate_rows(rows: list[dict[str, str]], max_examples: int) -> dict[str, Any]:
    signatures: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        payload = "\x1f".join(f"{key}={row[key]}" for key in sorted(row))
        signatures[hashlib.sha256(payload.encode("utf-8")).hexdigest()].append(index)

    groups = [indexes for indexes in signatures.values() if len(indexes) > 1]
    duplicate_count = sum(len(indexes) - 1 for indexes in groups)
    return {
        "duplicate_rows": duplicate_count,
        "groups": groups[:max_examples],
    }


def _numeric_outliers(rows: list[dict[str, str]], columns: list[str], max_examples: int) -> dict[str, Any]:
    outliers: dict[str, Any] = {}

    for column in columns:
        values: list[tuple[int, float]] = []
        for index, row in enumerate(rows, start=1):
            value = _to_float(row[column])
            if value is not None:
                values.append((index, value))

        if len(values) < 4:
            continue

        numbers = [value for _, value in values]
        q1 = _percentile(numbers, 25)
        q3 = _percentile(numbers, 75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        found = [
            {"row": index, "value": value}
            for index, value in values
            if value < lower or value > upper
        ]
        if found:
            outliers[column] = {
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
                "count": len(found),
                "examples": found[:max_examples],
            }

    return outliers


def _label_distribution(rows: list[dict[str, str]], label_column: str | None) -> dict[str, Any]:
    if not label_column:
        return {"status": "skipped", "reason": "No label column detected."}

    counts = Counter(row[label_column] or "<missing>" for row in rows)
    majority = counts.most_common(1)[0]
    minority = min(counts.items(), key=lambda item: item[1])

    return {
        "labels": dict(sorted(counts.items())),
        "majority_label": majority[0],
        "majority_rate": round(majority[1] / len(rows), 4),
        "minority_label": minority[0],
        "minority_rate": round(minority[1] / len(rows), 4),
    }


def _label_conflicts(
    rows: list[dict[str, str]],
    label_column: str | None,
    text_column: str | None,
    max_examples: int,
) -> dict[str, Any]:
    if not label_column or not text_column:
        return {"status": "skipped", "reason": "Requires both label and text columns."}

    labels_by_text: defaultdict[str, set[str]] = defaultdict(set)
    rows_by_text: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        text_key = _normalize_text(row[text_column])
        if not text_key:
            continue
        labels_by_text[text_key].add(row[label_column])
        rows_by_text[text_key].append(index)

    conflicts = [
        {
            "normalized_text": text,
            "labels": sorted(labels),
            "rows": rows_by_text[text],
        }
        for text, labels in labels_by_text.items()
        if len(labels) > 1
    ]

    return {
        "conflict_count": len(conflicts),
        "examples": conflicts[:max_examples],
    }


def _near_duplicate_text(rows: list[dict[str, str]], text_column: str | None, max_examples: int) -> dict[str, Any]:
    if not text_column:
        return {"status": "skipped", "reason": "No text column detected."}

    buckets: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        fingerprint = _text_fingerprint(row[text_column])
        if fingerprint:
            buckets[fingerprint].append(index)

    groups = [indexes for indexes in buckets.values() if len(indexes) > 1]
    return {
        "groups": groups[:max_examples],
        "near_duplicate_rows": sum(len(indexes) - 1 for indexes in groups),
    }


def _recommended_actions(checks: dict[str, Any]) -> list[str]:
    actions: list[str] = []

    if checks["duplicate_rows"]["duplicate_rows"]:
        actions.append("Review and remove exact duplicate rows.")
    if checks["missing_values"]["total_missing_cells"]:
        actions.append("Inspect columns with missing values before training.")
    if checks["numeric_outliers"]:
        actions.append("Validate numeric outliers; keep rare valid cases, fix data errors.")
    if checks["label_conflicts"].get("conflict_count"):
        actions.append("Relabel duplicated texts that currently map to different labels.")
    if checks["near_duplicate_text"].get("near_duplicate_rows"):
        actions.append("Review near-duplicate text examples to reduce dataset redundancy.")
    if checks["label_distribution"].get("majority_rate", 0) >= 0.8:
        actions.append("Collect or rebalance minority label examples.")

    return actions or ["No critical issue detected by the v0 checks."]


def _quality_score(row_count: int, checks: dict[str, Any]) -> int:
    if row_count == 0:
        return 0

    penalty = 0.0
    penalty += min(30, checks["duplicate_rows"]["duplicate_rows"] / row_count * 100)
    penalty += min(25, checks["missing_values"]["total_missing_cells"] / row_count * 10)
    penalty += min(20, sum(item["count"] for item in checks["numeric_outliers"].values()) / row_count * 100)
    penalty += min(20, checks["label_conflicts"].get("conflict_count", 0) * 5)
    penalty += min(15, checks["near_duplicate_text"].get("near_duplicate_rows", 0) / row_count * 50)

    majority_rate = checks["label_distribution"].get("majority_rate")
    if majority_rate and majority_rate >= 0.8:
        penalty += 10

    return max(0, min(100, round(100 - penalty)))


def _to_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if math.isfinite(number):
        return number
    return None


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _text_fingerprint(value: str) -> str:
    normalized = _normalize_text(value)
    tokens = normalized.split()
    if len(tokens) < 3:
        return normalized
    return " ".join(sorted(set(tokens)))
