from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.analyzer import AnalysisConfig, analyze_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dataset quality for AI training.")
    parser.add_argument("csv_path", type=Path, help="Path to a CSV dataset.")
    parser.add_argument("--label-column", help="Label column name. Auto-detected when omitted.")
    parser.add_argument("--text-column", help="Text column name. Auto-detected when omitted.")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path.")
    args = parser.parse_args()

    report = analyze_csv(
        args.csv_path,
        AnalysisConfig(label_column=args.label_column, text_column=args.text_column),
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False)

    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
