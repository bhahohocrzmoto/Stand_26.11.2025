#!/usr/bin/env python3
"""Utility to convert matrix JSON files into human-readable spreadsheets."""
from __future__ import annotations

import argparse
from pathlib import Path

from FastSolver.PlotGeneration.PlotGeneration import export_matrix_json_to_excel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_file", type=Path, help="Path to the matrix JSON produced by PlotGeneration")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output Excel path. Defaults to <json_file>_readable.xlsx",
    )
    args = parser.parse_args()

    output = export_matrix_json_to_excel(args.json_file, args.output)
    print(f"Readable workbook written to {output}")


if __name__ == "__main__":
    main()
