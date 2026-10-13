#!/usr/bin/env python3
"""Compare reproducible CSV artifacts with an explicit absolute tolerance."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


def numeric(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: compare-csv.py EXPECTED ACTUAL ABS_TOLERANCE")
    expected_path, actual_path = map(Path, sys.argv[1:3])
    tolerance = float(sys.argv[3])
    with expected_path.open(newline="", encoding="utf-8") as stream:
        expected = list(csv.reader(stream))
    with actual_path.open(newline="", encoding="utf-8") as stream:
        actual = list(csv.reader(stream))
    if len(expected) != len(actual):
        raise AssertionError(f"row count differs: {len(expected)} != {len(actual)}")
    for row_index, (expected_row, actual_row) in enumerate(zip(expected, actual), start=1):
        if len(expected_row) != len(actual_row):
            raise AssertionError(f"column count differs on row {row_index}")
        for column_index, (left, right) in enumerate(zip(expected_row, actual_row), start=1):
            left_number, right_number = numeric(left), numeric(right)
            if left_number is not None and right_number is not None:
                if math.isnan(left_number) and math.isnan(right_number):
                    continue
                if not math.isclose(left_number, right_number, rel_tol=0.0, abs_tol=tolerance):
                    raise AssertionError(
                        f"numeric difference at row {row_index}, column {column_index}: "
                        f"{left} != {right} (tolerance {tolerance})"
                    )
            elif left != right:
                raise AssertionError(
                    f"text difference at row {row_index}, column {column_index}: {left!r} != {right!r}"
                )
    print(f"CSV comparison passed at absolute tolerance {tolerance}: {actual_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
