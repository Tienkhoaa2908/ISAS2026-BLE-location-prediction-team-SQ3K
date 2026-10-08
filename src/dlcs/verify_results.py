#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TOL = 1e-9

EXPECTED_SUMMARY = {
    "baseline_macro_mean": 0.536686916710663,
    "candidate_macro_mean": 0.5486823604624447,
    "paired_delta_mean": 0.011995443751781687,
    "paired_accuracy_delta": 0.002324145104667802,
}

EXPECTED_REPS = {
    10: (0.534490805733374, 0.5385910274707544),
    11: (0.5351660691785216, 0.5519794595162316),
    12: (0.5524654655838526, 0.5695400526491061),
    13: (0.524625326346904, 0.5346189022136868),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the locked repeated session-grouped DLCS results."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def close(actual: float, expected: float) -> bool:
    return abs(float(actual) - float(expected)) <= TOL


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    summary_path = results_dir / "confirmation_summary.csv"
    by_rep_path = results_dir / "confirmation_by_rep.csv"

    if not summary_path.exists() or not by_rep_path.exists():
        missing = [
            str(path)
            for path in (summary_path, by_rep_path)
            if not path.exists()
        ]
        print("VERIFICATION FAILED")
        for path in missing:
            print(" - missing:", path)
        return 1

    summary_df = pd.read_csv(summary_path)
    by_rep = pd.read_csv(by_rep_path)

    if len(summary_df) != 1:
        print("VERIFICATION FAILED")
        print(f" - confirmation_summary.csv must contain one row, found {len(summary_df)}")
        return 1

    summary = summary_df.iloc[0]
    errors: list[str] = []

    for key, expected in EXPECTED_SUMMARY.items():
        if key not in summary.index:
            errors.append(f"summary column missing: {key}")
            continue
        actual = float(summary[key])
        if not close(actual, expected):
            errors.append(
                f"{key}: expected {expected:.15f}, got {actual:.15f}"
            )

    required_rep_columns = {"rep", "baseline_macro", "candidate_macro"}
    missing_columns = required_rep_columns.difference(by_rep.columns)
    if missing_columns:
        errors.append(
            "confirmation_by_rep.csv missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    else:
        for rep, (base_expected, candidate_expected) in EXPECTED_REPS.items():
            row = by_rep.loc[by_rep["rep"] == rep]
            if len(row) != 1:
                errors.append(f"rep {rep}: missing or duplicated")
                continue
            row = row.iloc[0]
            if not close(row["baseline_macro"], base_expected):
                errors.append(
                    f"rep {rep} baseline: expected {base_expected:.15f}, "
                    f"got {float(row['baseline_macro']):.15f}"
                )
            if not close(row["candidate_macro"], candidate_expected):
                errors.append(
                    f"rep {rep} DLCS: expected {candidate_expected:.15f}, "
                    f"got {float(row['candidate_macro']):.15f}"
                )

    if errors:
        print("VERIFICATION FAILED")
        for error in errors:
            print(" -", error)
        return 1

    print("VERIFICATION PASSED")
    print(f"Main Macro-F1: {float(summary['baseline_macro_mean']):.6f}")
    print(f"DLCS Macro-F1: {float(summary['candidate_macro_mean']):.6f}")
    print(f"Paired gain:   {float(summary['paired_delta_mean']):+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
