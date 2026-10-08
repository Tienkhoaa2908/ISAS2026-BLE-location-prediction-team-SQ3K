#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import evaluation_core as CORE

COMMON = CORE.COMMON
CONFIRM_REPS = [10, 11, 12, 13]
FOLDS = 5
DEFAULT_TREES = 1200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the locked repeated session-grouped DLCS confirmation."
    )
    parser.add_argument("--total-zip", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["confirm"],
        default="confirm",
        help="The public reproducibility package exposes the locked confirmation stage only.",
    )
    parser.add_argument("--trees", type=int, default=DEFAULT_TREES)
    parser.add_argument(
        "--confirm-config",
        default="strong_plus_local",
        choices=["strong_plus_local"],
        help="Frozen directional-gate policy used in the paper.",
    )
    return parser.parse_args()


def prepare_dataset(total_zip: Path, outdir: Path) -> Path:
    workspace = outdir / "_workspace"
    dataset_root = workspace / "Dataset"

    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(total_zip) as archive:
        archive.extractall(workspace)

    if not dataset_root.exists():
        candidates = [p for p in workspace.rglob("train_fingerprints.csv")]
        if len(candidates) != 1:
            raise FileNotFoundError(
                "Could not identify the extracted Dataset directory from --total-zip."
            )
        dataset_root = candidates[0].parent

    required = [
        dataset_root / "train_fingerprints.csv",
        dataset_root / "5f_ble_merged_clean(friend_data).csv",
        dataset_root / "model_core.py",
        dataset_root / "models" / "train_resample.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing runtime files:\n" + "\n".join(missing))

    return dataset_root


def smooth_predictions(
    prediction: np.ndarray,
    data: dict,
    classes: list[str],
) -> np.ndarray:
    class_to_idx = {room: index for index, room in enumerate(classes)}
    encoded = np.asarray([class_to_idx[str(value)] for value in prediction], dtype=int)
    smoothed = COMMON.apply_run_fixed(
        encoded,
        data["session_rows"],
        class_to_idx,
        classes,
    )
    return np.asarray([classes[int(index)] for index in smoothed], dtype=str)


def metrics(y_true: np.ndarray, prediction: np.ndarray, classes: list[str]) -> dict:
    return {
        "macro_f1": float(
            f1_score(
                y_true,
                prediction,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
    }


def run_confirmation(
    dataset_root: Path,
    outdir: Path,
    specialist_trees: int,
) -> None:
    sys.path[:0] = [str(dataset_root), str(dataset_root / "models")]
    import model_core

    train_resample = CORE.import_train_resample(dataset_root)
    sparse = COMMON.load_sparse_ble(dataset_root)
    data = COMMON.build_bridge_data(dataset_root, sparse, model_core)

    y = data["y"].astype(str).to_numpy()
    classes = sorted(np.unique(y).tolist())
    groups = np.asarray(data["session_groups"])
    X = data["original_X"]

    by_rep_rows: list[dict] = []
    gate_rows: list[dict] = []

    cache_dir = outdir / "prediction_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for rep in CONFIRM_REPS:
        splitter = StratifiedGroupKFold(
            n_splits=FOLDS,
            shuffle=True,
            random_state=rep,
        )

        base_oof = np.empty(len(y), dtype=object)
        dlcs_oof = np.empty(len(y), dtype=object)

        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(X, y, groups=groups)
        ):
            print(
                f"rep={rep} fold={fold + 1}/{FOLDS} "
                f"train={len(train_idx)} test={len(test_idx)}",
                flush=True,
            )

            cache = cache_dir / f"rep{rep}_fold{fold}_sp{specialist_trees}.npz"
            result = CORE.fit_predict_models(
                data,
                train_resample,
                np.asarray(train_idx, dtype=int),
                np.asarray(test_idx, dtype=int),
                classes,
                specialist_trees,
                cache_path=cache,
                force=False,
            )

            base = result["base_pred"].astype(str)
            corrected, audit = CORE.apply_v29_hard_gate(
                data,
                np.asarray(test_idx, dtype=int),
                base,
                result["base_proba"],
                result["spec_proba"],
                classes,
                alpha=0.75,
            )

            base_oof[test_idx] = base
            dlcs_oof[test_idx] = corrected

            row = {
                "rep": rep,
                "fold": fold,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "activated": int(audit.get("activated", 0)),
                "overridden": int(audit.get("overridden", 0)),
                "blocked_protect": int(audit.get("blocked_protect", 0)),
                "blocked_target": int(audit.get("blocked_target", 0)),
                "blocked_cluster": int(audit.get("blocked_cluster", 0)),
                "blocked_margin": int(audit.get("blocked_margin", 0)),
                "pairs": json.dumps(audit.get("pairs", {}), sort_keys=True),
            }
            gate_rows.append(row)

        if pd.isna(pd.Series(base_oof)).any() or pd.isna(pd.Series(dlcs_oof)).any():
            raise RuntimeError(f"Incomplete out-of-fold predictions for repetition {rep}")

        base_smoothed = smooth_predictions(base_oof.astype(str), data, classes)
        dlcs_smoothed = smooth_predictions(dlcs_oof.astype(str), data, classes)

        base_metrics = metrics(y, base_smoothed, classes)
        dlcs_metrics = metrics(y, dlcs_smoothed, classes)

        rep_gate = [row for row in gate_rows if row["rep"] == rep]
        activated = sum(row["activated"] for row in rep_gate)
        overridden = sum(row["overridden"] for row in rep_gate)

        by_rep_rows.append(
            {
                "rep": rep,
                "baseline_macro": base_metrics["macro_f1"],
                "candidate_macro": dlcs_metrics["macro_f1"],
                "paired_delta": dlcs_metrics["macro_f1"] - base_metrics["macro_f1"],
                "baseline_accuracy": base_metrics["accuracy"],
                "candidate_accuracy": dlcs_metrics["accuracy"],
                "accuracy_delta": dlcs_metrics["accuracy"] - base_metrics["accuracy"],
                "baseline_balanced_accuracy": base_metrics["balanced_accuracy"],
                "candidate_balanced_accuracy": dlcs_metrics["balanced_accuracy"],
                "activated": activated,
                "overridden": overridden,
            }
        )

        print(
            f"rep={rep} main={base_metrics['macro_f1']:.12f} "
            f"dlcs={dlcs_metrics['macro_f1']:.12f} "
            f"delta={dlcs_metrics['macro_f1'] - base_metrics['macro_f1']:+.12f}",
            flush=True,
        )

    by_rep = pd.DataFrame(by_rep_rows)
    by_rep.to_csv(outdir / "confirmation_by_rep.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(outdir / "confirmation_gate_audit.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "baseline_macro_mean": float(by_rep["baseline_macro"].mean()),
                "baseline_macro_std": float(by_rep["baseline_macro"].std(ddof=1)),
                "candidate_macro_mean": float(by_rep["candidate_macro"].mean()),
                "candidate_macro_std": float(by_rep["candidate_macro"].std(ddof=1)),
                "paired_delta_mean": float(by_rep["paired_delta"].mean()),
                "paired_delta_std": float(by_rep["paired_delta"].std(ddof=1)),
                "baseline_accuracy_mean": float(by_rep["baseline_accuracy"].mean()),
                "candidate_accuracy_mean": float(by_rep["candidate_accuracy"].mean()),
                "paired_accuracy_delta": float(by_rep["accuracy_delta"].mean()),
                "baseline_balanced_accuracy_mean": float(
                    by_rep["baseline_balanced_accuracy"].mean()
                ),
                "candidate_balanced_accuracy_mean": float(
                    by_rep["candidate_balanced_accuracy"].mean()
                ),
                "positive_macro_repetitions": int((by_rep["paired_delta"] > 0).sum()),
                "confirmation_repetitions": len(by_rep),
                "specialist_trees": specialist_trees,
                "gate_config": "strong_plus_local",
            }
        ]
    )
    summary.to_csv(outdir / "confirmation_summary.csv", index=False)

    print("\nConfirmation summary", flush=True)
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    total_zip = args.total_zip.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_root = prepare_dataset(total_zip, outdir)
    run_confirmation(dataset_root, outdir, args.trees)


if __name__ == "__main__":
    main()
