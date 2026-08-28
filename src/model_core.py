#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared data-selection, leakage-aware splitting, evaluation, and reporting utilities.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

from sklearn.model_selection import (
    train_test_split, GroupShuffleSplit, GroupKFold, StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, cohen_kappa_score, confusion_matrix, classification_report,
)

DEFAULT_META = ["date", "win_start", "win_end", "win_start_s", "win_end_s", "time"]
START_COL_CANDIDATES = ["win_start", "time", "start_time"]


def add_common_args(p):
    p.add_argument("-i", "--input", default="train_fingerprints.csv")
    p.add_argument("--label", default=None)
    p.add_argument("--drop", default="")
    p.add_argument("-o", "--outdir", default=".")
    p.add_argument("--split-mode", choices=["group", "day", "random"], default="group")
    p.add_argument("--cv", type=int, default=0)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-numeric-frac", type=float, default=0.5)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p


def resolve_start_col(df):
    for c in START_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def pick_label_column(df, requested):
    if requested:
        if requested not in df.columns:
            sys.exit(f"[ERROR] label column '{requested}' not found")
        return requested
    if "room" in df.columns:
        return "room"
    return df.columns[-1]


def select_features(df, label_col, extra_drop, min_numeric_frac):
    drop = set(DEFAULT_META) | set(extra_drop) | {label_col}
    meta_present = [c for c in df.columns if c in drop and c != label_col]
    candidates = [c for c in df.columns if c not in drop]
    feat, dropped_low, nan_counts = [], [], {}
    n = len(df)
    for c in candidates:
        col = pd.to_numeric(df[c], errors="coerce")
        frac = col.notna().sum() / max(n, 1)
        if frac >= min_numeric_frac:
            df[c] = col
            feat.append(c)
            miss = int(col.isna().sum())
            if miss:
                nan_counts[c] = miss
        else:
            dropped_low.append((c, round(frac, 3)))
    return feat, {"meta_dropped": meta_present,
                  "nonnumeric_dropped": dropped_low,
                  "nan_counts": nan_counts}


def make_groups(df, start_col="win_start"):
    p = df[start_col].astype(str).str.split(":", expand=True).astype(float)
    ws = p[0] * 3600 + p[1] * 60 + p[2]
    d = ws.diff()
    step = d[d > 0].median()
    if not np.isfinite(step) or step <= 0:
        step = 1.0
    new_group = (df["date"] != df["date"].shift()) | (d > 1.5 * step) | (d < 0)
    return new_group.cumsum().to_numpy()


def split_data(df, X, y, args):
    mode = args.split_mode
    if mode == "group":
        start_col = resolve_start_col(df)
        if start_col is None or "date" not in df.columns:
            mode = "random"
        else:
            groups = make_groups(df, start_col)
            gss = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
            tr, va = next(gss.split(X, y, groups=groups))
            return X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va], f"group (sessions={len(np.unique(groups))})"
    if mode == "day":
        if "date" not in df.columns:
            mode = "random"
        else:
            groups = df["date"].to_numpy()
            gss = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
            tr, va = next(gss.split(X, y, groups=groups))
            val_days = sorted(pd.unique(df["date"].to_numpy()[va]))
            return X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va], f"day, val={val_days}"
    vc = y.value_counts()
    stratify = y if vc.min() >= 2 else None
    Xtr, Xval, ytr, yval = train_test_split(
        X, y, test_size=args.val_size, random_state=args.seed, stratify=stratify)
    return Xtr, Xval, ytr, yval, "random stratified" if stratify is not None else "random"


def cv_groups(df, X, y, args):
    if args.split_mode == "group":
        start_col = resolve_start_col(df)
        if start_col is not None and "date" in df.columns:
            return GroupKFold(n_splits=args.cv), make_groups(df, start_col)
    if args.split_mode == "day" and "date" in df.columns:
        return GroupKFold(n_splits=args.cv), df["date"].to_numpy()
    return StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed), None


def compute_metrics(y_true, y_pred):
    labels = sorted(set(pd.unique(y_true)) | set(pd.unique(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "labels": labels, "cm": cm,
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_prec": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_rec": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "per_class_acc": cm.diagonal() / np.clip(cm.sum(axis=1), 1, None),
    }


def write_outputs(m, y_true, y_pred, header_lines, args, extra_block=""):
    stem = os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(args.outdir, exist_ok=True)
    cm_path = os.path.join(args.outdir, f"confusion_matrix_{stem}.csv")
    rep_path = os.path.join(args.outdir, f"report_{stem}.txt")
    labels, cm = m["labels"], m["cm"]
    pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels]).to_csv(cm_path)
    with open(rep_path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        if extra_block:
            f.write(extra_block + "\n")
        f.write(f"Accuracy: {m['acc']:.6f}\nBalanced accuracy: {m['bal_acc']:.6f}\nMacro F1: {m['macro_f1']:.6f}\n")
        f.write(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    return cm_path, rep_path


def run(args, build_model, model_name):
    if not os.path.exists(args.input):
        sys.exit(f"[ERROR] file not found: {args.input}")
    df = pd.read_csv(args.input)
    label_col = pick_label_column(df, args.label)
    extra_drop = [c.strip() for c in args.drop.split(",") if c.strip()]
    feat_cols, _ = select_features(df, label_col, extra_drop, args.min_numeric_frac)
    if not feat_cols:
        sys.exit("[ERROR] no numeric features found")
    df = df[df[label_col].notna()].copy().reset_index(drop=True)
    X = df[feat_cols].reset_index(drop=True)
    y = df[label_col].astype(str).reset_index(drop=True)

    if args.cv and args.cv >= 2:
        splitter, groups = cv_groups(df, X, y, args)
        oof_true = np.empty(len(y), dtype=object)
        oof_pred = np.empty(len(y), dtype=object)
        filled = np.zeros(len(y), dtype=bool)
        fold_f1, fold_acc, fold_bacc = [], [], []
        for k, (tr_idx, te_idx) in enumerate(splitter.split(X, y, groups), start=1):
            model = build_model(args)
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pk = model.predict(X.iloc[te_idx])
            oof_true[te_idx] = y.iloc[te_idx].to_numpy(); oof_pred[te_idx] = pk; filled[te_idx] = True
            fold_f1.append(f1_score(y.iloc[te_idx], pk, average="macro", zero_division=0))
            fold_acc.append(accuracy_score(y.iloc[te_idx], pk))
            fold_bacc.append(balanced_accuracy_score(y.iloc[te_idx], pk))
            print(k, fold_f1[-1], fold_acc[-1])
        yt, yp = oof_true[filled], oof_pred[filled]
        m = compute_metrics(yt, yp)
        header = [f"{model_name} cross-validation", f"input: {args.input}", f"features: {len(feat_cols)}"]
        write_outputs(m, yt, yp, header, args)
        print(m)
        return

    Xtr, Xval, ytr, yval, split_info = split_data(df, X, y, args)
    model = build_model(args)
    model.fit(Xtr, ytr)
    y_pred = model.predict(Xval)
    m = compute_metrics(yval, y_pred)
    header = [f"{model_name}", f"input: {args.input}", f"split: {split_info}", f"features: {len(feat_cols)}"]
    write_outputs(m, yval, y_pred, header, args)
    print(m)
