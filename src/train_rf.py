#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_rf.py — Random Forest (MÔ HÌNH TRỤC CHÍNH của dự án)
=========================================================

Đây là file model "gốc" dùng để phản ánh khả năng cải thiện của DỮ LIỆU/FEATURE.
Toàn bộ máy chung (nhận diện feature an toàn, chia dữ liệu CHỐNG RÒ RỈ, cross-
validation OOF, xuất confusion matrix + report) nằm ở model_core.py; file này chỉ
định nghĩa phần riêng của Random Forest. Các mô hình khác (hgb, lgbm, ...) là các
file độc lập cùng cấu trúc trong thư mục models/.

Đầu ra (đặt tên theo file train, ghi vào --outdir):
  - confusion_matrix_<stem>.csv, report_<stem>.txt

Ví dụ:
  python3 train_rf.py -i train_fingerprints.csv --cv 5 --outdir all_output/output
  python3 train_rf.py -i data.csv --split-mode day
  python3 train_rf.py --help
"""

import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import model_core

MODEL_NAME = "RANDOM FOREST"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Huấn luyện Random Forest phân loại phòng từ file fingerprint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    model_core.add_common_args(p)
    # Tham số riêng của Random Forest
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--min-samples-leaf", type=int, default=1)
    p.add_argument("--max-features", default="sqrt",
                   help="max_features của RF: 'sqrt','log2', số thực, hoặc 'none'.")
    p.add_argument("--class-weight", default="balanced_subsample",
                   help="'balanced', 'balanced_subsample', hoặc 'none'.")
    return p.parse_args(argv)


def build_model(args):
    """Pipeline SimpleImputer(median) -> RandomForest."""
    max_features = args.max_features
    if str(max_features).lower() == "none":
        max_features = None
    else:
        try:
            max_features = float(max_features)
        except ValueError:
            pass
    class_weight = None if str(args.class_weight).lower() == "none" else args.class_weight
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        )),
    ])


def main(argv=None):
    args = parse_args(argv)
    model_core.run(args, build_model, MODEL_NAME)


if __name__ == "__main__":
    main()
