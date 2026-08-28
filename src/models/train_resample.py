#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_resample.py — [5c] Resample lớp hiếm (đồng bộ từ explore_method_resample/)
================================================================================

Keeper của session phụ: **BorderlineSMOTE-1, nâng sàn T=200**, LEAK-SAFE chỉ trong fold train.
Ở đây bọc thành entry point chuẩn dự án (model_core.run) để đo bằng eval_robust/eval_model_ab
và **KẾT HỢP với hierarchical [4]** (--hier topo2).

Ví dụ:
  python3 eval_model_ab.py -i train_fingerprints.csv \
      -m hgb,resample:hgb:borderline1:200,resample:hgb:borderline1:200:topo2 --reps 10
  python3 models/train_resample.py -i train_fingerprints.csv --cv 5 --outdir all_output/exp_model
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_core
import hier_adj
import resample_wrap

MODEL_NAME = "RESAMPLE"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="[5c] Resample lớp hiếm (leak-safe in-fold), tuỳ chọn kết hợp hierarchical.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    model_core.add_common_args(p)
    p.add_argument("--base", choices=["rf", "hgb", "lgbm"], default="rf")
    p.add_argument("--method", default="borderline1",
                   help="ros,smote,borderline1,borderline2,smotetomek,smoteenn,... (keeper: borderline1)")
    p.add_argument("--target", type=int, default=200,
                   help="Nâng sàn: lớp có < T mẫu -> T. (keeper: 200)")
    p.add_argument("--hier", choices=["none", "geo5", "topo2"], default="none",
                   help="Kết hợp [4]: resample TRƯỚC, rồi hierarchical train trên dữ liệu đã resample.")
    return p.parse_args(argv)


def build_model(args):
    base_fac = hier_adj.make_base_factory(args.base, args.seed, args.n_jobs)
    if args.hier != "none":
        zmap = hier_adj.build_zone_map(args.hier)
        inner = lambda: hier_adj.HierarchicalClassifier(base_factory=base_fac, zone_map=zmap)  # noqa: E731
    else:
        inner = base_fac
    return resample_wrap.ResampledClassifier(
        base_factory=inner, method=args.method, target=args.target, seed=args.seed)


def main(argv=None):
    args = parse_args(argv)
    tag = f"{MODEL_NAME}({args.method},T={args.target},base={args.base}" \
          + ("" if args.hier == "none" else f",hier={args.hier}") + ")"
    model_core.run(args, build_model, tag)


if __name__ == "__main__":
    main()
