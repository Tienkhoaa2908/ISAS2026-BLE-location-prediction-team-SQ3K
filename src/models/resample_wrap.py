#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resample_wrap.py — ĐỒNG BỘ resampling (session phụ) vào pipeline chính
======================================================================

Nguồn gốc: `explore_method_resample/` (session song song, 2026-07-11). Keeper của họ =
**BorderlineSMOTE-1, nâng sàn T=200**, resample LEAK-SAFE chỉ trong fold train.

File này KHÔNG chép lại thuật toán — nó **tái dùng `resample_train()` của
`explore_method_resample/eval_resample.py`** (một nguồn sự thật, tránh trôi lệch giữa 2 folder).
Việc của nó là bọc resample thành **estimator sklearn** để:
  1. Cắm được vào `eval_robust.py` / `eval_model_ab.py` như một model bình thường.
  2. **KẾT HỢP** được với hierarchical [4] (resample trước, rồi hierarchical train trên dữ liệu đã resample).

⚠️ LEAK-SAFE do CẤU TRÚC: `fit()` chỉ nhìn thấy fold TRAIN (splitter gọi fit trên X_train),
nên mẫu tổng hợp KHÔNG bao giờ lọt sang test. Đây là lý do resample KHÔNG được "bake" sẵn
vào train CSV dùng chung (sẽ rò rỉ) — phải là bước TRONG pipeline train.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "models"), os.path.join(ROOT, "explore_method_resample")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_resample import resample_train  # noqa: E402  (một nguồn sự thật — session phụ)


class ResampledClassifier(BaseEstimator, ClassifierMixin):
    """Resample (X,y) của fold TRAIN rồi fit base. base_factory: callable -> estimator mới."""

    def __init__(self, base_factory=None, method="borderline1", target=200, seed=0):
        self.base_factory = base_factory
        self.method = method
        self.target = target
        self.seed = seed

    def fit(self, X, y):
        Xs = pd.DataFrame(np.asarray(X))
        ys = pd.Series(np.asarray(y))
        X2, y2 = resample_train(self.method, Xs, ys, self.target, None, self.seed)
        self.n_before_, self.n_after_ = len(ys), len(y2)
        self.base_ = self.base_factory()
        self.base_.fit(np.asarray(X2), np.asarray(y2))
        self.classes_ = np.unique(np.asarray(y2))
        return self

    def predict(self, X):
        return self.base_.predict(np.asarray(X))

    def predict_proba(self, X):
        return self.base_.predict_proba(np.asarray(X))
