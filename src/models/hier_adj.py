#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hier_adj.py — thành phần MODEL-SIDE dùng chung cho [4] hierarchical & [5b] adjacency-cost
=========================================================================================

CHỈ tầng model (kiến trúc/quyết định), KHÔNG đụng dữ liệu/feature. Dùng làm base cho
train_hier.py và train_adjcost.py; đo bằng eval_robust.py (chống rò rỉ, cross-model).

Gồm:
  - make_base_factory(base, seed, n_jobs): trả callable tạo base estimator MỚI (rf/hgb/lgbm).
  - build_zone_map(scheme): dict room->zone từ config/room_centers.json (geo5 / topo2).
  - build_adjacency(penalty, thresh): ma trận chi phí C[k,j] cho adjacency-aware.
  - HierarchicalClassifier: tier-1 route theo zone -> tier-2 phân loại phòng trong zone.
  - CostSensitiveClassifier: quyết định Bayes-risk tối thiểu dưới ma trận chi phí adjacency
    (áp lên predict_proba của base CỐ ĐỊNH — không đổi training).

Leak-safe: mọi thứ fit CHỈ trên train fold; zone_map/adjacency lấy từ TOẠ ĐỘ bản đồ
(config/), không từ phân bố nhãn -> không rò rỉ.
"""
import json
import os
import sys

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))

ROOM_CENTERS = os.path.join(ROOT, "config", "room_centers.json")
COMMON_ROOMS = {"kitchen", "cafeteria", "nurse station", "cleaning", "hallway"}


def _strip_class_weight(est):
    from sklearn.pipeline import Pipeline
    if isinstance(est, Pipeline):
        last = est.steps[-1][0]
        try:
            est.set_params(**{f"{last}__class_weight": None})
        except ValueError:
            pass
    return est


def make_base_factory(base, seed, n_jobs=-1, class_weight="keep"):
    def _wrap(f):
        return (lambda: _strip_class_weight(f())) if class_weight == "none" else f

    if base == "rf":
        import train_rf
        args = train_rf.parse_args([]); args.seed = seed; args.n_jobs = n_jobs
        return _wrap(lambda: train_rf.build_model(args))
    if base == "hgb":
        import train_hgb
        args = train_hgb.parse_args([]); args.seed = seed
        return _wrap(lambda: train_hgb.build_model(args))
    if base == "lgbm":
        import train_lgbm
        args = train_lgbm.parse_args([]); args.seed = seed
        return _wrap(lambda: train_lgbm.build_model(args))
    raise SystemExit(f"[LỖI] base không hợp lệ: {base}. Chọn rf/hgb/lgbm.")


def _load_centers():
    with open(ROOM_CENTERS) as f:
        data = json.load(f)
    return {k: tuple(v) for k, v in data.items() if not k.startswith("_")}


def build_zone_map(scheme="geo5"):
    centers = _load_centers()
    zmap = {}
    for room, (x, y) in centers.items():
        if room in COMMON_ROOMS:
            zmap[room] = "central"
            continue
        if scheme == "topo2":
            zmap[room] = "numbered"
        else:
            vert = "top" if y > 18 else "bottom"
            horiz = "left" if x < 30 else "right"
            zmap[room] = f"{vert}_{horiz}"
    return zmap


def build_cost_matrix(classes, penalty=0.3, thresh=11.0):
    centers = _load_centers()
    n = len(classes)
    C = np.ones((n, n))
    for i, ci in enumerate(classes):
        C[i, i] = 0.0
        pi = centers.get(ci)
        for j, cj in enumerate(classes):
            if i == j:
                continue
            pj = centers.get(cj)
            if pi is not None and pj is not None:
                d = np.hypot(pi[0] - pj[0], pi[1] - pj[1])
                if d <= thresh:
                    C[i, j] = penalty
    return C


def fit_with_sample_weight(est, X, y, w=None):
    from sklearn.pipeline import Pipeline
    if w is None:
        est.fit(X, y)
    elif isinstance(est, Pipeline):
        est.fit(X, y, **{f"{est.steps[-1][0]}__sample_weight": w})
    else:
        est.fit(X, y, sample_weight=w)
    return est


class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_factory=None, zone_map=None):
        self.base_factory = base_factory
        self.zone_map = zone_map

    def _zone_of(self, y):
        return np.array([self.zone_map.get(r, "central") for r in y])

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X)
        y = np.asarray(y)
        w = None if sample_weight is None else np.asarray(sample_weight)
        self.classes_ = np.unique(y)
        self.class_pos_ = {c: i for i, c in enumerate(self.classes_)}
        z = self._zone_of(y)
        self.tier1_ = fit_with_sample_weight(self.base_factory(), X, z, w)
        self.tier2_, self.zone_default_ = {}, {}
        for zone in np.unique(z):
            mask = z == zone
            yz = y[mask]
            if np.unique(yz).size == 1:
                self.zone_default_[zone] = yz[0]
            else:
                self.tier2_[zone] = fit_with_sample_weight(
                    self.base_factory(), X[mask], yz, None if w is None else w[mask])
        return self

    def predict(self, X):
        X = np.asarray(X)
        z_pred = self.tier1_.predict(X)
        out = np.empty(len(X), dtype=object)
        for zone in np.unique(z_pred):
            idx = np.where(z_pred == zone)[0]
            if zone in self.zone_default_:
                out[idx] = self.zone_default_[zone]
            elif zone in self.tier2_:
                out[idx] = self.tier2_[zone].predict(X[idx])
            else:
                out[idx] = self.classes_[0]
        return out

    def predict_proba(self, X):
        X = np.asarray(X)
        Pz = self.tier1_.predict_proba(X)
        P = np.zeros((len(X), len(self.classes_)))
        for zi, zone in enumerate(self.tier1_.classes_):
            pz = Pz[:, zi]
            if zone in self.zone_default_:
                P[:, self.class_pos_[self.zone_default_[zone]]] += pz
            elif zone in self.tier2_:
                m = self.tier2_[zone]
                P2 = m.predict_proba(X)
                for j, room in enumerate(m.classes_):
                    P[:, self.class_pos_[room]] += pz * P2[:, j]
        s = P.sum(axis=1, keepdims=True)
        return np.divide(P, s, out=np.full_like(P, 1.0 / P.shape[1]), where=s > 0)


class CostSensitiveClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_factory=None, penalty=0.3, thresh=11.0):
        self.base_factory = base_factory
        self.penalty = penalty
        self.thresh = thresh

    def fit(self, X, y):
        self.base_ = self.base_factory()
        self.base_.fit(np.asarray(X), np.asarray(y))
        self.classes_ = self.base_.classes_
        self.C_ = build_cost_matrix(list(self.classes_), self.penalty, self.thresh)
        return self

    def predict(self, X):
        P = self.base_.predict_proba(np.asarray(X))
        risk = P @ self.C_
        return self.classes_[risk.argmin(axis=1)]
