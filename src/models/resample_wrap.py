#!/usr/bin/env python3
"""Leak-safe in-fold resampling used by the locked hierarchical classifier."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE


def floor_strategy(y, target):
    if target is None or target <= 0:
        return 'auto'
    counts = y.value_counts()
    return {c: target for c, n in counts.items() if n < target}


def borderline_resample(X, y, target, seed):
    """Exact locked Borderline-SMOTE-1 procedure.

    Classes that would not have enough neighbors are first raised to six samples
    with random oversampling. Borderline-SMOTE then raises all classes below the
    requested target. Synthetic samples are generated inside the training fold only.
    """
    target_classes = list(floor_strategy(y, target).keys())
    counts = y.value_counts()
    pre = {c: 6 for c in target_classes if counts.get(c, 0) < 6}
    X_work, y_work = X, y
    if pre:
        X_work, y_work = RandomOverSampler(
            sampling_strategy=pre, random_state=seed).fit_resample(X_work, y_work)
    counts = y_work.value_counts()
    candidates = [counts[c] for c in target_classes if counts.get(c, 0) > 0]
    minimum = min(candidates) if candidates else 6
    k = max(1, min(5, int(minimum) - 1))
    sampler = BorderlineSMOTE(
        k_neighbors=k,
        kind='borderline-1',
        random_state=0,
        sampling_strategy=floor_strategy(y_work, target),
    )
    return sampler.fit_resample(X_work, y_work)


class ResampledClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_factory=None, method='borderline1', target=200, seed=0):
        self.base_factory = base_factory
        self.method = method
        self.target = target
        self.seed = seed

    def fit(self, X, y):
        X_frame = pd.DataFrame(np.asarray(X))
        y_series = pd.Series(np.asarray(y))
        if self.method != 'borderline1':
            raise ValueError('The reproducibility implementation supports the locked borderline1 method only.')
        X_resampled, y_resampled = borderline_resample(
            X_frame, y_series, self.target, self.seed)
        self.n_before_ = len(y_series)
        self.n_after_ = len(y_resampled)
        self.base_ = self.base_factory()
        self.base_.fit(np.asarray(X_resampled), np.asarray(y_resampled))
        self.classes_ = np.unique(np.asarray(y_resampled))
        return self

    def predict(self, X):
        return self.base_.predict(np.asarray(X))

    def predict_proba(self, X):
        return self.base_.predict_proba(np.asarray(X))
