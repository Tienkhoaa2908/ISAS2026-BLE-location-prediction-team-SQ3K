from __future__ import annotations
"""Core fitting, gating and fixed-22-class evaluation functions for DLCS."""
import importlib.util
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('dlcs_common', HERE / 'common.py')
COMMON = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(COMMON)
EPS = 1e-12
PAPER_SCOPE = ["501","502","503","506","508","510","511","512","513","515","518","520","522","523"]
HARD_PROTECT_BASE = {"501","502","516","522","523"}
TARGET_CLUSTERS = {
    "503":{"502","503","506"}, "508":{"508","511","520"}, "510":{"510","511","512"},
    "513":{"501","513","515"}, "515":{"501","513","515"},
    "518":{"515","517","518","520","522"}, "520":{"518","520","522","523"}}
V29_TARGETS = {"508","510","513","515","518","520"}
PAIR_MARGIN_OVERRIDES = {("502","503"):0.10,("522","520"):0.12,("523","520"):0.12,("501","513"):0.10,("501","515"):0.10}


def import_train_resample(root):
    spec = importlib.util.spec_from_file_location('train_resample_locked', root / 'models' / 'train_resample.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def aligned_probability(model, X, classes):
    raw = model.predict_proba(X)
    out = np.full((len(X), len(classes)), EPS, dtype=float)
    position = {c:i for i,c in enumerate(classes)}
    for column, room in enumerate(model.classes_):
        room = str(room)
        if room in position:
            out[:, position[room]] = np.clip(raw[:, column], EPS, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


def fit_predict_models(data, train_resample, train_idx, test_idx, classes, specialist_trees, cache_path=None, force=False):
    if cache_path is not None and Path(cache_path).exists() and not force:
        z = np.load(cache_path, allow_pickle=False)
        return {'test_idx':z['test_idx'].astype(int),'base_pred':z['base_pred'].astype(str),
                'base_proba':z['base_proba'].astype(float),'spec_proba':z['spec_proba'].astype(float)}

    X = data['original_X']; y = data['y'].astype(str).to_numpy(); context = data['context_X']
    context_key = data['frame']['context_key'].astype(str).to_numpy()
    base = COMMON.make_exact_champion(train_resample)
    base.fit(X.iloc[train_idx], y[train_idx])
    base_pred = np.asarray(base.predict(X.iloc[test_idx]), dtype=str)
    base_proba = aligned_probability(base, X.iloc[test_idx], classes)

    scope_train = train_idx[np.isin(y[train_idx], PAPER_SCOPE)]
    test_contexts = set(context_key[test_idx])
    scope_train = np.asarray([i for i in scope_train if context_key[i] not in test_contexts], dtype=int)
    spec_proba = np.zeros((len(test_idx), len(classes)), dtype=float)
    if len(scope_train) and len(np.unique(y[scope_train])) >= 2:
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(context.iloc[scope_train]); X_test = imputer.transform(context.iloc[test_idx])
        specialist = RandomForestClassifier(n_estimators=specialist_trees, class_weight='balanced_subsample',
                                            max_features='sqrt', random_state=42, n_jobs=-1)
        specialist.fit(X_train, y[scope_train]); raw = specialist.predict_proba(X_test)
        position = {c:i for i,c in enumerate(classes)}
        for column, room in enumerate(specialist.classes_):
            room = str(room)
            if room in position:
                spec_proba[:, position[room]] = raw[:, column]
        scope_indices = [position[r] for r in PAPER_SCOPE if r in position]
        local = spec_proba[:, scope_indices]; denom = local.sum(axis=1, keepdims=True)
        spec_proba[:, scope_indices] = np.divide(local, np.maximum(denom, EPS), out=np.zeros_like(local))

    if cache_path is not None:
        cache_path = Path(cache_path); cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, test_idx=np.asarray(test_idx,int), base_pred=base_pred.astype('U32'),
                            base_proba=base_proba, spec_proba=spec_proba)
    return {'test_idx':test_idx,'base_pred':base_pred,'base_proba':base_proba,'spec_proba':spec_proba}


def apply_v29_hard_gate(data, rows, base_pred, base_proba, spec_proba, classes, alpha=0.75):
    position = {c:i for i,c in enumerate(classes)}
    scope = [r for r in PAPER_SCOPE if r in position]; scope_indices = np.asarray([position[r] for r in scope], dtype=int)
    scope_set = set(scope)
    mass = base_proba[:, scope_indices].sum(axis=1, keepdims=True)
    normalized = np.divide(base_proba[:, scope_indices], np.maximum(mass, EPS), out=np.zeros_like(base_proba[:, scope_indices]))
    blend = (1-alpha) * normalized + alpha * spec_proba[:, scope_indices]
    candidate = np.asarray([scope[i] for i in blend.argmax(axis=1)], dtype=str)
    order = np.argsort(base_proba, axis=1)
    top1 = np.asarray([classes[i] for i in order[:, -1]], dtype=str); top2 = np.asarray([classes[i] for i in order[:, -2]], dtype=str)
    margin = base_proba[np.arange(len(rows)), order[:, -1]] - base_proba[np.arange(len(rows)), order[:, -2]]
    available = pd.to_numeric(data['frame'].iloc[rows]['ctx50_available'], errors='coerce').fillna(0).to_numpy() > 0
    output = base_pred.copy(); counters = Counter(); pairs = Counter(); local_column = {r:i for i,r in enumerate(scope)}
    for i in range(len(rows)):
        base_room, cand_room = str(base_pred[i]), str(candidate[i])
        if not available[i] or base_room not in scope_set or top1[i] not in scope_set or top2[i] not in scope_set or margin[i] > 0.30:
            continue
        counters['activated'] += 1
        if cand_room == base_room: continue
        if base_room in HARD_PROTECT_BASE: counters['blocked_protect'] += 1; continue
        if cand_room not in V29_TARGETS: counters['blocked_target'] += 1; continue
        cluster = TARGET_CLUSTERS.get(cand_room, set())
        if cluster and not {top1[i], top2[i]}.issubset(cluster): counters['blocked_cluster'] += 1; continue
        required = PAIR_MARGIN_OVERRIDES.get((base_room, cand_room), 0.03)
        source_score = blend[i, local_column[base_room]] if base_room in local_column else 0.0
        candidate_score = blend[i, local_column[cand_room]]
        if candidate_score - source_score < required: counters['blocked_margin'] += 1; continue
        output[i] = cand_room; counters['overridden'] += 1; pairs[f'{base_room}->{cand_room}'] += 1
    audit = dict(counters); audit['pairs'] = dict(pairs)
    return output, audit


def fixed_metrics(y_true, prediction, classes):
    per_class = f1_score(y_true, prediction, labels=list(classes), average=None, zero_division=0)
    return {'macro_f1':float(f1_score(y_true,prediction,labels=list(classes),average='macro',zero_division=0)),
            'accuracy':float(accuracy_score(y_true,prediction)),
            'balanced_accuracy':float(balanced_accuracy_score(y_true,prediction)),
            'zero_f1':int(np.sum(per_class == 0)), 'per_class':per_class}


def subset_session_rows(data, rows):
    global_to_local = {int(g):i for i,g in enumerate(rows)}; output = {}
    for session, global_rows in data['session_rows'].items():
        keep = [global_to_local[int(g)] for g in global_rows if int(g) in global_to_local]
        if keep: output[int(session)] = np.asarray(keep, dtype=int)
    return output


def exact_centered_smoothing(prediction, data, rows, classes):
    position = {c:i for i,c in enumerate(classes)}
    indices = np.asarray([position[str(v)] for v in prediction], dtype=int)
    smoothed = COMMON.apply_run_fixed(indices, subset_session_rows(data, rows), position, classes)
    return np.asarray([classes[int(i)] for i in smoothed], dtype=str)
