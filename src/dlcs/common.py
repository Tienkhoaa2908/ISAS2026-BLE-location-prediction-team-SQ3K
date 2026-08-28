from __future__ import annotations
"""Locked 50-second context construction and temporal utilities for DLCS."""
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

PAPER_SCOPE = ["501","502","503","506","508","510","511","512","513","515","518","520","522","523"]
PROTECTED_CLUSTER = {"501", "513", "515"}
PAPER_WINDOW_SECONDS = 50


def normalize_room_label(value):
    text = str(value)
    if text.startswith('Room_'):
        try:
            return str(500 + int(text.split('_', 1)[1]))
        except ValueError:
            return text
    return text


def safe_skew(values):
    if len(values) < 3 or np.allclose(values, values[0]):
        return 0.0
    value = float(skew(values, bias=False))
    return value if np.isfinite(value) else 0.0


def safe_kurtosis(values):
    if len(values) < 4 or np.allclose(values, values[0]):
        return 0.0
    value = float(kurtosis(values, bias=False))
    return value if np.isfinite(value) else 0.0


def paper_feature_names(prefix=''):
    names = []
    for beacon in range(1, 26):
        for statistic in ['mean', 'count', 'max', 'std']:
            names.append(f'{prefix}b{beacon}_{statistic}')
    names.extend(f'{prefix}{name}' for name in [
        'rssi_mean','rssi_std','rssi_min','rssi_max','rssi_median',
        'rssi_q10','rssi_q25','rssi_q75','rssi_q90','rssi_iqr',
        'rssi_range','rssi_skew','rssi_kurt','sample_count','rssi_count',
        'coverage_cell','beacon_count_std','tod_sin','tod_cos'])
    assert len(names) == 119
    return names


def extract_119_features(group, rssi_columns, prefix=''):
    values = group[list(rssi_columns)].to_numpy(dtype=float)
    valid = np.isfinite(values) & (values > -110.0)
    result, beacon_counts = {}, []
    for index in range(25):
        observed = values[:, index][valid[:, index]]
        beacon = index + 1
        beacon_counts.append(len(observed))
        result[f'{prefix}b{beacon}_mean'] = float(observed.mean()) if len(observed) else -110.0
        result[f'{prefix}b{beacon}_count'] = float(len(observed))
        result[f'{prefix}b{beacon}_max'] = float(observed.max()) if len(observed) else -110.0
        result[f'{prefix}b{beacon}_std'] = float(observed.std(ddof=0)) if len(observed) > 1 else 0.0
    all_values = values[valid]
    if len(all_values) == 0:
        all_values = np.asarray([-110.0])
    q = np.quantile(all_values, [0.10, 0.25, 0.50, 0.75, 0.90])
    result.update({
        f'{prefix}rssi_mean': float(all_values.mean()), f'{prefix}rssi_std': float(all_values.std(ddof=0)),
        f'{prefix}rssi_min': float(all_values.min()), f'{prefix}rssi_max': float(all_values.max()),
        f'{prefix}rssi_median': float(q[2]), f'{prefix}rssi_q10': float(q[0]), f'{prefix}rssi_q25': float(q[1]),
        f'{prefix}rssi_q75': float(q[3]), f'{prefix}rssi_q90': float(q[4]), f'{prefix}rssi_iqr': float(q[3] - q[1]),
        f'{prefix}rssi_range': float(all_values.max() - all_values.min()), f'{prefix}rssi_skew': safe_skew(all_values),
        f'{prefix}rssi_kurt': safe_kurtosis(all_values), f'{prefix}sample_count': float(len(group)),
        f'{prefix}rssi_count': float(valid.sum()), f'{prefix}coverage_cell': float(min(1.0, len(group) / 25.0)),
        f'{prefix}beacon_count_std': float(np.std(beacon_counts, ddof=0)),
    })
    timestamp = pd.Timestamp(group['timestamp'].iloc[0])
    seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
    result[f'{prefix}tod_sin'] = float(math.sin(2 * math.pi * seconds / 86400))
    result[f'{prefix}tod_cos'] = float(math.cos(2 * math.pi * seconds / 86400))
    return result


def load_sparse_ble(dataset_root: Path):
    frame = pd.read_csv(dataset_root / '5f_ble_merged_clean(friend_data).csv')
    frame['room'] = frame['room_class'].map(normalize_room_label)
    frame['timestamp'] = pd.to_datetime(frame['date'].astype(str) + ' ' + frame['time'].astype(str))
    return frame.sort_values('timestamp').reset_index(drop=True)


def build_label_free_context_bins(sparse):
    rssi_columns = [f'RSSI_{index}' for index in range(1, 26)]
    frame = sparse.copy()
    frame['context_start'] = frame['timestamp'].dt.floor('50s')
    rows = []
    for (date, context_start), group in frame.groupby(['date', 'context_start'], sort=True):
        features = extract_119_features(group, rssi_columns, prefix='ctx50_')
        features.update({'date': str(date), 'context_start': pd.Timestamp(context_start), 'context_rows': len(group)})
        rows.append(features)
    return pd.DataFrame(rows)


def build_bridge_data(dataset_root: Path, sparse, model_core):
    fingerprints = pd.read_csv(dataset_root / 'train_fingerprints.csv')
    label_col = model_core.pick_label_column(fingerprints, None)
    feature_columns, _ = model_core.select_features(fingerprints, label_col, [], 0.5)
    fingerprints = fingerprints[fingerprints[label_col].notna()].copy().reset_index(drop=True)
    fingerprints['room'] = fingerprints[label_col].astype(str)
    start_col = model_core.resolve_start_col(fingerprints)
    session_groups = model_core.make_groups(fingerprints, start_col)
    start_timestamp = pd.to_datetime(fingerprints['date'].astype(str) + ' ' + fingerprints[start_col].astype(str), errors='coerce')
    midpoint = start_timestamp + pd.to_timedelta(5, unit='s')
    fingerprints['context_start'] = midpoint.dt.floor('50s')
    fingerprints['context_key'] = fingerprints['date'].astype(str) + '|' + fingerprints['context_start'].astype(str)
    context = build_label_free_context_bins(sparse)
    merged = fingerprints.merge(context, on=['date', 'context_start'], how='left', validate='many_to_one')
    context_features = paper_feature_names(prefix='ctx50_')
    merged['ctx50_available'] = merged[context_features].notna().any(axis=1).astype(float)
    for column in context_features:
        merged[column] = pd.to_numeric(merged[column], errors='coerce')
    original_X = merged[feature_columns].apply(pd.to_numeric, errors='coerce')
    context_X = merged[context_features + ['ctx50_available']].apply(pd.to_numeric, errors='coerce')
    timestamp_ns = start_timestamp.astype('int64').to_numpy()
    session_rows = {}
    for session in np.unique(session_groups):
        rows = np.where(session_groups == session)[0]
        session_rows[int(session)] = rows[np.argsort(timestamp_ns[rows])]
    return {'frame': merged, 'y': merged['room'].astype(str).reset_index(drop=True),
            'original_X': original_X.reset_index(drop=True), 'context_X': context_X.reset_index(drop=True),
            'session_groups': session_groups, 'session_rows': session_rows,
            'original_feature_count': len(feature_columns), 'context_feature_count': len(context_features) + 1}


def make_exact_champion(train_resample):
    args = train_resample.parse_args(['-i', 'dummy.csv'])
    args.base = 'rf'; args.hier = 'topo2'; args.method = 'borderline1'; args.target = 200; args.seed = 42; args.n_jobs = -1
    return train_resample.build_model(args)


def centered_vote(seq, n_classes, radius):
    seq = np.asarray(seq, dtype=int)
    if len(seq) == 0 or radius == 0:
        return seq.copy(), None
    one_hot = np.zeros((len(seq), n_classes), dtype=np.int32)
    one_hot[np.arange(len(seq)), seq] = 1
    cumulative = np.vstack([np.zeros((1, n_classes), dtype=np.int32), np.cumsum(one_hot, axis=0)])
    positions = np.arange(len(seq)); lower = np.maximum(0, positions - radius); upper = np.minimum(len(seq), positions + radius + 1)
    counts = cumulative[upper] - cumulative[lower]; maximum = counts.max(axis=1); argmax = counts.argmax(axis=1)
    keep = counts[np.arange(len(seq)), seq] == maximum
    return np.where(keep, seq, argmax), (counts, lower, upper)


def apply_run_fixed(base_idx, session_rows, class_to_idx, classes):
    output = np.asarray(base_idx, dtype=int).copy()
    protected = {class_to_idx[room] for room in PROTECTED_CLUSTER if room in class_to_idx}
    hallway = class_to_idx.get('hallway')
    for rows in session_rows.values():
        seq = np.asarray(base_idx, dtype=int)[rows]
        vote15, info = centered_vote(seq, len(classes), 15); hallway_vote, _ = centered_vote(seq, len(classes), 2)
        counts, lower, upper = info; share = counts[np.arange(len(seq)), vote15] / np.maximum(1, upper - lower)
        run_length = np.ones(len(seq), dtype=int); start = 0
        for index in range(1, len(seq) + 1):
            if index == len(seq) or seq[index] != seq[start]:
                run_length[start:index] = index - start; start = index
        local = vote15.copy(); protected_mask = np.isin(seq, list(protected))
        preserve = protected_mask & ((run_length >= 3) | (share < 0.70)); local[preserve] = seq[preserve]
        if hallway is not None:
            mask = seq == hallway; local[mask] = hallway_vote[mask]
        output[rows] = local
    return output
