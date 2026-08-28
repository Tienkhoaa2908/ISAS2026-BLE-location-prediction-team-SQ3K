#!/usr/bin/env python3
"""Build the locked 175-feature BLE fingerprint representation used in the paper.

The reproducibility version intentionally supports only the frozen configuration:
10-second windows, 2-second stride, midpoint labels, five current-window beacon
statistics, and the previous-eight-window count/std context blocks.
"""
import argparse
import numpy as np
import pandas as pd

N_BEACONS = 25


def time_to_seconds(series):
    parts = series.astype(str).str.split(':', expand=True).astype(float)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def load_labels(path):
    labels = pd.read_csv(path, dtype=str).fillna('')
    labels['room'] = labels['room'].str.strip()
    labels = labels[labels['room'] != ''].copy()
    labels['start_s'] = time_to_seconds(labels['start_time'])
    labels['finish_s'] = time_to_seconds(labels['finish_time'])
    by_date = {}
    for date, group in labels.groupby('date'):
        group = group.sort_values('start_s')
        by_date[date] = (
            group.start_s.to_numpy(), group.finish_s.to_numpy(), group.room.to_numpy())
    return by_date


def label_at_midpoint(center, intervals):
    starts, finishes, rooms = intervals
    hit = np.where((starts <= center) & (center <= finishes))[0]
    return rooms[hit[0]] if hit.size else None


def window_statistics(beacon_index, rssi):
    count = np.bincount(beacon_index, minlength=N_BEACONS).astype(float)
    sums = np.bincount(beacon_index, weights=rssi, minlength=N_BEACONS)
    sums2 = np.bincount(beacon_index, weights=rssi * rssi, minlength=N_BEACONS)
    seen = count > 0
    mean = np.zeros(N_BEACONS)
    mean[seen] = sums[seen] / count[seen]
    var = np.zeros(N_BEACONS)
    var[seen] = sums2[seen] / count[seen] - mean[seen] ** 2
    std = np.sqrt(np.clip(var, 0, None))
    maximum = np.full(N_BEACONS, -np.inf)
    np.maximum.at(maximum, beacon_index, rssi)
    return count, mean, std, maximum


def seconds_to_hms(value):
    hour = int(value // 3600)
    minute = int((value % 3600) // 60)
    second = value % 60
    return f'{hour:02d}:{minute:02d}:{second:06.3f}'.rstrip('0').rstrip('.')


def build(input_path, label_path, output_path, window=10.0, step=2.0, history=8):
    ble = pd.read_csv(input_path)
    ble['date'] = ble['date'].astype(str)
    ble['sec'] = time_to_seconds(ble['time'])
    ble['b0'] = pd.to_numeric(ble['beacon']).astype(int) - 1
    ble['rssi'] = pd.to_numeric(ble['rssi']).astype(float)
    labels = load_labels(label_path)

    columns = []
    for prefix in ['mean', 'pres', 'std', 'max', 'count', 'pmean', 'pstd']:
        columns += [f'{prefix}_b{i}' for i in range(1, N_BEACONS + 1)]

    rows, metadata = [], []
    candidate_windows = 0

    for date, group in ble.groupby('date'):
        if date not in labels:
            continue
        intervals = labels[date]
        group = group.sort_values('sec')
        sec = group.sec.to_numpy()
        beacon = group.b0.to_numpy()
        rssi = group.rssi.to_numpy(float)
        starts = np.arange(np.floor(sec[0]), sec[-1] - window + 1e-9, step)

        history_rows = []
        previous_start = None
        for start in starts:
            candidate_windows += 1
            end = start + window
            room = label_at_midpoint(start + window / 2, intervals)
            if room is None:
                continue

            lo = np.searchsorted(sec, start, side='left')
            hi = np.searchsorted(sec, end, side='left')
            if hi - lo < 1:
                continue

            count, mean, std, maximum = window_statistics(beacon[lo:hi], rssi[lo:hi])
            seen = count > 0
            if previous_start is not None and (start - previous_start) > 1.5 * step:
                history_rows = []
            past = history_rows[-history:]
            past_count = np.mean([x[0] for x in past], axis=0) if past else np.zeros(N_BEACONS)
            past_std = np.mean([x[1] for x in past], axis=0) if past else np.zeros(N_BEACONS)

            row = {}
            row.update({f'mean_b{i+1}': v for i, v in enumerate(np.where(seen, mean, -125.0))})
            row.update({f'pres_b{i+1}': v for i, v in enumerate(np.where(seen, 1, 0))})
            row.update({f'std_b{i+1}': v for i, v in enumerate(np.where(seen, std, 0.0))})
            row.update({f'max_b{i+1}': v for i, v in enumerate(np.where(seen, maximum, -125.0))})
            row.update({f'count_b{i+1}': v for i, v in enumerate(count)})
            row.update({f'pmean_b{i+1}': v for i, v in enumerate(past_count)})
            row.update({f'pstd_b{i+1}': v for i, v in enumerate(past_std)})
            rows.append(row)
            metadata.append((date, start, end, room))
            history_rows.append((count.copy(), std.copy()))
            previous_start = start

    features = pd.DataFrame(rows, columns=columns)
    meta = pd.DataFrame(metadata, columns=['date', 'win_start_s', 'win_end_s', 'room'])
    features.insert(0, 'date', meta.date.values)
    features.insert(1, 'win_start', meta.win_start_s.map(seconds_to_hms).values)
    features.insert(2, 'win_end', meta.win_end_s.map(seconds_to_hms).values)
    features['room'] = meta.room.values
    features.to_csv(output_path, index=False)
    print('candidate_windows', candidate_windows)
    print('retained_windows', len(features))
    print('numeric_features', len(columns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-l', '--labels', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--window', type=float, default=10.0)
    parser.add_argument('--step', type=float, default=2.0)
    parser.add_argument('--features', default='mean,present,std,max,count')
    parser.add_argument('--past-context', type=int, default=8)
    parser.add_argument('--past-feats', default='count,std')
    args = parser.parse_args()
    if args.features != 'mean,present,std,max,count' or args.past_feats != 'count,std':
        raise SystemExit('This reproducibility implementation supports the locked feature configuration only.')
    build(args.input, args.labels, args.output, args.window, args.step, args.past_context)


if __name__ == '__main__':
    main()
