#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

REQUIRED_PRIVATE = {
    'data/derived/train_fingerprints.csv': 'train_fingerprints.csv',
    'data/derived/5f_ble_merged_clean(friend_data).csv': '5f_ble_merged_clean(friend_data).csv',
    'data/derived/BLE_processed.csv': 'BLE_processed.csv',
    'data/labels/user_97.csv': 'user_97.csv',
    'data/labels/5f_label_loc_train.csv': '5f_label_loc_train.csv',
    'data/test/BLE_Test_predict.csv': 'BLE_Test_predict.csv',
    'data/cache/cache_oof_train_fingerprints_stack_rf_r10_f5_proba.npz':
        'explore_temporal_smoothing/cache_oof_train_fingerprints_stack_rf_r10_f5_proba.npz',
}

EXPECTED_FINGERPRINT_SHA256 = 'd95ce428732af277c74bf8bd3fb16301c9aaeddbbb1c46a6b394905bfe5b3877'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def copy_sources(repo: Path, runtime: Path) -> None:
    src = repo / 'src'
    for p in src.rglob('*.py'):
        rel = p.relative_to(src)
        dst = runtime / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    for p in (repo / 'config').glob('*.json'):
        dst = runtime / 'config' / p.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description='Prepare the private challenge data for the reproducibility scripts.')
    ap.add_argument('--archive', type=Path, required=True, help='Private data.zip distributed separately from GitHub.')
    ap.add_argument('--out', type=Path, default=Path('runtime/Dataset'))
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    archive = args.archive.resolve()
    out = args.out.resolve()

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with zipfile.ZipFile(archive) as z:
            names = set(z.namelist())
            missing = [name for name in REQUIRED_PRIVATE if name not in names]
            if missing:
                raise FileNotFoundError('Private archive is missing:\n' + '\n'.join(missing))
            for member, rel in REQUIRED_PRIVATE.items():
                z.extract(member, tmp)
                dst = out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tmp / member, dst)

    copy_sources(repo, out)

    digest = sha256(out / 'train_fingerprints.csv')
    print('runtime_data_root =', out)
    print('train_fingerprint_sha256 =', digest)
    print('matches_reference =', digest == EXPECTED_FINGERPRINT_SHA256)


if __name__ == '__main__':
    main()
