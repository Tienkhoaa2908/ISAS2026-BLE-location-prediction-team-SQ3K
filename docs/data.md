# Data interface

The public repository does not redistribute the ISAS challenge dataset.

The reproducibility notebooks use a private archive named `data.zip`. The archive is expected to contain the following paths:

```text
data/derived/train_fingerprints.csv
data/derived/5f_ble_merged_clean(friend_data).csv
data/derived/BLE_processed.csv
data/labels/user_97.csv
data/labels/5f_label_loc_train.csv
data/test/BLE_Test_predict.csv
data/cache/cache_oof_train_fingerprints_stack_rf_r10_f5_proba.npz
```

`config/beacon_coords.json` and `config/room_centers.json` are versioned in the repository.

`python scripts/prepare_data.py --archive data.zip` creates the exact runtime layout used by the original implementation and copies the public source code into that layout.

## Locked fingerprints

The stored training fingerprint table must contain:

- 15,382 rows
- 22 location classes
- 175 numeric model features
- SHA-256 `d95ce428732af277c74bf8bd3fb16301c9aaeddbbb1c46a6b394905bfe5b3877`

The fingerprint table can be regenerated from `BLE_processed.csv` and `user_97.csv` with the public `src/build_fingerprints.py` implementation. `scripts/verify_fingerprints.py` performs this reconstruction and hash comparison.
