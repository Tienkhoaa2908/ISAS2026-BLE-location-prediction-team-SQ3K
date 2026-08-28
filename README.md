# BLE Indoor Location Prediction — ISAS 2026

Reproducibility repository for the SQ3K submission to the ISAS 2026 Location Recognition Challenge.

The implementation uses a 22-class hierarchical Random Forest on 10-second BLE fingerprints and a 14-room, 50-second Directional Long-Context Specialist Correction (DLCS) branch. Validation is reported under both repeated session-grouped cross-validation and fixed-22-class leave-one-day-out (LODO) evaluation.

> **Data access.** Challenge data are not redistributed in this repository. The notebooks expect the private `data.zip` bundle supplied separately to the organizers/team. The bundle is converted into the runtime layout by `scripts/prepare_data.py`.

## Results

### Repeated session-grouped validation

| System | Macro-F1 | Accuracy |
|---|---:|---:|
| Hierarchical RF + Borderline-SMOTE | 0.5367 | 0.7366 |
| DLCS | **0.5487** | **0.7389** |

### Fixed-22-class LODO

| System | Macro-F1 | Accuracy | Balanced accuracy |
|---|---:|---:|---:|
| Main, raw | 0.3647 | 0.6649 | 0.3746 |
| DLCS, raw | **0.3680** | **0.6655** | **0.3754** |
| Main + centered smoothing | **0.4026** | **0.6990** | **0.4051** |
| DLCS + centered smoothing | 0.3993 | 0.6984 | 0.4018 |

Because the common temporal smoother reverses the DLCS/main ordering under strict LODO, the locked hidden-test CSV was generated with the **hierarchical main model + centered temporal smoothing**. Its SHA-256 is:

```text
4b1a80a214b53b531a40467a515e2afbe4350b7b0533144ec1415c3eb9a43d80
```

## Method

The main representation is built from 10-second windows with a 2-second stride. For each of 25 beacons, the current window contributes mean RSSI, presence, RSSI standard deviation, maximum RSSI, and packet count. Two history blocks summarize mean packet count and mean RSSI standard deviation over the previous eight retained windows. This gives **175 features**.

The main classifier is a two-level hierarchical Random Forest. Tier 1 separates numbered patient rooms from common areas, and Tier 2 predicts the specific class inside the routed group. Each forest uses 400 trees, square-root feature subsampling, balanced-subsample class weights, and Borderline-SMOTE inside the training fold with minority target support 200.

The DLCS branch builds **119 label-free features** from non-overlapping 50-second BLE context bins and trains a 1,200-tree Random Forest specialist over 14 patient rooms. A frozen directional gate allows only selected uncertain corrections. The same centered temporal smoother is then applied to the main and DLCS sequences for paired comparison.

## Repository structure

```text
notebooks/
  training.ipynb                 full preprocessing and validation workflow
  test_prediction.ipynb          locked hidden-test inference workflow
src/
  build_fingerprints.py          10-s fingerprint construction
  model_core.py                  grouping and model utilities
  models/                        hierarchical RF / resampling implementation
  dlcs/                          long-context specialist and gating implementation
scripts/
  prepare_data.py                private-data bundle -> runtime/Dataset
  verify_fingerprints.py         reconstruction hash check
  session_grouped_validation.py  repeated grouped confirmation
  lodo_validation.py             fixed-22-class LODO evaluation
  test_inference.py              final hidden-test CSV generation
config/
  beacon_coords.json
  room_centers.json
results/
  session_grouped/               aggregate reference metrics
  lodo/                          aggregate/per-class reference metrics
```

## Colab

The recommended workflow is to open the notebooks directly in Colab:

- [Training notebook](https://colab.research.google.com/github/Tienkhoaa2908/ISAS2026-BLE-location-prediction-team-SQ3K/blob/main/notebooks/training.ipynb)
- [Test-prediction notebook](https://colab.research.google.com/github/Tienkhoaa2908/ISAS2026-BLE-location-prediction-team-SQ3K/blob/main/notebooks/test_prediction.ipynb)

Each notebook installs the pinned dependencies, asks for the private `data.zip` bundle if it is not already available, and prepares the runtime dataset automatically.

## Command-line reproduction

```bash
python -m pip install -r requirements.txt
python scripts/prepare_data.py --archive /path/to/data.zip --out runtime/Dataset
python scripts/verify_fingerprints.py --data-root runtime/Dataset --out runtime/rebuilt_train_fingerprints.csv
```

Expected fingerprint reconstruction:

```text
retained windows: 15382
numeric features: 175
SHA-256: d95ce428732af277c74bf8bd3fb16301c9aaeddbbb1c46a6b394905bfe5b3877
```

Repeated session-grouped confirmation:

```bash
python scripts/session_grouped_validation.py --data-root runtime/Dataset --outdir runtime/session_grouped --trees 1200
```

Strict LODO:

```bash
python scripts/lodo_validation.py --data-root runtime/Dataset --outdir runtime/lodo --specialist-trees 1200 --force
```

Final hidden-test inference:

```bash
python scripts/test_inference.py --data-root runtime/Dataset --out SQ3K_prediction.csv
```

The final inference command must report `matches_locked_submission True`.

## Reproducibility checks

The following quantities are treated as locked checks rather than tuning targets:

- fingerprint rows: **15,382**
- fingerprint dimensions: **175**
- fingerprint SHA-256: `d95ce428732af277c74bf8bd3fb16301c9aaeddbbb1c46a6b394905bfe5b3877`
- repeated grouped Main Macro-F1: **0.536686916710663**
- repeated grouped DLCS Macro-F1: **0.5486823604624447**
- LODO Main + centered Macro-F1: **0.402584**
- hidden-test rows: **62,222**
- hidden-test CSV SHA-256: `4b1a80a214b53b531a40467a515e2afbe4350b7b0533144ec1415c3eb9a43d80`

## Data policy

No raw BLE data, location labels, test packets, per-row validation predictions, or cached challenge predictions are committed. Aggregate metrics and normalized confusion matrices are included only to make the reported experiments auditable without redistributing the challenge dataset.
