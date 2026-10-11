# Temporal validation for route-volume forecasting

GRU forecasting with separate chronological training, validation, calibration and test periods. Inputs use 24 half-hour observations and predict 10 steps. The trailing two-hour target requires a two-hour embargo at evaluation boundaries. Scaling and clipping are fitted on the training prefix only. Holdout target windows do not overlap within each route.

Validation selects the checkpoint. A separate calibration period fits a global scale and a positive Ridge ensemble with persistence and rolling-mean baselines. Only then is the final test evaluated. Routes not observed in training are excluded; this protocol does not evaluate cold-start coverage.

## Run

```text
python pipeline.py --demo --output results_demo
python pipeline.py --train path/to/train.parquet --output results_real
python -m pytest -q test_validation.py
```

Required columns: `route_id`, `office_from_id`, `timestamp`, `target_2h`. Route and office identifiers must be integer-valued. Observations must use half-hour cadence; duplicate route/timestamp records are rejected. Short or discontinuous windows are excluded.

The available executed experiment uses explicitly synthetic data: 4,320 training windows and 72 windows each for validation, calibration and test. It verifies that training, checkpoint selection and calibration execute. These scores are not competition metrics or evidence of quality on actual WB data. The original competition parquet files were not available locally. Real-data evaluation remains outstanding.

Tests check future-data isolation of preprocessing, embargo boundaries, causal inputs, disjoint holdout target support and duplicate rejection. The original model and metric functions are retained in `reference_src/common.py`; original source files are preserved separately. The corrected protocol is a new experiment, not a claim that historical competition training used this split.
