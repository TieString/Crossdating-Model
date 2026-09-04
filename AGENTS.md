# Repository guide

This repository owns offline evidence construction, frozen file splits, LightGBM training, evaluation, model export and Python/TypeScript parity. It must not import source from `Crossdating_IDM`.

- Start at `scripts/reproduce_v5.py` for the stable end-to-end entry point.
- `configs/v5.yaml` is the frozen scientific configuration.
- `crossdating_model/evidence/reference_v5/` preserves the exact historical v5 dependency closure for audit.
- `model-releases/v5.0.0/` is immutable once published; create a new version rather than overwriting a released hash.
- Keep raw RWL data, large evidence matrices, official COFECHA executables and exploratory failures out of Git.
- Split development, calibration and final by complete RWL file. A consumed final remains a permanent result and cannot be used for tuning.
