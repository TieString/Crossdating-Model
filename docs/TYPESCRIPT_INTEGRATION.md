# TypeScript integration contract

1. Download one tagged `Crossdating-Model` release.
2. Verify every file against `SHA256SUMS.txt` and `model-manifest.json`.
3. Copy only `unifiedV5Model.json` and `model-manifest.json` into the application model asset directory.
4. Require exactly 207 feature names in the exact published order and the exact identity table.
5. Implement LightGBM numerical trees with `<=`, exported missing-value routing and already-scaled leaf values.
6. Compare the application scorer with `golden-fixtures/` before accepting the asset.
7. Run application workflow tests, model SHA checks and the production build.

The application must not load `.joblib`, invoke Python, infer a feature order, or fetch this repository's moving `main` branch at runtime.

Compatible production baseline:

| Crossdating_IDM | Crossdating-Model | cofecha-js |
| --- | --- | --- |
| 1.5.x | v5.0.0 | 0.2.0 |
