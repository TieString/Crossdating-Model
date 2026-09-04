# Reproducing v5

## Fresh-clone smoke run

The committed fixture contains 20 real frozen states: two independent complete-file groups for each of Clean/A/B/C/D in development and calibration. It is deliberately small and tests mechanics, not model quality.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.lock
npm ci
python -m pytest
python scripts/verify_release.py model-releases/v5.0.0
python scripts/reproduce_v5.py --smoke
```

The last command trains 600-tree LambdaRank with grouped folds, calibrates the event gate on Clean controls, exports a language-neutral tree JSON, generates Python scores and requires the independent TypeScript evaluator to match every committed gold candidate within `1e-12`.

## Full run

Obtain the two external assets declared in `datasets/source-manifest.json` and verify them before use:

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| packed-data.joblib | 5,766,558,747 | `547894f9680aa48520809a2bc467f78ff33e60dc7b37bb0ea7755fd1fa4883dc` |
| full-manifest.json | 51,897,205 | `41b17476e2dba417c43d95441a35784f9fb06a56b9e8eb379b127f91fbc2c74f` |

Place them under `datasets/cache/` and run:

```powershell
python scripts/reproduce_v5.py
```

The public asset URL is intentionally left null until a Zenodo/OSF/Release upload exists. Consequently, this repository currently provides complete code/config/contracts and a fresh-clone smoke reproduction, but does not claim that an anonymous user can yet reconstruct the accepted model from raw downloads alone.

## Evidence reconstruction audit

`crossdating_model/evidence/reference_v5/` contains the 15-file final research dependency closure used to create the frozen matrix. Historical command-line paths are preserved for audit. The stable public orchestration deliberately does not import the desktop application's source.

## Release discipline

Do not overwrite a published model. Create a new model version, produce its model/feature/identity/split/gold/metrics files, regenerate `SHA256SUMS.txt`, run parity and application integration tests, then pin that release and hash in `Crossdating_IDM`.
