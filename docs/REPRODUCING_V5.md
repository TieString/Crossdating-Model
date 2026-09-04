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

Run the complete public-data path:

```powershell
python scripts/rebuild_from_itrdb.py --download
```

This performs:

```text
420 public ITRDB RWL URLs + SHA-256
  → frozen development/calibration files and target lists
  → exact A/B/C/D/Clean case plans
  → bark-to-pith frontier states
  → cofecha-js 0.2.0 testingValues and leave-target-out references
  → operation candidates + constant-lag evidence
  → conditional 13-year window profiles
  → global baseline geometry
  → 207-field local packed cache
  → file-grouped LambdaRank training and calibration
  → JSON export and Python/TypeScript parity
```

The 420-source v5.0.0 reconstruction consists of 220 development and 200 calibration files. The accepted historical run did not assign or consume an independent final partition; this is explicitly recorded rather than retroactively manufacturing a final result.

Allow substantial disk space and runtime. The generated packed arrays are approximately 5.7 GB, live only under the ignored `work/` directory and may be deleted after the model/report has been exported. They are an optimization for repeated model-head experiments, not a downloadable prerequisite.

For a short structural check against locally downloaded RWL files:

```powershell
python scripts/rebuild_from_itrdb.py --rwl-root D:\path\to\measurements `
  --work work/pipeline-check --evidence-limit 10 --skip-training
```

The committed case plans carry the historical stage hashes. Generation stops immediately if event application, frontier order, input-unit normalization or the `cofecha-js` evidence key differs.

## Evidence reconstruction audit

`crossdating_model/evidence/reference_v5/` contains the minimal final research stages used to build the candidate matrix. Historical command-line shapes are preserved for audit. The stable public orchestration deliberately does not import the desktop application's source.

## Release discipline

Do not overwrite a published model. Create a new model version, produce its model/feature/identity/split/gold/metrics files, regenerate `SHA256SUMS.txt`, run parity and application integration tests, then pin that release and hash in `Crossdating_IDM`.
