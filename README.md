# Crossdating-Model

Reproducible training, evaluation, and TypeScript deployment pipeline for automated tree-ring crossdating suggestions.

## Repository boundary

| Repository | Responsibility | Main artifacts |
| --- | --- | --- |
| [`cofecha-js`](https://github.com/TieString/cofecha-js) | General COFECHA-compatible computation | npm package, PART 1–8 OUT, `testingValues` |
| [`Crossdating-Model`](https://github.com/TieString/Crossdating-Model) | Model research and reproducible validation | model JSON, manifests, training/evaluation reports, gold fixtures |
| [`Crossdating_IDM`](https://github.com/TieString/Crossdating_IDM) | Desktop product and production inference | Tauri application, TypeScript runtime and review UI |

Dependency direction is one-way:

```text
cofecha-js
   ↓
Crossdating-Model ── versioned release + SHA-256 ──→ Crossdating_IDM
   ↓                                                ↓
training / evaluation                         inference / interaction
```

This repository never imports `Crossdating_IDM` source. The application consumes a pinned model release; the model repository does not reach into the application checkout.

## Reproducibility status

- `smoke`: runnable from a fresh clone. It trains the same LambdaRank family on a small frozen real-candidate fixture, exports LightGBM JSON, generates Python gold, and verifies the independent TypeScript scorer.
- `full v5`: starts from the committed list of 420 public ITRDB RWL URLs and SHA-256 values. Every scenario, COFECHA testing value, candidate and 207-field evidence row is regenerated locally.
- The approximately 5.7 GB packed evidence is a disposable local build cache. It is neither published nor required as an input.
- The accepted v5 model is preserved in `model-releases/v5.0.0/` with a manifest and hashes.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.lock
npm ci
python scripts/reproduce_v5.py --smoke
```

Complete reconstruction from public ITRDB data:

```powershell
python scripts/rebuild_from_itrdb.py --download
```

See [`docs/REPRODUCING_V5.md`](docs/REPRODUCING_V5.md) for the exact stages, storage estimate and validation contract.

## Pipeline

```text
source manifest + SHA-256
  → immutable file splits
  → cofecha-js 0.2.0 preprocessing / testingValues
  → operation × exact-shift × 13-year-window candidates
  → 207-field frozen evidence matrix
  → file-grouped LambdaRank training and calibration
  → model JSON + manifest + Python gold
  → independent TypeScript score/parity check
  → A/B/C/D, r-bin, distance, Clean and clustered-bootstrap reports
```

The Tauri product does not need Python or LightGBM. Python is used only for offline training and release preparation.

## Commands

```text
python scripts/reproduce_v5.py --smoke
python datasets/download.py
python scripts/rebuild_from_itrdb.py
python -m pytest
npm test
npm run typecheck
python scripts/verify_release.py model-releases/v5.0.0
```

## Data policy

Git contains source URLs and hashes, static file/target/case manifests, compact fixtures and final reports. Raw public ITRDB files are downloaded from NCEI. Generated evidence and candidate matrices remain local and ignored; they are not release assets. Official COFECHA executables are never used or distributed here.

## Release consumption by Crossdating_IDM

See [`docs/TYPESCRIPT_INTEGRATION.md`](docs/TYPESCRIPT_INTEGRATION.md). IDM must pin a release tag and verify `SHA256SUMS.txt`; it must not download an unversioned model from this repository's `main` branch.

## License

GPL-3.0-only. Data retain their source attribution and redistribution terms.
