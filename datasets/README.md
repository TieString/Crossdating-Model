# Data and evidence

`source-manifest.json` records public source URLs and SHA-256 checksums. `download.py` downloads only declared URLs and refuses mismatched content.

The accepted full-v5 training matrix is intentionally not committed: its local size is about 5.77 GB. Publish it as an immutable release/Zenodo/OSF asset, then replace the null URL in `source-manifest.json` without changing its SHA-256.

`fixtures/v5-smoke.npz` is a compact extract of real frozen candidate states for CI mechanics. It is not an accuracy dataset and must never be included in A/B/C/D metrics.
