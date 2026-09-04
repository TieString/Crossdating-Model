# Data and evidence

`source-manifest.json` records all 420 public ITRDB RWL URLs, frozen roles and SHA-256 checksums. `download.py` downloads only declared URLs and refuses mismatched content.

The accepted full-v5 evidence matrix is intentionally not committed or published. It is regenerated under the ignored `work/` directory from these RWL files and may be deleted after export; its roughly 5.7 GB size is only a local cache concern.

`fixtures/v5-smoke.npz` is a compact extract of real frozen candidate states for CI mechanics. It is not an accuracy dataset and must never be included in A/B/C/D metrics.
