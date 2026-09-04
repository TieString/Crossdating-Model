#!/usr/bin/env python3
"""Add complete constant-lag correlations to a frozen operation table."""
from __future__ import annotations
import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

LAGS = np.arange(-100, 101)
WIDTHS = (5, 10, 15, 25, 50, 100, None)
NAMES = [f"constant{suffix}{width or 'Full'}" for width in WIDTHS for suffix in ("Corr", "Pairs", "Delta")]


def calculate(row, shifts):
    with gzip.open(row["evidencePath"], "rt", encoding="utf-8") as handle:
        sample = json.load(handle)
    years = np.asarray(sample["targetYears"])
    x = np.asarray(sample["targetValues"])[:, None]
    ref_years = np.asarray(sample["referenceYears"])
    dense = np.full(int(ref_years.max() - ref_years.min() + 1), np.nan)
    dense[ref_years - ref_years.min()] = sample["referenceValues"]
    indices = years[:, None] + LAGS[None, :] - ref_years.min()
    y = dense[np.clip(indices, 0, len(dense) - 1)]
    y[(indices < 0) | (indices >= len(dense))] = np.nan
    output = []
    for width in WIDTHS:
        use = np.ones(len(years), dtype=bool) if width is None else years >= years.max() - width + 1
        a, b = x[use], y[use]
        valid = np.isfinite(b)
        n = valid.sum(axis=0)
        xx = np.where(valid, a, 0.0)
        yy = np.where(valid, b, 0.0)
        sx, sy = xx.sum(axis=0), yy.sum(axis=0)
        covariance = (xx * yy).sum(axis=0) - sx * sy / np.maximum(n, 1)
        denominator = np.sqrt(np.maximum(0.0, ((xx * xx).sum(axis=0) - sx * sx / np.maximum(n, 1))
                                                   * ((yy * yy).sum(axis=0) - sy * sy / np.maximum(n, 1))))
        r = np.divide(covariance, denominator, out=np.full(len(LAGS), np.nan), where=(n >= 3) & (denominator > 1e-12))
        best = np.nanmax(r) if np.isfinite(r).any() else np.nan
        output.extend([r, n.astype(float), r - best])
    return row["stateHash"], np.asarray(output, dtype=np.float32).T[np.asarray(shifts) + 100]


def from_cache(row, shifts, root):
    path = Path(root) / row["evidenceKey"][:2] / (row["evidenceKey"] + ".npz")
    with np.load(path, allow_pickle=False) as data:
        return row["stateHash"], data["values"][np.asarray(shifts) + 100]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--cache-root")
    parser.add_argument("--uncompressed", action="store_true")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("choose a new frozen output")
    metadata = json.loads(Path(args.input + ".json").read_text(encoding="utf-8"))
    shifts = [identity[1] for identity in metadata["identities"]]
    output = {}
    with ProcessPoolExecutor(args.workers) as executor:
        unique = {row["stateHash"]: row for row in metadata["rows"]}
        futures = [executor.submit(from_cache, row, shifts, args.cache_root) if args.cache_root
                   else executor.submit(calculate, row, shifts) for row in unique.values()]
        for count, future in enumerate(as_completed(futures), 1):
            key, features = future.result()
            output[key] = features
            if count % 1000 == 0 or count == len(metadata["rows"]):
                print(json.dumps({"processed": count, "total": len(metadata["rows"])}), flush=True)
    with np.load(args.input, allow_pickle=False) as source:
        appended = np.stack([output[row["stateHash"]] for row in metadata["rows"]])
        combined = np.concatenate([source["features"], appended], axis=2)
        columns = list(source["columns"]) + NAMES
        save = np.savez if args.uncompressed else np.savez_compressed
        save(args.output, features=combined, windows=source["windows"],
                            stateHashes=source["stateHashes"], columns=np.asarray(columns))
    metadata.update({"method": "runtime-operation-candidates-with-exact-constant-lag-v2", "columns": columns,
                     "appendedEvidence": "constant-lag correlation, not a local counterfactual; invalid pairs remain NaN",
                     "sourceArtifact": args.input})
    Path(args.output + ".json").write_text(json.dumps(metadata, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"output": args.output, "shape": list(combined.shape)}))


if __name__ == "__main__":
    main()
