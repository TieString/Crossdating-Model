#!/usr/bin/env python3
"""Freeze one runtime candidate table for none/whole/missing/false/partial.

The operation model may rank this table, but receives no event class, distance
stratum, remaining count or clean target correlation. All positions are scanned
inside each operation before one conditional 13-year window is retained.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

PAIR = LOCAL = None
CACHE_ROOT = None
IDENTITIES = [("none", 0)] + [("whole", shift) for shift in range(-100, 101) if shift]
IDENTITIES += [("missing", -1), ("false", 1)] + [("partial", shift) for shift in range(-100, -1)]
CONFIGS = (("p50c6", 0.5, 6.0), ("p70c6", 0.7, 6.0), ("p70c12", 0.7, 12.0))
FAMILY_INDICES = {family: np.asarray([i for i, identity in enumerate(IDENTITIES) if identity[0] == family])
                  for family in ("none", "whole", "missing", "false", "partial")}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initialize(root, cache_root=None):
    global PAIR, LOCAL, CACHE_ROOT
    CACHE_ROOT = Path(cache_root) if cache_root else None
    PAIR = load("pair", str(Path(root) / "evaluate-engine-pairhmm.py"))
    PAIR.initialize()
    PAIR.BASE.LAGS = np.arange(-340, 121, dtype=np.int32)
    LOCAL = load("local", str(Path(root) / "evaluate-lag-platform-shift-oracle.py"))


def selected_correlations(prefix, first, last, lag):
    count, sx, sy, sxx, syy, sxy = prefix[:, last, lag] - prefix[:, first, lag]
    covariance = sxy - sx * sy / np.maximum(count, 1.)
    left = sxx - sx * sx / np.maximum(count, 1.)
    right = syy - sy * sy / np.maximum(count, 1.)
    denominator = np.sqrt(np.maximum(0., left * right))
    return np.divide(covariance, denominator, out=np.full_like(covariance, -1.),
                     where=(count >= 9) & (denominator > 1e-12))


def build(row):
    cache_path = None
    if CACHE_ROOT:
        key = hashlib.sha256(("unified-op-v1:" + row["evidenceKey"]).encode()).hexdigest()
        cache_path = CACHE_ROOT / key[:2] / (key + ".npz")
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as saved:
                return row["stateHash"], saved["features"], saved["windows"], saved["columns"].tolist(), True
    with gzip.open(row["evidencePath"], "rt", encoding="utf-8") as handle:
        sample = json.load(handle)
    if sample["evidenceVersion"] != "cofecha-js-0.2.0-explicit-units-testing-values-v2":
        raise ValueError("only corrected v2 runtime evidence is accepted")
    records = [{"shift": float(shift), "absShift": float(abs(shift)), **{
        f"is_{name}": float(operation == name) for name in ("none", "whole", "missing", "false", "partial")
    }} for operation, shift in IDENTITIES]
    windows = np.zeros(len(records), dtype=np.int32)
    by_beta = {}
    for name, beta, change in CONFIGS:
        if beta not in by_beta:
            by_beta[beta] = PAIR.emissions(sample, beta)
        years, values, null = by_beta[beta]
        prefix = PAIR.forward(values, null, change, "max", True)
        zero = int(np.searchsorted(PAIR.BASE.LAGS, 0))
        suffix = np.r_[np.cumsum(values[::-1, zero])[::-1], 0.0]
        splits = np.arange(1, len(years))
        none_score = float(suffix[0])
        scores = []
        local_windows = []
        masses = []
        for operation, shift in IDENTITIES:
            lag_index = int(np.searchsorted(PAIR.BASE.LAGS, shift))
            if operation == "none":
                score, start, mass = none_score, 0, 1.0
            elif operation == "whole":
                score, start, mass = float(prefix[-1, lag_index] - change), 0, 1.0
            else:
                cost = change * (0.5 if operation == "missing" else 0.75 if operation == "false" else 1.0)
                joint = prefix[splits - 1, lag_index] + suffix[splits] - cost
                if operation == "false":
                    joint = joint + null[splits - 1] - values[splits - 1, lag_index]
                # Missing/false events occur at the observed older-side year;
                # partial uses firstFixedYear, the first newer-side observation.
                boundary_years = years[splits] if operation == "partial" else years[splits - 1]
                start = PAIR.BASE.window_start(boundary_years, joint)
                score = float(logsumexp(joint))
                inside = (boundary_years >= start) & (boundary_years <= start + 12)
                mass = float(np.exp(logsumexp(joint[inside]) - score))
            scores.append(score)
            local_windows.append(start)
            masses.append(mass)
        scores = np.asarray(scores)
        for index, ((operation, _), record) in enumerate(zip(IDENTITIES, records)):
            family = FAMILY_INDICES[operation]
            rank = float(np.mean(scores[family] <= scores[index]))
            record.update({
                f"{name}Gain": float((scores[index] - none_score) / np.sqrt(len(years))),
                f"{name}GlobalDelta": float(scores[index] - np.max(scores)),
                f"{name}FamilyDelta": float(scores[index] - np.max(scores[family])),
                f"{name}FamilyRank": rank,
                f"{name}WindowMass": masses[index],
                f"{name}Distance": float(years[-1] - local_windows[index] - 6)
                    if operation not in ("none", "whole") else 0.0,
            })
        if name == "p70c6":
            windows = np.asarray(local_windows, dtype=np.int32)
    # Robust per-reference counterfactual support, with references selected only
    # by their agreement with the reference master (never by target truth).
    master = dict(zip(sample["referenceYears"], sample["referenceValues"]))
    def quality(reference):
        pairs = [(value, master[year]) for year, value in zip(reference["years"], reference["values"]) if year in master]
        return float(np.corrcoef(np.asarray(pairs).T)[0, 1]) if len(pairs) >= 20 else -1.0
    refs = sorted(sample["individualReferences"], key=quality, reverse=True)[:8]
    local_years = sorted(sample["targetYears"])
    lags = np.arange(-100, 101, dtype=np.int32)
    prefixes = [LOCAL.correlation_prefix({
        "targetYears": sample["targetYears"], "targetValues": sample["targetValues"],
        "referenceYears": ref["years"], "referenceValues": ref["values"],
    }, local_years, lags) for ref in refs]
    splits = np.asarray([int(np.searchsorted(local_years, windows[i] + 6))
                         if operation not in ("whole", "none") else len(local_years)
                         for i, (operation, _) in enumerate(IDENTITIES)])
    shift_indexes = np.asarray([shift + 100 for _, shift in IDENTITIES])
    for width in (20, 40):
        first = np.maximum(0, splits - width)
        before = np.stack([selected_correlations(prefix, first, splits, np.full(len(records), 100)) for prefix in prefixes], axis=1)
        after = np.stack([selected_correlations(prefix, first, splits, shift_indexes) for prefix in prefixes], axis=1)
        valid = (before > -.999) & (after > -.999)
        counts = valid.sum(axis=1)
        gains = after - before
        for label, values in (("Before", before), ("After", after), ("Gain", gains)):
            values = np.where(valid, values, np.nan)
            median, quartile = np.full(len(records), -1.), np.full(len(records), -1.)
            good = counts > 0
            median[good] = np.nanmedian(values[good], axis=1)
            quartile[good] = np.nanquantile(values[good], .25, axis=1)
            for i, record in enumerate(records):
                record[f"ref{width}{label}Median"] = float(median[i])
                record[f"ref{width}{label}Q25"] = float(quartile[i])
        positive = ((gains > 0) & valid).sum(axis=1) / np.maximum(counts, 1)
        for i, record in enumerate(records):
            record[f"ref{width}Positive"] = float(positive[i])
            record[f"ref{width}Count"] = float(counts[i])
    names = sorted(records[0])
    features = np.asarray([[record[key] for key in names] for record in records], dtype=np.float32)
    features = np.nan_to_num(features, nan=-1e6, posinf=1e6, neginf=-1e6)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, features=features, windows=windows, columns=np.asarray(names))
    return row["stateHash"], features, windows, names, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="append", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-root", default=".benchmark-results/unified99-engine-v1/unified-op-cache-v1")
    parser.add_argument("--reuse-artifact", action="append", default=[])
    args = parser.parse_args()
    rows = []
    for path in args.index:
        with Path(path).open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    opportunities = {}
    for row in rows:
        key = (row["role"], row["caseId"], row["stageIndex"])
        if key in opportunities and opportunities[key] != row:
            raise ValueError(f"same event opportunity has inconsistent metadata: {key}")
        opportunities[key] = row
    rows = sorted(opportunities.values(), key=lambda row: (row["stateHash"], row["caseId"], row["stageIndex"]))
    if args.limit:
        rows = rows[:args.limit]
    if Path(args.output).exists():
        raise FileExistsError("frozen candidate artifact exists; choose a new versioned output")
    results, cache_stats = {}, {"hits": 0, "misses": 0, "artifactReuses": 0}
    wanted = {row["stateHash"] for row in rows}
    unique_states = {row["stateHash"]: row for row in rows}
    for path in args.reuse_artifact:
        prior_metadata = json.loads(Path(path + ".json").read_text(encoding="utf-8"))
        assert prior_metadata["method"] == "runtime-unified-integer-operation-candidates-v1"
        assert prior_metadata["identities"] == [list(identity) for identity in IDENTITIES]
        with np.load(path, allow_pickle=False) as prior:
            columns = prior["columns"].tolist()
            saved_features, saved_windows = prior["features"], prior["windows"]
            for i, state_hash in enumerate(prior["stateHashes"]):
                if str(state_hash) in wanted:
                    results[str(state_hash)] = (saved_features[i], saved_windows[i])
                    cache_stats["artifactReuses"] += 1
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(str(Path(__file__).parent), args.cache_root)) as executor:
        futures = [executor.submit(build, row) for row in unique_states.values() if row["stateHash"] not in results]
        for count, future in enumerate(as_completed(futures), 1):
            state_hash, features, windows, columns, hit = future.result()
            cache_stats["hits" if hit else "misses"] += 1
            results[state_hash] = (features, windows)
            if count % 1000 == 0 or len(results) == len(wanted):
                print(json.dumps({"processedStates": len(results), "totalStates": len(wanted), "opportunities": len(rows), "cache": cache_stats}), flush=True)
    features = np.stack([results[row["stateHash"]][0] for row in rows])
    windows = np.stack([results[row["stateHash"]][1] for row in rows])
    np.savez_compressed(args.output, features=features, windows=windows, columns=np.asarray(columns),
                        stateHashes=np.asarray([row["stateHash"] for row in rows]))
    metadata = {"schemaVersion": 1, "method": "runtime-unified-integer-operation-candidates-v1",
                "locationConfig": "p70c6", "windowWidth": 13,
                "identities": IDENTITIES, "columns": columns, "rows": rows,
                "cacheStats": cache_stats,
                "uniqueStates": len(wanted), "opportunityRows": len(rows),
                "identicalInputsWithDifferentLabelsRetained": True,
                "prohibitedInferenceInputs": ["truth", "category", "distanceBand", "binName", "remainingEventCount", "masterCorrelation"]}
    Path(args.output + ".json").write_text(json.dumps(metadata, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"output": args.output, "shape": list(features.shape)}))


if __name__ == "__main__":
    main()
