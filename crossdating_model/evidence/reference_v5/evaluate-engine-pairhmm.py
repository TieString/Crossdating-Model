#!/usr/bin/env python3
"""Audit physical alignment with an unmatched observation for an inserted ring.

Every shift and split is scanned from frozen runtime evidence. Truth is read
only after inference to score exact displacement and one 13-year window.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.ndimage import maximum_filter1d

BASE = None
CONFIGS = tuple(
    (f"b{beta}_c{change}_{aggregation}_{'unmatched' if unmatched else 'matched'}",
     beta, change, aggregation, unmatched)
    for beta, change in ((0.5, 6.0), (0.7, 6.0), (0.7, 12.0))
    for aggregation in ("max", "sum")
    for unmatched in (False, True)
)


def initialize():
    global BASE
    spec = importlib.util.spec_from_file_location(
        "terminal", Path(__file__).with_name("evaluate-fixed-baseline-terminal-path.py"))
    BASE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(BASE)


def emissions(sample, beta):
    years, target, ref = BASE.aligned_values(sample)
    original = np.where(np.isfinite(ref),
                        -0.5 * (target[:, None] - beta * ref) ** 2 / (1 - beta ** 2), -12.0)
    # The no-match model has the same row centering as match emissions.
    center = np.median(original, axis=1)
    values = original - center[:, None]
    null = -0.5 * target ** 2 - center
    return years, values, null


def forward(values, null, change, aggregation, unmatched):
    result = np.empty_like(values)
    previous = values[0].copy()
    result[0] = previous
    for time in range(1, len(values)):
        # A partial loses 2..100 reference years. No unbounded old-side jumps.
        if aggregation == "max":
            partial = np.full_like(previous, -1e30)
            partial[2:] = maximum_filter1d(previous, size=99, origin=49, mode="constant", cval=-1e30)[:-2] - change
        else:
            padded = np.pad(previous, (100, 0), constant_values=-1e30)
            windows = np.lib.stride_tricks.sliding_window_view(padded, 99)[:len(previous)]
            partial = logsumexp(windows, axis=1) - change
        missing = np.full_like(previous, -1e30)
        missing[1:] = previous[:-1] - change * 0.5
        false = np.full_like(previous, -1e30)
        false[:-1] = previous[1:] - change * 0.75
        if unmatched:
            false[:-1] += null[time - 1] - values[time - 1, 1:]
        transitions = np.stack([previous, partial, missing, false])
        previous = values[time] + (np.max(transitions, axis=0) if aggregation == "max"
                                   else logsumexp(transitions, axis=0))
        result[time] = previous
    return result


def evaluate(row):
    with gzip.open(row["evidencePath"], "rt", encoding="utf-8") as handle:
        sample = json.load(handle)
    if sample["evidenceVersion"] != "cofecha-js-0.2.0-testing-values-v1":
        raise ValueError("wrong evidence version")
    by_beta, variants = {}, {}
    zero = int(np.searchsorted(BASE.LAGS, 0))
    indexes = np.searchsorted(BASE.LAGS, BASE.SHIFTS)
    for name, beta, change, aggregation, unmatched in CONFIGS:
        if beta not in by_beta:
            by_beta[beta] = emissions(sample, beta)
        years, values, null = by_beta[beta]
        prefix = forward(values, null, change, aggregation, unmatched)
        splits = np.arange(1, len(years))
        suffix = np.r_[np.cumsum(values[::-1, zero])[::-1], 0.0]
        joint = prefix[splits - 1][:, indexes].T + suffix[splits][None, :] - change
        marginal = logsumexp(joint, axis=1)
        order = np.argsort(marginal)[::-1]
        selected = int(order[0])
        truth_index = int(row["truth"]["shift"]) + 100
        start = BASE.window_start(years[splits], joint[selected])
        conditional = BASE.window_start(years[splits], joint[truth_index])
        truth_year = row["truth"]["year"]
        variants[name] = {
            "shift": int(BASE.SHIFTS[selected]),
            "top2": [int(BASE.SHIFTS[index]) for index in order[:2]],
            "windowStart": start, "conditionalWindowStart": conditional,
            "operationCorrect": selected == truth_index,
            "jointCorrect": selected == truth_index and start <= truth_year <= start + 12,
            "conditionalWindowCorrect": conditional <= truth_year <= conditional + 12,
        }
    return {**{key: row.get(key) for key in (
        "stateHash", "role", "relativePath", "category", "binName", "placement", "remainingBand")},
        "truth": row["truth"], "variants": variants}


def summary(rows):
    return {name: {
        "opportunities": len(rows),
        "operationAccuracy": sum(row["variants"][name]["operationCorrect"] for row in rows) / len(rows),
        "top2Recall": sum(row["truth"]["shift"] in row["variants"][name]["top2"] for row in rows) / len(rows),
        "jointWindowAccuracy": sum(row["variants"][name]["jointCorrect"] for row in rows) / len(rows),
        "conditionalWindowAccuracy": sum(row["variants"][name]["conditionalWindowCorrect"] for row in rows) / len(rows),
    } for name, *_ in CONFIGS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.index).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = sorted(rows, key=lambda row: hashlib.sha256(row["stateHash"].encode()).hexdigest())
    if args.limit:
        rows = rows[:args.limit]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as executor:
        futures = [executor.submit(evaluate, row) for row in rows]
        for count, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if count % 25 == 0 or count == len(rows):
                print(json.dumps({"processed": count, "total": len(rows)}), flush=True)
    groups = defaultdict(list)
    for row in results:
        for field in ("category", "binName", "placement", "remainingBand"):
            groups[f"{field}:{row[field]}"].append(row)
    report = {"schemaVersion": 1, "method": "bounded-physical-pairhmm-v1",
              "inferenceInputs": ["targetYears", "targetValues", "referenceYears", "referenceValues"],
              "final": False, "overall": summary(results),
              "groups": {key: summary(values) for key, values in groups.items()},
              "rows": sorted(results, key=lambda row: row["stateHash"])}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
