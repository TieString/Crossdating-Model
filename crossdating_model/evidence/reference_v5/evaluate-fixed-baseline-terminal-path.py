#!/usr/bin/env python3
"""Current partial = final d->0 transition with an unrestricted older lag path.

The input is a post-whole-correction state. Truth fields are used only for audit.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


LAGS = np.arange(-240, 25, dtype=np.int32)
SHIFTS = np.arange(-100, -1, dtype=np.int32)
CONFIGS = tuple(
    (f"b{beta}_c{change}_j{jump}", beta, change, jump)
    for beta in (0.4, 0.5, 0.7)
    for change in (6.0, 12.0, 24.0)
    for jump in (0.0, 0.1)
)


def aligned_values(sample: dict):
    order = np.argsort(np.asarray(sample["targetYears"], dtype=np.int32))
    years = np.asarray(sample["targetYears"], dtype=np.int32)[order]
    x = np.asarray(sample["targetValues"], dtype=np.float64)[order]
    master = {int(year): float(value) for year, value in zip(sample["referenceYears"], sample["referenceValues"])}
    low, high = min(master), max(master)
    dense = np.full(high - low + 1, np.nan)
    for year, value in master.items(): dense[year - low] = value
    indexes = years[:, None] + LAGS[None, :] - low
    y = dense[np.clip(indexes, 0, len(dense) - 1)]
    y[(indexes < 0) | (indexes >= len(dense))] = np.nan
    return years, x, y


def emissions(sample: dict, beta: float) -> tuple[np.ndarray, np.ndarray]:
    years, x, y = aligned_values(sample)
    valid = np.isfinite(y)
    residual = x[:, None] - beta * y
    score = -0.5 * residual * residual / (1.0 - beta * beta)
    score = np.where(valid, score, -12.0)
    score -= np.median(score, axis=1, keepdims=True)
    return years, score


def forward_scores(values: np.ndarray, change: float, jump: float) -> np.ndarray:
    result = np.empty_like(values)
    previous = values[0].copy()
    result[0] = previous
    position = np.arange(values.shape[1]) * jump
    for time in range(1, len(values)):
        prefix = np.maximum.accumulate(previous + position)
        partial = np.full_like(previous, -1e30)
        partial[2:] = prefix[:-2] - position[2:] - change
        missing = np.full_like(previous, -1e30)
        missing[1:] = previous[:-1] - change * 0.5
        false = np.full_like(previous, -1e30)
        false[:-1] = previous[1:] - change * 0.75
        previous = values[time] + np.maximum.reduce([previous, partial, missing, false])
        result[time] = previous
    return result


def window_start(years: np.ndarray, scores: np.ndarray) -> int:
    maximum = float(np.max(scores))
    mass = np.exp(np.clip(scores - maximum, -60.0, 0.0))
    minimum_year = int(years[0])
    maximum_year = int(years[-1])
    dense = np.zeros(maximum_year - minimum_year + 1)
    dense[years - minimum_year] = mass
    if len(dense) < 13:
        return minimum_year
    sums = np.convolve(dense, np.ones(13), mode="valid")
    return minimum_year + int(np.argmax(sums))


def evaluate(row: dict) -> dict:
    with gzip.open(row["evidencePath"], "rt", encoding="utf-8") as handle:
        sample = json.load(handle)
    by_beta = {}
    variants = {}
    zero_index = int(np.where(LAGS == 0)[0][0])
    shift_indexes = np.searchsorted(LAGS, SHIFTS)
    truth_index = int(row["truth"]["shift"]) + 100
    for name, beta, change, jump in CONFIGS:
        if beta not in by_beta:
            by_beta[beta] = emissions(sample, beta)
        years, values = by_beta[beta]
        forward = forward_scores(values, change, jump)
        suffix_zero = np.r_[np.cumsum(values[::-1, zero_index])[::-1], 0.0]
        # A boundary at years[k] uses the observed older prefix and only the
        # actually observed fixed newer suffix; nothing exists past the endpoint.
        splits = np.arange(1, len(years), dtype=np.int32)
        joint = forward[splits - 1][:, shift_indexes].T + suffix_zero[splits][None, :]
        joint -= (change + jump * np.abs(SHIFTS))[:, None]
        maxima = np.max(joint, axis=1)
        marginal = maxima + np.log(np.exp(np.clip(joint - maxima[:, None], -60, 0)).sum(axis=1))
        ranked = np.argsort(marginal, kind="stable")[::-1]
        selected = int(ranked[0])
        start = window_start(years[splits], joint[selected])
        conditional_start = window_start(years[splits], joint[truth_index])
        truth_year = int(row["truth"]["year"])
        variants[name] = {
            "shift": int(SHIFTS[selected]),
            "top2": [int(SHIFTS[index]) for index in ranked[:2]],
            "startYear": start,
            "conditionalStartYear": conditional_start,
            "operationCorrect": int(SHIFTS[selected]) == int(row["truth"]["shift"]),
            "windowCorrect": start <= truth_year <= start + 12,
            "conditionalWindowCorrect": conditional_start <= truth_year <= conditional_start + 12,
        }
    return {
        **{key: row.get(key) for key in (
            "stateHash", "role", "relativePath", "category", "placement", "binName", "remainingBand",
        )},
        "truth": row["truth"], "variants": variants,
    }


def summarize(rows):
    return {name: {
        "opportunities": len(rows),
        "operationAccuracy": sum(row["variants"][name]["operationCorrect"] for row in rows) / len(rows),
        "top2Recall": sum(row["truth"]["shift"] in row["variants"][name]["top2"] for row in rows) / len(rows),
        "conditionalWindowAccuracy": sum(row["variants"][name]["conditionalWindowCorrect"] for row in rows) / len(rows),
        "jointWindowAccuracy": sum(
            row["variants"][name]["operationCorrect"] and row["variants"][name]["windowCorrect"] for row in rows
        ) / len(rows),
    } for name, *_ in CONFIGS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="append", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for path in args.index:
        with Path(path).open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    rows = [row for row in rows if row.get("truth", {}).get("kind") == "partial"]
    if args.limit:
        rows = sorted(rows, key=lambda row: hashlib.sha256(row["stateHash"].encode()).hexdigest())[:args.limit]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate, row) for row in rows]
        for count, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if count % 50 == 0 or count == len(rows):
                print(json.dumps({"processed": count, "total": len(rows)}), flush=True)
    groups = defaultdict(list)
    for row in results:
        for field in ("role", "category", "placement", "binName", "remainingBand"):
            groups[f"{field}:{row[field]}"] .append(row)
    report = {
        "schemaVersion": 1, "method": "fixed-zero-physical-final-transition-v2",
        "parameters": {name: {"beta": beta, "change": change, "jump": jump} for name, beta, change, jump in CONFIGS},
        "inferenceInputs": ["targetYears", "targetValues", "referenceYears", "referenceValues"],
        "conditionalProtocol": "whole resolved; newest-side baseline fixed to zero",
        "overall": summarize(results), "groups": {name: summarize(current) for name, current in groups.items()},
        "rows": sorted(results, key=lambda row: row["stateHash"]),
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
