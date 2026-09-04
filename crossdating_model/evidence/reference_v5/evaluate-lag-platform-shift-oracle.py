#!/usr/bin/env python3
"""Evaluate exact partial shifts as differences between stable lag platforms."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


PATH = None
SHIFTS = np.arange(-100, -1, dtype=np.int32)
VARIANTS = tuple(
    (f"r_d{str(diff).replace('.', '')}_w{window or 'full'}", "residual", diff, window, top_k)
    for diff in (0.0, 0.25, 0.5)
    for window in (None, 15, 20, 30, 40, 60)
    for top_k in (1, 3)
) + tuple(
    (f"c_w{window or 'full'}", "correlation", 0.0, window, top_k)
    for window in (None, 15, 20, 30, 40, 60)
    for top_k in (1, 3)
) + tuple(
    (f"rf_d{str(diff).replace('.', '')}_w{window or 'full'}", "residualFixed", diff, window, top_k)
    for diff in (0.0, 0.25, 0.5)
    for window in (None, 20, 30, 40, 60, 90)
    for top_k in (1, 3)
) + tuple(
    (f"cf_w{window or 'full'}", "correlationFixed", 0.0, window, top_k)
    for window in (None, 20, 30, 40, 60, 90)
    for top_k in (1, 3)
)


def initialize(research_root: str) -> None:
    global PATH
    sys.path.insert(0, research_root)
    import right_censored_path_locator
    PATH = right_censored_path_locator


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for value in paths:
        role, path = value.split("=", 1)
        with Path(path).open("r", encoding="utf-8") as handle:
            rows.extend({**json.loads(line), "role": role} for line in handle if line.strip())
    return [row for row in rows if row.get("truth", {}).get("kind") == "partial"]


def candidate_scores(
    emissions: np.ndarray,
    valid: np.ndarray,
    window: int | None,
    top_k: int,
    fixed_new: bool = False,
) -> np.ndarray:
    values = np.where(valid, emissions, 0.0)
    prefix = np.vstack([np.zeros(emissions.shape[1]), np.cumsum(values, axis=0)])
    counts = np.vstack([np.zeros(emissions.shape[1]), np.cumsum(valid, axis=0)])
    n = len(emissions)
    output = np.full(len(SHIFTS), -1e30, dtype=np.float64)
    minimum = 9
    for split in range(minimum, n - minimum + 1):
        older_start = 0 if window is None else max(0, split - window)
        newer_end = n if window is None else min(n, split + window)
        older_count = counts[split] - counts[older_start]
        newer_count = counts[newer_end] - counts[split]
        older = prefix[split] - prefix[older_start]
        newer = prefix[newer_end] - prefix[split]
        older[older_count < minimum] = -1e30
        newer[newer_count < minimum] = -1e30
        same = float(np.max(older + newer))
        old_order = np.argpartition(older, -top_k)[-top_k:]
        new_order = np.asarray([100]) if fixed_new else np.argpartition(newer, -top_k)[-top_k:]
        span = max(minimum * 2, newer_end - older_start)
        for old_index in old_order:
            if older[old_index] < -1e20:
                continue
            for new_index in new_order:
                shift = int(old_index - new_index)
                if shift > -2 or shift < -100:
                    continue
                score = float(older[old_index] + newer[new_index] - same) / np.sqrt(span)
                output[shift + 100] = max(output[shift + 100], score)
    return output


def truth_boundary_prediction(
    emissions: np.ndarray,
    valid: np.ndarray,
    split: int,
    window: int | None,
    fixed_new: bool = False,
) -> int:
    values = np.where(valid, emissions, 0.0)
    older_start = 0 if window is None else max(0, split - window)
    newer_end = len(emissions) if window is None else min(len(emissions), split + window)
    older_count = valid[older_start:split].sum(axis=0)
    newer_count = valid[split:newer_end].sum(axis=0)
    older = values[older_start:split].sum(axis=0)
    newer = values[split:newer_end].sum(axis=0)
    older[older_count < 9] = -1e30
    newer[newer_count < 9] = -1e30
    return int(np.argmax(older) - (100 if fixed_new else np.argmax(newer)))


def correlation_prefix(sample: dict, years: list[int], lags: np.ndarray) -> np.ndarray:
    target = {int(year): float(value) for year, value in zip(sample["targetYears"], sample["targetValues"])}
    reference = {int(year): float(value) for year, value in zip(sample["referenceYears"], sample["referenceValues"])}
    current = np.zeros((6, len(years), len(lags)), dtype=np.float64)
    if reference:
        low, high = min(reference), max(reference)
        dense = np.full(high - low + 1, np.nan)
        for year, value in reference.items(): dense[year - low] = value
        indexes = np.asarray(years)[:, None] + lags[None, :] - low
        y = dense[np.clip(indexes, 0, len(dense) - 1)]
        valid = (indexes >= 0) & (indexes < len(dense)) & np.isfinite(y)
        x = np.asarray([target[year] for year in years])[:, None]
        current[0] = valid
        current[1] = np.where(valid, x, 0)
        current[2] = np.where(valid, y, 0)
        current[3] = np.where(valid, x * x, 0)
        current[4] = np.where(valid, y * y, 0)
        current[5] = np.where(valid, x * y, 0)
    return np.concatenate([
        np.zeros((6, 1, len(lags)), dtype=np.float64),
        np.cumsum(current, axis=1),
    ], axis=1)


def correlation_segment(prefix: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    values = prefix[:, end] - prefix[:, start]
    count, sx, sy, sxx, syy, sxy = values
    covariance = sxy - sx * sy / np.maximum(count, 1.0)
    left = sxx - sx * sx / np.maximum(count, 1.0)
    right = syy - sy * sy / np.maximum(count, 1.0)
    denominator = np.sqrt(np.maximum(0.0, left * right))
    correlation = np.divide(
        covariance, denominator,
        out=np.full_like(covariance, -1.0),
        where=(count >= 9) & (denominator > 1e-12),
    )
    return correlation, count


def correlation_candidate_scores(
    prefix: np.ndarray,
    window: int | None,
    top_k: int,
    fixed_new: bool = False,
) -> np.ndarray:
    n = prefix.shape[1] - 1
    output = np.full(len(SHIFTS), -1e30, dtype=np.float64)
    for split in range(9, n - 9 + 1):
        older_start = 0 if window is None else max(0, split - window)
        newer_end = n if window is None else min(n, split + window)
        older, older_count = correlation_segment(prefix, older_start, split)
        newer, newer_count = correlation_segment(prefix, split, newer_end)
        same_weight = older_count + newer_count
        same = np.divide(
            older * older_count + newer * newer_count,
            np.maximum(1.0, same_weight),
        )
        same_best = float(np.max(same))
        old_order = np.argpartition(older, -top_k)[-top_k:]
        new_order = np.asarray([100]) if fixed_new else np.argpartition(newer, -top_k)[-top_k:]
        balance = np.sqrt(max(1.0, (split - older_start) * (newer_end - split) / (newer_end - older_start)))
        for old_index in old_order:
            for new_index in new_order:
                shift = int(old_index - new_index)
                if shift > -2 or shift < -100:
                    continue
                weight = older_count[old_index] + newer_count[new_index]
                combined = (
                    older[old_index] * older_count[old_index]
                    + newer[new_index] * newer_count[new_index]
                ) / max(1.0, weight)
                output[shift + 100] = max(
                    output[shift + 100],
                    float(combined - same_best) * balance,
                )
    return output


def correlation_truth_boundary_prediction(
    prefix: np.ndarray,
    split: int,
    window: int | None,
    fixed_new: bool = False,
) -> int:
    n = prefix.shape[1] - 1
    older_start = 0 if window is None else max(0, split - window)
    newer_end = n if window is None else min(n, split + window)
    older, _ = correlation_segment(prefix, older_start, split)
    newer, _ = correlation_segment(prefix, split, newer_end)
    return int(np.argmax(older) - (100 if fixed_new else np.argmax(newer)))


def evaluate(row: dict) -> dict:
    with gzip.open(row["evidencePath"], "rt", encoding="utf-8") as handle:
        sample = json.load(handle)
    truth = int(row["truth"]["shift"])
    by_diff = {}
    predictions = {}
    top2 = {}
    truth_boundary = {}
    correlation = None
    for name, evidence, diff, window, top_k in VARIANTS:
        if diff not in by_diff:
            years, lags, emissions = PATH.emission_matrix(sample, -100, 15, diff)
            reference_years = set(int(year) for year in sample["referenceYears"])
            valid = np.asarray([
                [year + int(lag) in reference_years for lag in lags]
                for year in years
            ], dtype=bool)
            by_diff[diff] = emissions, valid
        fixed_new = evidence.endswith("Fixed")
        if evidence.startswith("correlation"):
            if correlation is None:
                correlation = correlation_prefix(sample, years, lags)
            scores = correlation_candidate_scores(correlation, window, top_k, fixed_new)
        else:
            scores = candidate_scores(*by_diff[diff], window, top_k, fixed_new)
        order = np.argsort(scores, kind="stable")[::-1]
        predictions[name] = int(SHIFTS[order[0]])
        top2[name] = [int(SHIFTS[index]) for index in order[:2]]
        if top_k == 1:
            split = int(np.searchsorted(years, int(row["truth"]["year"]), side="left"))
            truth_boundary[name] = (
                correlation_truth_boundary_prediction(correlation, split, window, fixed_new)
                if evidence.startswith("correlation")
                else truth_boundary_prediction(*by_diff[diff], split, window, fixed_new)
            )
    return {
        "stateHash": row["stateHash"],
        "role": row["role"],
        "relativePath": row["relativePath"],
        "category": row.get("category"),
        "placement": row.get("placement"),
        "binName": row.get("binName"),
        "truthShift": truth,
        "predictions": predictions,
        "top2": top2,
        "truthBoundaryPredictions": truth_boundary,
    }


def summary(rows: list[dict]) -> dict:
    return {
        name: {
            "opportunities": len(rows),
            "top1Accuracy": sum(row["predictions"][name] == row["truthShift"] for row in rows) / len(rows),
            "top2Recall": sum(row["truthShift"] in row["top2"][name] for row in rows) / len(rows),
            "truthBoundaryAccuracy": (
                sum(row["truthBoundaryPredictions"].get(name) == row["truthShift"] for row in rows) / len(rows)
                if name in rows[0]["truthBoundaryPredictions"] else None
            ),
        }
        for name, *_ in VARIANTS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="append", required=True)
    parser.add_argument("--research-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-per-role", type=int, default=0)
    args = parser.parse_args()
    rows = load_rows(args.index)
    if args.limit_per_role > 0:
        selected = []
        for role in sorted({row["role"] for row in rows}):
            current = sorted(
                (row for row in rows if row["role"] == role),
                key=lambda row: row["stateHash"],
            )
            selected.extend(current[:args.limit_per_role])
        rows = selected
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize,
        initargs=(args.research_root,),
    ) as executor:
        futures = [executor.submit(evaluate, row) for row in rows]
        for count, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if count % 25 == 0 or count == len(futures):
                print(json.dumps({"processed": count, "total": len(futures)}), flush=True)
    groups = defaultdict(list)
    for row in results:
        groups[f"role:{row['role']}"] .append(row)
        groups[f"category:{row['category']}"] .append(row)
        groups[f"placement:{row['placement']}"] .append(row)
        groups[f"bin:{row['binName']}"] .append(row)
    output = {
        "schemaVersion": 1,
        "method": "stable-lag-platform-difference-oracle-v1",
        "candidateRange": [-100, -2],
        "overall": summary(results),
        "groups": {name: summary(values) for name, values in sorted(groups.items())},
        "rows": sorted(results, key=lambda row: row["stateHash"]),
    }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {**output, "rows": f"{len(results)} rows"}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
