from __future__ import annotations
from collections import defaultdict
import numpy as np

def file_clustered_interval(rows: list[dict], iterations: int = 3000, seed: int = 20260914) -> dict:
    groups = defaultdict(lambda: np.zeros(5, dtype=np.int64))
    for row in rows:
        event = row["truth"]["kind"] != "none"
        groups[row["fileContentHash"]] += [event, event and row["correct"], event and row["answered"],
                                                   not event, not event and row["answered"]]
    counts = np.asarray(list(groups.values()))
    rng = np.random.default_rng(seed)
    totals = counts[rng.integers(len(counts), size=(iterations, len(counts)))].sum(axis=1)
    def interval(numerator: int, denominator: int) -> list[float]:
        values = totals[:, numerator] / np.maximum(1, totals[:, denominator])
        return np.quantile(values, [0.025, 0.975]).tolist()
    return {"unit": "complete RWL file hash", "files": len(counts), "iterations": iterations,
            "windowAccuracy95": interval(1, 0), "coverage95": interval(2, 0),
            "cleanFalsePositive95": interval(4, 3)}
