"""Frozen, label-only scenario and eligibility audits.

These fields are consumed by evaluation and never enter the candidate feature
matrix. Scenario generation remains separated from the unified model head.
"""
from __future__ import annotations

from collections import defaultdict


def expected_bin(correlation: float) -> str:
    if 0.60 <= correlation < 0.70:
        return "0.60-0.70"
    if 0.70 <= correlation < 0.80:
        return "0.70-0.80"
    if correlation >= 0.80:
        return "0.80+"
    raise ValueError(f"file correlation below eligibility floor: {correlation}")


def audit_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("manifest contains no states")
    roles_by_file: dict[str, set[str]] = defaultdict(set)
    categories: dict[str, int] = defaultdict(int)
    for row in rows:
        if row["targetMasterCorrelation"] <= 0.60:
            raise ValueError(f"target below master-correlation gate: {row['stateHash']}")
        if row["binName"] != expected_bin(float(row["fileIntercorrelation"])):
            raise ValueError(f"incorrect file-correlation bin: {row['stateHash']}")
        roles_by_file[row["fileContentHash"]].add(row["role"])
        categories[row["category"]] += 1
        truth = row["truth"]
        kind, shift = truth["kind"], int(truth["shift"])
        if kind == "whole" and (shift == 0 or not -100 <= shift <= 100):
            raise ValueError(f"whole displacement outside frozen scan: {row['stateHash']}")
        if kind == "partial" and not -100 <= shift <= -2:
            raise ValueError(f"partial displacement outside frozen scan: {row['stateHash']}")
        if kind == "missing" and shift != -1 or kind == "false" and shift != 1:
            raise ValueError(f"unit event displacement mismatch: {row['stateHash']}")
        if row["category"] == "D":
            # Frontier manifests repeat the current truth inside remainingTruths;
            # de-duplicate that event before checking adjacent physical events.
            years = sorted({event["year"] for event in [truth, *row.get("remainingTruths", [])]
                            if event.get("year") is not None})
            if any(right - left < 30 for left, right in zip(years, years[1:])):
                raise ValueError(f"D event spacing below 30 years: {row['stateHash']}")
    leakage = {file_hash: sorted(roles) for file_hash, roles in roles_by_file.items() if len(roles) > 1}
    if leakage:
        raise ValueError(f"complete-file split leakage: {leakage}")
    return {"states": len(rows), "files": len(roles_by_file),
            "roles": sorted({row["role"] for row in rows}),
            "categories": dict(sorted(categories.items())), "fileOverlap": 0}
