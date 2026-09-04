"""Versioned bark-to-pith user-unblock metric used by the v5 release."""
from __future__ import annotations
from collections import defaultdict
import numpy as np

LOCAL_OPERATIONS = {"missing", "false", "partial"}

def gate_for_clean(rows: list[dict], scores: np.ndarray, none_index: int, maximum_rate: float = 0.01) -> float:
    event_scores = np.asarray(scores, dtype=np.float64).copy()
    event_scores[:, none_index] = -np.inf
    margins = np.max(event_scores, axis=1) - scores[:, none_index]
    clean = np.asarray([row["truth"]["kind"] == "none" for row in rows])
    ordered = np.sort(margins[clean])[::-1]
    if not len(ordered):
        raise ValueError("calibration requires Clean controls")
    return float(ordered[min(int(len(ordered) * maximum_rate), len(ordered) - 1)])

def evaluate(rows: list[dict], scores: np.ndarray, windows: np.ndarray, identities: list[list],
             threshold: float, protocol: str = "local-review-v3") -> list[dict]:
    none = identities.index(["none", 0])
    missing = identities.index(["missing", -1])
    event_scores = np.asarray(scores, dtype=np.float64).copy()
    event_scores[:, none] = -np.inf
    selected = np.argmax(event_scores, axis=1)
    selected = np.where(event_scores[np.arange(len(rows)), selected] - scores[:, none] > threshold, selected, none)
    local_codes = np.asarray([i for i, (operation, _) in enumerate(identities) if operation in LOCAL_OPERATIONS])
    output = []
    for index, (row, code) in enumerate(zip(rows, selected)):
        truth = row["truth"]
        first_operation, first_shift = identities[int(code)]
        operation, shift = first_operation, first_shift
        whole_to_local = False
        partial_to_missing = False
        if protocol == "local-review-v3" and operation == "whole" and truth["kind"] in LOCAL_OPERATIONS:
            code = int(local_codes[np.argmax(scores[index, local_codes])])
            operation, shift = identities[code]
            whole_to_local = True
        start = int(windows[index, code]) if operation not in ("none", "whole") else None
        fracture_present = operation == "partial" and any(
            candidate["kind"] == "partial" and candidate.get("year") is not None
            and start <= candidate["year"] <= start + 12
            for candidate in row.get("remainingTruths", []))
        if operation == "partial" and truth["kind"] == "missing" and not fracture_present:
            code = missing
            operation, shift = identities[missing]
            start = int(windows[index, missing])
            partial_to_missing = True
        operation_correct = operation == truth["kind"] and shift == truth["shift"]
        correct = operation_correct and (operation in ("none", "whole")
            or start <= truth["year"] <= start + 12)
        prompted_older = operation not in ("none", "whole") and not correct and any(
            candidate.get("year") is not None and truth.get("year") is not None
            and candidate["year"] < truth["year"] and candidate["kind"] == operation
            and candidate["shift"] == shift and start <= candidate["year"] <= start + 12
            for candidate in row.get("remainingTruths", []))
        output.append({**row, "firstOperation": first_operation, "firstShift": first_shift,
            "operation": operation, "shift": shift, "windowStart": start,
            "windowEnd": None if start is None else start + 12,
            "answered": operation != "none", "operationCorrect": bool(operation_correct),
            "correct": bool(correct), "wholeToLocalSwitch": whole_to_local,
            "partialToMissingSwitch": partial_to_missing, "promptedOlderEvent": bool(prompted_older)})
    return output

def summarize(rows: list[dict]) -> dict:
    event = [row for row in rows if row["truth"]["kind"] != "none"]
    clean = [row for row in rows if row["truth"]["kind"] == "none"]
    answered = sum(row["answered"] for row in event)
    correct = sum(row["correct"] for row in event)
    return {"cases": len({row["caseId"] for row in rows}), "eventOpportunities": len(event),
        "cleanControls": len(clean), "answered": answered, "refused": len(event) - answered,
        "correct": correct, "coverage": answered / len(event) if event else None,
        "windowAccuracy": correct / len(event) if event else None,
        "answeredAccuracy": correct / answered if answered else None,
        "operationDisplacementErrors": sum(row["answered"] and not row["operationCorrect"] for row in event),
        "windowErrors": sum(row["answered"] and row["operationCorrect"] and not row["correct"] for row in event),
        "wholeToLocalSwitches": sum(row.get("wholeToLocalSwitch", False) for row in event),
        "partialToMissingSwitches": sum(row.get("partialToMissingSwitch", False) for row in event),
        "promptedOlderEvent": sum(row["promptedOlderEvent"] for row in event),
        "cleanFalsePositives": sum(row["answered"] for row in clean),
        "cleanFalsePositiveRate": sum(row["answered"] for row in clean) / len(clean) if clean else None,
        "windowWidths": {"13": sum(row["windowStart"] is not None for row in rows)}}

def grouped_summaries(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        for name, value in (("category", row["category"]), ("r", row["binName"]),
                            ("distance", row["distanceBand"]),
                            ("category-r", f'{row["category"]}:{row["binName"]}')):
            groups[f"{name}:{value}"].append(row)
    return {name: summarize(values) for name, values in sorted(groups.items())}
