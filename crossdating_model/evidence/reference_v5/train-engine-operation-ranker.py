#!/usr/bin/env python3
"""One file-grouped ranker for the complete runtime operation/shift table."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold
from event_distance import distance_band

CONFIGS = ({"num_leaves": 7, "min_child_samples": 50, "reg_lambda": 10.0},
           {"num_leaves": 15, "min_child_samples": 80, "reg_lambda": 15.0})


def gate_for_clean(rows, scores, none_index):
    event_scores = np.asarray(scores, dtype=np.float64).copy()
    event_scores[:, none_index] = -np.inf
    margins = np.max(event_scores, axis=1) - scores[:, none_index]
    clean = np.asarray([row["truth"]["kind"] == "none" for row in rows])
    ordered = np.sort(margins[clean])[::-1]
    allowed = int(len(ordered) * 0.01)
    # Strict > threshold leaves at most floor(1% * N) calibration Clean alerts.
    threshold = float(ordered[allowed]) if len(ordered) else 0.0
    return threshold


def evaluate(rows, scores, windows, identities, threshold):
    none = identities.index(["none", 0])
    missing = identities.index(["missing", -1])
    event_scores = np.asarray(scores, dtype=np.float64).copy()
    event_scores[:, none] = -np.inf
    best_event = np.argmax(event_scores, axis=1)
    selected = np.where(event_scores[np.arange(len(rows)), best_event] - scores[:, none] > threshold, best_event, none)
    output = []
    for i, (row, chosen) in enumerate(zip(rows, selected)):
        operation, shift = identities[int(chosen)]
        truth = row["truth"]
        first_operation = operation
        start = int(windows[i, chosen]) if operation not in ("none", "whole") else None
        direct_operation = operation == truth["kind"] and shift == truth["shift"]
        direct_correct = direct_operation and (operation in ("whole", "none") or start <= truth["year"] <= start + 12)
        fracture_present = operation == "partial" and any(
            t["kind"] == "partial" and t["year"] is not None and start <= t["year"] <= start + 12
            for t in row["remainingTruths"])
        switched = operation == "partial" and truth["kind"] == "missing" and not fracture_present
        if switched:
            operation, shift = "missing", -1
            start = int(windows[i, missing])
        operation_ok = operation == truth["kind"] and shift == truth["shift"]
        correct = operation_ok and (operation in ("whole", "none") or start <= truth["year"] <= start + 12)
        older = operation not in ("whole", "none") and not correct and truth["year"] is not None and any(
            t["year"] is not None and t["year"] < truth["year"] and t["kind"] == operation and t["shift"] == shift
            and start <= t["year"] <= start + 12 for t in row["remainingTruths"])
        output.append({
            **{key: row[key] for key in ("stateHash", "caseId", "relativePath", "fileContentHash", "category", "binName", "distanceBand", "stageIndex", "truth")},
            "distanceBand": distance_band(truth["barkDistanceYears"]) if "barkDistanceYears" in truth else row["distanceBand"],
            "primaryABCD": row.get("primaryABCD", not (row["category"] == "legacyMixedSpacingStress")),
            "firstOperation": first_operation, "firstShift": identities[int(chosen)][1],
            "operation": operation, "shift": shift, "windowStart": start,
            "windowEnd": None if start is None else start + 12,
            "answered": operation != "none", "firstDirectCorrect": bool(direct_correct),
            "partialToMissingSwitch": bool(switched), "operationCorrect": bool(operation_ok),
            "reviewBlockedByFracture": bool(first_operation == "partial" and truth["kind"] == "missing" and fracture_present),
            "correct": bool(correct), "promptedOlderEvent": bool(older),
        })
    return output


def summary(rows):
    event = [row for row in rows if row["truth"]["kind"] != "none"]
    clean = [row for row in rows if row["truth"]["kind"] == "none"]
    answered = sum(row["answered"] for row in event)
    correct = sum(row["correct"] for row in event)
    return {"cases": len({row["caseId"] for row in rows}), "eventOpportunities": len(event),
            "cleanControls": len(clean), "answered": answered, "refused": len(event) - answered,
            "correct": correct, "coverage": answered / len(event) if event else None,
            "windowAccuracy": correct / len(event) if event else None,
            "answeredAccuracy": correct / answered if answered else None,
            "firstDirectCorrect": sum(row["firstDirectCorrect"] for row in event),
            "partialToMissingSwitches": sum(row["partialToMissingSwitch"] for row in event),
            "recoveryCorrect": sum(row["partialToMissingSwitch"] and row["correct"] for row in event),
            "reviewBlockedByFracture": sum(row.get("reviewBlockedByFracture", False) for row in event),
            "operationDisplacementErrors": sum(row["answered"] and not row["operationCorrect"] for row in event),
            "windowErrors": sum(row["answered"] and row["operationCorrect"] and not row["correct"] for row in event),
            "promptedOlderEvent": sum(row["promptedOlderEvent"] for row in event),
            "cleanFalsePositives": sum(row["answered"] for row in clean),
            "cleanFalsePositiveRate": sum(row["answered"] for row in clean) / len(clean) if clean else None,
            "windowWidths": {"13": sum(row["windowStart"] is not None for row in rows)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--model-version", default="real-edit-runtime-unified-operation-v1")
    args = parser.parse_args()
    metadata = json.loads(Path(args.features + ".json").read_text(encoding="utf-8"))
    data = np.load(args.features, allow_pickle=False)
    features, windows = data["features"], data["windows"]
    rows, identities = metadata["rows"], metadata["identities"]
    assert list(data["stateHashes"]) == [row["stateHash"] for row in rows]
    assert all(row["protocol"] == "real-edit-frontier-explicit-units-v2" for row in rows)
    dev = np.asarray([i for i, row in enumerate(rows) if row["role"] == "development"])
    cal = np.asarray([i for i, row in enumerate(rows) if row["role"] == "calibration"])
    dev_files = {rows[i]["fileContentHash"] for i in dev}
    cal_files = {rows[i]["fileContentHash"] for i in cal}
    assert not dev_files & cal_files
    groups = np.asarray([rows[i]["fileContentHash"] for i in dev])
    labels = np.asarray([[int(identity == [row["truth"]["kind"], row["truth"]["shift"]])
                          for identity in identities] for row in rows], dtype=np.int8)
    assert np.all(labels.sum(axis=1) == 1)
    n_candidates, n_features = features.shape[1:]
    none = identities.index(["none", 0])
    def model(config, seed):
        return lgb.LGBMRanker(objective="lambdarank", label_gain=[0, 1], n_estimators=400,
                             learning_rate=0.03, lambdarank_truncation_level=8, reg_alpha=0.25,
                             n_jobs=args.jobs, verbosity=-1, random_state=seed, **config)
    def fit(current, indices):
        current.fit(features[indices].reshape(-1, n_features), labels[indices].reshape(-1),
                    group=np.full(len(indices), n_candidates))
    reports, all_oof = [], []
    dev_rows = [rows[i] for i in dev]
    for config_index, config in enumerate(CONFIGS):
        oof = np.zeros((len(dev), n_candidates))
        for fold, (train, test) in enumerate(GroupKFold(5).split(dev, groups=groups), 1):
            current = model(config, 20260903 + config_index * 10 + fold)
            fit(current, dev[train])
            oof[test] = current.predict(features[dev[test]].reshape(-1, n_features)).reshape(-1, n_candidates)
            print(json.dumps({"config": config_index, "fold": fold, "complete": True}), flush=True)
        threshold = gate_for_clean(dev_rows, oof, none)
        predictions = evaluate(dev_rows, oof, windows[dev], identities, threshold)
        reports.append({"config": config, "developmentGate": threshold, "oof": summary(predictions)})
        all_oof.append(predictions)
    best = max(range(len(reports)), key=lambda i: reports[i]["oof"]["windowAccuracy"])
    final = model(CONFIGS[best], 20260903)
    fit(final, dev)
    cal_rows = [rows[i] for i in cal]
    cal_scores = final.predict(features[cal].reshape(-1, n_features)).reshape(-1, n_candidates)
    threshold = gate_for_clean(cal_rows, cal_scores, none)
    predictions = evaluate(cal_rows, cal_scores, windows[cal], identities, threshold)
    grouped = defaultdict(list)
    for row in predictions:
        for name, value in (("category", row["category"]), ("r", row["binName"]), ("distance", row["distanceBand"]),
                            ("category-r", row["category"] + ":" + row["binName"]),
                            ("distance-operation-r", row["distanceBand"] + ":" + row["truth"]["kind"] + ":" + row["binName"])):
            grouped[f"{name}:{value}"].append(row)
    artifact = {"model": final, "columns": metadata["columns"], "identities": identities,
                "eventGate": threshold, "locationConfig": metadata["locationConfig"], "config": CONFIGS[best],
                "modelVersion": args.model_version, "newFinalConsumed": False}
    joblib.dump(artifact, args.model)
    report = {"schemaVersion": 1, "modelVersion": artifact["modelVersion"],
              "developmentFiles": len(dev_files), "calibrationFiles": len(cal_files), "fileOverlap": 0,
              "featureSha256": hashlib.sha256(Path(args.features).read_bytes()).hexdigest(),
              "configReports": reports, "selectedConfig": best, "calibrationGate": threshold,
              "developmentOof": reports[best]["oof"], "calibration": summary(predictions),
              "calibrationGroups": {key: summary(values) for key, values in sorted(grouped.items())},
              "developmentOofRows": all_oof[best], "calibrationRows": predictions,
              "newFinalConsumed": False, "positiveWholeGeneralizationValidated": False}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"developmentOof": report["developmentOof"], "calibration": report["calibration"]}, indent=2))


if __name__ == "__main__":
    main()
