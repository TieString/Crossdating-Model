"""One shared hypothesis ranker; operation/shift commitment precedes window output."""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold
from joint_window_decision import profile_operations


def load_candidate_arrays(rows, identities, feature_count, progress=None):
    """Load immutable rows once without a second full-size concatenation buffer."""
    sizes = np.asarray([row["candidateCount"] for row in rows], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(sizes)]
    count = int(offsets[-1])
    x = np.empty((count, feature_count), dtype=np.float32)
    y = np.empty(count, dtype=np.int8)
    codes = np.empty(count, dtype=np.int16)
    starts = np.empty(count, dtype=np.int32)
    for number, row in enumerate(rows):
        with np.load(row["candidatePath"], allow_pickle=False) as data:
            values, identity, window = data["features"], data["identity"], data["starts"]
        assert values.dtype == np.float32 and values.shape == (sizes[number], feature_count)
        assert len(identity) == len(window) == sizes[number]
        selection = slice(offsets[number], offsets[number+1])
        x[selection], codes[selection], starts[selection] = values, identity, window
        truth = row["truth"]
        good = identity == identities.index([truth["kind"], truth["shift"]])
        if truth["kind"] not in ("none", "whole"):
            good &= (window <= truth["year"]) & (truth["year"] <= window + 12)
        y[selection] = good
        if progress is not None: progress(number + 1)
    return {"x":x, "y":y, "offsets":offsets, "sizes":sizes, "codes":codes, "starts":starts}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--leaves", type=int, default=15)
    p.add_argument("--jobs", type=int, default=10)
    p.add_argument("--model-version", default="joint-window-ranker-v1")
    p.add_argument("--scores")
    p.add_argument("--packed-cache")
    p.add_argument("--rank-cutoff", type=int, default=10)
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows, identities = manifest["rows"], manifest["identities"]
    spec = importlib.util.spec_from_file_location("metric", Path(__file__).with_name("train-engine-operation-ranker.py"))
    metric = importlib.util.module_from_spec(spec); spec.loader.exec_module(metric)
    def primary_summary(predictions):
        current = [r for r in predictions if r.get("primaryABCD",True)
                   and r["category"] in ("A","B","C","D","Clean") and not r["caseId"].endswith(":D-broad-v7")]
        return metric.summary(current) if any(r["category"]=="D" for r in current) else None
    manifest_sha = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    if args.packed_cache and Path(args.packed_cache).exists():
        packed = joblib.load(args.packed_cache, mmap_mode="r")
        assert packed["manifestSha256"] == manifest_sha
        x, y, offsets, sizes = (packed[key] for key in ("x", "y", "offsets", "sizes"))
        codes = [packed["codes"][offsets[i]:offsets[i+1]] for i in range(len(rows))]
        starts = [packed["starts"][offsets[i]:offsets[i+1]] for i in range(len(rows))]
    else:
        def progress(number):
            if number % 3000 == 0: print(json.dumps({"loadedStates": number, "total": len(rows)}), flush=True)
        packed = load_candidate_arrays(rows, identities, len(manifest["columns"]), progress)
        packed["manifestSha256"] = manifest_sha
        x, y, offsets, sizes = (packed[key] for key in ("x", "y", "offsets", "sizes"))
        codes = [packed["codes"][offsets[i]:offsets[i+1]] for i in range(len(rows))]
        starts = [packed["starts"][offsets[i]:offsets[i+1]] for i in range(len(rows))]
        if args.packed_cache:
            joblib.dump(packed, args.packed_cache, compress=0)
            # Training folds need only part of the large table resident at once.
            del x, y, offsets, sizes, codes, starts, packed
            packed = joblib.load(args.packed_cache, mmap_mode="r")
            x, y, offsets, sizes = (packed[key] for key in ("x", "y", "offsets", "sizes"))
            codes = [packed["codes"][offsets[i]:offsets[i+1]] for i in range(len(rows))]
            starts = [packed["starts"][offsets[i]:offsets[i+1]] for i in range(len(rows))]
    dev = np.asarray([i for i, row in enumerate(rows) if row["role"] == "development"])
    cal = np.asarray([i for i, row in enumerate(rows) if row["role"] == "calibration"])
    groups = np.asarray([rows[i].get("evaluationClusterId", rows[i]["fileContentHash"]) for i in dev])
    assert not set(groups) & {rows[i].get("evaluationClusterId", rows[i]["fileContentHash"]) for i in cal}
    def flattened(indices):
        return np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in indices])
    def fit(indices, seed):
        selected = flattened(indices)
        model = lgb.LGBMRanker(objective="lambdarank", label_gain=[0, 1], lambdarank_truncation_level=args.rank_cutoff,
                              n_estimators=args.trees, learning_rate=.04, num_leaves=args.leaves,
                              min_child_samples=80, reg_lambda=20., reg_alpha=.25,
                              n_jobs=args.jobs, verbosity=-1, random_state=seed)
        model.fit(x[selected], y[selected], group=[sizes[i] for i in indices])
        return model
    def predict(indices, model):
        identity_scores = np.full((len(indices), len(identities)), -1e9)
        windows = np.zeros_like(identity_scores, dtype=np.int32)
        # Commit one operation/shift using its profile score. Location output is
        # then selected only within that committed identity, with no rewriting.
        for begin in range(0, len(indices), 128):
            batch = indices[begin:begin+128]
            values = model.booster_.predict(x[flattened(batch)])
            at = 0
            for position, i in enumerate(batch, begin):
                current = values[at:at+sizes[i]]; at += sizes[i]
                identity_scores[position], windows[position] = profile_operations(current, codes[i], starts[i], len(identities))
        return identity_scores, windows
    oof_scores = np.full((len(dev), len(identities)), -1e9)
    oof_windows = np.zeros_like(oof_scores, dtype=np.int32)
    started = time.perf_counter()
    for fold, (training, test) in enumerate(GroupKFold(5).split(dev, groups=groups), 1):
        current = fit(dev[training], 20260904 + fold)
        oof_scores[test], oof_windows[test] = predict(dev[test], current)
        print(json.dumps({"fold": fold, "complete": True, "seconds": time.perf_counter()-started}), flush=True)
    none = identities.index(["none", 0])
    dev_rows = [rows[i] for i in dev]
    dev_gate = metric.gate_for_clean(dev_rows, oof_scores, none)
    dev_prediction = metric.evaluate(dev_rows, oof_scores, oof_windows, identities, dev_gate)
    for prediction, row in zip(dev_prediction, dev_rows):
        if "evaluationClusterId" in row: prediction["evaluationClusterId"] = row["evaluationClusterId"]
    print(json.dumps({"developmentOof": metric.summary(dev_prediction)}), flush=True)
    final = fit(dev, 20260914)
    cal_rows = [rows[i] for i in cal]
    scores, windows = predict(cal, final)
    gate = metric.gate_for_clean(cal_rows, scores, none)
    prediction = metric.evaluate(cal_rows, scores, windows, identities, gate)
    for predicted, row in zip(prediction, cal_rows):
        if "evaluationClusterId" in row: predicted["evaluationClusterId"] = row["evaluationClusterId"]
    artifact = {"version": args.model_version, "model": final, "columns": manifest["columns"],
                "proposalSpec": manifest["spec"], "eventGate": gate, "identities": identities,
                "windowWidth": 13, "operationThenConditionalWindow": True}
    joblib.dump(artifact, args.model)
    if args.scores:
        np.savez_compressed(args.scores, developmentScores=oof_scores, developmentWindows=oof_windows,
                            calibrationScores=scores, calibrationWindows=windows)
    report = {"modelVersion": args.model_version, "candidateFeatures": x.shape[1], "hypotheses": len(x),
              "sourceManifest": args.manifest, "frozenScores": args.scores,
              "manifestSha256": manifest_sha,
              "calibrationGate": gate, "developmentGate": dev_gate,
              "developmentFiles": len({r["fileContentHash"] for r in dev_rows}), "calibrationFiles": len({r["fileContentHash"] for r in cal_rows}),
              "developmentClusters": len(set(groups)), "isolation": manifest.get("isolation"),
              "fileOverlap": 0, "newFinalConsumed": False, "operationThenConditionalWindow": True,
              "developmentOof": metric.summary(dev_prediction), "calibration": metric.summary(prediction),
              "developmentPrimaryABCD": primary_summary(dev_prediction), "calibrationPrimaryABCD": primary_summary(prediction),
              "developmentOofRows": dev_prediction, "calibrationRows": prediction,
              "parameters": {"leaves": args.leaves, "trees": args.trees, "lambda": 20., "rankCutoff": args.rank_cutoff}}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"calibration": report["calibration"], "primaryABCD":report["calibrationPrimaryABCD"],
                      "seconds": time.perf_counter()-started}), flush=True)


if __name__ == "__main__": main()
