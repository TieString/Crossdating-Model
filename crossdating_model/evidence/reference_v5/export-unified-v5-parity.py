"""Export the accepted model and same-state numerical gold; never fit or retune.

All COFECHA inputs/results and candidate evidence come from frozen caches.
Only deterministic path traces and model inference are evaluated here.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import joblib
import numpy as np


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream: return json.load(stream)


def serial(value):
    if isinstance(value, np.ndarray): return serial(value.tolist())
    if isinstance(value, list): return [serial(v) for v in value]
    if isinstance(value, dict): return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=".benchmark-results/unified99-joint-global-v5/full-manifest.json")
    parser.add_argument("--model", default=".benchmark-results/unified99-joint-global-v5/model-v5.joblib")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): raise FileExistsError("Use a new gold export directory")
    output.mkdir(parents=True)
    frozen = joblib.load(args.model)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    assert manifest["columns"] == frozen["columns"] and len(frozen["columns"]) == 207
    assert frozen["version"] == "joint-explicit-global-v5"
    dump = frozen["model"].booster_.dump_model()
    model = {k: v for k, v in frozen.items() if k != "model"}
    model.update(schemaVersion=1, objective=dump["objective"], averageOutput=dump["average_output"],
                 treeInfo=dump["tree_info"], sourceSha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest())
    (output / "model.json").write_text(json.dumps(model, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    buckets = defaultdict(list)
    for row in manifest["rows"]:
        if row.get("primaryABCD"):
            buckets[(row["category"], row["binName"], row["distanceBand"], row["truth"]["kind"])].append(row)
    selected, seen = [], set()
    while len(selected) < args.limit and any(buckets.values()):
        for key in sorted(buckets):
            while buckets[key] and buckets[key][0]["stateHash"] in seen: buckets[key].pop(0)
            if not buckets[key]: continue
            row = buckets[key].pop(0); seen.add(row["stateHash"]); selected.append(row)
            if len(selected) == args.limit: break
    selected.sort(key=lambda row: (row["fileContentHash"], row["stateHash"]))
    operation = load("v5_operations", "build-engine-operation-candidates.py")
    operation.initialize(str(Path(__file__).parent), ".benchmark-results/unified99-engine-v1/unified-op-cache-v1")
    constants = load("v5_constants", "augment-constant-lag-evidence.py")
    global_base = load("v5_global", "global_baseline_evidence.py"); global_base.initialize()
    decision = load("v5_decision", "joint_window_decision.py")
    index = []
    cache_stats = {"operationHits": 0, "operationDerivedFromFrozenInputs": 0}
    for number, row in enumerate(selected):
        sample, raw, refs = (read(row[key]) for key in ("evidencePath", "rawTargetPath", "rawReferencePath"))
        assert row["targetId"] == refs["excludedTargetId"]
        assert all(ref["id"] != row["targetId"] for ref in refs["references"])
        _, base, operation_windows, columns, hit = operation.build(row)
        cache_stats["operationHits" if hit else "operationDerivedFromFrozenInputs"] += 1
        _, appended = constants.from_cache(row, [i[1] for i in frozen["identities"]], ".benchmark-results/unified99-engine-v1/constant-lag-cache-v1")
        base = np.concatenate([base, appended], axis=1)
        assert columns + constants.NAMES == frozen["columns"][:62]
        with np.load(row["candidatePath"], allow_pickle=False) as data:
            features, codes, starts = data["features"], data["identity"], data["starts"]
        assert np.array_equal(base[codes], features[:, :62], equal_nan=True)
        scores = frozen["model"].booster_.predict(features, num_threads=1)
        identity_scores, windows = decision.profile_operations(scores, codes, starts, len(frozen["identities"]))
        traces = []
        for name, beta, change in operation.CONFIGS:
            years, values, null = operation.PAIR.emissions(sample, beta)
            # First, last, and 16 calendar-order rows of every path are traced.
            positions = np.unique(np.linspace(0, len(years)-1, 18).astype(int))
            history = operation.PAIR.forward(values, null, change, "max", True)
            _, x, ref = global_base.PAIR.BASE.aligned_values(sample)
            censored, no_match = global_base.censored_path_emissions(x, ref, beta)
            global_history = global_base.PAIR.forward(censored, no_match, change, "max", True)
            traces.append({"name": name, "positions": positions.tolist(), "values": values[positions],
                           "nulls": null[positions], "history": history[positions],
                           "censoredValues": censored[positions], "censoredHistory": global_history[positions]})
        filename = row["stateHash"] + ".json.gz"
        payload = {"stateHash": row["stateHash"], "fileContentHash": row["fileContentHash"], "targetId": row["targetId"],
                   "sample": sample, "rawTarget": raw["entries"], "rawReferences": refs["references"],
                   "base": base, "baseWindows": operation_windows, "features": features,
                   "codes": codes.tolist(), "starts": starts.tolist(), "scores": scores,
                   "identityScores": identity_scores, "windows": windows, "traces": traces}
        with gzip.open(output / filename, "wt", encoding="utf-8") as stream:
            json.dump(serial(payload), stream, separators=(",", ":"), allow_nan=False)
        index.append({key: row[key] for key in ("stateHash", "fileContentHash", "role", "category", "binName", "distanceBand")})
        index[-1]["file"] = filename
        if (number + 1) % 20 == 0: print(json.dumps({"exported": number+1, "states": len(selected), "cofechaCalls": 0}), flush=True)
    (output / "manifest.json").write_text(json.dumps({"schemaVersion": 1, "model": "model.json", "states": index,
        "columns": frozen["columns"], "cofechaCalls": 0, "cacheStats": cache_stats, "newFinalConsumed": False,
        "sourceManifestSha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
    print(json.dumps({"states": len(index), "output": str(output), "files": len({r["fileContentHash"] for r in selected})}))


if __name__ == "__main__": main()
