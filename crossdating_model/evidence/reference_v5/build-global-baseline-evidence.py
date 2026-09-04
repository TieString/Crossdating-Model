"""Freeze common G evidence once per state; all truth remains in the index."""
import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from global_baseline_evidence import NAMES, SHIFTS, build, initialize


def worker(job):
    path, evidence_path, target_id = job
    if Path(path).exists():
        with np.load(path,allow_pickle=False) as saved:
            assert saved["features"].shape == (len(SHIFTS),len(NAMES))
            assert np.array_equal(saved["shifts"],SHIFTS)
            assert not np.isinf(saved["features"]).any()
        return path, True
    with gzip.open(evidence_path, "rt", encoding="utf-8") as handle: sample = json.load(handle)
    assert sample["evidenceVersion"] == "cofecha-js-0.2.0-explicit-units-testing-values-v2"
    assert target_id not in sample["referenceIds"]
    values = build(sample)
    assert values.shape == (len(SHIFTS), len(NAMES)) and not np.isinf(values).any()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=values, shifts=SHIFTS)
    return path, False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--index", action="append", default=[])
    p.add_argument("--output", required=True)
    p.add_argument("--cache", default=".benchmark-results/unified99-global-baseline-v1/evidence")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = list(manifest["rows"])
    clusters = {r["fileContentHash"]:r.get("evaluationClusterId",r["fileContentHash"]) for r in rows}
    for path in args.index:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["evaluationClusterId"] = clusters[row["fileContentHash"]]
                rows.append(row)
    unique_opportunities = {}
    for row in rows:
        row["evaluationClusterId"] = row.get("evaluationClusterId",clusters.get(row["fileContentHash"],row["fileContentHash"]))
        key = row["role"],row["caseId"],row["stageIndex"]
        assert key not in unique_opportunities
        unique_opportunities[key] = row
    rows = sorted(rows, key=lambda r:(r["stateHash"],r["caseId"],r["stageIndex"]))
    if args.limit: rows = rows[:args.limit]
    files = defaultdict(set)
    for row in rows: files[row["role"]].add(row["evaluationClusterId"])
    assert not files["development"] & files["calibration"]
    source_files = ["global_baseline_evidence.py","evaluate-engine-pairhmm.py","evaluate-fixed-baseline-terminal-path.py"]
    source_sha = hashlib.sha256(b"".join(Path(__file__).with_name(name).read_bytes() for name in source_files)).hexdigest()
    jobs = {}
    for row in rows:
        key = hashlib.sha256((row["evidenceKey"]+":"+source_sha).encode()).hexdigest()
        path = str((Path(args.cache)/key[:2]/(key+".npz")).resolve())
        row["globalBaselinePath"] = path
        row["baselineTruth"] = next((t["shift"] for t in row["remainingTruths"] if t["kind"] == "whole"),0)
        row["globalBaselineCategory"] = "metamorphicControl" if row.get("primaryABCD") is False else (
            "legacyMixedSpacingStress" if row["category"] == "D" and row["caseId"].endswith(":D-broad-v7") else row["category"])
        jobs[path] = (path,row["evidencePath"],row["targetId"])
    print(json.dumps({"opportunities":len(rows),"uniqueStates":len(jobs),"cofechaCalls":0,
                      "features":len(NAMES),"globalCandidates":len(SHIFTS)}),flush=True)
    hits = 0
    with ProcessPoolExecutor(args.workers, initializer=initialize) as executor:
        futures = [executor.submit(worker,job) for job in jobs.values()]
        for count,future in enumerate(as_completed(futures),1):
            _,hit = future.result(); hits += int(hit)
            if count%500 == 0 or count == len(futures): print(json.dumps({"states":count,"total":len(futures),"hits":hits}),flush=True)
    grouped = defaultdict(set)
    for row in rows: grouped[row["stateHash"]].add(row["baselineTruth"])
    conflicts = {key:sorted(value) for key,value in grouped.items() if len(value)>1}
    report = {"version":"global-baseline-evidence-v1","sourceSha256":source_sha,"sources":source_files,
        "columns":NAMES,"shifts":SHIFTS.tolist(),"cofechaCalls":0,"cacheHits":hits,"uniqueStates":len(jobs),
        "baselineTruthConflicts":conflicts,"sourceManifest":args.manifest,"sourceIndexes":args.index,"rows":rows}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,separators=(",",":")),encoding="utf-8")


if __name__ == "__main__": main()
