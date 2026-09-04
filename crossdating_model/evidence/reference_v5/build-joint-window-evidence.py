"""Freeze truth-blind operation/window hypotheses; audit labels only afterward."""
import argparse
import gzip
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import numpy as np
from joint_window_candidates import build_state, Spec, LOCAL_COLUMNS


def calculate(row, base, columns, identities, spec_dict, root, code_sha):
    raw_bytes = Path(row["rawTargetPath"]).read_bytes()
    key = hashlib.sha256((row["evidenceKey"] + code_sha + json.dumps(spec_dict, sort_keys=True)).encode()
                         + hashlib.sha256(raw_bytes).digest() + base.tobytes()).hexdigest()
    path = Path(root) / key[:2] / (key + ".npz")
    hit = path.exists()
    if hit:
        with np.load(path, allow_pickle=False) as data:
            hypothesis, starts = data["identity"], data["starts"]
    else:
        raw = json.loads(gzip.decompress(raw_bytes))
        assert raw["stateHash"] == row["stateHash"]
        with gzip.open(row["evidencePath"], "rt", encoding="utf-8") as handle: sample = json.load(handle)
        x, hypothesis, starts = build_state(sample, raw["entries"], base, columns, identities, Spec(**spec_dict))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, features=x, identity=hypothesis, starts=starts)
    # This is test-framework scoring, separate from all feature construction.
    truth = row["truth"]
    expected = identities.index([truth["kind"], truth["shift"]])
    op = hypothesis == expected
    valid = op if truth["kind"] in ("none", "whole") else op & (starts <= truth["year"]) & (truth["year"] <= starts + 12)
    return {**row, "candidatePath": str(path.resolve()), "candidateCount": len(hypothesis),
            "operationCandidatePresent": bool(op.any()), "validCandidatePresent": bool(valid.any()),
            "positiveCandidateCount": int(valid.sum())}, hit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", action="append", required=True)
    p.add_argument("--base-features", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cache", default=".benchmark-results/unified99-joint-v1/evidence")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--selfcheck-per-cell", type=int, default=0)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--stride", type=int, default=13)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists(): raise FileExistsError("choose a new frozen manifest")
    meta = json.loads(Path(args.base_features + ".json").read_text(encoding="utf-8"))
    with np.load(args.base_features, allow_pickle=False) as data: base = data["features"]
    by_state = {row["stateHash"]: i for i, row in enumerate(meta["rows"])}
    rows = {}
    for path in args.index:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["role"], row["caseId"], row["stageIndex"]] = row
    selected = sorted(rows.values(), key=lambda row: (row["stateHash"], row["caseId"], row["stageIndex"]))
    if args.selfcheck_per_cell:
        cells = {}
        for row in selected:
            key = (row["role"], row["binName"], row["truth"]["kind"], row["distanceBand"])
            cells.setdefault(key, set()).add(row["caseId"])
        case_ids = set()
        for key, values in cells.items():
            ranked = sorted(values, key=lambda value: hashlib.sha256((str(key)+value).encode()).hexdigest())
            case_ids.update(ranked[:args.selfcheck_per_cell])
        selected = [row for row in selected if row["caseId"] in case_ids]
    if args.limit:
        chosen = np.random.default_rng(20260904).choice(len(selected), min(args.limit, len(selected)), replace=False)
        selected = [selected[i] for i in sorted(chosen)]
    assert all(row["stateHash"] in by_state and Path(row["rawTargetPath"]).exists() for row in selected)
    spec = asdict(Spec(top_k=args.top_k, stride=args.stride))
    code_sha = hashlib.sha256(Path(__file__).with_name("joint_window_candidates.py").read_bytes()).hexdigest()
    start, hits, result = time.perf_counter(), 0, []
    runtime_key = lambda row: (row["stateHash"], row["evidenceKey"], row["rawTargetPath"])
    unique = {runtime_key(row): row for row in selected}
    calculated = {}
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(calculate, row, base[by_state[row["stateHash"]]], meta["columns"], meta["identities"], spec, args.cache, code_sha): key
                   for key, row in unique.items()}
        for number, future in enumerate(as_completed(futures), 1):
            row, hit = future.result(); hits += int(hit); calculated[futures[future]] = row
            if number % 500 == 0 or number == len(unique):
                print(json.dumps({"states": number, "total": len(unique), "hits": hits, "seconds": time.perf_counter()-start}), flush=True)
    for row in selected:
        current = calculated[runtime_key(row)]
        if current["truth"] == row["truth"]:
            scored = {key: current[key] for key in ("candidateCount", "operationCandidatePresent", "validCandidatePresent", "positiveCandidateCount")}
        else:
            with np.load(current["candidatePath"], allow_pickle=False) as data:
                identity, windows = data["identity"], data["starts"]
            truth = row["truth"]
            operation = identity == meta["identities"].index([truth["kind"], truth["shift"]])
            valid = operation if truth["kind"] in ("none", "whole") else operation & (windows <= truth["year"]) & (truth["year"] <= windows + 12)
            scored = {"candidateCount": len(identity), "operationCandidatePresent": bool(operation.any()),
                      "validCandidatePresent": bool(valid.any()), "positiveCandidateCount": int(valid.sum())}
        result.append({**row, **scored, "candidatePath": current["candidatePath"]})
    result.sort(key=lambda row: (row["stateHash"], row["caseId"], row["stageIndex"]))
    groups = {}
    for role in sorted({row["role"] for row in result}):
        for label in ("overall", "A", "B", "C", "D"):
            current = [r for r in result if r["role"] == role and r["truth"]["kind"] != "none" and (label == "overall" or r["category"] == label)]
            if current:
                groups[f"{role}:{label}"] = {"events": len(current), "candidateRecall": sum(r["validCandidatePresent"] for r in current)/len(current)}
    manifest = {"version": "joint-window-candidate-evidence-v1", "spec": spec, "codeSha256": code_sha,
                "columns": meta["columns"] + LOCAL_COLUMNS, "identities": meta["identities"], "rows": result,
                "opportunities": sum(r["truth"]["kind"] != "none" for r in result),
                "totalHypotheses": sum(r["candidateCount"] for r in result), "cacheHits": hits, "cofechaCalls": 0,
                "oracleOnly": True, "groups": groups, "seconds": time.perf_counter()-start}
    manifest["uniqueRuntimeStates"] = len(unique)
    if args.selfcheck_per_cell:
        counts = {}
        for row in result:
            key = ":".join((row["role"], row["binName"], row["truth"]["kind"], row["distanceBand"]))
            counts[key] = counts.get(key, 0) + 1
        manifest["selfcheckCellCounts"] = counts
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key not in ("rows", "columns", "identities")}, indent=2))


if __name__ == "__main__": main()
