"""Add complete likelihood profiles without recomputing frozen COFECHA results."""
import argparse
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from joint_profile_evidence import build, initialize, NAMES


def append(row, identities, root, code_hash):
    with np.load(row["candidatePath"], allow_pickle=False) as data:
        base, codes, starts = data["features"], data["identity"], data["starts"]
    key = hashlib.sha256((row["evidenceKey"]+code_hash).encode()+base.tobytes()+codes.tobytes()+starts.tobytes()).hexdigest()
    path = Path(root)/key[:2]/(key+".npz")
    hit = path.exists()
    if not hit:
        with gzip.open(row["evidencePath"], "rt", encoding="utf-8") as handle: sample = json.load(handle)
        extra = build(sample, identities, codes, starts)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, features=np.concatenate([base, extra], axis=1), identity=codes, starts=starts)
    return {**row, "candidatePath": str(path.resolve())}, hit


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--output", required=True)
    p.add_argument("--cache", default=".benchmark-results/unified99-joint-v2/evidence")
    p.add_argument("--workers", type=int, default=8)
    args=p.parse_args()
    source=json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    code_hash=hashlib.sha256(Path(__file__).with_name("joint_profile_evidence.py").read_bytes()).hexdigest()
    output, hits=[],0
    unique = {(row["candidatePath"], row["evidenceKey"]): row for row in source["rows"]}
    calculated = {}
    with ProcessPoolExecutor(args.workers, initializer=initialize) as pool:
        futures={pool.submit(append,row,source["identities"],args.cache,code_hash): key for key,row in unique.items()}
        for number,future in enumerate(as_completed(futures),1):
            row,hit=future.result(); calculated[futures[future]]=row["candidatePath"]; hits+=int(hit)
            if number%500==0 or number==len(futures):print(json.dumps({"states":number,"total":len(futures),"hits":hits}),flush=True)
    output=[{**row,"candidatePath":calculated[row["candidatePath"],row["evidenceKey"]]} for row in source["rows"]]
    output.sort(key=lambda r:(r["stateHash"],r["caseId"],r["stageIndex"]))
    source.update(version="joint-window-candidate-evidence-v2", rows=output,columns=source["columns"]+NAMES,
                  profileCodeSha256=code_hash,profileCacheHits=hits,cofechaCalls=0)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(source,separators=(",",":")),encoding="utf-8")
    print(json.dumps({"features":len(source["columns"]),"states":len(output)}))


if __name__=="__main__":main()
