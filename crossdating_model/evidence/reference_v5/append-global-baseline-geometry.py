"""Append latest-path geometry without changing the frozen baseline features."""
import argparse
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import global_baseline_evidence as base
import global_baseline_geometry as geometry


def worker(job):
    path, old_path, evidence_path = job
    if Path(path).exists():
        with np.load(path,allow_pickle=False) as saved:
            assert saved["features"].shape == (len(base.SHIFTS),len(base.NAMES)+len(geometry.NAMES))
            assert np.array_equal(saved["shifts"],base.SHIFTS)
        return True
    with gzip.open(evidence_path,"rt",encoding="utf-8") as handle: sample=json.load(handle)
    extra=geometry.build(sample)
    with np.load(old_path,allow_pickle=False) as old: values=old["features"]; shifts=old["shifts"]
    assert values.shape==(len(base.SHIFTS),len(base.NAMES)) and np.array_equal(shifts,base.SHIFTS)
    combined=np.concatenate([values,extra],axis=1)
    assert not np.isinf(combined).any()
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,features=combined,shifts=shifts)
    return False


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True);p.add_argument("--output",required=True)
    p.add_argument("--cache",default=".benchmark-results/unified99-global-baseline-v2/evidence")
    p.add_argument("--workers",type=int,default=8);p.add_argument("--limit",type=int,default=0)
    args=p.parse_args()
    old=json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    source_sha=hashlib.sha256(Path(__file__).with_name("global_baseline_geometry.py").read_bytes()+old["sourceSha256"].encode()).hexdigest()
    rows=old["rows"][:args.limit] if args.limit else old["rows"]
    jobs={}
    for row in rows:
        previous=row["globalBaselinePath"]
        key=hashlib.sha256((source_sha+":"+previous).encode()).hexdigest()
        path=str((Path(args.cache)/key[:2]/(key+".npz")).resolve())
        row["priorGlobalBaselinePath"]=previous;row["globalBaselinePath"]=path
        jobs[path]=(path,previous,row["evidencePath"])
    hits=0
    print(json.dumps({"states":len(jobs),"features":len(base.NAMES)+len(geometry.NAMES),"cofechaCalls":0}),flush=True)
    with ProcessPoolExecutor(args.workers,initializer=base.initialize) as executor:
        futures=[executor.submit(worker,job) for job in jobs.values()]
        for count,future in enumerate(as_completed(futures),1):
            hits+=int(future.result())
            if count%500==0 or count==len(futures):print(json.dumps({"states":count,"total":len(futures),"hits":hits}),flush=True)
    old.update(version="global-baseline-geometry-v2",rows=rows,sourceSha256=source_sha,
               sourceManifest=args.manifest,columns=old["columns"]+geometry.NAMES,cacheHits=hits,cofechaCalls=0,
               uniqueStates=len(jobs),partialEvidenceAssembly=bool(args.limit))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(old,separators=(",",":")),encoding="utf-8")


if __name__=="__main__":main()
