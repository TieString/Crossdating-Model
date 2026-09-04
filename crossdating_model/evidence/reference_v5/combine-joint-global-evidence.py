"""One joint operation/window model with an explicit, separate global baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from event_distance import distance_band
from joint_global_binding import bind_global_evidence


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base-manifest",required=True);p.add_argument("--d-manifest")
    p.add_argument("--baseline-manifest",required=True);p.add_argument("--output",required=True)
    p.add_argument("--cache",default=".benchmark-results/unified99-joint-global-v5/evidence")
    p.add_argument("--packed-base")
    p.add_argument("--replace-existing-global",action="store_true")
    p.add_argument("--version",default="joint-explicit-global-baseline-v5")
    p.add_argument("--limit",type=int,default=0)
    args=p.parse_args()
    original=json.loads(Path(args.base_manifest).read_text(encoding="utf-8"))
    distant=json.loads(Path(args.d_manifest).read_text(encoding="utf-8")) if args.d_manifest else None
    global_data=json.loads(Path(args.baseline_manifest).read_text(encoding="utf-8"))
    packed=joblib.load(args.packed_base,mmap_mode="r") if args.packed_base else None
    if packed is not None:
        assert args.d_manifest is None
        assert packed["manifestSha256"]==hashlib.sha256(Path(args.base_manifest).read_bytes()).hexdigest()
    kept=[i for i,name in enumerate(original["columns"]) if not args.replace_existing_global or not name.startswith("global_")]
    if args.replace_existing_global:assert len(kept)<len(original["columns"])
    base_columns=[original["columns"][i] for i in kept]
    combined_columns=base_columns+["global_"+name for name in global_data["columns"]]
    unchanged_prefix=combined_columns[:len(original["columns"])]==original["columns"]
    if distant is not None:
        assert original["columns"]==distant["columns"] and original["identities"]==distant["identities"]
        assert original["spec"]==distant["spec"]
        assert all(row["category"]=="D" and row.get("classDefinitionId")=="distant-D-minimum-30-v11" for row in distant["rows"])
    lookup={r["stateHash"]:r for r in global_data["rows"]}
    rows=original["rows"]+(distant["rows"] if distant is not None else [])
    if args.limit:rows=rows[:args.limit]
    assert len({(r["role"],r["caseId"],r["stageIndex"]) for r in rows})==len(rows)
    digest=hashlib.sha256(Path(__file__).with_name("joint_global_binding.py").read_bytes()).hexdigest()
    output=[];hits=0
    for number,row in enumerate(rows,1):
        entry=lookup[row["stateHash"]]
        assert entry["role"]==row["role"] and entry["fileContentHash"]==row["fileContentHash"]
        key=hashlib.sha256((digest+row["candidatePath"]+entry["globalBaselinePath"]+json.dumps(kept)).encode()).hexdigest()
        target=Path(args.cache)/key[:2]/(key+".npz")
        if target.exists():
            with np.load(target,allow_pickle=False) as data:
                assert data["features"].shape==(row["candidateCount"],len(base_columns)+len(global_data["columns"]))
            hits+=1
        else:
            if packed is not None:
                first,last=packed["offsets"][number-1:number+1]
                base,codes,starts=packed["x"][first:last],packed["codes"][first:last],packed["starts"][first:last]
            else:
                with np.load(row["candidatePath"],allow_pickle=False) as data:
                    base,codes,starts=data["features"],data["identity"],data["starts"]
            previous_values=base
            base=base[:,kept]
            with np.load(entry["globalBaselinePath"],allow_pickle=False) as data:
                extra=bind_global_evidence(codes,original["identities"],data["shifts"],data["features"])
            combined=np.concatenate([base,extra],axis=1)
            assert combined.dtype==np.float32 and not np.isinf(combined).any()
            if unchanged_prefix:
                assert np.array_equal(combined[:,:previous_values.shape[1]],previous_values,equal_nan=True)
            target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,features=combined,identity=codes,starts=starts)
        legacy=row["category"]=="D" and row["caseId"].endswith(":D-broad-v7")
        output.append({**row,"candidatePath":str(target.resolve()),"evaluationClusterId":entry["evaluationClusterId"],
                       "category":"legacyMixedSpacingStress" if legacy else row["category"],
                       "primaryABCD":not legacy and row.get("primaryABCD",row["role"] in ("development","calibration") and row["category"] in ("A","B","C","D","Clean")),
                       "distanceBand":distance_band(row["truth"].get("barkDistanceYears"))})
        if number%3000==0:print(json.dumps({"states":number,"total":len(rows),"hits":hits}),flush=True)
    output.sort(key=lambda r:(r["stateHash"],r["caseId"],r["stageIndex"]))
    assert not {r["evaluationClusterId"] for r in output if r["role"]=="development"}&{
        r["evaluationClusterId"] for r in output if r["role"]=="calibration"}
    original.update(version=args.version,rows=output,
        columns=combined_columns,
        sourceManifests=[path for path in (args.base_manifest,args.d_manifest,args.baseline_manifest) if path],
        globalBindingSha256=digest,cofechaCalls=0,cacheHits=hits,
        totalHypotheses=sum(r["candidateCount"] for r in output),
        opportunities=sum(r["truth"]["kind"]!="none" for r in output),
        correctedDDefinition="adjacent local events >=30 years; old mixed-spacing D is a separate stress set" if any(r.get("classDefinitionId")=="distant-D-minimum-30-v11" for r in output) else None,
        partialEvidenceAssembly=bool(args.limit),replacedGlobalFeatures=args.replace_existing_global)
    original["currentRoles"]={role:{"states":sum(r["role"]==role for r in output),
        "primaryEvents":sum(r["role"]==role and r["primaryABCD"] and r["truth"]["kind"]!="none" for r in output),
        "stressEvents":sum(r["role"]==role and not r["primaryABCD"] and r["truth"]["kind"]!="none" for r in output),
        "files":len({r["fileContentHash"] for r in output if r["role"]==role})} for role in sorted({r["role"] for r in output})}
    original.pop("groups",None)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(original,separators=(",",":")),encoding="utf-8")
    print(json.dumps({"states":len(output),"features":len(original["columns"]),"hypotheses":original["totalHypotheses"]}))


if __name__=="__main__":main()
