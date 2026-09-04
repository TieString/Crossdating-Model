"""Maintainer tool: reduce historical manifests to the public v5 design contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    final = json.loads(args.final_manifest.read_text(encoding="utf-8"))
    for role in ("development", "calibration"):
        folder = args.artifact_root / "distant-d-v11"
        source_manifest = json.loads((folder / f"{role}-manifest.json").read_text(encoding="utf-8"))
        source_manifest["itrdbRoot"] = "datasets/rwl"
        source_manifest["designConfigSha256"] = source_manifest.pop("configSha256", None)
        source_manifest["configPath"] = "configs/v5.yaml"
        source_manifest["configSha256"] = hashlib.sha256(
            (args.output_root / "configs/v5.yaml").read_bytes()).hexdigest()
        source_manifest["cofechaJsVersion"] = "0.2.0"
        source_manifest.pop("cofechaSha256", None)
        source_manifest["historicalSourceManifestSha256"] = source_manifest.pop(
            "sourceManifestSha256", None)
        source_manifest.pop("sourceManifestPath", None)
        write(args.output_root / "splits" / f"{role}-targets.json", source_manifest)
        case_maps = []
        for design in ("inclusive-v10", "distant-d-v11"):
            source = json.loads((args.artifact_root / design / f"{role}-cases.json").read_text(encoding="utf-8"))
            case_maps.append({case["caseId"]: case for case in source["cases"]})
        selected_ids = sorted({row["caseId"] for row in final["rows"] if row["role"] == role})
        rows_by_case: dict[str, list[dict]] = {}
        for row in final["rows"]:
            if row["role"] == role:
                rows_by_case.setdefault(row["caseId"], []).append(row)
        selected = []
        for case_id in selected_ids:
            found = [mapping[case_id] for mapping in case_maps if case_id in mapping]
            if not found:
                raise ValueError(f"missing historical case definition: {case_id}")
            canonical = {key: value for key, value in found[0].items() if key != "index"}
            if any({key: value for key, value in item.items() if key != "index"} != canonical for item in found[1:]):
                raise ValueError(f"case definition disagreement: {case_id}")
            stages = sorted(rows_by_case[case_id], key=lambda row: row["stageIndex"])
            selected.append({
                **canonical,
                "index": len(selected),
                "scenarioGeneratorVersion": 11 if case_id.endswith(":D-distant-v11") else 10,
                "primaryABCD": bool(stages[0].get("primaryABCD", True)),
                "expectedStages": [{
                    "stageIndex": row["stageIndex"],
                    "stateHash": row["stateHash"],
                    "evidenceKey": row["evidenceKey"],
                    "truth": row["truth"],
                } for row in stages],
            })
        write(args.output_root / "scenarios" / f"v5-{role}-cases.json", {
            "schemaVersion": 1,
            "protocolVersion": "real-edit-frontier-explicit-units-v2",
            "scenarioGeneratorVersions": [10, 11],
            "role": role,
            "primaryAndStressCases": True,
            "cases": selected,
        })
        expected = {row["caseId"] for row in final["rows"] if row["role"] == role}
        if expected != {case["caseId"] for case in selected}:
            raise ValueError(f"case plan mismatch for {role}")
        print(json.dumps({"role": role, "files": len(source_manifest["files"]),
                          "targets": source_manifest["counts"]["eligibleTargets"],
                          "cases": len(selected)}))


if __name__ == "__main__":
    main()
