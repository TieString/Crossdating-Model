from __future__ import annotations
import json
from pathlib import Path
import joblib
from crossdating_model.config import file_sha256

def export_model(artifact_path: str | Path, output_path: str | Path) -> dict:
    artifact_path, output_path = Path(artifact_path), Path(output_path)
    artifact = joblib.load(artifact_path)
    dump = artifact["model"].booster_.dump_model()
    columns, identities = artifact["columns"], artifact["identities"]
    if len(columns) != 207 or len(set(columns)) != 207:
        raise ValueError("release requires exactly 207 unique ordered features")
    if len(dump["tree_info"]) != int(artifact["config"]["lightgbm"]["n_estimators"]):
        raise ValueError("incomplete LightGBM tree dump")
    model = {"schemaVersion": 1, "version": artifact["version"],
        "releaseVersion": artifact.get("releaseVersion", "v5.0.0"),
        "columns": columns, "identities": identities, "proposalSpec": artifact["proposalSpec"],
        "eventGate": artifact["eventGate"], "windowWidth": artifact["windowWidth"],
        "sourceSha256": file_sha256(artifact_path), "objective": dump["objective"],
        "averageOutput": dump["average_output"], "treeInfo": dump["tree_info"]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return model
