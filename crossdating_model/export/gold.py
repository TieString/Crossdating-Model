from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
from crossdating_model.training.ranker import load_packed

def generate_gold(artifact_path: str | Path, dataset_path: str | Path, manifest_path: str | Path,
                  output_path: str | Path, limit: int = 5) -> dict:
    artifact = joblib.load(artifact_path)
    packed, manifest = load_packed(dataset_path, manifest_path)
    rows = manifest["rows"]
    candidates = [i for i, row in enumerate(rows) if row["role"] == "calibration"][:limit]
    states = []
    for index in candidates:
        selection = slice(packed["offsets"][index], packed["offsets"][index + 1])
        features = np.asarray(packed["x"][selection], dtype=np.float32)
        scores = artifact["model"].booster_.predict(features, num_threads=1)
        states.append({"stateHash": rows[index]["stateHash"], "features": [
            [None if not np.isfinite(value) else float(value) for value in row] for row in features],
            "scores": scores.tolist(), "codes": packed["codes"][selection].tolist(),
            "starts": packed["starts"][selection].tolist()})
    payload = {"schemaVersion": 1, "featureCount": len(artifact["columns"]), "states": states}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return payload
