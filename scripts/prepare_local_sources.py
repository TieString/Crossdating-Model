"""Prepare public, compact artifacts from the frozen local v5 research outputs.

This maintainer-only command is not needed by a fresh clone. It verifies every
input before deriving a small real-candidate fixture and the versioned release
contract that can safely live in Git.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossdating_model.config import file_sha256
from crossdating_model.export.gold import generate_gold


EXPECTED = {
    "packed": "547894f9680aa48520809a2bc467f78ff33e60dc7b37bb0ea7755fd1fa4883dc",
    "manifest": "41b17476e2dba417c43d95441a35784f9fb06a56b9e8eb379b127f91fbc2c74f",
    "joblib": "ea06e86c66cf26eabe9a571a41e7088ca7d5146f217298a4b799df883dbd109c",
    "model_json": "4a355af59c22a9cd53e20d233256d94b44b83aea222d7ace55e6cb7e9bbb5c1e",
}


def write_text_lf(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8", newline="\n")


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual}")


def choose_rows(rows: list[dict], per_cell: int) -> list[int]:
    chosen: list[int] = []
    used_by_role: dict[str, set[str]] = defaultdict(set)
    for role in ("development", "calibration"):
        for category in ("Clean", "A", "B", "C", "D"):
            pool = sorted(
                (i for i, row in enumerate(rows)
                 if row["role"] == role and row["category"] == category
                 and row.get("primaryABCD", True)),
                key=lambda i: rows[i]["stateHash"],
            )
            cell: list[int] = []
            for index in pool:
                group = rows[index].get("evaluationClusterId", rows[index]["fileContentHash"])
                if group in used_by_role[role]:
                    continue
                used_by_role[role].add(group)
                cell.append(index)
                if len(cell) == per_cell:
                    break
            if len(cell) != per_cell:
                raise ValueError(f"not enough independent states for {role}/{category}")
            chosen.extend(cell)
    return chosen


def public_row(row: dict) -> dict:
    keep = (
        "role", "protocol", "stateHash", "caseId", "stageIndex", "relativePath",
        "fileContentHash", "targetId", "fileIntercorrelation", "targetMasterCorrelation",
        "category", "binName", "distanceBand", "placement", "truth", "remainingTruths",
        "remainingEventCount", "remainingBand", "candidateCount", "operationCandidatePresent",
        "validCandidatePresent", "positiveCandidateCount", "evaluationClusterId", "primaryABCD",
    )
    return {key: row[key] for key in keep if key in row}


def make_fixture(packed_path: Path, manifest_path: Path, destination: Path,
                 per_cell: int) -> tuple[Path, Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packed = joblib.load(packed_path, mmap_mode="r")
    selected = choose_rows(manifest["rows"], per_cell)
    x_parts, y_parts, code_parts, start_parts = [], [], [], []
    sizes: list[int] = []
    for index in selected:
        start, stop = int(packed["offsets"][index]), int(packed["offsets"][index + 1])
        x_parts.append(np.asarray(packed["x"][start:stop], dtype=np.float32))
        y_parts.append(np.asarray(packed["y"][start:stop], dtype=np.int8))
        code_parts.append(np.asarray(packed["codes"][start:stop], dtype=np.int16))
        start_parts.append(np.asarray(packed["starts"][start:stop], dtype=np.int32))
        sizes.append(stop - start)
    offsets = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
    destination.mkdir(parents=True, exist_ok=True)
    data_path = destination / "v5-smoke.npz"
    manifest_out = destination / "v5-smoke-manifest.json"
    np.savez_compressed(
        data_path, x=np.concatenate(x_parts), y=np.concatenate(y_parts), offsets=offsets,
        sizes=np.asarray(sizes, dtype=np.int64), codes=np.concatenate(code_parts),
        starts=np.concatenate(start_parts),
    )
    payload = {
        "schemaVersion": 1,
        "purpose": "pipeline mechanics only; not a scientific accuracy benchmark",
        "sourceManifestSha256": EXPECTED["manifest"],
        "spec": manifest.get("spec", {}),
        "columns": manifest["columns"],
        "identities": manifest["identities"],
        "rows": [public_row(manifest["rows"][index]) for index in selected],
    }
    write_text_lf(manifest_out, json.dumps(payload, indent=2))
    return data_path, manifest_out


def make_splits(manifest_path: Path, destination: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    grouped: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)
    for row in manifest["rows"]:
        key = (row["relativePath"], row["fileContentHash"])
        grouped[row["role"]][key] = {
            "relativePath": row["relativePath"],
            "sha256": row["fileContentHash"],
            "evaluationClusterId": row.get("evaluationClusterId"),
            "rBin": row["binName"],
        }
    roles = {role: sorted(values.values(), key=lambda item: (item["relativePath"], item["sha256"]))
             for role, values in sorted(grouped.items())}
    roles.setdefault("final", [])
    payload = {
        "schemaVersion": 1,
        "sourceManifestSha256": EXPECTED["manifest"],
        "fileIsolation": True,
        "newFinalConsumed": False,
        "roles": roles,
        "counts": {role: len(values) for role, values in roles.items()},
        "finalStatus": "unassigned; no independent final was consumed by v5.0.0",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(destination, json.dumps(payload, indent=2))


def make_release(model_json: Path, model_joblib: Path, fixture: Path,
                 fixture_manifest: Path, destination: Path,
                 source_commit: str | None) -> None:
    model = json.loads(model_json.read_text(encoding="utf-8"))
    if len(model["columns"]) != 207 or len(model["treeInfo"]) != 600:
        raise ValueError("accepted model does not satisfy the frozen v5 contract")
    destination.mkdir(parents=True, exist_ok=True)
    copied_model = destination / "unifiedV5Model.json"
    shutil.copyfile(model_json, copied_model)
    write_text_lf(destination / "feature-schema.json", json.dumps({
        "schemaVersion": 5, "featureCount": 207, "orderedFeatures": model["columns"],
    }, indent=2))
    write_text_lf(destination / "identities.json", json.dumps({
        "schemaVersion": 1, "identities": model["identities"],
    }, indent=2))
    gold_path = destination / "golden-fixtures" / "python-scores.json"
    generate_gold(model_joblib, fixture, fixture_manifest, gold_path, limit=5)
    manifest = {
        "schemaVersion": 1,
        "modelVersion": "v5.0.0",
        "assetModelVersion": model["version"],
        "runtimeVersion": "local-review-v3",
        "featureSchemaVersion": 5,
        "featureCount": 207,
        "treeCount": 600,
        "windowWidth": 13,
        "cofechaJsVersion": "0.2.0",
        "modelSha256": file_sha256(copied_model),
        "sourceJoblibSha256": EXPECTED["joblib"],
        "trainingConfigSha256": file_sha256(ROOT / "configs/v5.yaml"),
        "splitManifestSha256": file_sha256(ROOT / "splits/FILE_SPLITS.json"),
        "sourceRepository": "TieString/Crossdating-Model",
        "sourceCommit": source_commit,
        "importedFromApplicationCommit": "f2d48dd0393785cdf456dff78d4977a816626d0c",
        "provenance": "Imported from the frozen pre-publication v5 research run",
    }
    write_text_lf(destination / "model-manifest.json", json.dumps(manifest, indent=2))
    shutil.copyfile(ROOT / "MODEL_CARD.md", destination / "MODEL_CARD.md")
    shutil.copyfile(ROOT / "reports/v5-accepted-evaluation.md", destination / "METRICS.md")
    shutil.copyfile(ROOT / "splits/FILE_SPLITS.json", destination / "FILE_SPLITS.json")


def write_sums(destination: Path) -> None:
    files = sorted(path for path in destination.rglob("*") if path.is_file()
                   and path.name != "SHA256SUMS.txt")
    lines = [f"{file_sha256(path)}  {path.relative_to(destination).as_posix()}" for path in files]
    write_text_lf(destination / "SHA256SUMS.txt", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-joblib", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--per-cell", type=int, default=2)
    parser.add_argument("--source-commit", help="Auditable repository commit containing the imported pipeline")
    args = parser.parse_args()
    for label, path in (("packed", args.packed), ("manifest", args.manifest),
                        ("joblib", args.model_joblib), ("model_json", args.model_json)):
        require_hash(path, EXPECTED[label], label)
    fixture, fixture_manifest = make_fixture(
        args.packed, args.manifest, ROOT / "datasets/fixtures", args.per_cell)
    make_splits(args.manifest, ROOT / "splits/FILE_SPLITS.json")
    release = ROOT / "model-releases/v5.0.0"
    make_release(args.model_json, args.model_joblib, fixture, fixture_manifest, release,
                 args.source_commit)
    write_sums(release)
    print(json.dumps({"fixture": str(fixture), "fixtureSha256": file_sha256(fixture),
                      "releaseModelSha256": file_sha256(release / "unifiedV5Model.json")}, indent=2))


if __name__ == "__main__":
    main()
