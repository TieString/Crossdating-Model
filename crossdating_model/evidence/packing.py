"""Pack versioned per-state candidate files into memory-mappable arrays."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from crossdating_model.config import file_sha256


def pack_candidates(manifest_path: str | Path, output_dir: str | Path) -> Path:
    manifest_path, output = Path(manifest_path), Path(output_dir)
    if output.exists():
        raise FileExistsError(f"packed output already exists: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows, identities = manifest["rows"], manifest["identities"]
    sizes = np.asarray([row["candidateCount"] for row in rows], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
    total, feature_count = int(offsets[-1]), len(manifest["columns"])
    output.mkdir(parents=True)
    arrays = {
        "x": np.lib.format.open_memmap(output / "x.npy", mode="w+", dtype=np.float32,
                                        shape=(total, feature_count)),
        "y": np.lib.format.open_memmap(output / "y.npy", mode="w+", dtype=np.int8, shape=(total,)),
        "codes": np.lib.format.open_memmap(output / "codes.npy", mode="w+", dtype=np.int16, shape=(total,)),
        "starts": np.lib.format.open_memmap(output / "starts.npy", mode="w+", dtype=np.int32, shape=(total,)),
    }
    for index, row in enumerate(rows):
        with np.load(row["candidatePath"], allow_pickle=False) as source:
            values, codes, starts = source["features"], source["identity"], source["starts"]
        if values.shape != (sizes[index], feature_count):
            raise ValueError(f"candidate shape mismatch: {row['stateHash']}")
        selection = slice(int(offsets[index]), int(offsets[index + 1]))
        arrays["x"][selection] = values
        arrays["codes"][selection] = codes
        arrays["starts"][selection] = starts
        truth = row["truth"]
        correct_identity = identities.index([truth["kind"], truth["shift"]])
        good = codes == correct_identity
        if truth["kind"] not in ("none", "whole"):
            good &= (starts <= truth["year"]) & (truth["year"] <= starts + 12)
        arrays["y"][selection] = good
        if (index + 1) % 1000 == 0:
            print(json.dumps({"packedStates": index + 1, "totalStates": len(rows)}), flush=True)
    for array in arrays.values():
        array.flush()
    np.save(output / "offsets.npy", offsets)
    np.save(output / "sizes.npy", sizes)
    (output / "manifest-sha256.txt").write_text(file_sha256(manifest_path) + "\n",
                                                  encoding="ascii", newline="\n")
    (output / "metadata.json").write_text(json.dumps({
        "schemaVersion": 1, "states": len(rows), "candidates": total,
        "featureCount": feature_count, "manifestSha256": file_sha256(manifest_path),
        "derivedCache": True, "publish": False,
    }, indent=2), encoding="utf-8", newline="\n")
    return output
