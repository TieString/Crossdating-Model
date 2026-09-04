"""Verify a model release without trusting filenames or Git metadata."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crossdating_model.config import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    args = parser.parse_args()
    release = args.release.resolve()
    sums = release / "SHA256SUMS.txt"
    for line in sums.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        target = (release / relative).resolve()
        if release not in target.parents:
            raise ValueError(f"unsafe release path: {relative}")
        actual = file_sha256(target)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {relative}: {actual}")
    contract = json.loads((release / "model-manifest.json").read_text(encoding="utf-8"))
    model_path = release / "unifiedV5Model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    checks = {
        "modelSha256": file_sha256(model_path) == contract["modelSha256"],
        "featureCount": len(model["columns"]) == contract["featureCount"] == 207,
        "uniqueFeatures": len(set(model["columns"])) == 207,
        "treeCount": len(model["treeInfo"]) == contract["treeCount"] == 600,
        "windowWidth": model["windowWidth"] == contract["windowWidth"] == 13,
        "cofechaJs": contract["cofechaJsVersion"] == "0.2.0",
    }
    if not all(checks.values()):
        raise ValueError(f"release contract failed: {checks}")
    print(json.dumps({"release": str(release), "filesVerified": len(sums.read_text().splitlines()),
                      "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
