import json
from pathlib import Path


def test_public_manifest_contains_only_rwl_inputs() -> None:
    manifest = json.loads(Path("datasets/source-manifest.json").read_text(encoding="utf-8"))
    assert manifest["sourceCount"] == len(manifest["sources"]) == 420
    assert manifest["derivedEvidencePublished"] is False
    assert {row["kind"] for row in manifest["sources"]} == {"rwl"}
    assert sum(row["role"] == "development" for row in manifest["sources"]) == 220
    assert sum(row["role"] == "calibration" for row in manifest["sources"]) == 200
    assert all(row["url"].startswith("https://www.ncei.noaa.gov/") for row in manifest["sources"])
    assert len({row["relativePath"] for row in manifest["sources"]}) == 420
