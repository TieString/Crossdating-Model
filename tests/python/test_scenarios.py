import json
from pathlib import Path

from crossdating_model.scenarios import audit_rows


def test_smoke_manifest_obeys_frozen_scenario_contract() -> None:
    manifest = json.loads(Path("datasets/fixtures/v5-smoke-manifest.json").read_text(encoding="utf-8"))
    result = audit_rows(manifest["rows"])
    assert result["states"] == 20
    assert result["files"] == 20
    assert result["fileOverlap"] == 0
    assert result["categories"] == {"A": 4, "B": 4, "C": 4, "Clean": 4, "D": 4}
