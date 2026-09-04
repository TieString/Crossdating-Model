import json
from pathlib import Path


def test_frozen_target_and_case_counts() -> None:
    expected = {
        "development": {"files": 220, "targets": 3000, "cases": 6554},
        "calibration": {"files": 200, "targets": 2924, "cases": 6437},
    }
    for role, counts in expected.items():
        targets = json.loads(Path(f"splits/{role}-targets.json").read_text(encoding="utf-8"))
        cases = json.loads(Path(f"scenarios/v5-{role}-cases.json").read_text(encoding="utf-8"))
        assert len(targets["files"]) == counts["files"]
        assert targets["counts"]["eligibleTargets"] == counts["targets"]
        assert len(cases["cases"]) == counts["cases"]
        assert all(case["expectedStages"] for case in cases["cases"])
        assert all("D:\\" not in case["relativePath"] for case in cases["cases"])
    final = json.loads(Path("splits/final-targets.json").read_text(encoding="utf-8"))
    assert final["status"] == "unassigned"
    assert final["files"] == []


def test_distant_d_definition_is_at_least_thirty_years() -> None:
    for role in ("development", "calibration"):
        cases = json.loads(Path(f"scenarios/v5-{role}-cases.json").read_text(encoding="utf-8"))["cases"]
        distant = [case for case in cases if case["caseId"].endswith(":D-distant-v11")]
        assert distant
        for case in distant:
            years = sorted(truth["year"] for truth in case["truths"] if truth["year"] is not None)
            assert len(years) >= 2
            assert all(right - left >= 30 for left, right in zip(years, years[1:]))
