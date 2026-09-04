import numpy as np

from crossdating_model.evaluation.workflow import evaluate, summarize


IDENTITIES = [["none", 0], ["whole", -3], ["partial", -3], ["missing", -1], ["false", 1]]


def row(kind: str, shift: int, year: int | None, remaining=None) -> dict:
    return {"caseId": f"case-{kind}", "stateHash": f"state-{kind}", "fileContentHash": kind,
            "evaluationClusterId": kind, "category": "A", "binName": "0.80+",
            "distanceBand": "1-15", "truth": {"kind": kind, "shift": shift, "year": year},
            "remainingTruths": remaining or []}


def test_window_coverage_not_exact_top_year_is_correct() -> None:
    rows = [row("missing", -1, 2000)]
    scores = np.asarray([[0.0, -2.0, -1.0, 3.0, -2.0]])
    windows = np.asarray([[0, 0, 0, 1995, 0]])
    result = evaluate(rows, scores, windows, IDENTITIES, threshold=-10)
    assert result[0]["correct"] is True
    assert (result[0]["windowStart"], result[0]["windowEnd"]) == (1995, 2007)


def test_whole_to_local_and_partial_to_missing_review() -> None:
    rows = [row("missing", -1, 2000)]
    scores = np.asarray([[0.0, 4.0, 3.0, 2.0, 1.0]])
    windows = np.asarray([[0, 0, 1990, 1995, 1990]])
    result = evaluate(rows, scores, windows, IDENTITIES, threshold=-10)
    assert result[0]["wholeToLocalSwitch"] is True
    assert result[0]["partialToMissingSwitch"] is True
    assert result[0]["correct"] is True


def test_whole_requires_exact_shift_and_no_window() -> None:
    rows = [row("whole", -3, None)]
    scores = np.asarray([[0.0, 3.0, 2.0, 1.0, -1.0]])
    windows = np.zeros((1, len(IDENTITIES)), dtype=int)
    result = evaluate(rows, scores, windows, IDENTITIES, threshold=-10)
    assert result[0]["correct"] is True
    assert result[0]["windowStart"] is None
    assert summarize(result)["windowAccuracy"] == 1.0
