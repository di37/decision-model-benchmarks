import json
from pathlib import Path

import pytest

from scripts.common import mcnemar, provider_config
from scripts.run_tool_selection import summarize, tool_selection_request


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "tool_selection" / "bfcl_v3_multiple.jsonl"


def load_cases() -> list[dict]:
    """Load the normalized BFCL cases.

    Returns:
        All cases from ``bfcl_v3_multiple.jsonl``.
    """
    return [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_bfcl_has_200_aligned_expected_tools() -> None:
    """All 200 cases are unique and their expected tool is a candidate."""
    cases = load_cases()
    assert len(cases) == 200
    assert len({case["id"] for case in cases}) == 200
    for case in cases:
        candidate_names = {tool["name"] for tool in case["candidate_tools"]}
        assert case["expected_tool"] in candidate_names
        assert case["conversation"]


def test_request_preserves_candidates_and_is_deterministic() -> None:
    """Requests are repeatable and keep each tool's description and schema."""
    case = load_cases()[1]
    first = tool_selection_request(case)
    second = tool_selection_request(case)
    first_criteria = first["questions"]["tool"]["criteria"]
    second_criteria = second["questions"]["tool"]["criteria"]
    expected_names = {tool["name"] for tool in case["candidate_tools"]}

    assert first["state"]["conversation"] == case["conversation"]
    assert first_criteria == second_criteria
    assert list(first_criteria) == list(second_criteria)
    assert set(first_criteria) == expected_names
    for tool in case["candidate_tools"]:
        criterion = first_criteria[tool["name"]]
        assert tool["description"] in criterion
        assert "Parameter schema:" in criterion


def test_tool_selection_exposes_the_four_compared_providers() -> None:
    """Tool selection resolves all four providers and rejects others."""
    for name in ("jev", "clef-flash", "laya", "clm"):
        assert provider_config(name)["name"] == name
    with pytest.raises(ValueError, match="Unknown provider"):
        provider_config("other")


def test_tool_metrics_and_paired_statistic() -> None:
    """Tool-selection metrics and the paired McNemar counts are correct."""
    cases = [
        {
            "id": "one",
            "candidate_tools": [{"name": "a"}, {"name": "b"}],
        },
        {
            "id": "two",
            "candidate_tools": [{"name": "c"}, {"name": "d"}],
        },
    ]
    rows = [
        {
            "case_id": "one",
            "provider": "jev",
            "expected_tool": "a",
            "predicted_tool": "a",
            "probabilities": {"a": 0.8, "b": 0.2},
            "latency_ms": 10.0,
            "error": None,
        },
        {
            "case_id": "two",
            "provider": "jev",
            "expected_tool": "c",
            "predicted_tool": "d",
            "probabilities": {"c": 0.25, "d": 0.75},
            "latency_ms": 30.0,
            "error": None,
        },
    ]
    summary = summarize(rows, {case["id"]: case for case in cases})
    assert summary["accuracy"] == 0.5
    assert summary["macro_f1"] == 0.5
    assert summary["latency_p50_ms"] == 20.0
    assert summary["log_loss"] is not None
    assert summary["brier_score"] is not None
    assert summary["ece"] is not None

    paired = mcnemar({"one": True, "two": False}, {"one": False, "two": False})
    assert paired["left_correct_right_wrong"] == 1
    assert paired["p_exact_two_sided"] == 1.0
