import math
from pathlib import Path

import pytest

from scripts.common import (
    MAX_ATTEMPTS,
    calibration_metrics,
    macro_f1,
    read_jsonl,
    resume_rows,
)
from scripts.run_public_benchmarks import (
    dcg,
    ndcg,
    parse_suites,
    run_classification,
)


KEY_FIELDS = ("provider", "case_id")


def saved_row(provider: str, case_id: str, error: str | None = None) -> dict:
    """Build a minimal saved result row.

    Args:
        provider: Provider name.
        case_id: Case id.
        error: Recorded request error, if any.

    Returns:
        A row with the fields ``resume_rows`` reads.
    """
    return {"provider": provider, "case_id": case_id, "error": error}


class StubClient:
    """Client stand-in that returns a fixed response body or fails."""

    def __init__(self, body: dict | None) -> None:
        """Store the response body; ``None`` makes every request fail.

        Args:
            body: Decoded SystemOne response to return.
        """
        self.config = {"model": "configured-model"}
        self.body = body

    def request(self, request: dict) -> tuple[dict, float, int]:
        """Return the stored body, or raise when there is none.

        Args:
            request: Ignored request payload.

        Returns:
            The body, a fixed latency, and one attempt.

        Raises:
            RuntimeError: If no body was stored.
        """
        if self.body is None:
            raise RuntimeError("server down")
        return self.body, 12.5, 1


def test_resume_keeps_successes_and_retries_selected_errors() -> None:
    """Without --force, only selected rows that recorded an error are rerun."""
    saved = [
        saved_row("laya", "1"),
        saved_row("laya", "2", error="RuntimeError: timeout"),
        saved_row("jev", "2", error="RuntimeError: timeout"),
    ]
    selected = {("laya", "1"), ("laya", "2")}
    kept = resume_rows(saved, selected, KEY_FIELDS, force=False)
    assert kept == [saved[0], saved[2]]


def test_force_with_limit_replaces_only_selected_cases() -> None:
    """--force drops the selected cases and keeps the provider's other rows."""
    saved = [saved_row("laya", str(index)) for index in range(5)]
    selected = {("laya", "0"), ("laya", "1")}
    kept = resume_rows(saved, selected, KEY_FIELDS, force=True)
    assert kept == saved[2:]
    assert len(saved) == 5


def test_read_jsonl_is_strict_only_when_required(tmp_path: Path) -> None:
    """A missing file is empty by default and an error when required.

    Args:
        tmp_path: Fixture providing an empty directory.
    """
    missing = tmp_path / "missing.jsonl"
    assert read_jsonl(missing) == []
    with pytest.raises(FileNotFoundError):
        read_jsonl(missing, required=True)


def test_calibration_metrics_match_hand_computed_values() -> None:
    """Log loss, Brier score, and ECE match a hand-computed case."""
    metrics = calibration_metrics([({"a": 0.8, "b": 0.2}, "a", "a", ("a", "b"))])
    assert metrics["log_loss"] == pytest.approx(-math.log(0.8))
    assert metrics["brier_score"] == pytest.approx(0.08)
    assert metrics["ece"] == pytest.approx(0.2)
    assert calibration_metrics([]) == {
        "log_loss": None,
        "brier_score": None,
        "ece": None,
    }


def test_macro_f1_averages_over_explicit_labels() -> None:
    """Explicit labels count even when they never occur."""
    assert macro_f1(["a"], ["a"]) == 1.0
    assert macro_f1(["a"], ["a"], ("a", "b")) == 0.5


def test_dcg_uses_exponential_gains() -> None:
    """ESCI grades 3/2/1/0 contribute gains 7/3/1/0."""
    assert dcg([3, 2, 1, 0]) == pytest.approx(7 + 3 / math.log2(3) + 1 / 2)


def test_ndcg_ranks_candidates_by_score() -> None:
    """nDCG is 1 for the ideal order and 0 when nothing is relevant."""
    ideal_order = [
        {"label": "Exact", "score": 2.0},
        {"label": "Irrelevant", "score": 0.0},
    ]
    reversed_order = [
        {"label": "Exact", "score": 0.0},
        {"label": "Irrelevant", "score": 2.0},
    ]
    assert ndcg(ideal_order) == 1.0
    assert ndcg(reversed_order) == pytest.approx(1 / math.log2(3))
    assert ndcg([{"label": "Irrelevant", "score": 1.0}]) == 0.0


def test_parse_suites_rejects_unknown_names() -> None:
    """Known suites parse and a typo stops the run."""
    assert parse_suites("llm_evals, reranking") == {"llm_evals", "reranking"}
    with pytest.raises(SystemExit):
        parse_suites("mmlu")


def test_public_rows_record_the_reported_model(tmp_path: Path) -> None:
    """Successful public rows save the model id from the response.

    Args:
        tmp_path: Fixture providing an empty directory.
    """
    body = {
        "model": "jev-1.13.0",
        "answers": {"queue": {"choice": "a", "probabilities": {"a": 0.9, "b": 0.1}}},
    }
    raw_path = tmp_path / "raw.jsonl"
    rows = run_classification(
        StubClient(body),
        "jev",
        "inbox_triage",
        [{"id": "case-1", "expected": "a"}],
        lambda case: {},
        "expected",
        raw_path,
    )
    assert rows[0]["model"] == "jev-1.13.0"
    assert rows[0]["predicted"] == "a"
    assert read_jsonl(raw_path) == rows


def test_public_error_rows_record_configured_model(tmp_path: Path) -> None:
    """Failed public rows save the error, attempt count, and configured model.

    Args:
        tmp_path: Fixture providing an empty directory.
    """
    rows = run_classification(
        StubClient(None),
        "jev",
        "llm_evals",
        [{"id": "case-1", "expected": "A"}],
        lambda case: {},
        "expected",
        tmp_path / "raw.jsonl",
    )
    assert rows[0]["error"] == "RuntimeError: server down"
    assert rows[0]["attempts"] == MAX_ATTEMPTS
    assert rows[0]["model"] == "configured-model"
