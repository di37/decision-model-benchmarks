import math

import pytest

from scripts.analyze_public_benchmarks import wilson
from scripts.common import (
    PROVIDERS,
    Client,
    all_pairs_mcnemar,
    holm,
    macro_f1,
    mcnemar,
    provider_config,
)
from scripts.run_model_routing import (
    routing_request,
    summarize as summarize_routing,
)
from scripts.run_public_benchmarks import dcg, mmlu_request, shuffled_mapping


def test_option_shuffle_is_deterministic() -> None:
    """Option shuffling is repeatable and keeps every option."""
    options = {"a": "one", "b": "two", "c": "three", "d": "four"}
    first = shuffled_mapping(options, "case-1")
    second = shuffled_mapping(options, "case-1")
    assert first == second
    assert set(first.items()) == set(options.items())


def test_mmlu_request_preserves_all_choices() -> None:
    """MMLU requests keep all four choices under their letters."""
    case = {
        "id": "mmlu-test-01",
        "subject": "test_subject",
        "question": "Which option?",
        "choices": ["one", "two", "three", "four"],
    }
    request = mmlu_request(case)
    assert request["state"]["subject"] == "test subject"
    assert set(request["questions"]["answer"]["criteria"].items()) == {
        ("A", "one"),
        ("B", "two"),
        ("C", "three"),
        ("D", "four"),
    }


def test_only_comparison_providers_are_exposed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the four compared providers resolve, at their default URLs.

    Args:
        monkeypatch: Fixture used to clear URL overrides from the environment.
    """
    for name in ("JEV_URL", "CLEF_FLASH_URL", "LAYA_URL", "CLM_URL"):
        monkeypatch.delenv(name, raising=False)
    assert provider_config("jev")["url"] == "https://api.typesafe.ai/v1/systemone"
    assert (
        provider_config("clef-flash")["url"]
        == "http://127.0.0.1:8710/v1/systemone"
    )
    assert provider_config("laya")["url"] == "http://127.0.0.1:8004/v1/systemone"
    assert provider_config("clm")["url"] == "http://127.0.0.1:8700/v1/systemone"
    assert PROVIDERS == ("clef-flash", "clm", "jev", "laya")
    with pytest.raises(ValueError, match="Unknown provider"):
        provider_config("other")


def test_laya_uses_router_and_raised_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Laya omits ``model`` and gets the raised budget; others get neither.

    Args:
        monkeypatch: Fixture used to clear Laya overrides from the environment.
    """
    for name in ("LAYA_MODEL", "LAYA_MAX_LEN", "LAYA_HEAD_MAX_LEN"):
        monkeypatch.delenv(name, raising=False)
    config = provider_config("laya")
    assert config["model"] is None
    client = Client(config)
    payload = client.payload({"state": "x", "questions": {}})
    client.close()
    assert payload["max_len"] == 2048
    assert payload["head_max_len"] == 1024
    assert "model" not in payload

    for name in ("jev", "clef-flash", "clm"):
        other_client = Client(provider_config(name))
        other = other_client.payload({"state": "x", "questions": {}})
        other_client.close()
        assert "max_len" not in other and "head_max_len" not in other


def test_all_pairs_are_holm_adjusted() -> None:
    """Every provider pair is tested and Holm never lowers a p-value."""
    correctness = {
        "a": {str(i): True for i in range(10)},
        "b": {str(i): i < 2 for i in range(10)},
        "c": {str(i): i < 9 for i in range(10)},
    }
    pairs = all_pairs_mcnemar(correctness)
    assert [(p["left"], p["right"]) for p in pairs] == [("a", "b"), ("a", "c"), ("b", "c")]
    for pair in pairs:
        assert pair["p_holm"] >= pair["p_exact_two_sided"]
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_macro_f1_handles_missing_predictions() -> None:
    """A missing prediction counts as a miss in macro-F1."""
    assert macro_f1(["a", "b"], ["a", None]) == 0.5


def test_dcg_rewards_relevant_items_near_the_top() -> None:
    """DCG is higher when relevant items rank first."""
    assert dcg([3, 2, 1, 0]) > dcg([0, 1, 2, 3])


def test_paired_statistics_helpers() -> None:
    """McNemar counts and the Wilson interval match known values."""
    result = mcnemar({"1": True}, {"1": False})
    assert result["left_correct_right_wrong"] == 1
    assert result["left_wrong_right_correct"] == 0
    assert result["p_exact_two_sided"] == 1.0

    low, high = wilson(90, 100)
    assert low < 0.9 < high
    assert math.isclose(low, 0.8256326956, rel_tol=1e-8)


def test_routing_payload_uses_native_systemone_schema() -> None:
    """The routing request uses the native SystemOne choice schema."""
    request = routing_request({"prompt": "Fix this ordinary Python bug."})
    assert request["state"] == {"prompt": "Fix this ordinary Python bug."}
    question = request["questions"]["route"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"small", "medium", "large"}
    assert "minimum model capability tier" in question["instructions"]


def test_routing_metrics_and_provider_restrictions() -> None:
    """Routing metrics are correct and unknown providers are rejected."""
    rows = []
    for route, predicted, latency in (
        ("small", "small", 10.0),
        ("medium", "medium", 20.0),
        ("large", "medium", 30.0),
    ):
        probabilities = {name: 0.0 for name in ("small", "medium", "large")}
        probabilities[predicted] = 1.0
        rows.append(
            {
                "provider": "jev",
                "expected_route": route,
                "predicted_route": predicted,
                "probabilities": probabilities,
                "latency_ms": latency,
                "error": None,
            }
        )
    summary = summarize_routing(rows)
    assert summary["accuracy"] == 2 / 3
    assert summary["small_recall"] == 1.0
    assert summary["medium_recall"] == 1.0
    assert summary["large_recall"] == 0.0
    assert summary["latency_p50_ms"] == 20.0

    paired = mcnemar({"1": True}, {"1": False})
    assert paired["left_correct_right_wrong"] == 1
    with pytest.raises(ValueError, match="Unknown provider"):
        provider_config("other")
