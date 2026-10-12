#!/usr/bin/env python3
"""Run and report the 90-case model-routing benchmark via native SystemOne."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    MAX_ATTEMPTS,
    PROVIDERS,
    Client,
    all_pairs_mcnemar,
    append_jsonl,
    calibration_metrics,
    macro_f1,
    parse_providers,
    percentile,
    read_jsonl,
    record_run,
    resume_rows,
    row_key,
    write_csv,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "model_routing" / "routing_benchmark.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "model_routing"
KEY_FIELDS = ("provider", "case_id")
# The saved results used this fixed option order for every provider; shuffling
# it per case would change the requests and require a full rerun.
ROUTES = ("small", "medium", "large")
ROUTING_QUESTION = {
    "type": "choice",
    "instructions": (
        "Choose the minimum model capability tier that can reliably complete "
        "the entire user request. Account for reasoning depth, ambiguity, "
        "specialist knowledge, risk, and number of interacting constraints. "
        "Apply these tie-breaking rules in order: "
        "(1) route to the lowest tier that can reliably complete the whole "
        "request; "
        "(2) high consequence raises a request to the largest tier, even when "
        "the prose is short; "
        "(3) a request combining multiple otherwise-medium deliverables is "
        "large; "
        "(4) input length alone does not raise the tier if the operation "
        "remains mechanical."
    ),
    "criteria": {
        "small": (
            "Deterministic, low-risk, single-step extraction, formatting, "
            "classification, arithmetic, or short factual work."
        ),
        "medium": (
            "Bounded multi-step reasoning, ordinary coding, debugging, "
            "analysis, planning, or synthesis with reversible mistakes."
        ),
        "large": (
            "Difficult, ambiguous, high-risk, specialist, long-horizon, or "
            "many-constraint work such as architecture, security, legal, "
            "medical, financial, research, or formal reasoning."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the routing runner.

    Returns:
        Parsed options: providers, dataset, output directory, limit, and force.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        default=",".join(PROVIDERS),
        help=f"Comma-separated subset of {','.join(PROVIDERS)}.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace selected providers and cases instead of resuming.",
    )
    return parser.parse_args()


def routing_request(case: dict[str, Any]) -> dict[str, Any]:
    """Build the SystemOne routing request for one case.

    Args:
        case: Routing case with a ``prompt``.

    Returns:
        A provider-neutral request asking the shared ``route`` question.
    """
    return {
        "state": {"prompt": case["prompt"]},
        "questions": {"route": ROUTING_QUESTION},
    }


def normalized_result(
    case: dict[str, Any],
    provider: str,
    body: dict[str, Any],
    latency_ms: float,
    attempts: int,
) -> dict[str, Any]:
    """Convert a routing response into a saved result row.

    Args:
        case: The routing case that was sent.
        provider: Provider name.
        body: Decoded SystemOne response.
        latency_ms: Latency of the successful attempt in milliseconds.
        attempts: Number of attempts made.

    Returns:
        The prediction, route probabilities, confidence, latency, token usage,
        and raw response.

    Raises:
        KeyError: If the response has no ``route`` answer.
        ValueError: If the predicted route is not small, medium, or large.
    """
    answer = body["answers"]["route"]
    predicted = str(answer["choice"]).lower()
    if predicted not in ROUTES:
        raise ValueError(f"unexpected route {predicted!r}")
    probabilities = {
        str(key).lower(): float(value)
        for key, value in (answer.get("probabilities") or {}).items()
        if str(key).lower() in ROUTES
    }
    usage = body.get("usage") or {}
    confidence = answer.get("confidence")
    return {
        "case_id": case["id"],
        "provider": provider,
        "expected_route": case["expected_route"],
        "predicted_route": predicted,
        "probabilities": probabilities,
        "confidence": (
            float(confidence)
            if confidence is not None
            else probabilities.get(predicted)
        ),
        "latency_ms": latency_ms,
        "model": body.get("model"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "error": None,
        "raw_response": body,
        "attempts": attempts,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute quality, calibration, and latency metrics for one provider.

    Args:
        rows: Saved routing rows for a single provider.

    Returns:
        Accuracy, macro-F1, per-route recall, latency percentiles, log loss,
        Brier score, and ECE. Calibration metrics are ``None`` when no row has
        probabilities for all three routes.
    """
    completed = [row for row in rows if not row.get("error")]
    expected = [row["expected_route"] for row in rows]
    predicted = [row.get("predicted_route") for row in rows]
    correct = sum(a == p for a, p in zip(expected, predicted))
    recalls = {
        route: (
            sum(a == route and p == route for a, p in zip(expected, predicted))
            / expected.count(route)
            if route in expected
            else 0.0
        )
        for route in ROUTES
    }
    calibration = calibration_metrics(
        [
            (
                row["probabilities"],
                row["expected_route"],
                row["predicted_route"],
                ROUTES,
            )
            for row in completed
            if all(route in (row.get("probabilities") or {}) for route in ROUTES)
        ]
    )

    latencies = [
        float(row["latency_ms"])
        for row in completed
        if row.get("latency_ms") is not None
    ]
    return {
        "provider": rows[0]["provider"],
        "cases": len(rows),
        "completed": len(completed),
        "errors": len(rows) - len(completed),
        "accuracy": correct / len(rows),
        "macro_f1": macro_f1(expected, predicted, ROUTES),
        "small_recall": recalls["small"],
        "medium_recall": recalls["medium"],
        "large_recall": recalls["large"],
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_mean_ms": statistics.mean(latencies) if latencies else None,
        **calibration,
    }


def write_reports(
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Write summaries, per-case predictions, and paired tests.

    Args:
        cases: All routing cases, in dataset order.
        rows: Saved result rows for every provider.
        output_dir: Directory for the summary, predictions, and analysis files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["provider"]].append(row)
    summaries = [summarize(grouped[name]) for name in sorted(grouped)]
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "summary.csv", summaries)

    case_by_id = {case["id"]: case for case in cases}
    case_order = {case["id"]: index for index, case in enumerate(cases)}
    predictions = []
    for row in sorted(
        rows, key=lambda item: (item["provider"], case_order[item["case_id"]])
    ):
        case = case_by_id[row["case_id"]]
        probabilities = row.get("probabilities") or {}
        predictions.append(
            {
                "case_id": row["case_id"],
                "provider": row["provider"],
                "category": case["category"],
                "prompt": case["prompt"],
                "expected_route": row["expected_route"],
                "predicted_route": row.get("predicted_route"),
                "correct": row.get("predicted_route") == row["expected_route"]
                and not row.get("error"),
                "confidence": row.get("confidence"),
                "max_probability": (
                    max(probabilities.values()) if probabilities else None
                ),
                "p_small": probabilities.get("small"),
                "p_medium": probabilities.get("medium"),
                "p_large": probabilities.get("large"),
                "latency_ms": row.get("latency_ms"),
                "model": row.get("model"),
                "error": row.get("error"),
                "rationale": case["rationale"],
            }
        )
    write_csv(output_dir / "predictions.csv", predictions)

    correctness = {
        provider: {
            row["case_id"]: (
                row.get("predicted_route") == row["expected_route"]
                and not row.get("error")
            )
            for row in provider_rows
        }
        for provider, provider_rows in grouped.items()
    }
    comparisons = all_pairs_mcnemar(correctness)
    analysis = {"classification_pairwise_mcnemar": comparisons}
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "classification_pairwise.csv", comparisons)


def main() -> int:
    """Run or resume the routing benchmark and regenerate its reports.

    Returns:
        Process exit code 0.

    Raises:
        SystemExit: If no providers or cases are selected.
        FileNotFoundError: If the dataset file is missing.
        ValueError: If saved results contain an unsupported provider.
    """
    args = parse_args()
    load_dotenv(ROOT / ".env")
    configs = parse_providers(args.providers)

    all_cases = read_jsonl(args.dataset, required=True)
    cases = all_cases[: args.limit] if args.limit is not None else all_cases
    if not cases:
        raise SystemExit("No benchmark cases selected")
    raw_path = args.output_dir / "raw_results.jsonl"
    saved = read_jsonl(raw_path)
    unsupported = {row.get("provider") for row in saved} - set(PROVIDERS)
    if unsupported:
        raise ValueError(
            "raw_results.jsonl contains unsupported providers: "
            + ", ".join(sorted(str(value) for value in unsupported))
        )

    selected = {
        (str(config["name"]), case["id"]) for config in configs for case in cases
    }
    existing = resume_rows(saved, selected, KEY_FIELDS, args.force)
    if len(existing) != len(saved):
        write_jsonl(raw_path, existing)
    completed = {row_key(row, KEY_FIELDS) for row in existing}

    for config in configs:
        provider = str(config["name"])
        pending = [
            case for case in cases if (provider, case["id"]) not in completed
        ]
        if not pending:
            continue
        record_run(args.output_dir, config, "model_routing")
        client = Client(config)
        try:
            client.warm_up()
            for index, case in enumerate(pending, start=1):
                print(
                    f"[{provider} {index}/{len(pending)}] {case['id']}",
                    flush=True,
                )
                try:
                    body, latency, attempts = client.request(routing_request(case))
                    row = normalized_result(
                        case, provider, body, latency, attempts
                    )
                except Exception as exc:
                    row = {
                        "case_id": case["id"],
                        "provider": provider,
                        "expected_route": case["expected_route"],
                        "predicted_route": None,
                        "probabilities": {},
                        "confidence": None,
                        "latency_ms": None,
                        "model": config["model"],
                        "input_tokens": None,
                        "output_tokens": None,
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_response": None,
                        "attempts": MAX_ATTEMPTS,
                    }
                append_jsonl(raw_path, row)
                existing.append(row)
                completed.add((provider, case["id"]))
        finally:
            client.close()

    all_case_ids = {case["id"] for case in all_cases}
    report_rows = [
        row
        for row in existing
        if row["case_id"] in all_case_ids and row["provider"] in PROVIDERS
    ]
    write_reports(all_cases, report_rows, args.output_dir)
    errors = sum(bool(row.get("error")) for row in report_rows)
    print(f"Results: {len(report_rows)}, errors: {errors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
