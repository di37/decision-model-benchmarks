#!/usr/bin/env python3
"""Run BFCL v3 multiple as native-SystemOne tool selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
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


DEFAULT_DATASET = ROOT / "data" / "tool_selection" / "bfcl_v3_multiple.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "tool_selection"
SEED = 20261009
KEY_FIELDS = ("provider", "case_id")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the tool-selection runner.

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


def shuffled_tools(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Shuffle a case's candidate tools deterministically.

    Args:
        case: BFCL case with an ``id`` and ``candidate_tools``.

    Returns:
        The candidate tools in an order that depends only on ``SEED`` and the
        case id.
    """
    tools = list(case["candidate_tools"])
    digest = hashlib.sha256(f"{SEED}:{case['id']}".encode()).digest()
    random.Random(digest).shuffle(tools)
    return tools


def tool_selection_request(case: dict[str, Any]) -> dict[str, Any]:
    """Build the SystemOne tool-selection request for one BFCL case.

    Args:
        case: BFCL case with ``conversation`` and ``candidate_tools``.

    Returns:
        A request with a single ``tool`` choice whose options are the exact
        function names, each described by its description and parameter
        schema.
    """
    criteria = {}
    for tool in shuffled_tools(case):
        schema = json.dumps(
            tool["parameter_schema"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        criteria[tool["name"]] = (
            f"{tool['description']}\nParameter schema: {schema}"
        )
    return {
        "state": {"conversation": case["conversation"]},
        "questions": {
            "tool": {
                "type": "choice",
                "instructions": (
                    "Select the one candidate function that should be called "
                    "to satisfy the user request. Return the exact function name."
                ),
                "criteria": criteria,
            }
        },
    }


def normalized_result(
    case: dict[str, Any],
    provider: str,
    body: dict[str, Any],
    latency_ms: float,
    attempts: int,
) -> dict[str, Any]:
    """Convert a tool-selection response into a saved result row.

    Args:
        case: The BFCL case that was sent.
        provider: Provider name.
        body: Decoded SystemOne response.
        latency_ms: Latency of the successful attempt in milliseconds.
        attempts: Number of attempts made.

    Returns:
        The predicted tool, candidate probabilities, confidence, latency, token
        usage, and raw response.

    Raises:
        KeyError: If the response has no ``tool`` answer.
        ValueError: If the predicted tool is not one of the candidates.
    """
    answer = body["answers"]["tool"]
    predicted = str(answer["choice"])
    candidate_names = {tool["name"] for tool in case["candidate_tools"]}
    if predicted not in candidate_names:
        raise ValueError(f"unexpected tool {predicted!r}")
    probabilities = {
        str(key): float(value)
        for key, value in (answer.get("probabilities") or {}).items()
        if str(key) in candidate_names
    }
    confidence = answer.get("confidence")
    usage = body.get("usage") or {}
    return {
        "case_id": case["id"],
        "provider": provider,
        "expected_tool": case["expected_tool"],
        "predicted_tool": predicted,
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


def summarize(
    rows: list[dict[str, Any]], case_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Compute quality, calibration, and latency metrics for one provider.

    Args:
        rows: Saved tool-selection rows for a single provider.
        case_by_id: Case id to BFCL case, used to find each case's candidates.

    Returns:
        Accuracy, macro-F1, latency percentiles, log loss, Brier score, and ECE.
        Calibration metrics are ``None`` when no row has probabilities for all
        of its candidates.
    """
    completed = [row for row in rows if not row.get("error")]
    expected = [row["expected_tool"] for row in rows]
    predicted = [row.get("predicted_tool") for row in rows]
    correct = sum(a == p for a, p in zip(expected, predicted))

    calibrated = []
    for row in completed:
        # Sorted so Brier terms are summed in a fixed order on every run.
        candidates = sorted(
            {tool["name"] for tool in case_by_id[row["case_id"]]["candidate_tools"]}
        )
        if set(candidates) <= set(row.get("probabilities") or {}):
            calibrated.append(
                (
                    row["probabilities"],
                    row["expected_tool"],
                    row["predicted_tool"],
                    candidates,
                )
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
        "macro_f1": macro_f1(expected, predicted),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_mean_ms": statistics.mean(latencies) if latencies else None,
        **calibration_metrics(calibrated),
    }


def write_reports(
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Write summaries, per-case predictions, and paired tests.

    Args:
        cases: All BFCL cases, in dataset order.
        rows: Saved result rows for every provider.
        output_dir: Directory for the summary, predictions, and analysis files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    case_by_id = {case["id"]: case for case in cases}
    case_order = {case["id"]: index for index, case in enumerate(cases)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["provider"]].append(row)
    summaries = [
        summarize(grouped[provider], case_by_id) for provider in sorted(grouped)
    ]
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "summary.csv", summaries)

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
                "conversation": case["conversation"],
                "candidate_count": len(case["candidate_tools"]),
                "expected_tool": row["expected_tool"],
                "predicted_tool": row.get("predicted_tool"),
                "correct": row.get("predicted_tool") == row["expected_tool"]
                and not row.get("error"),
                "confidence": row.get("confidence"),
                "max_probability": (
                    max(probabilities.values()) if probabilities else None
                ),
                "probabilities": json.dumps(
                    probabilities, ensure_ascii=False, sort_keys=True
                ),
                "latency_ms": row.get("latency_ms"),
                "model": row.get("model"),
                "error": row.get("error"),
            }
        )
    write_csv(output_dir / "predictions.csv", predictions)

    correctness = {
        provider: {
            row["case_id"]: (
                row.get("predicted_tool") == row["expected_tool"]
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
    """Run or resume BFCL tool selection and regenerate its reports.

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
        record_run(args.output_dir, config, "tool_selection")
        client = Client(config)
        try:
            client.warm_up()
            for index, case in enumerate(pending, start=1):
                print(
                    f"[{provider} {index}/{len(pending)}] {case['id']}",
                    flush=True,
                )
                try:
                    body, latency, attempts = client.request(
                        tool_selection_request(case)
                    )
                    row = normalized_result(
                        case, provider, body, latency, attempts
                    )
                except Exception as exc:
                    row = {
                        "case_id": case["id"],
                        "provider": provider,
                        "expected_tool": case["expected_tool"],
                        "predicted_tool": None,
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
    raise SystemExit(main())
