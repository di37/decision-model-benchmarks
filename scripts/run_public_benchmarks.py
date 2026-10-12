#!/usr/bin/env python3
"""Run public inbox-triage, reranking, and MMLU suites via SystemOne."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    MAX_ATTEMPTS,
    PROVIDERS,
    Client,
    append_jsonl,
    macro_f1,
    parse_providers,
    percentile,
    read_jsonl,
    record_run,
    resume_rows,
    row_key,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public_benchmarks"
DEFAULT_OUTPUT = ROOT / "results" / "public_benchmarks"
SEED = 20261009
SUITES = ("inbox_triage", "reranking", "llm_evals")
KEY_FIELDS = ("provider", "suite", "case_id")
RELEVANCE = {"irrelevant": 0, "complement": 1, "substitute": 2, "exact": 3}
RELEVANCE_CRITERIA = {
    "exact": "Directly satisfies the search query as the requested product.",
    "substitute": "A plausible replacement for the requested product.",
    "complement": "Related accessory or companion, but not a replacement.",
    "irrelevant": "Does not usefully satisfy the search query.",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the public-benchmark runner.

    Returns:
        Parsed options: providers, output directory, suites, limit, and force.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        default=",".join(PROVIDERS),
        help=f"Comma-separated subset of {','.join(PROVIDERS)}.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--suites",
        default=",".join(SUITES),
        help=f"Comma-separated subset of {','.join(SUITES)}.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace selected providers, suites, and cases instead of resuming.",
    )
    return parser.parse_args()


def shuffled_mapping(
    values: dict[str, str], key: str
) -> dict[str, str]:
    """Shuffle a mapping's insertion order deterministically.

    Args:
        values: Option key to option text.
        key: Per-case string mixed with the global seed.

    Returns:
        The same items in an order that depends only on ``SEED`` and ``key``.
    """
    items = list(values.items())
    digest = hashlib.sha256(f"{SEED}:{key}".encode()).digest()
    random.Random(digest).shuffle(items)
    return dict(items)


def triage_request(case: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    """Build the SystemOne request for one Banking77 message.

    Args:
        case: Triage case with an ``id`` and message ``text``.
        labels: All 77 intent labels.

    Returns:
        A request with a single ``queue`` choice over every intent.
    """
    criteria = {
        label: label.replace("_", " ")
        for label in labels
    }
    return {
        "state": {"message": case["text"]},
        "questions": {
            "queue": {
                "type": "choice",
                "instructions": (
                    "Route this customer-support inbox message to exactly one "
                    "intent queue."
                ),
                "criteria": shuffled_mapping(criteria, case["id"]),
            }
        },
    }


def mmlu_request(case: dict[str, Any]) -> dict[str, Any]:
    """Build the SystemOne request for one MMLU question.

    Args:
        case: MMLU case with ``id``, ``subject``, ``question``, and four
            ``choices``.

    Returns:
        A request with a single ``answer`` choice over options A to D.
    """
    criteria = dict(zip("ABCD", case["choices"]))
    return {
        "state": {
            "subject": case["subject"].replace("_", " "),
            "question": case["question"],
        },
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": (
                    "Select the one correct answer to the multiple-choice question."
                ),
                "criteria": shuffled_mapping(criteria, case["id"]),
            }
        },
    }


def reranking_request(case: dict[str, Any]) -> dict[str, Any]:
    """Build the SystemOne request for one ESCI query.

    Args:
        case: Reranking case with a ``query`` and ten ``candidates``.

    Returns:
        A request with one relevance choice per candidate product.
    """
    questions = {}
    for index, candidate in enumerate(case["candidates"]):
        questions[f"candidate_{index}"] = {
            "type": "choice",
            "instructions": (
                "Classify how relevant this product is to the search query. "
                f"Product title: {candidate['title']}"
            ),
            "criteria": shuffled_mapping(
                RELEVANCE_CRITERIA,
                f"{case['id']}:{candidate['id']}",
            ),
        }
    return {
        "state": {"search_query": case["query"]},
        "questions": questions,
    }


def answer_data(answer: dict[str, Any]) -> tuple[str | None, dict[str, float]]:
    """Extract the chosen option and probabilities from an answer.

    Args:
        answer: One SystemOne choice answer.

    Returns:
        The chosen option (or ``None``) and the option probabilities.
    """
    choice = answer.get("choice")
    probabilities = {
        str(key): float(value)
        for key, value in (answer.get("probabilities") or {}).items()
    }
    return str(choice) if choice is not None else None, probabilities


def run_classification(
    client: Client,
    provider: str,
    suite: str,
    cases: list[dict[str, Any]],
    request_builder,
    expected_key: str,
    raw_path: Path,
) -> list[dict[str, Any]]:
    """Run a single-choice suite and append each result as it completes.

    Args:
        client: Client for the provider being evaluated.
        provider: Provider name stored on each row.
        suite: Suite name stored on each row.
        cases: Cases still to run.
        request_builder: Function that turns a case into a request.
        expected_key: Case field holding the gold label.
        raw_path: JSON Lines file that receives each row.

    Returns:
        The new result rows, including rows that record request errors. Each
        row stores the model id the response reports, or the configured model
        id when the request failed.
    """
    rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[{suite} {index}/{len(cases)}] {provider}: {case['id']}", flush=True)
        try:
            body, latency, attempts = client.request(request_builder(case))
            answer = next(iter(body["answers"].values()))
            predicted, probabilities = answer_data(answer)
            row = {
                    "provider": provider,
                    "suite": suite,
                    "case_id": case["id"],
                    "expected": case[expected_key],
                    "predicted": predicted,
                    "probabilities": probabilities,
                    "latency_ms": latency,
                    "attempts": attempts,
                    "model": body.get("model"),
                    "subject": case.get("subject"),
                    "error": None,
                }
        except Exception as exc:
            row = {
                    "provider": provider,
                    "suite": suite,
                    "case_id": case["id"],
                    "expected": case[expected_key],
                    "predicted": None,
                    "probabilities": {},
                    "latency_ms": None,
                    "attempts": MAX_ATTEMPTS,
                    "model": client.config["model"],
                    "subject": case.get("subject"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
        append_jsonl(raw_path, row)
        rows.append(row)
    return rows


def run_reranking(
    client: Client,
    provider: str,
    cases: list[dict[str, Any]],
    raw_path: Path,
) -> list[dict[str, Any]]:
    """Run the ESCI suite and append each result as it completes.

    Args:
        client: Client for the provider being evaluated.
        provider: Provider name stored on each row.
        cases: Queries still to run.
        raw_path: JSON Lines file that receives each row.

    Returns:
        The new result rows. Each candidate gets an expected-relevance score
        from its probabilities, used later for ranking. Each row stores the
        model id the response reports, or the configured model id when the
        request failed.
    """
    rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[reranking {index}/{len(cases)}] {provider}: {case['id']}", flush=True)
        try:
            body, latency, attempts = client.request(reranking_request(case))
            candidates = []
            for candidate_index, candidate in enumerate(case["candidates"]):
                answer = body["answers"][f"candidate_{candidate_index}"]
                predicted, probabilities = answer_data(answer)
                score = sum(
                    RELEVANCE[label] * probabilities.get(label, 0.0)
                    for label in RELEVANCE
                )
                if not probabilities and predicted in RELEVANCE:
                    score = float(RELEVANCE[predicted])
                candidates.append(
                    {
                        **candidate,
                        "predicted": predicted,
                        "probabilities": probabilities,
                        "score": score,
                    }
                )
            row = {
                    "provider": provider,
                    "suite": "reranking",
                    "case_id": case["id"],
                    "query": case["query"],
                    "candidates": candidates,
                    "latency_ms": latency,
                    "attempts": attempts,
                    "model": body.get("model"),
                    "error": None,
                }
        except Exception as exc:
            row = {
                    "provider": provider,
                    "suite": "reranking",
                    "case_id": case["id"],
                    "query": case["query"],
                    "candidates": [],
                    "latency_ms": None,
                    "attempts": MAX_ATTEMPTS,
                    "model": client.config["model"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
        append_jsonl(raw_path, row)
        rows.append(row)
    return rows


def dcg(relevances: list[int]) -> float:
    """Compute discounted cumulative gain with exponential gains.

    Args:
        relevances: Graded relevance values in ranked order.

    Returns:
        The DCG, using gain ``2**relevance - 1`` (7/3/1/0 for ESCI's
        Exact/Substitute/Complement/Irrelevant grades 3/2/1/0) and a
        ``log2(rank + 1)`` discount.
    """
    return sum(
        (2**relevance - 1) / math.log2(index + 2)
        for index, relevance in enumerate(relevances)
    )


def ndcg(candidates: list[dict[str, Any]]) -> float:
    """Compute nDCG for one ranked query.

    Args:
        candidates: Candidates carrying a predicted ``score`` and the gold
            ESCI ``label``. Ties keep the saved candidate order.

    Returns:
        nDCG over all candidates ranked by descending score, or 0.0 when no
        candidate is relevant.
    """
    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
    values = [RELEVANCE[item["label"].lower()] for item in ranked]
    ideal = dcg(sorted(values, reverse=True))
    return dcg(values) / ideal if ideal else 0.0


def summarize_classification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute metrics for one provider on one single-choice suite.

    Args:
        rows: Result rows for a single provider and suite.

    Returns:
        Case counts, accuracy, macro-F1, log loss, Brier score, and latency
        percentiles. Calibration metrics are ``None`` when no row has a
        probability for its gold label.
    """
    expected = [row["expected"] for row in rows]
    predicted = [row["predicted"] for row in rows]
    completed = [row for row in rows if not row["error"]]
    latencies = [row["latency_ms"] for row in completed]
    correct = sum(a == p for a, p in zip(expected, predicted))
    calibrated = [
        row
        for row in completed
        if row["expected"] in row["probabilities"] and row["probabilities"]
    ]
    log_loss = (
        statistics.mean(
            -math.log(max(row["probabilities"][row["expected"]], 1e-12))
            for row in calibrated
        )
        if calibrated
        else None
    )
    brier = (
        statistics.mean(
            sum(
                (
                    probability
                    - (1.0 if label == row["expected"] else 0.0)
                )
                ** 2
                for label, probability in row["probabilities"].items()
            )
            for row in calibrated
        )
        if calibrated
        else None
    )
    return {
        "cases": len(rows),
        "completed": len(completed),
        "accuracy": correct / len(rows),
        "macro_f1": macro_f1(expected, predicted),
        "log_loss": log_loss,
        "brier_score": brier,
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
    }


def summarize_reranking(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute ranking metrics for one provider on ESCI.

    Args:
        rows: Reranking rows for a single provider.

    Returns:
        Case counts, nDCG@10, exact-match MRR, per-candidate label accuracy and
        macro-F1, and latency percentiles.
    """
    ndcgs, reciprocal_ranks = [], []
    expected_labels, predicted_labels = [], []
    completed = [row for row in rows if not row["error"]]
    for row in completed:
        ndcgs.append(ndcg(row["candidates"]))
        ranked = sorted(row["candidates"], key=lambda item: item["score"], reverse=True)
        first_exact = next(
            (index for index, item in enumerate(ranked, start=1) if item["label"] == "Exact"),
            None,
        )
        reciprocal_ranks.append(1 / first_exact if first_exact else 0.0)
        expected_labels.extend(item["label"].lower() for item in row["candidates"])
        predicted_labels.extend(item["predicted"] for item in row["candidates"])
    latencies = [row["latency_ms"] for row in completed]
    return {
        "cases": len(rows),
        "completed": len(completed),
        "ndcg_at_10": statistics.mean(ndcgs) if ndcgs else 0.0,
        "mrr_exact": statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "candidate_accuracy": (
            sum(a == p for a, p in zip(expected_labels, predicted_labels))
            / len(expected_labels)
            if expected_labels
            else 0.0
        ),
        "candidate_macro_f1": (
            macro_f1(expected_labels, predicted_labels) if expected_labels else 0.0
        ),
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
    }


def summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize every provider and suite present in the results.

    Args:
        rows: All saved public-benchmark rows.

    Returns:
        One summary dict per (provider, suite), sorted by provider then suite.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["provider"], row["suite"])].append(row)
    output = []
    for (provider, suite), items in sorted(grouped.items()):
        summary = (
            summarize_reranking(items)
            if suite == "reranking"
            else summarize_classification(items)
        )
        output.append({"provider": provider, "suite": suite, **summary})
    return output


def parse_suites(value: str) -> set[str]:
    """Parse and validate a comma-separated suite list.

    Args:
        value: Comma-separated suite names, for example ``"llm_evals"``.

    Returns:
        The selected suite names.

    Raises:
        SystemExit: If no suite is given or a name is not a known suite.
    """
    suites = {item.strip() for item in value.split(",") if item.strip()}
    if not suites:
        raise SystemExit("No suites selected")
    unknown = suites - set(SUITES)
    if unknown:
        raise SystemExit(
            f"Unknown suites: {', '.join(sorted(unknown))}; "
            f"choose from {', '.join(SUITES)}"
        )
    return suites


def main() -> int:
    """Run or resume the public suites and regenerate the summaries.

    Returns:
        Process exit code 0.

    Raises:
        SystemExit: If no providers or suites are selected, or a suite is
            unknown.
        FileNotFoundError: If a prepared dataset is missing.
    """
    args = parse_args()
    load_dotenv(ROOT / ".env")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_results.jsonl"
    configs = parse_providers(args.providers)
    suites = parse_suites(args.suites)

    triage = read_jsonl(DATA / "inbox_triage.jsonl", required=True)
    # Labels come from the full dataset so --limit smoke runs send the same
    # 77-option question as a full run.
    labels = sorted({case["expected"] for case in triage})
    suite_cases = {
        "inbox_triage": triage,
        "reranking": read_jsonl(DATA / "reranking.jsonl", required=True),
        "llm_evals": read_jsonl(DATA / "llm_evals.jsonl", required=True),
    }
    if args.limit is not None:
        suite_cases = {
            suite: cases[: args.limit] for suite, cases in suite_cases.items()
        }
    builders = {
        "inbox_triage": lambda case: triage_request(case, labels),
        "llm_evals": mmlu_request,
    }

    selected = {
        (config["name"], suite, case["id"])
        for config in configs
        for suite in suites
        for case in suite_cases[suite]
    }
    saved = read_jsonl(raw_path)
    existing = resume_rows(saved, selected, KEY_FIELDS, args.force)
    if len(existing) != len(saved):
        write_jsonl(raw_path, existing)
    completed = {row_key(row, KEY_FIELDS) for row in existing}

    for config in configs:
        pending = {
            suite: [
                case
                for case in suite_cases[suite]
                if (config["name"], suite, case["id"]) not in completed
            ]
            for suite in SUITES
            if suite in suites
        }
        if not any(pending.values()):
            continue
        record_run(args.output_dir, config, ",".join(sorted(suites)))
        client = Client(config)
        try:
            client.warm_up()
            for suite, cases in pending.items():
                if not cases:
                    continue
                if suite == "reranking":
                    new_rows = run_reranking(
                        client, config["name"], cases, raw_path
                    )
                else:
                    new_rows = run_classification(
                        client,
                        config["name"],
                        suite,
                        cases,
                        builders[suite],
                        "expected",
                        raw_path,
                    )
                existing.extend(new_rows)
        finally:
            client.close()

    summary = summaries(existing)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(summary).to_csv(args.output_dir / "summary.csv", index=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
