#!/usr/bin/env python3
"""Produce paired statistical comparisons for the public benchmark run."""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import PROVIDERS, holm, mcnemar, read_jsonl  # noqa: E402
from run_public_benchmarks import RELEVANCE, dcg, ndcg  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "public_benchmarks"
DATA = ROOT / "data" / "public_benchmarks"


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute a Wilson score confidence interval for a proportion.

    Args:
        successes: Number of successes.
        total: Number of trials; must be positive.
        z: Standard-normal quantile; 1.96 gives a 95% interval.

    Returns:
        The lower and upper interval bounds.
    """
    proportion = successes / total
    denominator = 1 + z * z / total
    center = proportion + z * z / (2 * total)
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return (center - margin) / denominator, (center + margin) / denominator


def bootstrap_difference(
    left: list[float], right: list[float], seed: int = 20261009
) -> dict[str, float]:
    """Bootstrap the paired mean difference between two score lists.

    Args:
        left: Per-query scores for the first provider.
        right: Per-query scores for the second provider, aligned with ``left``.
        seed: Random seed for the 20,000 resamples.

    Returns:
        The observed mean difference (left minus right), the 95% percentile
        interval, and a two-sided bootstrap p-value.
    """
    rng = random.Random(seed)
    differences = []
    for _ in range(20_000):
        indices = [rng.randrange(len(left)) for _ in left]
        differences.append(
            sum(left[index] - right[index] for index in indices) / len(indices)
        )
    differences.sort()
    return {
        "mean_difference": sum(a - b for a, b in zip(left, right)) / len(left),
        "ci95_low": differences[499],
        "ci95_high": differences[19_499],
        "bootstrap_two_sided_p": min(
            1.0,
            2
            * min(
                sum(value <= 0 for value in differences) / len(differences),
                sum(value >= 0 for value in differences) / len(differences),
            ),
        ),
    }


def binomial_upper_tail(successes: int, total: int, probability: float) -> float:
    """Compute the upper tail of a binomial distribution.

    Args:
        successes: Observed number of successes.
        total: Number of trials.
        probability: Success probability under the null hypothesis.

    Returns:
        The probability of observing ``successes`` or more.
    """
    return sum(
        math.comb(total, value)
        * probability**value
        * (1 - probability) ** (total - value)
        for value in range(successes, total + 1)
    )


def random_reranking_baseline() -> dict[str, float]:
    """Simulate nDCG@10 and MRR for random product orderings.

    Returns:
        Mean nDCG@10 and exact-match MRR over 20,000 random permutations of each
        query's candidates, and the number of simulations.
    """
    cases = read_jsonl(DATA / "reranking.jsonl", required=True)
    rng = random.Random(20261009)
    ndcgs, reciprocal_ranks = [], []
    for _ in range(20_000):
        per_query_ndcg, per_query_rr = [], []
        for case in cases:
            values = [
                RELEVANCE[candidate["label"].lower()]
                for candidate in case["candidates"]
            ]
            rng.shuffle(values)
            ideal = dcg(sorted(values, reverse=True))
            per_query_ndcg.append(dcg(values) / ideal if ideal else 0.0)
            first_exact = next(
                (
                    index
                    for index, value in enumerate(values, start=1)
                    if value == RELEVANCE["exact"]
                ),
                None,
            )
            per_query_rr.append(1 / first_exact if first_exact else 0.0)
        ndcgs.append(sum(per_query_ndcg) / len(per_query_ndcg))
        reciprocal_ranks.append(sum(per_query_rr) / len(per_query_rr))
    return {
        "ndcg_at_10": sum(ndcgs) / len(ndcgs),
        "mrr_exact": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "simulations": len(ndcgs),
    }


def main() -> int:
    """Write confidence intervals, chance tests, and paired comparisons.

    Reads ``results/public_benchmarks/raw_results.jsonl`` and writes
    ``analysis.json`` plus the CSV tables next to it.

    Returns:
        Process exit code 0.

    Raises:
        FileNotFoundError: If the raw results or ESCI dataset are missing.
        ValueError: If the raw results contain an unsupported provider.
    """
    rows = read_jsonl(RESULTS / "raw_results.jsonl", required=True)
    providers = sorted({row["provider"] for row in rows})
    unsupported = set(providers) - set(PROVIDERS)
    if unsupported:
        raise ValueError(
            "raw_results.jsonl contains unsupported providers: "
            + ", ".join(sorted(unsupported))
        )
    classification = [
        row for row in rows if row["suite"] in {"inbox_triage", "llm_evals"}
    ]

    intervals = []
    correctness: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for provider in providers:
        for suite in ("inbox_triage", "llm_evals"):
            selected = [
                row
                for row in classification
                if row["provider"] == provider and row["suite"] == suite
            ]
            successes = sum(row["expected"] == row["predicted"] for row in selected)
            low, high = wilson(successes, len(selected))
            intervals.append(
                {
                    "provider": provider,
                    "suite": suite,
                    "correct": successes,
                    "cases": len(selected),
                    "accuracy": successes / len(selected),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
            correctness[(provider, suite)] = {
                row["case_id"]: row["expected"] == row["predicted"]
                for row in selected
            }

    pairs = []
    for suite in ("inbox_triage", "llm_evals"):
        for left_index, left in enumerate(providers):
            for right in providers[left_index + 1 :]:
                pairs.append(
                    {
                        "suite": suite,
                        **mcnemar(
                            correctness[(left, suite)],
                            correctness[(right, suite)],
                            left,
                            right,
                        ),
                    }
                )
    for suite in ("inbox_triage", "llm_evals"):
        family = [pair for pair in pairs if pair["suite"] == suite]
        for pair, adjusted in zip(
            family, holm([pair["p_exact_two_sided"] for pair in family])
        ):
            pair["p_holm"] = adjusted

    subjects = []
    for provider in providers:
        selected = [
            row
            for row in classification
            if row["provider"] == provider and row["suite"] == "llm_evals"
        ]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            grouped[row["subject"]].append(row)
        for subject, items in sorted(grouped.items()):
            subjects.append(
                {
                    "provider": provider,
                    "subject": subject,
                    "accuracy": sum(
                        row["expected"] == row["predicted"] for row in items
                    )
                    / len(items),
                    "cases": len(items),
                }
            )

    rerank_by_provider: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["suite"] == "reranking":
            rerank_by_provider[row["provider"]][row["case_id"]] = ndcg(
                row["candidates"]
            )
    rerank_pairs = []
    for left_index, left in enumerate(providers):
        for right in providers[left_index + 1 :]:
            case_ids = sorted(
                set(rerank_by_provider[left]) & set(rerank_by_provider[right])
            )
            rerank_pairs.append(
                {
                    "left": left,
                    "right": right,
                    **bootstrap_difference(
                        [rerank_by_provider[left][key] for key in case_ids],
                        [rerank_by_provider[right][key] for key in case_ids],
                    ),
                }
            )
    for pair, adjusted in zip(
        rerank_pairs, holm([pair["bootstrap_two_sided_p"] for pair in rerank_pairs])
    ):
        pair["p_holm"] = adjusted

    chance_tests = []
    for item in intervals:
        probability = 1 / 77 if item["suite"] == "inbox_triage" else 0.25
        chance_tests.append(
            {
                "provider": item["provider"],
                "suite": item["suite"],
                "chance_accuracy": probability,
                "one_sided_binomial_p": binomial_upper_tail(
                    item["correct"], item["cases"], probability
                ),
            }
        )

    analysis = {
        "baselines": {
            "inbox_triage_accuracy": 1 / 77,
            "llm_evals_accuracy": 0.25,
            "reranking_random_permutation": random_reranking_baseline(),
        },
        "accuracy_intervals": intervals,
        "classification_vs_chance": chance_tests,
        "classification_pairwise_mcnemar": pairs,
        "reranking_pairwise_bootstrap": rerank_pairs,
    }
    (RESULTS / "analysis.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    pd.DataFrame(intervals).to_csv(RESULTS / "accuracy_intervals.csv", index=False)
    pd.DataFrame(pairs).to_csv(RESULTS / "classification_pairwise.csv", index=False)
    pd.DataFrame(subjects).to_csv(RESULTS / "mmlu_by_subject.csv", index=False)
    pd.DataFrame(rerank_pairs).to_csv(RESULTS / "reranking_pairwise.csv", index=False)
    pd.DataFrame(chance_tests).to_csv(
        RESULTS / "classification_vs_chance.csv", index=False
    )
    print(json.dumps(analysis, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
