#!/usr/bin/env python3
"""Download and deterministically sample three public decision benchmarks."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "public_benchmarks"
CACHE = OUTPUT / "source_cache"
SEED = 20261009
MMLU_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "business_ethics",
    "college_computer_science",
    "conceptual_physics",
    "high_school_psychology",
    "international_law",
    "machine_learning",
    "professional_medicine",
    "world_religions",
)


def parquet_urls(dataset: str, config: str, split: str) -> list[str]:
    """List the Hugging Face parquet files for one dataset split.

    Args:
        dataset: Hugging Face dataset id, for example ``mteb/banking77``.
        config: Dataset configuration name.
        split: Split name, for example ``test``.

    Returns:
        Download URLs for the matching parquet files.

    Raises:
        requests.HTTPError: If the datasets-server request fails.
    """
    response = requests.get(
        "https://datasets-server.huggingface.co/parquet",
        params={"dataset": dataset},
        timeout=60,
    )
    response.raise_for_status()
    return [
        item["url"]
        for item in response.json()["parquet_files"]
        if item["config"] == config and item["split"] == split
    ]


def download(url: str, name: str) -> Path:
    """Download a file into the source cache unless it is already there.

    Args:
        url: Source URL.
        name: File name inside the cache directory.

    Returns:
        Path to the cached file.

    Raises:
        requests.HTTPError: If the download fails.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / name
    if path.exists():
        return path
    temporary = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(path)
    return path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a JSON Lines file, replacing its contents.

    Args:
        path: File to write.
        rows: Rows to serialize, one per line.
    """
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_record(path: Path, source: str) -> dict[str, Any]:
    """Describe a cached source file for the manifest.

    Args:
        path: Cached file.
        source: Dataset id the file came from.

    Returns:
        The source id, repository-relative path, size in bytes, and SHA-256.
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "source": source,
        "cached_file": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }


def prepare_triage(rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample the Banking77 inbox-triage cases.

    Args:
        rng: Seeded random generator shared across all preparation steps.

    Returns:
        The shuffled cases (two test messages per intent) and their manifest
        entry.
    """
    dataset = "mteb/banking77"
    url = parquet_urls(dataset, "default", "test")[0]
    source = download(url, "banking77-test.parquet")
    frame = pd.read_parquet(source)
    labels = sorted(frame["label_text"].unique())
    rows: list[dict[str, Any]] = []
    for label in labels:
        group = frame[frame["label_text"] == label].to_dict("records")
        for index, item in enumerate(rng.sample(group, 2), start=1):
            rows.append(
                {
                    "id": f"banking77-{label}-{index}",
                    "text": item["text"],
                    "expected": label,
                }
            )
    rng.shuffle(rows)
    return rows, {
        **file_record(source, dataset),
        "cases": len(rows),
        "classes": len(labels),
        "sampling": "two test examples per label",
    }


def prepare_reranking(
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample the Amazon ESCI reranking cases.

    Args:
        rng: Seeded random generator shared across all preparation steps.

    Returns:
        Fifty US queries with ten products each, including at least one
        ``Exact`` and one ``Irrelevant`` product, and their manifest entry.
    """
    dataset = "tasksource/esci"
    url = parquet_urls(dataset, "default", "test")[0]
    source = download(url, "esci-test-shard-0000.parquet")
    frame = pd.read_parquet(
        source,
        columns=[
            "query_id",
            "query",
            "product_id",
            "product_locale",
            "esci_label",
            "product_title",
        ],
    )
    frame = frame[
        (frame["product_locale"] == "us")
        & frame["product_title"].notna()
        & frame["esci_label"].isin(["Exact", "Substitute", "Complement", "Irrelevant"])
    ]
    eligible = []
    for query_id, group in frame.groupby("query_id", sort=True):
        labels = set(group["esci_label"])
        if len(group) >= 10 and "Exact" in labels and "Irrelevant" in labels:
            eligible.append((query_id, group))
    selected = rng.sample(eligible, 50)
    rows: list[dict[str, Any]] = []
    for query_id, group in selected:
        records = group.to_dict("records")
        exact = rng.choice([item for item in records if item["esci_label"] == "Exact"])
        irrelevant = rng.choice(
            [item for item in records if item["esci_label"] == "Irrelevant"]
        )
        remaining = [
            item
            for item in records
            if item["product_id"] not in {exact["product_id"], irrelevant["product_id"]}
        ]
        candidates = [exact, irrelevant, *rng.sample(remaining, 8)]
        rng.shuffle(candidates)
        rows.append(
            {
                "id": f"esci-{query_id}",
                "query": str(group.iloc[0]["query"]),
                "candidates": [
                    {
                        "id": item["product_id"],
                        "title": item["product_title"],
                        "label": item["esci_label"],
                    }
                    for item in candidates
                ],
            }
        )
    rows.sort(key=lambda item: item["id"])
    return rows, {
        **file_record(source, dataset),
        "cases": len(rows),
        "candidates_per_case": 10,
        "eligible_queries_in_shard": len(eligible),
        "sampling": (
            "50 US queries with >=10 products and both Exact and Irrelevant; "
            "10 products per query"
        ),
    }


def prepare_mmlu(rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample the MMLU multiple-choice cases.

    Args:
        rng: Seeded random generator shared across all preparation steps.

    Returns:
        The shuffled cases (20 questions from each fixed subject) and their
        manifest entry.
    """
    dataset = "cais/mmlu"
    url = parquet_urls(dataset, "all", "test")[0]
    source = download(url, "mmlu-all-test.parquet")
    frame = pd.read_parquet(source)
    rows: list[dict[str, Any]] = []
    for subject in MMLU_SUBJECTS:
        group = frame[frame["subject"] == subject].to_dict("records")
        for index, item in enumerate(rng.sample(group, 20), start=1):
            answer = int(item["answer"])
            rows.append(
                {
                    "id": f"mmlu-{subject}-{index:02d}",
                    "subject": subject,
                    "question": item["question"],
                    "choices": list(item["choices"]),
                    "expected": "ABCD"[answer],
                }
            )
    rng.shuffle(rows)
    counts = Counter(item["subject"] for item in rows)
    return rows, {
        **file_record(source, dataset),
        "cases": len(rows),
        "subjects": dict(sorted(counts.items())),
        "sampling": "20 test questions from each of 10 fixed subjects",
    }


def main() -> int:
    """Download all sources and write the prepared datasets and manifest.

    Returns:
        Process exit code 0.
    """
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    triage, triage_meta = prepare_triage(rng)
    reranking, reranking_meta = prepare_reranking(rng)
    mmlu, mmlu_meta = prepare_mmlu(rng)
    write_jsonl(OUTPUT / "inbox_triage.jsonl", triage)
    write_jsonl(OUTPUT / "reranking.jsonl", reranking)
    write_jsonl(OUTPUT / "llm_evals.jsonl", mmlu)
    manifest = {
        "seed": SEED,
        "inbox_triage": triage_meta,
        "reranking": reranking_meta,
        "llm_evals": mmlu_meta,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
