#!/usr/bin/env python3
"""Download and normalize the official BFCL v3 multiple split."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "tool_selection"
CACHE = OUTPUT / "source_cache"
INPUT_URL = (
    "https://huggingface.co/datasets/gorilla-llm/"
    "Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_multiple.json"
)
ANSWER_URL = (
    "https://huggingface.co/datasets/gorilla-llm/"
    "Berkeley-Function-Calling-Leaderboard/resolve/main/"
    "possible_answer/BFCL_v3_multiple.json"
)


def download(url: str, filename: str) -> Path:
    """Download a file into the source cache unless it is already there.

    Args:
        url: Source URL.
        filename: File name inside the cache directory.

    Returns:
        Path to the cached file.

    Raises:
        requests.HTTPError: If the download fails.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / filename
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file.

    Args:
        path: File to read.

    Returns:
        One dict per non-blank line.
    """
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten_messages(value: Any) -> list[dict[str, str]]:
    """Flatten BFCL's nested message lists into role/content pairs.

    Args:
        value: A message dict, or an arbitrarily nested list of them.

    Returns:
        Messages in order, with non-string content serialized as JSON.
    """
    messages: list[dict[str, str]] = []
    if isinstance(value, list):
        for item in value:
            messages.extend(flatten_messages(item))
    elif isinstance(value, dict) and "content" in value:
        content = value["content"]
        messages.append(
            {
                "role": str(value.get("role", "user")),
                "content": (
                    content
                    if isinstance(content, str)
                    else json.dumps(content, ensure_ascii=False, sort_keys=True)
                ),
            }
        )
    return messages


def normalized_case(
    input_row: dict[str, Any], answer_row: dict[str, Any]
) -> dict[str, Any]:
    """Combine one BFCL input row and its answer into a benchmark case.

    Args:
        input_row: Official ``BFCL_v3_multiple`` input row.
        answer_row: Matching ``possible_answer`` row.

    Returns:
        The case id, flattened conversation text, candidate tools with their
        descriptions and parameter schemas, and the expected function name.

    Raises:
        ValueError: If the rows are unaligned, the answer is not exactly one
            function call, the conversation is empty, candidate names repeat,
            or the expected function is not a candidate.
    """
    if input_row["id"] != answer_row["id"]:
        raise ValueError(
            f"unaligned IDs: {input_row['id']!r} != {answer_row['id']!r}"
        )
    ground_truth = answer_row.get("ground_truth")
    if not isinstance(ground_truth, list) or len(ground_truth) != 1:
        raise ValueError(f"{input_row['id']}: expected one ground-truth call")
    if not isinstance(ground_truth[0], dict) or len(ground_truth[0]) != 1:
        raise ValueError(f"{input_row['id']}: expected one ground-truth function")
    expected_tool = next(iter(ground_truth[0]))

    messages = flatten_messages(input_row.get("question"))
    if not messages:
        raise ValueError(f"{input_row['id']}: no conversation messages")
    conversation = "\n".join(
        f"{message['role'].upper()}: {message['content']}" for message in messages
    )
    tools = [
        {
            "name": str(tool["name"]),
            "description": str(tool.get("description") or ""),
            "parameter_schema": tool.get("parameters") or {},
        }
        for tool in input_row.get("function", [])
    ]
    names = [tool["name"] for tool in tools]
    if len(names) != len(set(names)):
        raise ValueError(f"{input_row['id']}: duplicate candidate tool names")
    if expected_tool not in names:
        raise ValueError(
            f"{input_row['id']}: expected tool {expected_tool!r} is not a candidate"
        )
    return {
        "id": input_row["id"],
        "conversation": conversation,
        "candidate_tools": tools,
        "expected_tool": expected_tool,
    }


def source_record(path: Path, url: str) -> dict[str, Any]:
    """Describe a downloaded source file for the manifest.

    Args:
        path: Cached file.
        url: URL the file was downloaded from.

    Returns:
        The URL, SHA-256, and size in bytes.
    """
    return {
        "url": url,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def main() -> int:
    """Download BFCL v3 multiple and write the normalized cases and manifest.

    Returns:
        Process exit code 0.

    Raises:
        ValueError: If the official files do not contain 200 aligned rows.
    """
    input_path = download(INPUT_URL, "BFCL_v3_multiple.json")
    answer_path = download(ANSWER_URL, "BFCL_v3_multiple_answers.json")
    input_rows = read_jsonl(input_path)
    answer_rows = read_jsonl(answer_path)
    if len(input_rows) != 200 or len(answer_rows) != 200:
        raise ValueError(
            f"expected 200 aligned rows, got {len(input_rows)} and {len(answer_rows)}"
        )
    input_ids = [row["id"] for row in input_rows]
    answer_ids = [row["id"] for row in answer_rows]
    if input_ids != answer_ids:
        raise ValueError("official input and answer IDs are not aligned in order")

    cases = [
        normalized_case(input_row, answer_row)
        for input_row, answer_row in zip(input_rows, answer_rows)
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT / "bfcl_v3_multiple.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    manifest = {
        "benchmark": "BFCL v3 multiple",
        "case_count": len(cases),
        "sources": {
            "inputs": source_record(input_path, INPUT_URL),
            "possible_answers": source_record(answer_path, ANSWER_URL),
        },
        "attribution": {
            "name": "Berkeley Function Calling Leaderboard (BFCL)",
            "creator": "Gorilla LLM / UC Berkeley",
            "dataset_repository": (
                "https://huggingface.co/datasets/gorilla-llm/"
                "Berkeley-Function-Calling-Leaderboard"
            ),
            "project": "https://gorilla.cs.berkeley.edu/leaderboard.html",
        },
        "normalization": (
            "Aligned BFCL_v3_multiple inputs and possible answers by ordered ID; "
            "flattened nested conversation messages; retained candidate names, "
            "descriptions, parameter schemas, and expected function name."
        ),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
