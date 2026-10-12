"""Shared provider configuration, HTTP client, resume logic, and metrics."""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


PROVIDERS = ("clef-flash", "clm", "jev", "laya")
REQUEST_TIMEOUT_S = 120
MAX_ATTEMPTS = 3
# Must not match any benchmark case, or a server-side cache answers the first
# measured request.
WARMUP_REQUEST = {
    "state": {"message": "Warm-up request; not part of any benchmark."},
    "questions": {
        "warmup": {
            "type": "choice",
            "instructions": "Is this a warm-up request?",
            "criteria": {"yes": "It is a warm-up.", "no": "It is not a warm-up."},
        }
    },
}


def _optional_int(name: str, default: int) -> int | None:
    """Read an optional integer from the environment.

    Args:
        name: Environment variable name.
        default: Value used when the variable is unset.

    Returns:
        The parsed integer, ``default`` when the variable is unset, or ``None``
        when it is set to an empty string.

    Raises:
        ValueError: If the variable is set to a non-integer value.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return int(raw) if raw else None


def provider_config(name: str) -> dict[str, Any]:
    """Build the connection settings for one provider.

    Args:
        name: Provider name; one of ``clef-flash``, ``clm``, ``jev``, or ``laya``
            (case-insensitive).

    Returns:
        A dict with ``name``, ``url``, ``model``, ``key``, and ``extra``, where
        ``extra`` holds vendor-documented request fields merged into every
        payload.

    Raises:
        ValueError: If ``name`` is not a supported provider.
    """
    normalized = name.strip().lower()
    if normalized == "jev":
        return {
            "name": "jev",
            "url": os.getenv("JEV_URL", "https://api.typesafe.ai/v1/systemone"),
            "model": os.getenv("JEV_MODEL", "jev-latest"),
            "key": os.getenv("JEV_API_KEY"),
            "extra": {},
        }
    if normalized == "clef-flash":
        return {
            "name": "clef-flash",
            "url": os.getenv(
                "CLEF_FLASH_URL", "http://127.0.0.1:8710/v1/systemone"
            ),
            "model": os.getenv("CLEF_FLASH_MODEL", "clef-flash"),
            "key": os.getenv("CLEF_FLASH_API_KEY"),
            "extra": {},
        }
    if normalized == "clm":
        return {
            "name": "clm",
            "url": os.getenv("CLM_URL", "http://127.0.0.1:8700/v1/systemone"),
            "model": os.getenv("CLM_MODEL", "clm-latest"),
            "key": os.getenv("CLM_API_KEY"),
            "extra": {},
        }
    if normalized == "laya":
        # Laya maps a `model` of "laya" to the English checkpoint, so it is omitted
        # by default to let the Router choose. At head_max_len=512, Banking77's 77
        # options collapse to 72 distinct spans; 1024 keeps every option distinct.
        extra = {
            key: value
            for key, value in (
                ("max_len", _optional_int("LAYA_MAX_LEN", 2048)),
                ("head_max_len", _optional_int("LAYA_HEAD_MAX_LEN", 1024)),
            )
            if value is not None
        }
        return {
            "name": "laya",
            "url": os.getenv("LAYA_URL", "http://127.0.0.1:8004/v1/systemone"),
            "model": os.getenv("LAYA_MODEL") or None,
            "key": os.getenv("LAYA_API_KEY"),
            "extra": extra,
        }
    raise ValueError(f"Unknown provider: {name}")


def parse_providers(value: str) -> list[dict[str, Any]]:
    """Parse a comma-separated provider list into configurations.

    Args:
        value: Comma-separated provider names, for example ``"jev,laya"``.

    Returns:
        One configuration per provider, in the given order.

    Raises:
        SystemExit: If no providers are given or a provider is repeated.
        ValueError: If a name is not a supported provider.
    """
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise SystemExit("No providers selected")
    configs = [provider_config(name) for name in names]
    if len({config["name"] for config in configs}) != len(configs):
        raise SystemExit("Duplicate providers are not allowed")
    return configs


class Client:
    """Sequential SystemOne client with a shared timeout and retry policy.

    Every provider uses the same client, so request handling, timeouts, retries,
    and latency measurement are identical across the comparison.

    Attributes:
        config: Provider configuration from ``provider_config``.
        client: Underlying ``httpx.Client`` carrying the auth headers.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Create an HTTP client for one provider.

        Args:
            config: Provider configuration from ``provider_config``.
        """
        self.config = config
        headers = {"Content-Type": "application/json"}
        if config["key"]:
            headers["Authorization"] = f"Bearer {config['key']}"
        self.client = httpx.Client(headers=headers, timeout=REQUEST_TIMEOUT_S)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self.client.close()

    def payload(self, request: dict[str, Any]) -> dict[str, Any]:
        """Add the provider's model id and extra fields to a request.

        Args:
            request: Provider-neutral SystemOne request with ``state`` and
                ``questions``.

        Returns:
            A new dict ready to POST; ``request`` is not modified.
        """
        payload = dict(request)
        if self.config["model"]:
            payload["model"] = self.config["model"]
        payload.update(self.config["extra"])
        return payload

    def warm_up(self) -> None:
        """Send one request that is not part of any benchmark dataset.

        Raises:
            RuntimeError: If every attempt fails.
        """
        self.request(WARMUP_REQUEST)

    def request(self, request: dict[str, Any]) -> tuple[dict[str, Any], float, int]:
        """POST a SystemOne request, retrying failed attempts.

        Args:
            request: Provider-neutral SystemOne request with ``state`` and
                ``questions``.

        Returns:
            A tuple of the decoded response body, the latency of the successful
            attempt in milliseconds, and the number of attempts made.

        Raises:
            RuntimeError: If all ``MAX_ATTEMPTS`` attempts fail.
        """
        payload = self.payload(request)
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            started = time.perf_counter()
            try:
                response = self.client.post(str(self.config["url"]), json=payload)
                response.raise_for_status()
                body = response.json()
                return body, (time.perf_counter() - started) * 1000, attempt + 1
            except (httpx.HTTPError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"{type(last_error).__name__}: {last_error}")


def _health_url(url: str) -> str:
    """Derive a server's health URL from its SystemOne URL.

    Args:
        url: SystemOne endpoint, for example
            ``http://127.0.0.1:8004/v1/systemone``.

    Returns:
        The ``/health`` URL on the same host.
    """
    return url.split("/v1/", 1)[0] + "/health"


def _gpu_snapshot() -> list[str]:
    """Read GPU memory usage from ``nvidia-smi``.

    Returns:
        One CSV line per GPU (index, name, used and total memory), or an empty
        list when ``nvidia-smi`` is unavailable.
    """
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def record_run(output_dir: Path, config: dict[str, Any], suite: str) -> None:
    """Append provider settings and environment details for one run.

    Local servers also have their ``/health`` response recorded; the hosted Jev
    API is not probed.

    Args:
        output_dir: Results directory that holds ``run_manifest.jsonl``.
        config: Provider configuration from ``provider_config``.
        suite: Suite name or comma-separated suite names covered by the run.
    """
    health: Any = None
    if not config["url"].startswith("https://api.typesafe.ai"):
        try:
            health = httpx.get(_health_url(config["url"]), timeout=5).json()
        except (httpx.HTTPError, ValueError) as exc:
            health = f"{type(exc).__name__}: {exc}"
    append_jsonl(
        output_dir / "run_manifest.jsonl",
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "suite": suite,
            "provider": config["name"],
            "url": config["url"],
            "model": config["model"],
            "extra_request_fields": config["extra"],
            "timeout_s": REQUEST_TIMEOUT_S,
            "max_attempts": MAX_ATTEMPTS,
            "health": health,
            "gpus": _gpu_snapshot(),
            "host": platform.node(),
        },
    )


def read_jsonl(path: Path, required: bool = False) -> list[dict[str, Any]]:
    """Read a JSON Lines file.

    Args:
        path: File to read.
        required: Raise instead of returning an empty list when the file is
            missing. Use it for inputs such as datasets.

    Returns:
        One dict per non-blank line, or an empty list if the file is missing
        and not ``required``.

    Raises:
        FileNotFoundError: If ``required`` is set and the file is missing.
    """
    if not required and not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a JSON Lines file, replacing its contents atomically.

    The rows go to a temporary file that then replaces ``path``, so an
    interrupted write leaves the previous contents intact.

    Args:
        path: File to write; parent directories are created.
        rows: Rows to serialize, one per line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one row to a JSON Lines file.

    Args:
        path: File to append to; parent directories are created.
        row: Row to serialize as a single line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a CSV file using the first row's keys as the header.

    Args:
        path: File to write.
        rows: Rows to write; an empty list produces an empty file.
    """
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    """Build the resume key of a saved result row.

    Args:
        row: Saved result row.
        fields: Row fields that identify one provider/case result, for example
            ``("provider", "case_id")``.

    Returns:
        The values of ``fields`` in order.
    """
    return tuple(row[field] for field in fields)


def resume_rows(
    saved: list[dict[str, Any]],
    selected: set[tuple[Any, ...]],
    fields: tuple[str, ...],
    force: bool,
) -> list[dict[str, Any]]:
    """Choose which saved rows a resumed run keeps.

    A selected row is dropped, and therefore rerun, when ``force`` is set or
    when it recorded an error. Rows outside the selection are always kept, so
    ``--force --limit`` replaces only the limited cases.

    Args:
        saved: Rows already in ``raw_results.jsonl``.
        selected: Keys, built like ``row_key``, of the results this run covers.
        fields: Row fields that make up a key.
        force: Whether to rerun every selected row.

    Returns:
        A new list of the rows to keep, in their saved order.
    """
    return [
        row
        for row in saved
        if row_key(row, fields) not in selected or not (force or row.get("error"))
    ]


def percentile(values: list[float], quantile: float) -> float | None:
    """Compute a percentile with linear interpolation.

    Args:
        values: Sample values.
        quantile: Quantile between 0 and 1, for example ``0.95``.

    Returns:
        The interpolated percentile, or ``None`` if ``values`` is empty.
    """
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def macro_f1(
    expected: Sequence[str],
    predicted: Sequence[str | None],
    labels: Sequence[str] | None = None,
) -> float:
    """Compute macro-averaged F1.

    Args:
        expected: Gold labels.
        predicted: Predicted labels; ``None`` counts as a miss.
        labels: Labels to average over; defaults to every label in
            ``expected``.

    Returns:
        The unweighted mean F1 across the averaged labels.
    """
    averaged = sorted(set(expected)) if labels is None else labels
    scores = []
    for label in averaged:
        tp = sum(a == label and p == label for a, p in zip(expected, predicted))
        fp = sum(a != label and p == label for a, p in zip(expected, predicted))
        fn = sum(a == label and p != label for a, p in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return statistics.mean(scores)


def calibration_metrics(
    items: list[tuple[dict[str, float], str, str, Sequence[str]]],
) -> dict[str, float | None]:
    """Compute log loss, Brier score, and ten-bin ECE.

    Args:
        items: One ``(probabilities, expected, predicted, options)`` tuple per
            case whose probabilities cover every option. Brier terms are summed
            in ``options`` order, so pass a fixed order for reproducible
            floating-point results.

    Returns:
        ``log_loss``, ``brier_score``, and ``ece``; each is ``None`` when
        ``items`` is empty.
    """
    if not items:
        return {"log_loss": None, "brier_score": None, "ece": None}
    log_loss = statistics.mean(
        -math.log(max(min(probabilities[expected], 1.0), 1e-12))
        for probabilities, expected, _, _ in items
    )
    brier = statistics.mean(
        sum(
            (probabilities[option] - (1.0 if option == expected else 0.0)) ** 2
            for option in options
        )
        for probabilities, expected, _, options in items
    )
    bins: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for probabilities, expected, predicted, _ in items:
        confidence = max(probabilities.values())
        bins[min(int(confidence * 10), 9)].append(
            (confidence, float(predicted == expected))
        )
    ece = sum(
        len(group)
        / len(items)
        * abs(
            statistics.mean(conf for conf, _ in group)
            - statistics.mean(acc for _, acc in group)
        )
        for group in bins.values()
    )
    return {"log_loss": log_loss, "brier_score": brier, "ece": ece}


def mcnemar(
    left: dict[str, bool],
    right: dict[str, bool],
    left_name: str = "left",
    right_name: str = "right",
) -> dict[str, Any]:
    """Run an exact two-sided McNemar test on paired per-case correctness.

    Args:
        left: Case id to correctness for the first provider.
        right: Case id to correctness for the second provider.
        left_name: Label stored for the first provider.
        right_name: Label stored for the second provider.

    Returns:
        The shared case count, the four agreement/disagreement counts, and the
        exact two-sided p-value. Only case ids present in both inputs are used.
    """
    shared = sorted(set(left) & set(right))
    both_correct = sum(left[key] and right[key] for key in shared)
    left_only = sum(left[key] and not right[key] for key in shared)
    right_only = sum(not left[key] and right[key] for key in shared)
    both_wrong = sum(not left[key] and not right[key] for key in shared)
    discordant = left_only + right_only
    if discordant:
        lower = min(left_only, right_only)
        p_value = min(
            1.0,
            2
            * sum(math.comb(discordant, value) for value in range(lower + 1))
            / 2**discordant,
        )
    else:
        p_value = 1.0
    return {
        "left": left_name,
        "right": right_name,
        "shared_cases": len(shared),
        "both_correct": both_correct,
        "left_correct_right_wrong": left_only,
        "left_wrong_right_correct": right_only,
        "both_wrong": both_wrong,
        "p_exact_two_sided": p_value,
    }


def holm(p_values: list[float]) -> list[float]:
    """Apply the Holm-Bonferroni correction.

    Args:
        p_values: Unadjusted p-values for one family of tests.

    Returns:
        Adjusted p-values in the same order as ``p_values``.
    """
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def all_pairs_mcnemar(
    correctness: dict[str, dict[str, bool]],
) -> list[dict[str, Any]]:
    """Run McNemar for every provider pair with Holm correction.

    Args:
        correctness: Provider name to a mapping of case id to correctness.

    Returns:
        One ``mcnemar`` result per provider pair in sorted order, each with an
        added ``p_holm`` adjusted across all pairs.
    """
    providers = sorted(correctness)
    rows = [
        mcnemar(correctness[left], correctness[right], left, right)
        for index, left in enumerate(providers)
        for right in providers[index + 1 :]
    ]
    for row, adjusted in zip(rows, holm([row["p_exact_two_sided"] for row in rows])):
        row["p_holm"] = adjusted
    return rows
