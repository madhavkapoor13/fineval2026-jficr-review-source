#!/usr/bin/env python3
"""Reproducible Maddies NLP inference workflow for FinNLP 2026 JF-ICR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from prompts import REPAIR_PROMPT, STAGE1_PROMPT, STAGE2_PROMPT

VALID_LABELS = {"+2", "+1", "0", "-1", "-2"}
RUN_ID = "maddies_nlp_gemma_tagged_label_first"
PROMPT_RE = re.compile(
    r"Financial Question:\s*(.*?)\nCompany Response:\s*(.*?)\n\nDirectly output",
    re.S,
)
PROMPT_FALLBACK_RE = re.compile(
    r"Financial Question:\s*(.*?)\nCompany Response:\s*(.*?)(?:\n\n|$)",
    re.S,
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_original_query(query: str) -> tuple[str, str]:
    match = PROMPT_RE.search(query)
    if match is None:
        match = PROMPT_FALLBACK_RE.search(query)
    if match is None:
        raise ValueError("Could not parse Financial Question / Company Response")
    return match.group(1).strip(), match.group(2).strip()


def read_input(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".jsonl", ".ndjson"}:
        frame = pd.read_json(path, lines=True)
    else:
        raise ValueError("Input must be Parquet, CSV, or JSONL")

    required_direct = {"id", "financial_question", "company_response"}
    if required_direct.issubset(frame.columns):
        parsed = frame[["id", "financial_question", "company_response"]].copy()
    elif {"id", "query"}.issubset(frame.columns):
        questions: list[str] = []
        responses: list[str] = []
        for query in frame["query"].astype(str):
            question, response = parse_original_query(query)
            questions.append(question)
            responses.append(response)
        parsed = pd.DataFrame(
            {
                "id": frame["id"],
                "financial_question": questions,
                "company_response": responses,
            }
        )
    else:
        raise ValueError(
            "Input needs id/query or id/financial_question/company_response columns"
        )

    if parsed.empty:
        raise ValueError("Input contains no examples")
    if parsed["id"].duplicated().any():
        raise ValueError("Input IDs must be unique")
    if parsed[["financial_question", "company_response"]].isna().any().any():
        raise ValueError("Question and response must not be missing")
    parsed["financial_question"] = parsed["financial_question"].astype(str)
    parsed["company_response"] = parsed["company_response"].astype(str)
    if (parsed["financial_question"].str.strip() == "").any() or (
        parsed["company_response"].str.strip() == ""
    ).any():
        raise ValueError("Question and response must not be empty")
    return parsed.sort_values("id").to_dict("records")


def response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content)


def usage_dict(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def request_hash(value: dict[str, Any]) -> str:
    stable = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return sha256_bytes(stable.encode("utf-8"))


def stage1_ok(text: str) -> bool:
    return "<STAGE1_ANALYSIS" in (text or "") and "<GLOBAL>" in (text or "")


def extract_stage2_label(text: str) -> tuple[str | None, str]:
    if not text:
        return None, "empty"
    strict = re.search(
        r"<LABEL>\s*(\+2|\+1|0|-1|-2)\s*</LABEL>", text, flags=re.I
    )
    if strict:
        return strict.group(1), "strict_label_tag"
    patterns = [
        r"<LABEL>\s*(\+2|\+1|0|-1|-2)",
        r"\bFINAL\s*LABEL\s*[:=]\s*(\+2|\+1|0|-1|-2)",
        r"\bLABEL\s*[:=]\s*(\+2|\+1|0|-1|-2)",
        r"\bFINAL\s*ANSWER\s*[:=]\s*(\+2|\+1|0|-1|-2)",
    ]
    for pattern in patterns:
        near_miss = re.search(pattern, text, flags=re.I)
        if near_miss:
            return near_miss.group(1), "near_miss_label_marker"
    return None, "invalid"


def openrouter_call(
    url: str, payload: dict[str, Any], api_key: str
) -> tuple[dict[str, Any], float]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "Maddies NLP FinNLP 2026 JF-ICR",
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    start = time.time()
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    return json.loads(body), time.time() - start


class Budget:
    def __init__(self, cap: float) -> None:
        self.cap = cap
        self.reserved = 0.0
        self.lock = threading.Lock()

    def reserve(self, amount: float, cached: bool) -> None:
        if cached:
            return
        with self.lock:
            if self.reserved + amount > self.cap:
                raise RuntimeError(f"Estimated spend would exceed ${self.cap:.2f}")
            self.reserved += amount


def call_cached(
    *,
    root: Path,
    config: dict[str, Any],
    budget: Budget,
    item_id: Any,
    stage: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    api_key: str,
    refresh: bool,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": config["model"],
        "messages": messages,
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "max_tokens": max_tokens,
        "seed": config["seed"],
        "provider": config["provider"],
    }
    key = request_hash(
        {"run_id": RUN_ID, "item_id": item_id, "stage": stage, "payload": payload}
    )
    cache_path = root / "cache" / f"{key}.json"
    cached = cache_path.exists() and not refresh
    estimated_input_tokens = max(
        1, sum(len(message.get("content", "")) for message in messages) // 3
    )
    conservative_cost = estimated_input_tokens / 1_000_000 + max_tokens * 6 / 1_000_000
    budget.reserve(conservative_cost, cached)
    if cached:
        saved = load_json(cache_path)
        response = saved["response"]
        latency = float(saved.get("latency_seconds", 0.0))
    else:
        response, latency = openrouter_call(
            config["openrouter_url"], payload, api_key
        )
        write_json(
            cache_path,
            {
                "response": response,
                "latency_seconds": latency,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    raw = response_text(response)
    return raw, {
        "id": item_id,
        "stage": stage,
        "cache_key": key,
        "cache_hit": cached,
        "latency_seconds": latency,
        "usage": usage_dict(response),
        "raw_response": raw,
    }


def process_row(
    root: Path,
    config: dict[str, Any],
    budget: Budget,
    row: dict[str, Any],
    api_key: str,
    refresh: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    item_id = row["id"]
    question = str(row["financial_question"])
    answer = str(row["company_response"])
    stage1_messages = [
        {"role": "system", "content": STAGE1_PROMPT},
        {
            "role": "user",
            "content": (
                f"<financial_question>\n{question}\n</financial_question>\n\n"
                f"<company_response>\n{answer}\n</company_response>\n\n"
                "Return Stage-1 tagged analysis only."
            ),
        },
    ]
    stage1_raw, stage1_log = call_cached(
        root=root,
        config=config,
        budget=budget,
        item_id=item_id,
        stage="stage1_tagged_extraction",
        messages=stage1_messages,
        max_tokens=int(config["stage1_max_tokens"]),
        api_key=api_key,
        refresh=refresh,
    )
    logs = [stage1_log]
    stage1_retry_used = False
    if not stage1_ok(stage1_raw):
        stage1_retry_used = True
        retry_messages = [
            *stage1_messages,
            {"role": "assistant", "content": stage1_raw},
            {
                "role": "user",
                "content": "The previous Stage-1 output was incomplete. Return only the requested tagged format.",
            },
        ]
        stage1_raw, retry_log = call_cached(
            root=root,
            config=config,
            budget=budget,
            item_id=item_id,
            stage="stage1_format_retry",
            messages=retry_messages,
            max_tokens=int(config["stage1_max_tokens"]),
            api_key=api_key,
            refresh=refresh,
        )
        logs.append(retry_log)

    stage2_messages = [
        {"role": "system", "content": STAGE2_PROMPT},
        {
            "role": "user",
            "content": (
                f"Raw Japanese Financial Question:\n{question}\n\n"
                f"Raw Japanese Company Response:\n{answer}\n\n"
                f"Stage-1 Tagged Evidence:\n{stage1_raw}\n\n"
                "Return Stage-2 tagged decision now. The first line must be the <LABEL> line."
            ),
        },
    ]
    stage2_raw, stage2_log = call_cached(
        root=root,
        config=config,
        budget=budget,
        item_id=item_id,
        stage="stage2_label_first",
        messages=stage2_messages,
        max_tokens=int(config["stage2_max_tokens"]),
        api_key=api_key,
        refresh=refresh,
    )
    logs.append(stage2_log)
    label, parse_method = extract_stage2_label(stage2_raw)
    repair_raw = ""
    if label not in VALID_LABELS:
        repair_messages = [
            {"role": "system", "content": REPAIR_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Previous Stage-2 adjudication:\n{stage2_raw}\n\n"
                    "Return the final label only."
                ),
            },
        ]
        repair_raw, repair_log = call_cached(
            root=root,
            config=config,
            budget=budget,
            item_id=item_id,
            stage="stage2_label_repair",
            messages=repair_messages,
            max_tokens=int(config["repair_max_tokens"]),
            api_key=api_key,
            refresh=refresh,
        )
        logs.append(repair_log)
        label, repair_method = extract_stage2_label(repair_raw)
        parse_method = f"repair:{repair_method}"
    if label not in VALID_LABELS:
        raise RuntimeError(f"ID {item_id} has no valid final label")

    return {
        "id": item_id,
        "prediction": label,
        "parse_method": parse_method,
        "stage1_valid": stage1_ok(stage1_raw),
        "stage1_retry_used": stage1_retry_used,
        "stage2_repair_used": bool(repair_raw),
        "financial_question": question,
        "company_response": answer,
        "stage1_raw": stage1_raw,
        "stage2_raw": stage2_raw,
        "stage2_repair_raw": repair_raw,
    }, logs


def validate_predictions(rows: list[dict[str, Any]], expected_rows: int | None) -> None:
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} predictions, received {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Output IDs must be unique")
    if any(row["prediction"] not in VALID_LABELS for row in rows):
        raise ValueError("Output contains an invalid label")


def write_outputs(
    output_path: Path,
    run_dir: Path,
    results: list[dict[str, Any]],
    logs: list[dict[str, Any]],
    budget: Budget,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "prediction"])
        writer.writeheader()
        writer.writerows(
            {"id": row["id"], "prediction": row["prediction"]} for row in results
        )
    with (run_dir / "raw_outputs.jsonl").open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (run_dir / "api_log.jsonl").open("w", encoding="utf-8") as handle:
        for log in logs:
            handle.write(json.dumps(log, ensure_ascii=False) + "\n")
    prediction_counts = pd.Series(
        [row["prediction"] for row in results], dtype="string"
    ).value_counts()
    summary = {
        "rows": len(results),
        "valid_predictions": len(results),
        "prediction_counts": {
            str(label): int(count) for label, count in prediction_counts.items()
        },
        "stage1_retries": sum(bool(row["stage1_retry_used"]) for row in results),
        "stage2_repairs": sum(bool(row["stage2_repair_used"]) for row in results),
        "api_calls": len(logs),
        "cache_hits": sum(bool(log["cache_hit"]) for log in logs),
        "conservative_reserved_cost_usd": budget.reserved,
        "submission": str(output_path),
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Parquet, CSV, or JSONL input")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--run-dir", default="run_artifacts")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--budget-cap", type=float, default=2.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_path = (root / args.config).resolve()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    run_dir = Path(args.run_dir).resolve()
    config = load_json(config_path)
    if int(config["workers"]) != 4:
        raise SystemExit("The frozen workflow uses exactly four workers")
    rows = read_input(input_path)
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise SystemExit(
            f"Expected {args.expected_rows} input rows, received {len(rows)}"
        )

    manifest = {
        "created_before_calls": datetime.now(timezone.utc).isoformat(),
        "team": "Maddies NLP",
        "task": "FinNLP 2026 Subtask 2: Japanese Implicit Communication Recognition",
        "provider": "OpenRouter",
        "model": config["model"],
        "model_url": config["model_url"],
        "architecture": "tagged Stage-1 extraction plus label-first Stage-2 adjudication",
        "rows": len(rows),
        "settings": {
            key: config[key]
            for key in [
                "temperature",
                "top_p",
                "seed",
                "workers",
                "stage1_max_tokens",
                "stage2_max_tokens",
                "repair_max_tokens",
                "provider",
            ]
        },
        "hashes": {
            "inference": sha256_file(Path(__file__)),
            "input": sha256_file(input_path),
            "config": sha256_file(config_path),
            "stage1_prompt": sha256_bytes(STAGE1_PROMPT.encode("utf-8")),
            "stage2_prompt": sha256_bytes(STAGE2_PROMPT.encode("utf-8")),
            "repair_prompt": sha256_bytes(REPAIR_PROMPT.encode("utf-8")),
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    load_dotenv(root / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    budget = Budget(args.budget_cap)
    results: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="maddies-gemma-final"
    ) as executor:
        futures = {
            executor.submit(
                process_row, root, config, budget, row, api_key, args.refresh
            ): row["id"]
            for row in rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result, row_logs = future.result()
            results.append(result)
            logs.extend(row_logs)
            if completed == 1 or completed % 10 == 0 or completed == len(rows):
                print(f"completed {completed}/{len(rows)}", flush=True)

    results.sort(key=lambda row: row["id"])
    logs.sort(key=lambda row: (row["id"], row["stage"]))
    validate_predictions(results, args.expected_rows)
    write_outputs(output_path, run_dir, results, logs, budget)


if __name__ == "__main__":
    main()
