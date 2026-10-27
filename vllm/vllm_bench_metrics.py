#!/usr/bin/env python3
"""Shared vLLM benchmark parsing and Prefill backtest helpers."""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable


NUMBER_RE = re.compile(
    r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$"
)
TTFT_METRICS = {
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
}

PREFILL_TRIAL_FIELDS = [
    "case_id",
    "case_name",
    "input_len",
    "output_len",
    "parallel_config",
    "SLO",
    "ttft_metric",
    "ttft_targets_s",
    "request_rate",
    "duration_s",
    "rps",
    "input_throughput_tok_s",
    "generate_throughput_tok_s",
    "total_throughput_tok_s",
    "successful_requests",
    "failed_requests",
    "num_prompts",
    "max_concurrency",
    "concurrency",
    "peak_output_throughput",
    "peak_concurrency",
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p75_ttft_ms",
    "p90_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "p999_ttft_ms",
    "max_ttft_ms",
    "mean_tpot_ms",
    "p50_tpot_ms",
    "p75_tpot_ms",
    "p90_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "p999_tpot_ms",
    "max_tpot_ms",
    "mean_itl_ms",
    "p50_itl_ms",
    "p75_itl_ms",
    "p90_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "p999_itl_ms",
    "max_itl_ms",
    "accept_length",
    "mean_e2e_latency_ms",
    "p50_e2e_latency_ms",
    "p75_e2e_latency_ms",
    "p90_e2e_latency_ms",
    "p95_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "p999_e2e_latency_ms",
    "max_e2e_latency_ms",
    "exit_code",
    "case_status",
    "search_phase",
    "log_file",
]

PREFILL_SUMMARY_FIELDS = [
    "case_name",
    "input_len",
    "output_len",
    "parallel_config",
    "SLO",
    "successful_requests",
    "failed_requests",
    "num_prompts",
    "max_concurrency",
    "concurrency",
    "peak_output_throughput",
    "peak_concurrency",
    "request_rate",
    "duration_s",
    "rps",
    "input_throughput_tok_s",
    "generate_throughput_tok_s",
    "total_throughput_tok_s",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "accept_length",
    "mean_e2e_latency_ms",
    "p95_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "p50_ttft_ms",
    "p50_tpot_ms",
    "p50_itl_ms",
    "p50_e2e_latency_ms",
]


@dataclass(frozen=True)
class PrefillCase:
    case_id: str
    input_tokens: int
    output_tokens: int
    metric: str
    targets_s: tuple[float, ...]

    @property
    def case_name(self) -> str:
        return self.case_id

    @property
    def slo_json(self) -> str:
        key = self.metric.removesuffix("_ttft_ms")
        if key.startswith("p"):
            key = "P" + key[1:]
        payload = {"ttft": {key: list(self.targets_s)}}
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _clean_number(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    return str(value).strip()


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"inf", "+inf", "-inf", "nan", "none", "null", "n/a"}:
        return value
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if NUMBER_RE.fullmatch(value):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in ("", None):
            return row[key]
    return ""


def has_value(row: dict[str, Any], key: str) -> bool:
    return key in row and row[key] not in ("", None)


def percentile_from_list(values: Any, q: int) -> Any:
    if not values:
        return ""
    if isinstance(values, dict):
        for key, value in values.items():
            try:
                if abs(float(key) - q) < 1e-6:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return ""
    if not isinstance(values, list):
        return ""
    for item in values:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                if abs(float(item[0]) - q) < 1e-6:
                    return float(item[1])
            except (TypeError, ValueError):
                continue
    return ""


def parse_percentiles_from_text(text: str) -> dict[str, float]:
    patterns = [
        re.compile(
            r"P\s*(?P<pct>50|95|99)\s+"
            r"(?P<metric>TTFT|TPOT|ITL|E2E(?:L| Latency)?)"
            r"[^\d\n]*?(?P<val>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?P<metric>TTFT|TPOT|ITL|E2E(?:L| Latency)?)\s+"
            r"P\s*(?P<pct>50|95|99)"
            r"[^\d\n]*?(?P<val>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            re.IGNORECASE,
        ),
    ]
    metric_map = {
        "ttft": "ttft",
        "tpot": "tpot",
        "itl": "itl",
        "e2e": "e2e_latency",
        "e2el": "e2e_latency",
        "e2e latency": "e2e_latency",
    }
    result: dict[str, float] = {}
    for pattern in patterns:
        for match in pattern.finditer(text):
            metric = re.sub(r"\s+", " ", match.group("metric").strip().lower())
            metric = metric_map.get(metric, metric)
            if metric not in {"ttft", "tpot", "itl", "e2e_latency"}:
                continue
            try:
                result[f"p{int(match.group('pct'))}_{metric}_ms"] = float(
                    match.group("val")
                )
            except ValueError:
                continue
    return result


def parse_percentiles_from_log(log_path: Path) -> dict[str, float]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    return parse_percentiles_from_text(text)


def normalize_log_key(label: str) -> str:
    key = label.strip().lower()
    key = key.replace("%", "percent")
    key = re.sub(r"\(req/s\)", "", key)
    key = re.sub(r"\(tok/s\)", "", key)
    key = re.sub(r"\(ms\)", "_ms", key)
    key = re.sub(r"\(s\)", "_s", key)
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return re.sub(r"_+", "_", key).strip("_")


def parse_summary_text(text: str) -> dict[str, Any]:
    aliases = {
        "traffic_request_rate": "request_rate",
        "input_token_throughput": "input_throughput",
        "output_token_throughput": "output_throughput",
        "total_token_throughput": "total_throughput",
        "peak_output_token_throughput": "peak_output_throughput",
        "peak_concurrent_requests": "peak_concurrency",
        "mean_e2el_ms": "mean_e2e_latency_ms",
    }
    known_prefixes = (
        "successful_requests",
        "failed_requests",
        "completed",
        "benchmark_duration",
        "duration_s",
        "total_input",
        "total_generated",
        "request_throughput",
        "request_rate",
        "traffic_request_rate",
        "input_token",
        "output_token",
        "total_token",
        "peak_output",
        "peak_concurrency",
        "peak_concurrent",
        "concurrency",
        "max_",
        "mean_",
        "median_",
        "p50_",
        "p75_",
        "p90_",
        "p95_",
        "p99_",
        "p999_",
        "accept_length",
    )
    result: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", raw_line).strip()
        match = re.match(r"^(.+?)\s*:\s*(.*?)\s*$", line)
        if not match:
            continue
        key = normalize_log_key(match.group(1).strip(" -="))
        if not key or not key.startswith(known_prefixes):
            continue
        key = aliases.get(key, key)
        result[key] = parse_scalar(match.group(2))
    result.update(parse_percentiles_from_text(text))
    return result


def parse_summary_from_log(log_path: Path) -> dict[str, Any]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    return parse_summary_text(text)


def normalize_row(row: dict[str, Any], log_path: Path | None = None) -> dict[str, Any]:
    row = dict(row)
    row["num_prompts"] = first_present(
        row, "num_prompts", "successful_requests", "completed", "global_concurrency"
    )
    row["max_concurrency"] = first_present(
        row, "max_concurrency", "max_concurrent_requests",
        "max_request_concurrency", "global_concurrency"
    )
    row["concurrency"] = first_present(
        row, "concurrency", "max_concurrent_requests", "global_concurrency"
    )
    row["peak_output_throughput"] = first_present(
        row, "peak_output_throughput",
        "peak_output_token_throughput",
        "max_output_tokens_per_s",
    )
    row["peak_concurrency"] = first_present(
        row, "peak_concurrency",
        "peak_concurrent_requests",
        "max_concurrent_requests",
    )
    row["request_rate"] = first_present(
        row, "request_rate", "traffic_request_rate"
    )
    if row["request_rate"] in ("", None):
        row["request_rate"] = "inf"
    row["duration_s"] = first_present(
        row, "duration_s", "benchmark_duration_s", "duration", "elapsed_s"
    )
    row["rps"] = first_present(
        row, "rps", "request_throughput", "request_throughput_req_s"
    )
    row["input_throughput_tok_s"] = first_present(
        row, "input_throughput_tok_s",
        "input_throughput",
        "input_token_throughput",
    )
    row["generate_throughput_tok_s"] = first_present(
        row, "generate_throughput_tok_s",
        "output_throughput",
        "output_throughput_tok_s",
    )
    row["total_throughput_tok_s"] = first_present(
        row, "total_throughput_tok_s",
        "total_token_throughput",
        "total_throughput",
    )
    if row["input_throughput_tok_s"] in ("", None):
        try:
            input_tokens = float(first_present(row, "total_input_tokens", "prompt_tokens"))
            duration_s = float(row["duration_s"])
            if duration_s > 0:
                row["input_throughput_tok_s"] = round(input_tokens / duration_s, 2)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    row["mean_e2e_latency_ms"] = first_present(
        row, "mean_e2e_latency_ms", "mean_e2el_ms"
    )
    row["accept_length"] = first_present(
        row, "accept_length", "mean_acceptance_length", "accept_len"
    )
    if log_path is not None:
        for key, value in parse_percentiles_from_log(log_path).items():
            if not has_value(row, key):
                row[key] = value

    metric_map = [
        ("ttft", "ttft"),
        ("tpot", "tpot"),
        ("itl", "itl"),
        ("e2el", "e2e_latency"),
    ]
    completed = row.get("completed", row.get("successful_requests", ""))
    try:
        completed_n = int(completed)
    except (TypeError, ValueError):
        completed_n = -1

    for source, destination in metric_map:
        plist = row.get(f"percentiles_{source}_ms")
        for q in (50, 95, 99):
            output_key = f"p{q}_{destination}_ms"
            if not has_value(row, output_key) and plist not in ("", None):
                value = percentile_from_list(plist, q)
                if value != "":
                    row[output_key] = value

        p50_key = f"p50_{destination}_ms"
        if not has_value(row, p50_key):
            row[p50_key] = first_present(
                row, p50_key, f"median_{destination}_ms", f"median_{source}_ms"
            )
        mean_key = (
            f"mean_{destination}_ms"
            if destination != "e2e_latency"
            else "mean_e2e_latency_ms"
        )
        mean_alt = f"mean_{source}_ms"
        base = first_present(row, p50_key, mean_key, mean_alt, f"median_{source}_ms")
        if base == "":
            continue
        if completed_n in (0, 1):
            for q in (50, 95, 99):
                output_key = f"p{q}_{destination}_ms"
                if not has_value(row, output_key):
                    row[output_key] = base
        mean_value = first_present(row, mean_key, mean_alt)
        median_value = first_present(
            row, f"median_{destination}_ms", f"median_{source}_ms", p50_key
        )
        try:
            same_single_value = (
                mean_value != ""
                and median_value != ""
                and abs(float(mean_value) - float(median_value)) < 1e-9
            )
        except (TypeError, ValueError):
            same_single_value = False
        if same_single_value:
            for q in (95, 99):
                output_key = f"p{q}_{destination}_ms"
                if not has_value(row, output_key):
                    row[output_key] = mean_value
    return row


def parse_token_value(raw: str, token_unit: int = 1024) -> int:
    if token_unit <= 0:
        raise ValueError("token_unit must be positive")
    text = str(raw).strip().lower()
    if not text:
        raise ValueError("token length is empty")
    has_k = text.endswith("k")
    number = text[:-1] if has_k else text
    try:
        value = Decimal(number)
    except InvalidOperation as exc:
        raise ValueError(f"invalid token length: {raw}") from exc
    if value <= 0:
        raise ValueError("token length must be positive")
    if has_k:
        value *= Decimal(token_unit)
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _safe_case_id(raw_input: str, metric: str, targets: Iterable[float]) -> str:
    target_text = "_".join(format(target, "g") for target in targets)
    raw = f"{raw_input}_{metric}_{target_text}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def parse_prefill_case(spec: str, token_unit: int = 1024) -> PrefillCase:
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError(
            "case format must be INPUT,TTFT_METRIC,TTFT_TARGETS_S"
        )
    input_raw, metric, targets_raw = parts
    metric = metric.lower()
    if metric not in TTFT_METRICS:
        raise ValueError(f"unsupported TTFT metric: {metric}")
    targets: list[float] = []
    for raw_target in targets_raw.split("|"):
        try:
            target = float(raw_target)
        except ValueError as exc:
            raise ValueError(f"invalid TTFT target: {raw_target}") from exc
        if not math.isfinite(target) or target <= 0:
            raise ValueError("TTFT targets must be positive finite seconds")
        targets.append(target)
    input_tokens = parse_token_value(input_raw, token_unit)
    case_id = _safe_case_id(input_raw, metric, targets)
    return PrefillCase(
        case_id=case_id,
        input_tokens=input_tokens,
        output_tokens=1,
        metric=metric,
        targets_s=tuple(targets),
    )


def case_to_dict(case: PrefillCase) -> dict[str, Any]:
    payload = asdict(case)
    payload["targets_s"] = list(case.targets_s)
    payload["slo_json"] = case.slo_json
    return payload


def case_from_dict(payload: dict[str, Any]) -> PrefillCase:
    return PrefillCase(
        case_id=str(payload["case_id"]),
        input_tokens=int(payload["input_tokens"]),
        output_tokens=int(payload.get("output_tokens", 1)),
        metric=str(payload["metric"]),
        targets_s=tuple(float(value) for value in payload["targets_s"]),
    )


def format_rate(rate: float | str) -> str:
    if isinstance(rate, str) and rate.lower() == "inf":
        return "inf"
    value = float(rate)
    if not math.isfinite(value):
        return "inf"
    return f"{value:.2f}"


def build_rate_sequence(start_rps: float, step: float = 0.1) -> list[float]:
    try:
        step_decimal = Decimal(str(step))
        start_decimal = Decimal(str(start_rps))
    except InvalidOperation as exc:
        raise ValueError("request rate and step must be numeric") from exc
    if step_decimal < Decimal("0.1"):
        raise ValueError("request-rate step must be at least 0.1")
    if not start_decimal.is_finite() or start_decimal < Decimal("0.1"):
        return []
    step_cents = (step_decimal * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    current = (start_decimal * 100).to_integral_value(rounding=ROUND_FLOOR) / Decimal(100)
    rates: list[float] = []
    while current >= Decimal("0.1"):
        rates.append(float(current.quantize(Decimal("0.01"))))
        current -= step_cents / Decimal(100)
    return rates


def _as_float(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def row_is_valid(row: dict[str, Any]) -> bool:
    try:
        exit_code = int(row.get("exit_code", 1))
    except (TypeError, ValueError):
        return False
    if exit_code != 0:
        return False
    successful = _as_float(row, "successful_requests")
    failed = _as_float(row, "failed_requests")
    if successful is not None and successful <= 0:
        return False
    if failed is not None and failed > 0:
        return False
    return _as_float(row, "rps") is not None


def select_sla_rows(
    rows: list[dict[str, Any]], case: PrefillCase
) -> dict[float, dict[str, Any] | None]:
    selected: dict[float, dict[str, Any] | None] = {}
    for target in case.targets_s:
        target_ms = target * 1000.0
        candidates = []
        for row in rows:
            if not row_is_valid(row):
                continue
            value = _as_float(row, case.metric)
            rps = _as_float(row, "rps")
            if value is None or rps is None or value > target_ms:
                continue
            candidates.append(row)
        selected[target] = max(
            candidates,
            key=lambda row: _as_float(row, "rps") or float("-inf"),
            default=None,
        )
    return selected


def _json_number(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def append_trial_row(path: Path, row: dict[str, Any]) -> None:
    rows = _load_csv(path)
    key = (row.get("case_id", ""), row.get("request_rate", ""), row.get("log_file", ""))
    rows = [
        old for old in rows
        if (old.get("case_id", ""), old.get("request_rate", ""), old.get("log_file", "")) != key
    ]
    rows.append(row)
    _write_csv(path, rows, PREFILL_TRIAL_FIELDS)


def make_trial_row(
    log_path: Path,
    case: PrefillCase,
    request_rate: str,
    num_prompts: int,
    max_concurrency: str,
    parallel_config: str,
    exit_code: int,
    search_phase: str,
) -> dict[str, Any]:
    metrics = normalize_row(parse_summary_from_log(log_path), log_path)
    configured_rate = format_rate(request_rate)
    row: dict[str, Any] = {field: "" for field in PREFILL_TRIAL_FIELDS}
    row.update(metrics)
    row.update(
        {
            "case_id": case.case_id,
            "case_name": case.case_name,
            "input_len": case.input_tokens,
            "output_len": case.output_tokens,
            "parallel_config": parallel_config,
            "SLO": case.slo_json,
            "ttft_metric": case.metric,
            "ttft_targets_s": "|".join(format(target, "g") for target in case.targets_s),
            "request_rate": configured_rate,
            "num_prompts": num_prompts,
            "max_concurrency": max_concurrency,
            "exit_code": exit_code,
            "case_status": "PASS" if exit_code == 0 else "FAIL",
            "search_phase": search_phase,
            "log_file": str(log_path),
        }
    )
    return row


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in PREFILL_SUMMARY_FIELDS}


def write_prefill_summary(
    trials_path: Path,
    summary_path: Path,
    cases: list[PrefillCase],
) -> None:
    trials = _load_csv(trials_path)
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in trials:
        by_case.setdefault(row.get("case_id", ""), []).append(row)

    output: list[dict[str, Any]] = []
    for case in cases:
        rows = [row for row in by_case.get(case.case_id, []) if row_is_valid(row)]
        inf_rows = [row for row in rows if row.get("request_rate", "").lower() == "inf"]
        finite_rows = [row for row in rows if row.get("request_rate", "").lower() != "inf"]
        best_inf = max(
            inf_rows,
            key=lambda row: _as_float(row, "input_throughput_tok_s") or float("-inf"),
            default=None,
        )
        rows = ([best_inf] if best_inf is not None else []) + finite_rows
        rows.sort(
            key=lambda row: (
                0 if row.get("request_rate", "").lower() == "inf" else 1,
                -(_as_float(row, "rps") or float("-inf")),
            )
        )
        for row in rows:
            enriched = dict(row)
            enriched["SLO"] = case.slo_json
            enriched["case_name"] = case.case_name
            output.append(_summary_row(enriched))

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        handle.write("\t".join(PREFILL_SUMMARY_FIELDS) + "\n")
        for row in output:
            values = []
            for field in PREFILL_SUMMARY_FIELDS:
                value = row.get(field, "")
                values.append(str(value).replace("\t", " ").replace("\r", " ").replace("\n", "") or "-")
            handle.write("\t".join(values) + "\n")


def case_targets_met(trials_path: Path, case: PrefillCase) -> bool:
    rows = [
        row
        for row in _load_csv(trials_path)
        if row.get("case_id", "") == case.case_id
        and row.get("request_rate", "").lower() != "inf"
        and row_is_valid(row)
    ]
    selected = select_sla_rows(rows, case)
    return all(selected[target] is not None for target in case.targets_s)


def _load_cases_json(path: Path) -> list[PrefillCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [case_from_dict(item) for item in payload]


def _usage() -> str:
    return (
        "usage: vllm_bench_metrics.py "
        "parse-log LOG | parse-case SPEC TOKEN_UNIT | rate-list START STEP | "
        "append-trial TRIALS_CSV LOG CASE_JSON REQUEST_RATE NUM_PROMPTS MAX_CONCURRENCY "
        "PARALLEL_CONFIG EXIT_CODE SEARCH_PHASE | summary TRIALS_CSV SUMMARY_TSV CASES_JSON | "
        "targets-met TRIALS_CSV CASE_JSON"
    )


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(_usage(), file=sys.stderr)
        return 2
    command = argv[1]
    if command == "parse-log" and len(argv) == 3:
        print(json.dumps(normalize_row(parse_summary_from_log(Path(argv[2]))), ensure_ascii=False))
        return 0
    if command == "parse-case" and len(argv) == 4:
        print(json.dumps(case_to_dict(parse_prefill_case(argv[2], int(argv[3]))), ensure_ascii=False))
        return 0
    if command == "rate-list" and len(argv) == 4:
        print(json.dumps(build_rate_sequence(float(argv[2]), float(argv[3]))))
        return 0
    if command == "append-trial" and len(argv) == 11:
        trials_path = Path(argv[2])
        log_path = Path(argv[3])
        case = case_from_dict(json.loads(argv[4]))
        row = make_trial_row(
            log_path,
            case,
            argv[5],
            int(argv[6]),
            argv[7],
            argv[8],
            int(argv[9]),
            argv[10],
        )
        append_trial_row(trials_path, row)
        return 0
    if command == "summary" and len(argv) == 5:
        cases = _load_cases_json(Path(argv[4]))
        write_prefill_summary(Path(argv[2]), Path(argv[3]), cases)
        return 0
    if command == "targets-met" and len(argv) == 4:
        case = case_from_dict(json.loads(argv[3]))
        print("1" if case_targets_met(Path(argv[2]), case) else "0")
        return 0
    print(_usage(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
