#!/usr/bin/env python3
"""Parse SGLang benchmark logs and persist GSP trial rows."""

import csv
import json
import math
import os
import re
import sys


ALIASES = {
    "Backend": "backend",
    "Traffic request rate": "traffic_request_rate",
    "Max request concurrency": "max_request_concurrency",
    "Max concurrent requests": "max_request_concurrency",
    "Request throughput (req/s)": "request_throughput",
    "Input token throughput (tok/s)": "input_throughput",
    "Output token throughput (tok/s)": "output_throughput",
    "Peak output token throughput (tok/s)": "peak_output_throughput",
    "Total token throughput (tok/s)": "total_throughput",
    "Peak concurrency": "peak_concurrency",
    "Peak concurrent requests": "peak_concurrency",
    "Concurrency": "concurrency",
    "Accept length": "accept_length",
    "Mean E2E Latency (ms)": "mean_e2e_latency_ms",
    "Median E2E Latency (ms)": "p50_e2e_latency_ms",
    "P50 E2E Latency (ms)": "p50_e2e_latency_ms",
    "P75 E2E Latency (ms)": "p75_e2e_latency_ms",
    "P90 E2E Latency (ms)": "p90_e2e_latency_ms",
    "P95 E2E Latency (ms)": "p95_e2e_latency_ms",
    "P99 E2E Latency (ms)": "p99_e2e_latency_ms",
    "P999 E2E Latency (ms)": "p999_e2e_latency_ms",
    "Max E2E Latency (ms)": "max_e2e_latency_ms",
    "Mean TTFT (ms)": "mean_ttft_ms",
    "Median TTFT (ms)": "p50_ttft_ms",
    "P50 TTFT (ms)": "p50_ttft_ms",
    "P75 TTFT (ms)": "p75_ttft_ms",
    "P90 TTFT (ms)": "p90_ttft_ms",
    "P95 TTFT (ms)": "p95_ttft_ms",
    "P99 TTFT (ms)": "p99_ttft_ms",
    "P999 TTFT (ms)": "p999_ttft_ms",
    "Max TTFT (ms)": "max_ttft_ms",
    "Mean TPOT (ms)": "mean_tpot_ms",
    "Median TPOT (ms)": "p50_tpot_ms",
    "P50 TPOT (ms)": "p50_tpot_ms",
    "P75 TPOT (ms)": "p75_tpot_ms",
    "P90 TPOT (ms)": "p90_tpot_ms",
    "P95 TPOT (ms)": "p95_tpot_ms",
    "P99 TPOT (ms)": "p99_tpot_ms",
    "P999 TPOT (ms)": "p999_tpot_ms",
    "Max TPOT (ms)": "max_tpot_ms",
    "Mean ITL (ms)": "mean_itl_ms",
    "Median ITL (ms)": "p50_itl_ms",
    "P50 ITL (ms)": "p50_itl_ms",
    "P75 ITL (ms)": "p75_itl_ms",
    "P90 ITL (ms)": "p90_itl_ms",
    "P95 ITL (ms)": "p95_itl_ms",
    "P99 ITL (ms)": "p99_itl_ms",
    "P999 ITL (ms)": "p999_itl_ms",
    "Max ITL (ms)": "max_itl_ms",
    "Total prompt tokens": "total_prompt_tokens",
    "Total cached tokens": "total_cached_tokens",
    "Total input tokens": "total_input_tokens",
    "Total input text tokens": "total_input_text_tokens",
    "Total generated tokens": "total_generated_tokens",
    "Total generated tokens (retokenized)": "total_generated_tokens_retokenized",
    "Cache hit rate": "cache_hit_rate_pct",
    "Successful requests": "successful_requests",
    "Benchmark duration (s)": "benchmark_duration_s",
}

TRIAL_FIELDS = [
    "case_id", "input", "output", "specified_cache_hit_rate_pct", "system_prompt_len",
    "question_len", "theoretical_cache_hit_rate_pct", "requested_theoretical_cache_hit_rate_pct",
    "gsp_page_size", "parallel_config", "page_aligned", "gsp_num_groups",
    "gsp_prompts_per_group", "num_prompts", "ttft_metric", "ttft_targets_s",
    "tpot_metric", "tpot_target_ms", "configured_request_rate", "search_phase",
    "exit_code", "actual_rps", "backend", "traffic_request_rate", "max_request_concurrency",
    "successful_requests", "benchmark_duration_s",
    "request_throughput", "input_throughput", "output_throughput", "total_throughput",
    "peak_output_throughput", "peak_concurrency", "max_concurrency", "concurrency",
    "mean_ttft_ms", "p50_ttft_ms", "p75_ttft_ms", "p90_ttft_ms", "p95_ttft_ms",
    "p99_ttft_ms", "p999_ttft_ms", "max_ttft_ms", "mean_tpot_ms", "p50_tpot_ms",
    "p75_tpot_ms", "p90_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "p50_itl_ms", "p75_itl_ms", "p90_itl_ms",
    "p95_itl_ms", "p99_itl_ms", "accept_length",
    "mean_e2e_latency_ms", "p50_e2e_latency_ms", "p75_e2e_latency_ms", "p90_e2e_latency_ms",
    "p95_e2e_latency_ms", "p99_e2e_latency_ms", "p999_e2e_latency_ms", "max_e2e_latency_ms",
    "actual_cache_hit_rate_pct", "duration_s",
    "log_file",
]

NUMBER = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?$")


def convert(value):
    value = value.strip()
    if value.endswith("%"):
        value = value[:-1].strip()
    if NUMBER.fullmatch(value):
        parsed = float(value)
        if math.isfinite(parsed):
            return int(parsed) if parsed.is_integer() else parsed
    return value


def parse_log(path):
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return {}
    values = {}
    for line in lines:
        if ":" not in line:
            continue
        label, raw = line.split(":", 1)
        label = label.strip()
        if label in ALIASES:
            values[ALIASES[label]] = convert(raw)
    if "cache_hit_rate_pct" not in values:
        prompt = values.get("total_prompt_tokens")
        cached = values.get("total_cached_tokens")
        if isinstance(prompt, (int, float)) and isinstance(cached, (int, float)) and prompt > 0:
            values["cache_hit_rate_pct"] = 100.0 * cached / prompt
    return values


def append_trial(argv):
    (path, case_id, input_raw, output_raw, target_hit, system_len, question_len,
     theory_hit, requested_theory_hit, page_size, parallel_config, page_aligned, groups,
     prompts_per_group, num_prompts, ttft_metric, ttft_targets, tpot_metric, tpot_target,
     request_rate, phase, max_concurrency, log_file, exit_code, metrics_raw) = argv
    try:
        metrics = json.loads(metrics_raw)
    except json.JSONDecodeError:
        metrics = {}
    row = {key: "" for key in TRIAL_FIELDS}
    row.update({
        "case_id": case_id, "input": input_raw, "output": output_raw,
        "specified_cache_hit_rate_pct": target_hit, "system_prompt_len": system_len,
        "question_len": question_len, "theoretical_cache_hit_rate_pct": theory_hit,
        "requested_theoretical_cache_hit_rate_pct": requested_theory_hit,
        "gsp_page_size": page_size, "parallel_config": parallel_config,
        "page_aligned": page_aligned, "gsp_num_groups": groups,
        "gsp_prompts_per_group": prompts_per_group, "num_prompts": num_prompts,
        "ttft_metric": ttft_metric, "ttft_targets_s": ttft_targets,
        "tpot_metric": tpot_metric, "tpot_target_ms": tpot_target,
        "configured_request_rate": request_rate, "search_phase": phase,
        "max_concurrency": max_concurrency, "exit_code": exit_code, "log_file": log_file,
    })
    row.update({key: value for key, value in metrics.items() if key in TRIAL_FIELDS})
    row["actual_rps"] = metrics.get("request_throughput", "")
    row["request_throughput"] = metrics.get("request_throughput", "")
    row["actual_cache_hit_rate_pct"] = metrics.get("cache_hit_rate_pct", "")

    rows = []
    if os.path.exists(path):
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
        except (OSError, csv.Error):
            rows = []
    key = (case_id, request_rate, phase, groups, prompts_per_group, system_len, question_len)
    rows = [r for r in rows if (
        r.get("case_id", ""), r.get("configured_request_rate", ""), r.get("search_phase", ""),
        r.get("gsp_num_groups", ""), r.get("gsp_prompts_per_group", ""),
        r.get("system_prompt_len", ""), r.get("question_len", "")
    ) != key]
    rows.append(row)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: gsp_prefix_metrics.py parse-log LOG | append-trial ARGS...")
    if sys.argv[1] == "parse-log":
        print(json.dumps(parse_log(sys.argv[2]), ensure_ascii=False, separators=(",", ":")))
    elif sys.argv[1] == "append-trial":
        append_trial(sys.argv[2:])
    else:
        raise SystemExit(f"unknown command: {sys.argv[1]}")


if __name__ == "__main__":
    main()
