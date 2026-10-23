#!/usr/bin/env python3
"""Build and prune the GSP summary TSV."""
import sys

def write_summary(args):
    import csv
    import json
    import math
    import os
    import sys

    trials_path, summary_tsv_path, baseline_tolerance_s, token_unit_s = args
    baseline_tolerance = float(baseline_tolerance_s)
    token_unit = int(token_unit_s)
    rows = []
    if os.path.exists(trials_path):
        with open(trials_path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

    def num(row, key):
        try:
            value = float(row.get(key, ""))
            return value if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    def token_value(raw):
        """Convert the case's token notation using the shared TOKEN_UNIT."""
        text = str(raw or "").strip().lower()
        if not text:
            return None
        multiplier = token_unit if text.endswith("k") else 1
        if multiplier != 1:
            text = text[:-1]
        try:
            value = float(text) * multiplier
            return int(value) if value.is_integer() else value
        except ValueError:
            return None

    def csv_number(value):
        if value is None:
            return ""
        return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)

    def same_parameters(row, baseline):
        keys = ("gsp_num_groups", "gsp_prompts_per_group", "system_prompt_len", "question_len")
        return all(row.get(key, "") == baseline.get(key, "") for key in keys)

    source_fields = list(rows[0].keys()) if rows else []
    metadata_fields = [
        "result_type", "parameter_status", "confirmed_inf_actual_rps",
        "confirmed_inf_cache_hit_rate_pct", "confirmed_inf_phase", "request_rate_order",
        "actual_cache_hit_rate_delta_pct",
    ]
    preferred_fields = [
        # Keep the complete public metric block identical to sglc.sh. GSP-only
        # workload fields are appended after this block so the two summaries can
        # be pasted into the same table without moving the standard metrics.
        "case_name", "input_len", "output_len",
        "parallel_config",
        # Cache-hit requirements are the primary GSP case identity and should be
        # visible before the common capacity/performance metrics.
        "specified_cache_hit_rate_pct", "actual_cache_hit_rate_pct", "SLO",
        "max_running_requests", "max_total_tokens", "tokens_per_request",
        "token_theoretical_per_dp_bs", "effective_theoretical_per_dp_bs",
        "per_dp_bs", "num_prompts",
        "max_concurrency", "concurrency", "peak_output_throughput", "peak_concurrency",
        "request_rate", "duration_s", "rps", "input_throughput_tok_s",
        "generate_throughput_tok_s", "total_throughput_tok_s",
        "mean_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
        "mean_itl_ms", "p95_itl_ms", "p99_itl_ms",
        "accept_length", "mean_e2e_latency_ms",
        "total_input_tokens", "total_input_text_tokens", "total_generated_tokens",
        "total_generated_tokens_retokenized",
        "std_e2e_latency_ms", "p75_e2e_latency_ms", "p90_e2e_latency_ms",
        "p999_e2e_latency_ms", "max_e2e_latency_ms", "std_ttft_ms", "p75_ttft_ms",
        "p90_ttft_ms", "p999_ttft_ms", "max_ttft_ms", "std_tpot_ms", "p75_tpot_ms",
        "p90_tpot_ms", "p999_tpot_ms", "max_tpot_ms", "std_itl_ms", "p75_itl_ms",
        "p90_itl_ms", "p999_itl_ms", "max_itl_ms",
        # Remaining GSP workload fields come after the shared client metrics.
        "system_prompt_len", "question_len", "gsp_page_size", "gsp_num_groups",
        "gsp_prompts_per_group", "total_prompt_tokens", "total_cached_tokens",
        # P50 is retained as an extra after the shared columns.
        "p50_ttft_ms", "p50_tpot_ms", "p50_itl_ms",
    ]
    excluded_fields = {
        "result_type", "parameter_status", "confirmed_inf_phase", "request_rate_order",
        "actual_cache_hit_rate_delta_pct", "exit_code", "actual_rps", "traffic_request_rate",
        "page_aligned", "confirmed_inf_actual_rps", "search_phase", "log_file", "backend",
        "max_request_concurrency", "ttft_metric", "ttft_targets_s", "tpot_metric", "tpot_target_ms",
        "theoretical_cache_hit_rate_pct", "requested_theoretical_cache_hit_rate_pct",
        "confirmed_inf_cache_hit_rate_pct", "configured_request_rate", "benchmark_duration_s",
        "case_id", "input", "output", "successful_requests", "request_throughput",
        "input_throughput", "output_throughput", "total_throughput", "max_tpot_ms",
        "p50_e2e_latency_ms", "p95_e2e_latency_ms", "p99_e2e_latency_ms",
        "case_status", "case_exit_code", "benchmark_result_found",
    }
    fields = preferred_fields + [
        field for field in source_fields
        if field not in preferred_fields and field not in metadata_fields and field not in excluded_fields
    ]

    summary = []
    case_groups = {}
    for row in rows:
        case_groups.setdefault(row.get("case_id", ""), []).append(row)

    for case_id, case_rows in case_groups.items():
        baseline_rows = [
            row for row in case_rows
            if row.get("configured_request_rate", "") == "inf"
            and row.get("exit_code", "") == "0"
            and num(row, "actual_cache_hit_rate_pct") is not None
            and num(row, "specified_cache_hit_rate_pct") is not None
            and abs(num(row, "actual_cache_hit_rate_pct") - num(row, "specified_cache_hit_rate_pct")) <= baseline_tolerance
        ]
        if not baseline_rows:
            continue
        # Keep the latest layout after calibration/duration adjustment, then choose
        # the highest-throughput inf trial within that layout.
        latest_layout = baseline_rows[-1]
        layout_rows = [row for row in baseline_rows if same_parameters(row, latest_layout)]
        baseline = max(layout_rows, key=lambda row: (num(row, "input_throughput") or float("-inf")))
        finite_rows = [
            row for row in case_rows
            if row.get("search_phase", "") == "descending"
            and row.get("configured_request_rate", "") != "inf"
            and same_parameters(row, baseline)
        ]
        inf_rps = num(baseline, "request_throughput")
        inf_hit = num(baseline, "actual_cache_hit_rate_pct")
        metric = baseline.get("ttft_metric", "")
        if metric == "mean_ttft_ms":
            ttft_key = "mean"
        elif metric.startswith("p") and metric.endswith("_ttft_ms"):
            ttft_key = "P" + metric[1:-len("_ttft_ms")]
        else:
            ttft_key = metric
        slo_values = []
        for target_s in baseline.get("ttft_targets_s", "").split("|"):
            if not target_s:
                continue
            try:
                target_seconds = float(target_s)
            except ValueError:
                continue
            slo_values.append(int(target_seconds) if target_seconds.is_integer() else target_seconds)
        slo_payload = {"ttft": {ttft_key: slo_values}} if ttft_key and slo_values else {}
        if baseline.get("tpot_metric") and baseline.get("tpot_target_ms"):
            try:
                tpot_target = float(baseline["tpot_target_ms"])
                tpot_metric = baseline["tpot_metric"]
                if tpot_metric == "mean_tpot_ms":
                    tpot_key = "mean"
                elif tpot_metric.startswith("p") and tpot_metric.endswith("_tpot_ms"):
                    tpot_key = "P" + tpot_metric[1:-len("_tpot_ms")]
                else:
                    tpot_key = tpot_metric
                slo_payload.setdefault("tpot", {})[tpot_key] = [
                    int(tpot_target) if tpot_target.is_integer() else tpot_target
                ]
            except ValueError:
                pass
        slo_json = json.dumps(slo_payload, ensure_ascii=False, separators=(",", ":"))
        final_rows = [(baseline, "inf")] + [(row, "finite") for row in finite_rows]
        final_rows.sort(key=lambda item: (
            0 if item[1] == "inf" else 1,
            -(num(item[0], "configured_request_rate") or float("-inf"))
        ))
        for order, (row, result_type) in enumerate(final_rows, start=1):
            output = dict(row)
            target_hit = num(row, "specified_cache_hit_rate_pct")
            actual_hit = num(row, "actual_cache_hit_rate_pct")
            input_len = num(row, "system_prompt_len")
            question_len = num(row, "question_len")
            if input_len is not None and question_len is not None:
                input_len += question_len
            else:
                input_len = token_value(row.get("input", ""))
            output_len = token_value(row.get("output", ""))
            request_rps = row.get("request_throughput", "")
            input_tp = row.get("input_throughput", "")
            output_tp = row.get("output_throughput", "")
            total_tp = row.get("total_throughput", "")
            output.update({
                "case_name": row.get("case_id", ""),
                "input_len": csv_number(input_len),
                "output_len": csv_number(output_len),
                "tokens_per_request": csv_number(
                    input_len + output_len if input_len is not None and output_len is not None else None
                ),
                "rps": request_rps,
                "input_throughput_tok_s": input_tp,
                "generate_throughput_tok_s": output_tp,
                "total_throughput_tok_s": total_tp,
                "result_type": result_type,
                "parameter_status": "confirmed",
                "SLO": slo_json,
                "request_rate": row.get("configured_request_rate", ""),
                "duration_s": row.get("benchmark_duration_s", row.get("duration_s", "")),
                "confirmed_inf_actual_rps": f"{inf_rps:.2f}" if inf_rps is not None else "",
                "confirmed_inf_cache_hit_rate_pct": f"{inf_hit:.4f}" if inf_hit is not None else "",
                "confirmed_inf_phase": baseline.get("search_phase", ""),
                "request_rate_order": str(order),
                "actual_cache_hit_rate_delta_pct": (
                    f"{actual_hit - target_hit:.4f}"
                    if actual_hit is not None and target_hit is not None else ""
                ),
            })
            summary.append(output)

    summary.sort(key=lambda row: (
        -(num(row, "confirmed_inf_actual_rps") or float("-inf")),
        row.get("case_id", ""),
        0 if row.get("result_type") == "inf" else 1,
        -(num(row, "configured_request_rate") or float("-inf")),
    ))

    output_rows = [{field: row.get(field, "") for field in fields} for row in summary]
    clipboard_rows = [
        {field: (row.get(field, "") or "-") for field in fields}
        for row in output_rows
    ]
    # This file is intended for direct copy/paste into a spreadsheet. Keep it as
    # plain TSV instead of CSV-quoting every cell; JSON quotes inside SLO remain
    # part of the value, while ordinary cells have no surrounding quotes.
    with open(summary_tsv_path, "w", newline="", encoding="utf-8-sig") as f:
        f.write("\t".join(fields) + "\n")
        for row in clipboard_rows:
            values = []
            for field in fields:
                value = str(row.get(field, "") or "")
                values.append(value.replace("\t", " ").replace("\r", " ").replace("\n", " "))
            f.write("\t".join(values) + "\n")

def prune_confirmed_trials(args):
    import csv
    import os
    import sys

    trials_path, summary_tsv_path = args
    if not os.path.exists(trials_path) or not os.path.exists(summary_tsv_path):
        raise SystemExit(0)

    with open(summary_tsv_path, newline="", encoding="utf-8-sig") as f:
        confirmed = list(csv.DictReader(f, delimiter="\t"))
    if not confirmed:
        raise SystemExit(0)

    def key(row):
        return (
            row.get("case_id") or row.get("case_name", ""),
            row.get("configured_request_rate") or row.get("request_rate", ""),
            row.get("gsp_num_groups", ""),
            row.get("gsp_prompts_per_group", ""), row.get("system_prompt_len", ""),
            row.get("question_len", ""),
        )

    confirmed_keys = {key(row) for row in confirmed}
    with open(trials_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys()) if rows else []
    remaining = [row for row in rows if key(row) not in confirmed_keys]
    with open(trials_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(remaining)

def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: gsp_prefix_summary.py summary|prune ...")
    if sys.argv[1] == "summary":
        write_summary(sys.argv[2:])
    elif sys.argv[1] == "prune":
        prune_confirmed_trials(sys.argv[2:])
    else:
        raise SystemExit(f"unknown command: {sys.argv[1]}")

if __name__ == "__main__":
    main()
