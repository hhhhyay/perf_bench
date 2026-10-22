#!/usr/bin/env python3
"""Render the GSP prefix capacity matrix."""
import sys

def plot_summary(args):
    import csv
    import math
    import sys
    from pathlib import Path

    summary_path, trials_path, output_path, cell, finite_tolerance_s = args
    finite_tolerance = float(finite_tolerance_s)
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit(f"matplotlib/numpy 不可用，无法作图: {exc}")

    with open(summary_path, newline="", encoding="utf-8-sig") as f:
        summary_rows = list(csv.DictReader(f, delimiter="\t"))
    if not summary_rows:
        raise SystemExit("summary TSV 没有可作图数据")
    with open(trials_path, newline="", encoding="utf-8-sig") as f:
        trial_rows = list(csv.DictReader(f))

    def row_key(row):
        return (
            row.get("case_id") or row.get("case_name", ""),
            row.get("configured_request_rate") or row.get("request_rate", ""),
            row.get("gsp_num_groups", ""),
            row.get("gsp_prompts_per_group", ""), row.get("system_prompt_len", ""),
            row.get("question_len", ""),
        )

    confirmed_keys = {row_key(row) for row in summary_rows}
    rows = [row for row in trial_rows if row_key(row) in confirmed_keys]
    if not rows:
        raise SystemExit("summary 对应的正式回测数据不存在")

    def num(row, key):
        try:
            value = float(row.get(key, ""))
            return value if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    columns = []
    for row in rows:
        label = f"{row.get('input','')} / {row.get('output','')}\nhit {row.get('specified_cache_hit_rate_pct','')}%"
        if label not in columns:
            columns.append(label)
    targets = []
    for row in rows:
        for target_s in row.get("ttft_targets_s", "").split("|"):
            if not target_s:
                continue
            label = f"{row.get('ttft_metric','')} < {target_s} s"
            if label not in targets:
                targets.append(label)

    matrix = np.full((len(targets), len(columns)), np.nan, dtype=float)
    annotations = [["NO\nMEET" for _ in columns] for _ in targets]
    for i, target in enumerate(targets):
        metric, _, target_s = target.partition(" < ")
        target_ms = float(target_s.removesuffix(" s")) * 1000.0
        for j, column in enumerate(columns):
            candidates = []
            for row in rows:
                row_column = f"{row.get('input','')} / {row.get('output','')}\nhit {row.get('specified_cache_hit_rate_pct','')}%"
                if row_column != column or row.get("configured_request_rate") == "inf":
                    continue
                measured_hit = num(row, "actual_cache_hit_rate_pct")
                target_hit = num(row, "specified_cache_hit_rate_pct")
                ttft = num(row, metric)
                actual_rps = num(row, "request_throughput")
                if None in (measured_hit, target_hit, ttft, actual_rps):
                    continue
                if abs(measured_hit - target_hit) > finite_tolerance or ttft > target_ms:
                    continue
                tpot_metric = row.get("tpot_metric", "")
                if tpot_metric:
                    tpot = num(row, tpot_metric)
                    tpot_limit = num(row, "tpot_target_ms")
                    if tpot is None or tpot_limit is None or tpot > tpot_limit:
                        continue
                value = actual_rps if cell == "rps" else num(row, "input_throughput")
                if value is not None:
                    candidates.append((actual_rps, value))
            if candidates:
                _, value = max(candidates, key=lambda item: item[0])
                matrix[i, j] = value
                annotations[i][j] = f"{value:.2f}"

    fig_width = max(7.0, 2.6 * len(columns) + 2.0)
    fig_height = max(4.5, 1.2 * len(targets) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap="GnBu", aspect="auto")
    ax.set_xticks(range(len(columns)), columns, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(targets)), targets, fontsize=11)
    ax.set_xlabel("GSP workload case", fontsize=12, fontweight="bold")
    ax.set_ylabel("TTFT SLA", fontsize=12, fontweight="bold")
    ax.set_title("GSP Prefix Capacity Matrix", fontsize=18, fontweight="bold", pad=18)
    ax.text(0.5, 1.02, f"Maximum actual client {'RPS' if cell == 'rps' else 'input throughput'} under TTFT + TPOT + cache-hit constraints", transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#5d6775")
    for i in range(len(targets)):
        for j in range(len(columns)):
            text = annotations[i][j]
            value = matrix[i, j]
            color = "white" if not math.isnan(value) and value > np.nanmedian(matrix) else "#132238"
            ax.text(j, i, text, ha="center", va="center", fontsize=14, fontweight="bold", color=color)
    fig.colorbar(image, ax=ax, shrink=0.82, pad=0.03)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(output_path)

def main():
    if len(sys.argv) != 6:
        raise SystemExit("usage: gsp_prefix_plot.py SUMMARY_TSV TRIALS_CSV OUTPUT_PNG CELL HIT_TOLERANCE")
    plot_summary(sys.argv[1:])

if __name__ == "__main__":
    main()
