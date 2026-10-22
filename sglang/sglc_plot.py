#!/usr/bin/env python3
"""Render a decode capacity matrix from sglc.sh's sum_all.csv."""

import csv
import math
import sys
from pathlib import Path


def number(row, key):
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def parse_rules(raw):
    rules = []
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            continue
        metric, target, near = parts
        try:
            rules.append((metric, float(target), float(near)))
        except ValueError:
            continue
    return rules


def token_label(value, token_unit):
    tokens = number({"value": value}, "value")
    if tokens is None:
        return str(value or "-")
    amount = tokens / token_unit
    return f"{amount:g}K"


def main(args):
    if len(args) != 6:
        raise SystemExit(
            "usage: sglc_plot.py SUM_ALL_CSV OUTPUT_PNG SLA_RULES CELL TOKEN_UNIT"
        )
    csv_path, output_path, rules_raw, cell, token_unit_raw = args
    try:
        token_unit = int(token_unit_raw)
    except ValueError:
        token_unit = 1024
    rules = parse_rules(rules_raw)
    if not rules:
        raise SystemExit("SLA_RULES 没有可作图的规则")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit(f"matplotlib/numpy 不可用，无法作图: {exc}")

    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if number(row, "rps") is not None]
    if not rows:
        raise SystemExit("sum_all.csv 没有可作图的有效 RPS 数据")

    def column_key(row):
        return (
            number(row, "input_len") or float("inf"),
            number(row, "output_len") or float("inf"),
            number(row, "per_dp_bs") or float("inf"),
        )

    columns = []
    column_rows = {}
    for row in sorted(rows, key=column_key):
        key = (
            row.get("case_name", ""),
            row.get("input_len", ""),
            row.get("output_len", ""),
            row.get("per_dp_bs", ""),
        )
        if key in column_rows:
            continue
        label = (
            f"{token_label(row.get('input_len'), token_unit)} / "
            f"{token_label(row.get('output_len'), token_unit)}\n"
            f"BS {row.get('per_dp_bs', '-') }"
        )
        columns.append((key, label))
        column_rows[key] = row

    target_labels = [f"{metric} < {target:g} ms" for metric, target, _ in rules]
    matrix = np.full((len(rules), len(columns)), np.nan, dtype=float)
    annotations = [["NO\nMEET" for _ in columns] for _ in rules]

    for rule_index, (metric, target, _near) in enumerate(rules):
        for column_index, (key, _label) in enumerate(columns):
            candidates = []
            for row in rows:
                row_key = (
                    row.get("case_name", ""),
                    row.get("input_len", ""),
                    row.get("output_len", ""),
                    row.get("per_dp_bs", ""),
                )
                if row_key != key:
                    continue
                measured = number(row, metric)
                actual_rps = number(row, "rps")
                value = actual_rps if cell == "rps" else number(row, "generate_throughput_tok_s")
                if measured is None or actual_rps is None or value is None:
                    continue
                if measured <= target:
                    candidates.append((actual_rps, value))
            if candidates:
                _, value = max(candidates, key=lambda item: item[0])
                matrix[rule_index, column_index] = value
                annotations[rule_index][column_index] = f"{value:.2f}"

    fig_width = max(8.0, 2.0 * len(columns) + 2.5)
    fig_height = max(4.5, 1.3 * len(rules) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(columns)), [label for _, label in columns], fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(rules)), target_labels, fontsize=11)
    ax.set_xlabel("Decode workload / per-DP BS", fontsize=12, fontweight="bold")
    ax.set_ylabel("SLA rule", fontsize=12, fontweight="bold")
    ax.set_title("Decode Capacity Matrix", fontsize=18, fontweight="bold", pad=18)
    shown = "actual client RPS" if cell == "rps" else "generated token throughput"
    ax.text(
        0.5,
        1.02,
        f"Maximum {shown} under SLA constraints",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#5d6775",
    )
    finite_values = matrix[np.isfinite(matrix)]
    median = float(np.median(finite_values)) if finite_values.size else 0.0
    for i in range(len(rules)):
        for j in range(len(columns)):
            value = matrix[i, j]
            color = "white" if not math.isnan(value) and value > median else "#132238"
            ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=12, fontweight="bold", color=color)
    fig.colorbar(image, ax=ax, shrink=0.82, pad=0.03)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(output_path)


if __name__ == "__main__":
    main(sys.argv[1:])
