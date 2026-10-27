#!/usr/bin/env python3
"""Plot the vLLM Prefill TTFT capacity matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _number(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _metric_field(metric_key: str) -> str:
    if metric_key == "mean":
        return "mean_ttft_ms"
    return f"p{metric_key.lower().lstrip('p')}_ttft_ms"


def _metric_label(metric_key: str) -> str:
    return "Mean TTFT" if metric_key == "mean" else f"P{metric_key.lower().lstrip('p')} TTFT"


def _slo_targets(row: dict[str, str]) -> list[tuple[str, float]]:
    try:
        payload = json.loads(row.get("SLO", "{}"))
    except json.JSONDecodeError:
        return []
    ttft = payload.get("ttft", {})
    result: list[tuple[str, float]] = []
    for metric_key, targets in ttft.items():
        if not isinstance(targets, list):
            targets = [targets]
        for target in targets:
            try:
                target_value = float(target)
            except (TypeError, ValueError):
                continue
            if math.isfinite(target_value) and target_value > 0:
                result.append((metric_key, target_value))
    return result


def build_matrix(
    summary_path: Path,
    cell: str = "rps",
) -> dict[str, Any]:
    if cell not in {"rps", "input_throughput"}:
        raise ValueError("cell must be rps or input_throughput")
    with summary_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    columns: list[str] = []
    row_keys: list[tuple[str, float]] = []
    winner_pairs: dict[tuple[str, tuple[str, float]], tuple[float, float]] = {}
    for row in rows:
        case_name = row.get("case_name", "")
        if case_name not in columns:
            columns.append(case_name)
        if row.get("request_rate", "").lower() == "inf":
            continue
        rps = _number(row, "rps")
        value = _number(row, cell if cell == "rps" else "input_throughput_tok_s")
        if rps is None or value is None:
            continue
        for metric_key, target in _slo_targets(row):
            key = (metric_key, target)
            if key not in row_keys:
                row_keys.append(key)
            metric_value = _number(row, _metric_field(metric_key))
            if metric_value is None or metric_value > target * 1000:
                continue
            current = winner_pairs.get((case_name, key))
            if current is None or rps > current[0]:
                winner_pairs[(case_name, key)] = (rps, value)

    row_keys.sort(key=lambda item: (item[1], item[0]))
    return {
        "columns": columns,
        "rows": row_keys,
        "winners": {key: pair[1] for key, pair in winner_pairs.items()},
        "cell": cell,
    }


def plot_prefill_matrix(
    summary_path: Path,
    output_path: Path,
    cell: str = "rps",
) -> None:
    matrix = build_matrix(summary_path, cell)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("matplotlib/numpy is not installed") from exc

    columns = matrix["columns"]
    row_keys = matrix["rows"]
    winners = matrix["winners"]
    if not columns or not row_keys:
        raise RuntimeError("summary 中没有可作图的有效 Prefill 结果")

    values = np.full((len(row_keys), len(columns)), np.nan)
    for row_index, row_key in enumerate(row_keys):
        for column_index, column in enumerate(columns):
            value = winners.get((column, row_key))
            if value is not None:
                values[row_index, column_index] = value

    fig_width = max(8.0, 2.4 * len(columns) + 2.5)
    fig_height = max(4.5, 1.0 * len(row_keys) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=20, ha="right")
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels(
        [f"{_metric_label(metric)} < {target:g}s" for metric, target in row_keys]
    )
    ax.set_xlabel("Input case")
    ax.set_ylabel("TTFT SLA")
    ax.set_title(
        "vLLM Prefill TTFT Capacity Matrix\n"
        + ("Maximum actual client RPS" if cell == "rps" else "Input token throughput")
    )
    for row_index, row_key in enumerate(row_keys):
        for column_index, column in enumerate(columns):
            value = winners.get((column, row_key))
            label = "NO MEET" if value is None else f"{value:.2f}"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="white" if value is not None else "#555555",
                fontsize=11,
                fontweight="bold",
            )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("RPS" if cell == "rps" else "tok/s")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cell", choices=("rps", "input_throughput"), default="rps")
    args = parser.parse_args()
    plot_prefill_matrix(args.summary, args.output, args.cell)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
