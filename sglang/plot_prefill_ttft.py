#!/usr/bin/env python3
"""Draw a dependency-free Prefill TTFT chart from a benchmark CSV.

The CSV needs: case, input_k, bs, request_rate, actual_rps, input_throughput,
mean_ttft_ms, p99_ttft_ms, phase. Pillow is used only for PNG rendering.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "mean": (27, 94, 142),
    "p99": (211, 84, 0),
    "grid": (221, 226, 232),
    "axis": (65, 72, 80),
    "text": (32, 38, 45),
    "sla2": (46, 125, 50),
    "sla3": (156, 39, 176),
    "panel": (250, 251, 252),
}


def font(path: str | None, size: int, bold: bool = False):
    candidates = []
    if path:
        candidates.append(path)
    candidates += [
        r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("plot", "1") != "0"]
    normalized = []
    for row in rows:
        input_k = row.get("input_k", "")
        if not input_k and row.get("input_len"):
            try:
                input_k = str(float(row["input_len"]) / 1024).rstrip("0").rstrip(".")
            except (TypeError, ValueError):
                input_k = ""
        bs = row.get("bs", "") or row.get("per_dp_bs", "")
        row["input_k"] = input_k
        row["bs"] = bs
        row["case"] = row.get("case", "") or f"{input_k}K / BS{bs}"
        row["request_rate"] = (
            row.get("request_rate", "")
            or row.get("input_request_rate", "")
            or row.get("traffic_request_rate", "")
        )
        row["actual_rps"] = row.get("actual_rps", "") or row.get("request_throughput", "")
        normalized.append(row)
    return normalized


def grouped(rows):
    groups = {}
    for row in rows:
        key = row["case"]
        groups.setdefault(key, []).append(row)
    for values in groups.values():
        values.sort(key=lambda item: float(item["request_rate"]))
    return groups


def draw_panel(draw, box, rows, title, fonts):
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=8, fill=COLORS["panel"], outline=(226, 230, 235))
    draw.text((left + 18, top + 14), title, font=fonts["subtitle"], fill=COLORS["text"])

    plot = (left + 64, top + 56, right - 24, bottom - 54)
    px0, py0, px1, py1 = plot
    rates = [float(row["request_rate"]) for row in rows]
    values = [float(row["mean_ttft_ms"]) for row in rows] + [float(row["p99_ttft_ms"]) for row in rows]
    x_min, x_max = min(rates), max(rates)
    if math.isclose(x_min, x_max):
        x_min -= 0.1
        x_max += 0.1
    y_max = max(5000.0, max(values) * 1.08)

    def x(value):
        return px0 + (value - x_min) / (x_max - x_min) * (px1 - px0)

    def y(value):
        return py1 - value / y_max * (py1 - py0)

    for value in (0, 2000, 3000, 5000, 10000, 20000, 50000, 100000):
        if value > y_max:
            continue
        yy = y(value)
        draw.line((px0, yy, px1, yy), fill=COLORS["grid"], width=1)
        draw.text((px0 - 10, yy), f"{value / 1000:g}s", font=fonts["small"], fill=COLORS["axis"], anchor="ra")

    for rate in rates:
        xx = x(rate)
        draw.line((xx, py0, xx, py1), fill=(239, 241, 244), width=1)
        draw.text((xx, py1 + 10), f"{rate:g}", font=fonts["small"], fill=COLORS["axis"], anchor="ma")

    draw.line((px0, py0, px0, py1), fill=COLORS["axis"], width=2)
    draw.line((px0, py1, px1, py1), fill=COLORS["axis"], width=2)
    for value, color, label in ((2000, COLORS["sla2"], "SLA 2s"), (3000, COLORS["sla3"], "SLA 3s")):
        if value <= y_max:
            yy = y(value)
            draw.line((px0, yy, px1, yy), fill=color, width=2)
            draw.text((px1 - 4, yy - 4), label, font=fonts["small"], fill=color, anchor="rs")

    for metric, color, key in (("Mean TTFT", COLORS["mean"], "mean_ttft_ms"), ("P99 TTFT", COLORS["p99"], "p99_ttft_ms")):
        points = [(x(float(row["request_rate"])), y(float(row[key]))) for row in rows]
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        for xx, yy in points:
            draw.ellipse((xx - 5, yy - 5, xx + 5, yy + 5), fill=color, outline="white", width=2)
    draw.text((px0, bottom - 30), "请求率 req/s（按请求率从低到高展示）", font=fonts["small"], fill=COLORS["axis"])
    return [(COLORS["mean"], "Mean TTFT"), (COLORS["p99"], "P99 TTFT")]


def heat_color(seconds):
    stops = [
        (0.0, (46, 125, 50)),
        (2.0, (139, 195, 74)),
        (3.0, (255, 193, 7)),
        (6.0, (239, 108, 0)),
        (12.0, (198, 40, 40)),
    ]
    value = max(0.0, min(float(seconds), stops[-1][0]))
    for (left_value, left_color), (right_value, right_color) in zip(stops, stops[1:]):
        if value <= right_value:
            ratio = (value - left_value) / (right_value - left_value)
            return tuple(round(a + (b - a) * ratio) for a, b in zip(left_color, right_color))
    return stops[-1][1]


def draw_heatmap(draw, box, rows, metric, title, fonts, all_rates):
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=8, fill=COLORS["panel"], outline=(226, 230, 235))
    draw.text((left + 18, top + 14), title, font=fonts["subtitle"], fill=COLORS["text"])
    plot_left, plot_top = left + 150, top + 58
    plot_right, plot_bottom = right - 24, bottom - 38
    row_keys = sorted({f"{row['input_k']}K / BS{row['bs']}" for row in rows})
    row_index = {key: index for index, key in enumerate(row_keys)}
    rate_index = {rate: index for index, rate in enumerate(all_rates)}
    cell_w = (plot_right - plot_left) / max(1, len(all_rates))
    cell_h = (plot_bottom - plot_top) / max(1, len(row_keys))
    values = {}
    for row in rows:
        key = f"{row['input_k']}K / BS{row['bs']}"
        rate = float(row["request_rate"])
        values[(key, rate)] = float(row[metric]) / 1000.0

    for key, index in row_index.items():
        yy = plot_top + index * cell_h
        draw.text((plot_left - 12, yy + cell_h / 2), key, font=fonts["small"], fill=COLORS["axis"], anchor="ra")
        for rate, rate_pos in rate_index.items():
            xx = plot_left + rate_pos * cell_w
            value = values.get((key, rate))
            fill = (245, 246, 248) if value is None else heat_color(value)
            draw.rectangle((xx + 2, yy + 2, xx + cell_w - 2, yy + cell_h - 2), fill=fill, outline="white", width=2)
            if value is not None:
                text_color = "white" if value >= 6 else COLORS["text"]
                draw.text((xx + cell_w / 2, yy + cell_h / 2), f"{value:.2f}s", font=fonts["small"], fill=text_color, anchor="mm")
            draw.text((xx + cell_w / 2, plot_bottom + 10), f"{rate:g}", font=fonts["small"], fill=COLORS["axis"], anchor="ma")


def draw_heatmap_view(rows, output, args):
    width, height = 1800, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = {
        "title": font(args.font, 34, True),
        "subtitle": font(args.font, 24, True),
        "small": font(args.font, 17),
        "legend": font(args.font, 19),
    }
    draw.text((58, 28), "811 Prefill TTFT 热力图", font=fonts["title"], fill=COLORS["text"])
    draw.text((60, 76), "绿色代表更接近 SLA，黄色为 3s 附近，橙红色表示明显超时；输出长度=1，flush-cache", font=fonts["small"], fill=COLORS["axis"])
    rates = sorted({float(row["request_rate"]) for row in rows})
    draw_heatmap(draw, (60, 124, 1740, 402), rows, "mean_ttft_ms", "Mean TTFT", fonts, rates)
    draw_heatmap(draw, (60, 438, 1740, 716), rows, "p99_ttft_ms", "P99 TTFT", fonts, rates)
    legend_y = height - 24
    x0 = 1160
    for seconds, label in ((1, "<=2s"), (2.5, "约3s"), (6, ">6s"), (12, ">=12s")):
        draw.rectangle((x0, legend_y - 12, x0 + 25, legend_y + 12), fill=heat_color(seconds))
        draw.text((x0 + 34, legend_y), label, font=fonts["legend"], fill=COLORS["text"], anchor="lm")
        x0 += 140
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)
    print(output)


def capacity_color(value, low, high):
    start = (164, 205, 226)
    end = (0, 103, 55)
    if high <= low:
        ratio = 1.0
    else:
        ratio = (value - low) / (high - low)
    ratio = max(0.0, min(1.0, ratio))
    return tuple(round(a + (b - a) * ratio) for a, b in zip(start, end))


def input_label(input_k):
    label = str(input_k).upper()
    if not label.endswith("K"):
        label += "K"
    return label


def draw_capacity_matrix(rows, output, args, targets):
    metric = args.metric
    cases = sorted(
        {(row["input_k"], row["bs"], row["case"]) for row in rows},
        key=lambda item: (float(item[0]), int(item[1])),
    )
    case_metrics = {
        case: next(
            (row.get("target_metric") for row in rows if row["case"] == case and row.get("target_metric")),
            "",
        )
        for _, _, case in cases
    }
    columns = [
        {"key": case, "label": f"{input_label(input_k)} / BS{bs}"}
        for input_k, bs, case in cases
    ]
    if metric == "auto" and args.targets.strip().lower() == "auto":
        target_specs = sorted(
            {
                (float(row["target_value_ms"]) / 1000.0, row["target_metric"])
                for row in rows
                if row.get("target_value_ms", "") and row.get("target_metric", "")
            },
            key=lambda item: (item[0], item[1]),
        )
    else:
        target_specs = [(target, None if metric == "auto" else metric) for target in targets]

    def cell_value(row):
        if args.cell == "rps":
            raw = row.get("actual_rps", "")
        else:
            raw = row.get("input_throughput", "") or row.get("input_token_throughput", "")
            if not raw:
                try:
                    raw = float(row["total_input_tokens"]) / float(row["benchmark_duration_s"])
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def metric_passes(row, target, target_metric):
        try:
            return bool(row.get(target_metric, "")) and float(row[target_metric]) <= target * 1000
        except (TypeError, ValueError):
            return False

    matrix = {}
    for target, requested_metric in target_specs:
        for column in columns:
            cell_metric = requested_metric or case_metrics[column["key"]]
            if requested_metric and case_metrics[column["key"]] != requested_metric:
                matrix[(target, requested_metric, column["key"])] = None
                continue
            candidates = [
                value
                for row in rows
                if row["case"] == column["key"]
                and metric_passes(row, target, cell_metric)
                for value in [cell_value(row)]
                if value is not None
            ]
            matrix[(target, requested_metric, column["key"])] = max(candidates) if candidates else None

    values = [value for value in matrix.values() if value is not None]
    low = min(values) if values else 0.0
    high = max(values) if values else 1.0
    width = max(1500, 300 + len(columns) * 330 + 50)
    height = max(860, 260 + len(target_specs) * 132 + 120)
    image = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(image)
    fonts = {
        "title": font(args.font, 38, True),
        "subtitle": font(args.font, 23),
        "header": font(args.font, 28, True),
        "row": font(args.font, 25),
        "cell": font(args.font, 32, True),
        "small": font(args.font, 17),
    }
    draw.text((width / 2, 42), "Prefill TTFT Capacity Matrix", font=fonts["title"], fill=(35, 40, 48), anchor="ma")
    if metric == "auto":
        metric_label = "case-specific TTFT"
    else:
        metric_label = metric.replace("_ms", "").replace("_", " ").title().replace("Ttft", "TTFT")
    if args.cell == "rps" and metric == "auto":
        subtitle = "Maximum actual client RPS under each case's TTFT target"
    elif args.cell == "rps":
        subtitle = f"Maximum actual client RPS under {metric_label} target"
    else:
        subtitle = f"Maximum client input token throughput under {metric_label} target"
    draw.text((width / 2, 92), subtitle, font=fonts["subtitle"], fill=(98, 108, 122), anchor="ma")

    left = 300
    top = 210
    cell_w = 330
    cell_h = 132
    draw.text((70, top - 42), metric_label, font=fonts["header"], fill=(42, 48, 58))
    for index, column in enumerate(columns):
        xx = left + index * cell_w + cell_w / 2
        label = column["label"]
        if metric == "auto":
            case_metric = case_metrics[column["key"]]
            metric_short = case_metric.replace("_ttft_ms", "").upper() if case_metric else "SLA"
            label = f"{label}\n{metric_short}"
            draw.multiline_text((xx, top - 58), label, font=fonts["small"], fill=(63, 72, 85), anchor="ma", align="center", spacing=2)
        else:
            draw.text((xx, top - 42), label, font=fonts["header"], fill=(63, 72, 85), anchor="ma")

    for row_index, (target, requested_metric) in enumerate(target_specs):
        yy = top + row_index * cell_h
        if metric == "auto" and requested_metric:
            metric_short = requested_metric.replace("_ttft_ms", "").upper()
            row_label = f"{metric_short} {target:g} s"
        else:
            row_label = f"{target:g} s"
        draw.text((left - 62, yy + cell_h / 2), row_label, font=fonts["row"], fill=(63, 72, 85), anchor="ra")
        for col_index, column in enumerate(columns):
            xx = left + col_index * cell_w
            value = matrix[(target, requested_metric, column["key"])]
            fill = (232, 235, 239) if value is None else capacity_color(value, low, high)
            draw.rounded_rectangle((xx, yy, xx + cell_w - 22, yy + cell_h - 22), radius=10, fill=fill)
            if value is None:
                label = "--"
            elif args.cell == "rps":
                label = f"{value:.2f}"
            else:
                label = f"{value:.0f}"
            text_color = (35, 48, 62) if value is None or (value is not None and value < (low + high) / 2) else "white"
            draw.text((xx + (cell_w - 22) / 2, yy + (cell_h - 22) / 2), label, font=fonts["cell"], fill=text_color, anchor="mm")

    draw.text((width / 2, top + len(target_specs) * cell_h + 28), "Input length / batch size", font=fonts["header"], fill=(42, 48, 58), anchor="ma")
    if args.cell == "rps":
        note = "数值为满足目标的最大实际客户端 RPS；-- 表示当前已测数据没有满足该 SLA 的点"
    else:
        note = "数值为满足目标的客户端输入 token 吞吐（tok/s）；-- 表示当前已测数据没有满足该 SLA 的点"
    draw.text((width / 2, height - 34), note, font=fonts["small"], fill=(98, 108, 122), anchor="ma")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--font", default=None)
    parser.add_argument("--kind", choices=("line", "heatmap", "capacity"), default="line")
    parser.add_argument("--metric", default="mean_ttft_ms", help="TTFT 指标，或 auto 按每个 case 的 target_metric 使用各自 SLA")
    parser.add_argument("--targets", default="3,5,10", help="秒数列表，或 auto 从 CSV 中按 metric 自动读取")
    parser.add_argument(
        "--cell",
        choices=("rps", "input_throughput", "input_tp"),
        default="rps",
        help="格子中央显示的指标；input_tp 是 input_throughput 的简写",
    )
    args = parser.parse_args()
    if args.cell == "input_tp":
        args.cell = "input_throughput"
    rows = read_rows(args.csv)
    groups = grouped(rows)
    if not groups:
        raise SystemExit("CSV 中没有可绘图数据")

    if args.kind == "heatmap":
        output = args.output or args.csv.with_name(args.csv.stem + "_heatmap.png")
        draw_heatmap_view(rows, output, args)
        return
    if args.kind == "capacity":
        output = args.output or args.csv.with_name(args.csv.stem + "_capacity_matrix.png")
        if args.targets.strip().lower() == "auto":
            target_values = {
                float(row["target_value_ms"]) / 1000.0
                for row in rows
                if row.get("target_value_ms", "")
                and (args.metric == "auto" or row.get("target_metric") == args.metric)
            }
            targets = sorted(target_values) or [3.0, 5.0, 10.0]
        else:
            targets = [float(value) for value in args.targets.replace(" ", ",").split(",") if value]
        draw_capacity_matrix(rows, output, args, targets)
        return

    width, height = 1800, 1120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    fonts = {
        "title": font(args.font, 34, True),
        "subtitle": font(args.font, 24, True),
        "small": font(args.font, 17),
        "legend": font(args.font, 19),
    }
    draw.text((58, 30), "811 Prefill TTFT 压测", font=fonts["title"], fill=COLORS["text"])
    draw.text((60, 78), "输出长度=1，flush-cache；横轴为请求率，纵轴为 TTFT", font=fonts["small"], fill=COLORS["axis"])

    panels = list(groups.items())
    panel_w = (width - 180) // 2
    panel_h = 430
    for index, (case, values) in enumerate(panels):
        col, row = index % 2, index // 2
        left = 60 + col * (panel_w + 40)
        top = 130 + row * (panel_h + 42)
        title = f"{case}（{values[0]['input_k']}K / BS{values[0]['bs']}）"
        draw_panel(draw, (left, top, left + panel_w, top + panel_h), values, title, fonts)

    legend_y = height - 28
    x0 = 70
    for color, label in ((COLORS["mean"], "Mean TTFT"), (COLORS["p99"], "P99 TTFT"), (COLORS["sla2"], "SLA 2s"), (COLORS["sla3"], "SLA 3s")):
        draw.line((x0, legend_y, x0 + 28, legend_y), fill=color, width=4)
        draw.text((x0 + 38, legend_y), label, font=fonts["legend"], fill=COLORS["text"], anchor="lm")
        x0 += 190

    output = args.output or args.csv.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)
    print(output)


if __name__ == "__main__":
    main()
