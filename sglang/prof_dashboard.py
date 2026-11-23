#!/usr/bin/env python3
"""prof_analyze.py 产物 → 自包含 HTML 可视化看板 (无第三方依赖)。

用法:
    python3 sglang/prof_dashboard.py <结果目录> [--mode decode] [--num-steps 10]
                                     [--bs 32] [--case 4096o1024h0] [--out prof_dashboard.html]

输入 (由 prof_analyze.py 在同目录生成):
    <dir>/prof_kernels.csv   kernel,count,total_ms,mean_us,share_pct
    <dir>/prof_ranks.csv     file,span_s,gpu_busy_s,busy_pct,events,parse_s

可选输入 (自动探测, 用于端到端对账面板):
    <dir>/driver.jsonl           抓取轮驱动输出 (bench_serving / prefix_hit.py)
    <dir>/perf_pass/driver.jsonl 第一轮性能基准 (--perf-first)

对账口径:
    decode : E2E 隐含 round = 客户端 Mean TPOT × 实测 accept_length, 对比 span/num_steps
    prefill: 客户端 Mean TTFT 对比 span/num_steps (每 pass wall)
    经验签名: prof wall 比 E2E 推算值高 ~12~17% (profiler 固定开销), busy 高 ~3~8%。

输出:
    <dir>/prof_dashboard.html
"""
import argparse
import csv
import json
import math
import os
import re
import sys

PAL = ["#5b8ff9", "#5ad8a6", "#f6bd16", "#e8684a", "#6dc8ec", "#9270ca",
       "#5d7092", "#f4a261", "#2ec4b6", "#ef767a"]
CIRC = 2 * math.pi * 80.0  # donut r=80


# ---------------- 数据读取 ----------------

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_driver(path):
    """返回 (汇总dict, 行数)。兼容两种形态: 每请求一行 / 单行汇总。"""
    if not os.path.isfile(path):
        return None, 0
    recs, last = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                recs.append(obj)
                last = obj
    if last is None:
        return None, 0
    if "mean_tpot_ms" in last or "mean_ttft_ms" in last:  # 单行汇总形态
        return last, len(recs)
    # 逐请求形态: 现场聚合
    tpots, ttfts, accepts, itls, outlens = [], [], [], [], []
    for r in recs:
        lat, ttft, n = r.get("latency"), r.get("ttft"), r.get("output_len") or 0
        if lat is not None and ttft is not None and n and n > 1:
            tpots.append((lat - ttft) * 1000.0 / (n - 1))
        if ttft is not None:
            ttfts.append(ttft * 1000.0)
        a = r.get("spec_accept_length") or r.get("accept_length")
        if isinstance(a, (int, float)):
            accepts.append(a)
        iv = r.get("itl")
        if isinstance(iv, list) and len(iv) > 1:
            gaps = iv[1:] if not isinstance(iv[0], list) else iv[0][1:]
            itls.extend(g for g in gaps if isinstance(g, (int, float)))
        if n:
            outlens.append(n)
    def pctl(xs, p):
        if not xs:
            return None
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(len(xs) * p))]
    m = {
        "mean_tpot_ms": sum(tpots) / len(tpots) if tpots else None,
        "mean_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else None,
        "accept_length": sum(accepts) / len(accepts) if accepts else None,
        "p99_itl_ms": pctl(itls, 0.99),
        "median_itl_ms": pctl(itls, 0.50),
        "output_tok_per_s": None,
    }
    return m, len(recs)


def gather_client(dirpath):
    """基准轮优先, 其次抓取轮。返回 (标签, 指标dict, 记录数)。"""
    pp, n1 = read_driver(os.path.join(dirpath, "perf_pass", "driver.jsonl"))
    if pp and (pp.get("mean_tpot_ms") or pp.get("mean_ttft_ms")):
        return "perf 第一轮 (无 profiler)", pp, n1
    dr, n2 = read_driver(os.path.join(dirpath, "driver.jsonl"))
    if dr and (dr.get("mean_tpot_ms") or dr.get("mean_ttft_ms")):
        return "抓取轮 (profiler 开启)", dr, n2
    return "", None, 0


# ---------------- kernel 分类 ----------------

def categorize(name):
    n = name.lower()
    if "grouped_fp8_gemm" in n:
        if "masked" in n:
            return "MoE grouped GEMM (masked)"
        if "contiguous" in n:
            return "MoE grouped GEMM (contiguous)"
        return "MoE grouped GEMM"
    if "deep_ep" in n:
        return "deep_ep 通信"
    if "sparse_attn" in n or "mla" in n or "attn" in n or "attention" in n or "fmha" in n:
        return "Attention (decode)" if ("mla" in n or "headshare" in n or "decode" in n) else "Attention"
    if "cijk" in n or "gemm" in n:
        return "Dense GEMM"
    if "memcpy" in n or "memset" in n:
        return "Memcpy/Memset"
    for k in ("rotary", "rms", "act_and_mul", "quant", "gate", "topk", "softmax",
              "gather", "allgather", "all_reduce", "allreduce", "concat", "copy",
              "cast", "norm", "index", "arange", "fill", "cache", "every"):
        if k in n:
            return "逐元素/辅助/集合通信"
    return "其他 kernel"


# ---------------- HTML 生成 ----------------

def cards(cards_list):
    cells = "".join(
        f'<div class="card"><b>{v}</b><i>{i}</i></div>' for v, i in cards_list)
    return f'<div class="cards">{cells}</div>'


def bar(pct, width=280, color=PAL[0]):
    pct = max(0.0, min(100.0, pct))
    return (f'<span class="bar" style="width:{width}px">'
            f'<span style="width:{pct:.2f}%;background:{color}"></span></span>')


def short(name, limit=52):
    n = name.strip()
    return n[:limit] + ("…" if len(n) > limit else "")


def donut(items, center_top, center_bot):
    segs, off = [], 0.0
    for i, (_, _, pct) in enumerate(items):
        ln = CIRC * pct / 100.0
        segs.append(f'<circle cx="110" cy="110" r="80" fill="none" stroke="{PAL[i % len(PAL)]}" '
                    f'stroke-width="34" stroke-dasharray="{ln:.1f} {CIRC:.1f}" '
                    f'stroke-dashoffset="{-off:.1f}"/>')
        off += ln
    body = "".join(segs)
    return (f'<svg width="260" height="260" viewBox="0 0 220 220">'
            f'<g transform="rotate(-90 110 110)">{body}</g>'
            f'<text x="110" y="105" text-anchor="middle" fill="#fff" font-size="20" font-weight="bold">{center_top}</text>'
            f'<text x="110" y="125" text-anchor="middle" fill="#94a3b8" font-size="11">{center_bot}</text></svg>')


def fmt(v, nd=1, suf=""):
    if v is None:
        return "-"
    return f"{v:.{nd}f}{suf}"


def verdict(delta):
    if delta is None:
        return '<span style="color:#94a3b8">无数据</span>'
    if -0.05 <= delta <= 0.25:
        return f'<span style="color:#5ad8a6"><b>{delta * 100:+.1f}% ✓</b> 对账闭环</span>'
    return (f'<span style="color:#f6bd16"><b>{delta * 100:+.1f}% △</b></span>'
            f'<span style="color:#94a3b8"> 偏离签名(±12~17%), 检查窗口稳态</span>')


def build(args):
    d = os.path.abspath(args.path)
    kc = os.path.join(d, "prof_kernels.csv")
    rc = os.path.join(d, "prof_ranks.csv")
    if not (os.path.isfile(kc) and os.path.isfile(rc)):
        print(f"[ERROR] 缺少 prof_kernels.csv / prof_ranks.csv: {d}\n"
              f"        先执行 prof_analyze.py {d} --mode {args.mode}", file=sys.stderr)
        return 2
    krows = read_csv(kc)
    rrows = read_csv(rc)
    # 可选: 调度/通信扩展数据 (新版 prof_analyze 自动生成, 同目录)
    sched_rows, gap_rows = [], []
    sp_path = os.path.join(d, "prof_sched.csv")
    gp_path = os.path.join(d, "prof_gaps_top.csv")
    if os.path.isfile(sp_path):
        sched_rows = read_csv(sp_path)
        if sched_rows and "gaps_total_ms" not in sched_rows[0]:
            sched_rows = []  # 列不匹配 (文件损坏/旧格式) → 按缺数据降级
    if os.path.isfile(gp_path):
        gap_rows = read_csv(gp_path)
        if gap_rows and "rel_ms" not in gap_rows[0]:
            gap_rows = []
    ktotal = sum(float(r["total_ms"]) for r in krows) / 1000.0  # s
    spans = [float(r["span_s"]) for r in rrows]
    busys = [float(r["gpu_busy_s"]) for r in rrows]
    span_avg = sum(spans) / len(spans)
    busy_avg = sum(busys) / len(busys)
    busy_pct_avg = busy_avg / span_avg * 100.0 if span_avg else 0.0
    steps = args.num_steps or (5 if args.mode == "prefill" else 10)
    wall_step = span_avg / steps * 1000.0          # 整窗口径 (对账基线, 含窗口边界空档)
    busy_step = busy_avg / steps * 1000.0
    # 有效窗口 (去窗口边界空档): 只用于推导面板 TPOT/吞吐, 对账面板仍用整窗 wall
    act_avg = span_avg
    if sched_rows and "span_active_ms" in sched_rows[0]:
        acts = [float(r["span_active_ms"]) / 1000.0 for r in sched_rows]
        if acts:
            act_avg = sum(acts) / len(acts)
    act_step = act_avg / steps * 1000.0
    bnd_avg = max(span_avg - act_avg, 0.0)

    lab, cli, nrec = gather_client(d)
    # 对账
    implied = delta_w = delta_b = None
    cli_rows = ""
    if cli:
        tpot = cli.get("mean_tpot_ms")
        ttft = cli.get("mean_ttft_ms")
        acc = cli.get("accept_length")
        if args.mode == "decode" and tpot and acc:
            implied = tpot * acc
            basis = f"TPOT {tpot:.2f} ms × accept {acc:.2f}"
        elif args.mode != "decode" and ttft:
            implied = ttft
            basis = f"Mean TTFT {ttft:.0f} ms"
        else:
            basis = None
        if implied:
            delta_w = wall_step / implied - 1.0
            delta_b = busy_step / implied - 1.0
            itl99 = cli.get("p99_itl_ms")
            cli_rows = (f"<tr><td>基准来源</td><td>{lab}</td>"
                        f"<td>{basis}<br><span style='color:#94a3b8'>ITL p99 {fmt(itl99, 1, ' ms')} · {nrec} 条记录</span></td></tr>"
                        f"<tr><td>E2E 隐含 round</td><td><b>{implied:.1f} ms</b></td>"
                        f"<td>{bar(min(implied / max(wall_step, implied) * 100.0, 100.0), 300, PAL[1])}</td></tr>"
                        f"<tr><td>prof wall / round</td><td><b>{wall_step:.1f} ms</b></td>"
                        f"<td>Δ {verdict(delta_w)}</td></tr>"
                        f"<tr><td>prof GPU busy / round</td><td>{busy_step:.1f} ms</td>"
                        f"<td>Δ {verdict(delta_b)}</td></tr>")
        else:
            cli_rows = ("<tr><td>驱动数据</td><td colspan='2' style='color:#94a3b8'>"
                        "client 记录缺少对账所需字段 (decode 需 TPOT×accept, prefill 需 TTFT)</td></tr>")
    else:
        cli_rows = ("<tr><td colspan='3' style='color:#94a3b8'>目录内无 driver.jsonl"
                    " (加 --perf-first 重抓或补拷康基准轮数据后重新生成)</td></tr>")

    # 类目
    cats = {}
    for r in krows:
        c = categorize(r["kernel"])
        cats[c] = cats.get(c, 0.0) + float(r["total_ms"]) / 1000.0
    cat_items = sorted(cats.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(v for _, v in cat_items) or 1.0
    cat_pie = [(k, v, v / total * 100.0) for k, v in cat_items]
    cat_rows = "".join(
        f"<tr><td><span style='color:{PAL[i % len(PAL)]}'>■</span></td><td>{k}</td>"
        f"<td>{pct:.1f}%</td><td>{v:.2f} s</td></tr>"
        for i, (k, v, pct) in enumerate(cat_pie))

    # rank 表
    max_pct = max((float(r["busy_pct"]) for r in rrows), default=0.0)
    rank_rows = ""
    for i, r in enumerate(rrows):
        p = float(r["busy_pct"])
        if max_pct - p > 1.5:
            flag = " <b style='color:#fca5a5'>⚠</b>"
            color = PAL[3]
        else:
            flag = ""
            color = PAL[1]
        rank_rows += (f"<tr><td>{short(os.path.basename(r['file']), 40)}</td>"
                      f"<td>{float(r['span_s']):.3f}</td><td>{float(r['gpu_busy_s']):.3f}</td>"
                      f"<td>{bar(p, 180, color)} {p:.1f}%{flag}</td>"
                      f"<td>{r['events']}</td></tr>")

    # 空泡 (GPU idle) 诊断与改进
    bubbles = [float(r["span_s"]) - float(r["gpu_busy_s"]) for r in rrows]
    bubble_avg = sum(bubbles) / len(bubbles) if bubbles else 0.0
    bubble_pct = 100.0 - busy_pct_avg
    # 有效空泡 = 去掉窗口边界空档 (采集起停+调度残留) 后的空泡率
    bubble_eff = max(act_avg - busy_avg, 0.0) / act_avg * 100.0 if act_avg else 0.0

    def _grade_s(lv):
        if lv < 5:
            return '<span style="color:#5ad8a6"><b>✓ 健康 (&lt;5%)</b></span>'
        if lv < 15:
            return '<span style="color:#f6bd16"><b>△ 关注 (5~15%)</b></span>'
        return '<span style="color:#fca5a5"><b>⚠ 异常 (&gt;15%)</b></span>'

    bl_flag = _grade_s(bubble_eff)
    sched_map = {r["file"]: r for r in sched_rows}
    bubble_rows = ""
    for r in rrows:
        f = os.path.basename(r["file"])
        sr = sched_map.get(f)
        act_s = (float(sr["span_active_ms"]) / 1000.0
                 if sr and "span_active_ms" in sr else float(r["span_s"]))
        eff = max(act_s - float(r["gpu_busy_s"]), 0.0) / act_s * 100.0 if act_s else 0.0
        bubble_rows += (f"<tr><td>{short(f, 40)}</td>"
                        f"<td>{(float(r['span_s']) - float(r['gpu_busy_s'])) * 1000:.0f}</td>"
                        f"<td>{100.0 - float(r['busy_pct']):.1f}%</td>"
                        f"<td>{eff:.1f}%</td>"
                        f"<td>{bar(eff, 180, PAL[3])}</td></tr>")
    caveat = ("<li>CC 抓取方式的本底放大: CUPTI 拖慢调度, kernel launch 间隙全部计入空泡; "
              "若服务端以 --disable-cuda-graph 抓取 (prof 与图不兼容时常见), 该放大更明显。"
              "prof 空泡天然高于 E2E 开图实况, 只做相对比较, 不直接外推 E2E 空泡率。</li>")
    bubble_panel = (
        '<table><tr><th>rank 文件</th><th>空泡 ms/rank</th><th>整窗占比</th>'
        '<th>去边界有效占比</th><th style="width:32%"></th></tr>' + bubble_rows + "</table>"
        '<div class="note" style="margin-bottom:6px">判级: 整窗 '
        f'{bubble_pct:.1f}% / 去窗口边界空档后 <b>{bubble_eff:.1f}%</b> (每 rank 边界空档 {bnd_avg * 1000:.1f} ms)'
        f' → {bl_flag}</div>'
        '<table><tr><th>Perfetto 空档形态</th><th>典型归因</th><th>改进方向</th></tr>'
        "<tr><td>密集短空档 (几十 µs, 前后无相邻 kernel)</td><td>CPU 调度 / launch gap</td>"
        "<td>E2E 开启 CUDA Graph / torch.compile; 已开图则认定为本抓取的放大假象, 不处理</td></tr>"
        "<tr><td>长空档紧邻 deep_ep / all-reduce, 个别 rank 空泡显著偏大 (④ 中 ⚠ rank)</td>"
        "<td>TP/EP 同步等待, MoE 负载不均</td><td>two-batch overlap (tbo) 通算重叠; "
        "MoE 负载均衡 / 专家路由热度核查</td></tr>"
        "<tr><td>空档带内混有大量 memcpy/memset</td><td>KV cache 换页 / 显存管理</td>"
        "<td>调大 gpu-memory-utilization, 复查 swap / PD 迁移配置</td></tr>"
        "<tr><td>空档集中在 draft 与 verify 交替处 (decode)</td><td>MTP/spec 解码切换开销</td>"
        "<td>核对 speculative 配置, 联调 draft 步数与 accept 长度</td></tr>"
        "<tr><td>窗口边界空档 (采集起停 + 调度残留)</td><td>profiler 起停 + 上一批残留</td>"
        "<td>已由 span_active 自动扣除 (⑦ 中 band=B), 不计入每步耗时; 加大 --num-steps 可摊薄</td></tr></table>"
        '<div class="note"><ul style="margin:4px 0 0 18px">' + caveat +
        "<li>经验参考: 本集群 decode (per-DP) 空泡约 3% 属计算密集健康态; prefill 的 "
        "deep_ep 占 kernel 累计比例高, tbo (通算重叠) 收益最大; 个别 rank 空泡显著偏大时"
        "先怀疑该 rank 自身调度/降频/混部, 不要一律归为同步等待。</li>"
        "<li>注意: 通信 kernel 时长含「早到者」在集体通信上的自旋等待, comm 占比高更可能是<b>快 rank</b>; "
        "慢 rank 应以 busy 最低 + 内部空档最多 (band=C/G) 识别, 不要用 comm 高低排序定慢侧。</li></ul></div>")

    # 实测数据驱动的空泡改进优先级 (依赖 prof_sched.csv / prof_gaps_top.csv)
    def _grade_html(p):
        if p < 5:
            return '<span style="color:#5ad8a6">✓ 健康</span>'
        if p < 15:
            return '<span style="color:#f6bd16">△ 关注</span>'
        return '<span style="color:#fca5a5">⚠ 异常</span>'

    tips = []
    span_ms = span_avg * 1000.0
    if sched_rows:
        nr = len(sched_rows)
        g_tot = sum(float(r["gaps_total_ms"]) for r in sched_rows)
        has_band = bool(gap_rows) and "band" in gap_rows[0]
        # 窗口边界空档: 优先用 prof_sched 的 head/tail 列 (精确), 旧数据回退 band=B, 再回退 0
        bnd_ms = 0.0
        if "head_gap_ms" in sched_rows[0]:
            bnd_ms = sum(float(r["head_gap_ms"]) + float(r["tail_gap_ms"]) for r in sched_rows)
        elif has_band:
            bnd_ms = sum(float(g["dur_ms"]) for g in gap_rows if g.get("band") == "B")
        # 内部紧邻通信的空档 = 同步等待线索 (band=C); 旧数据无该列时置 0
        sync_ms = sum(float(g["dur_ms"]) for g in gap_rows
                      if g.get("band") == "C" and float(g["dur_ms"]) >= 0.5) if has_band else 0.0
        short_ms = sum(float(r["b0_ms"]) + float(r["b1_ms"]) for r in sched_rows)
        real_tot = max(g_tot - bnd_ms, 0.0)
        short_sh = min(short_ms / real_tot, 1.0) if real_tot > 0 else 0.0
        sync_sh = min(sync_ms / real_tot, 1.0) if real_tot > 0 else 0.0
        adj_pct = bubble_eff
        head = f"判级基准: 整窗空泡 <b>{bubble_pct:.1f}%/rank</b>, 去窗口边界后有效 <b>{adj_pct:.1f}%</b> → {_grade_html(adj_pct)}"
        if bnd_ms > 0.5:
            head += (f"；窗口边界空档 ≈ <b>{bnd_ms / nr:.1f} ms/rank</b>"
                     f" (profiler 起停 + 调度残留, 已扣除; 见 ⑦ band=B)")
        tips.append(head + "。")
        if real_tot < 0.5 * nr:
            tips.append("扣除窗口边界空档后每 rank 有效空档 &lt;0.5 ms: 空泡侧无需改造, 保持回归盯防即可。")
        elif adj_pct >= 5 and short_sh >= 0.5 and sync_sh < 0.25:
            tips.append(f"<b>优先级 1 · launch gap 主导</b> (密集 &lt;100µs 短空档 {short_ms:.0f} ms, "
                        f"占有效空档 {short_sh:.0%}): E2E 确认 CUDA Graph 开启 (抓取态关图会放大此形态, "
                        "量级以 E2E 回归为准); 进一步 --enable-torch-compile 与 CPU 独占核/numactl 绑核。")
        elif adj_pct >= 5:
            tips.append(f"<b>优先级 1 · 同步等待待解</b> (内部紧邻通信的空档 {sync_ms:.1f} ms, "
                        f"占有效空档 {sync_sh:.0%}): 优先试 --enable-two-batch-overlap (tbo)"
                        " 通算重叠; MoE 场景做专家负载均衡 (EPLB/路由热度); 慢 rank 按 busy 最低 + "
                        "内部空档最多识别, 再单查 dcu-smi 降频、他人占卡与 NUMA 绑定。")
        else:
            txt = (f"空档形态: 短空档 {short_sh:.0%} / 通信相邻空档 {sync_sh:.0%}, 有效量级 "
                   f"{adj_pct:.1f}% < 5% → 无需为空泡专门改造")
            if sync_ms >= 1:
                txt += "；通信相邻部分与 deep_ep/all-reduce 优化合并为 tbo 一并评估"
            tips.append(txt + "。")
        # 通信暴露 (未与算力重叠) 汇总: 串行流下≈100%, 即 tbo 可覆盖上限
        ex_tot = sum(float(r["comm_exposed_ms"]) for r in sched_rows if "comm_exposed_ms" in r)
        cm_tot = sum(float(r["comm_ms"]) for r in sched_rows)
        act_tot = sum(float(r["span_active_ms"]) for r in sched_rows) if "span_active_ms" in sched_rows[0] else 0.0
        if cm_tot > 0 and act_tot > 0:
            ex_sh = ex_tot / cm_tot
            tips.append(f"通信暴露 (未与算力重叠, 全 rank 合计) {ex_tot:.0f} ms = 通信总量 {ex_sh:.0%}, "
                        f"占有效窗口 {ex_tot / act_tot:.1%}"
                        + (" → 通算几乎零重叠, tbo 可覆盖上限即该比例" if ex_sh >= 0.98 else
                           f" → 已有 {1 - ex_sh:.0%} 通算重叠, tbo 剩余收益以暴露部分为上限"))
        de = cats.get("deep_ep 通信", 0.0)
        if ktotal and de / ktotal >= 0.05:
            tips.append(f"deep_ep 通信占 kernel 累计 {de / ktotal:.0%}: 通算 overlap 是该项直接抓手 — "
                        "tbo 开启后 dispatch/combine 空档多数可被计算覆盖; combine 显著大于 dispatch 时 "
                        "EPLB/专家迁移亦有收益。")
        gs = [float(r["gaps_total_ms"]) for r in sched_rows]
        g_avg = sum(gs) / len(gs)
        if max(gs) - min(gs) > max(2.0, 0.3 * g_avg):
            slow = sched_rows[gs.index(max(gs))]["file"]
            tips.append(f"rank 空档不均: 最短 {min(gs):.1f} ms vs 最长 {max(gs):.1f} ms "
                        f"({short(slow, 30)}): 若最长者同时 busy 最低则为慢侧源点; "
                        "若其空档主要是边界/band=B 则只是起停残留差异, 不能据此定慢 rank。")
    else:
        tips.append("本目录暂无 prof_sched.csv: 对同一 trace 重跑 prof_analyze.py 即可给出上面的"
                    "实测归因 (空档分桶/band 归因/通信暴露), 期间可先按下方通用方向表排查。")
    memcpy_s = sum(float(r["total_ms"]) for r in krows
                   if "memcpy" in r["kernel"].lower() or "memset" in r["kernel"].lower()) / 1000.0
    if ktotal and memcpy_s / ktotal > 0.03:
        tips.append(f"memcpy/memset 占 kernel 累计 {memcpy_s / ktotal:.0%} (KV 换页/显存管理): "
                    "调大 gpu-memory-utilization, 复查 swap 与 PD 迁移配置。")
    bubble_panel += ('<div class="panel" style="margin-top:12px">'
                     '<b>本窗口实测 → 空泡改进优先级</b>'
                     '<ul style="margin:6px 0 0 18px;font-size:12.5px">'
                     + "".join(f"<li style='margin:3px 0'>{t}</li>" for t in tips)
                     + "</ul></div>")

    # kernel Top10
    top_rows = ""
    for i, r in enumerate(krows[:10]):
        s = float(r["share_pct"])
        top_rows += (f"<tr><td>{i + 1}</td><td title=\"{r['kernel']}\">{short(r['kernel'])}</td>"
                     f"<td>{bar(s, 240, PAL[i % len(PAL)])}</td><td>{int(r['count']):,}</td>"
                     f"<td>{float(r['mean_us']):.0f}µs</td><td>{s:.1f}%</td></tr>")

    # prof 直接推导 TPOT / 吞吐 (prof 本身无 token 口径, 需 accept/输入长度补充)
    deriv_rows = ""
    acc = args.accept or ((cli or {}).get("accept_length") if cli else None)

    def _len0(s):
        s = s.strip().lower()
        if s.endswith("k"):
            return int(float(s[:-1]) * 1024)
        if s.endswith("m"):
            return int(float(s[:-1]) * 1024 * 1024)
        return int(float(s))

    bs_n = 0
    if args.bs:
        try:
            bs_n = _len0(str(args.bs).split(",")[0])
        except ValueError:
            bs_n = 0
    if args.mode == "decode" and acc:
        tpot_w, tpot_b = act_step / acc, busy_step / acc
        deriv_rows += (f"<tr><td>prof 推导 TPOT</td><td><b>{tpot_w:.2f} ms</b>"
                       f"<span style='color:#94a3b8'> = 有效步长 {act_step:.1f} ms ÷ accept {acc:g}"
                       f" (已去窗口边界空档 {bnd_avg * 1000:.1f} ms; busy 口径 {tpot_b:.2f} ms, busy 含通信/memcpy)</span></td><td></td></tr>")
        if bs_n:
            tput = bs_n * acc / (act_step / 1000.0)
            tput_cap = bs_n * acc / (busy_step / 1000.0)
            deriv_rows += (f"<tr><td>prof 推导吞吐</td><td><b>{tput:.0f} tok/s</b>"
                           f"<span style='color:#94a3b8'> = 并发 {bs_n} × accept {acc:g} ÷ {act_step / 1000.0:.3f} s (有效步长)"
                           f" (busy 口径算力上限 {tput_cap:.0f} tok/s)</span></td><td></td></tr>")
    elif args.mode and args.mode != "decode":
        try:
            in_len = _len0((args.case or "0").split(",")[0])
        except ValueError:
            in_len = 0
        if in_len:
            deriv_rows += (f"<tr><td>prof 推导 TTFT</td><td><b>{act_step:.0f} ms</b>"
                           f"<span style='color:#94a3b8'> = 每 pass 有效步长 (去窗口边界 {bnd_avg * 1000:.1f} ms)</span></td><td></td></tr>"
                           f"<tr><td>prof 推导 prefill 吞吐</td><td><b>{in_len / (act_step / 1000.0):.0f} tok/s</b>"
                           f"<span style='color:#94a3b8'> = {in_len} tok ÷ {act_step / 1000.0:.2f} s/pass</span></td><td></td></tr>")
    if not deriv_rows:
        deriv_rows = ("<tr><td>prof 推导 TPOT/吞吐</td><td colspan='2' style='color:#94a3b8'>"
                      "decode 需 accept (实测或 --accept) 与并发 (--bs); prefill 需 --case 含输入长度 (如 4096,1,0)</td></tr>")

    # ⑦ 调度与通信 (prof_sched.csv / prof_gaps_top.csv; 旧产物缺这两个 CSV 时给提示)
    if not sched_rows:
        sched_panel = ('<div class="panel"><div class="note">无 prof_sched.csv / prof_gaps_top.csv '
                       "(旧版 prof_analyze 产物)。对同一 trace 目录重跑 "
                       f"<code>prof_analyze.py {d} --mode {args.mode}</code> 即可补齐, 无需重抓。</div></div>")
    else:
        busy_ms_map = {r["file"]: float(r["gpu_busy_s"]) * 1000.0 for r in rrows}
        has_band = bool(gap_rows) and "band" in gap_rows[0]
        has_active = "span_active_ms" in sched_rows[0]
        rows = ""
        for r in sched_rows:
            gt = float(r["gaps_total_ms"])
            b = [float(r[f"b{i}_ms"]) for i in range(4)]
            bnd = (float(r["head_gap_ms"]) + float(r["tail_gap_ms"])) if has_active else 0.0
            act = float(r["span_active_ms"]) if has_active else 0.0
            eff_gap = max(gt - bnd, 0.0)
            eff_pct = eff_gap / act * 100.0 if act else 0.0
            ex = float(r.get("comm_exposed_ms") or 0.0)
            cm = float(r["comm_ms"])
            long_sh = (b[2] + b[3] - bnd) / eff_gap if eff_gap > 0 else 0.0
            if gt < 0.5:
                note = "几乎无空档 ✓"
            elif eff_gap < 0.5:
                note = "空档全在窗口边界 (起停残留) ✓"
            elif long_sh > 0.5:
                note = "长空洞为主 (内部, 见 Top 表 band)"
            elif long_sh > 0.2:
                note = "混合形态"
            else:
                note = "短空档为主 → launch gap"
            rows += (f"<tr><td>{short(r['file'], 36)}</td>"
                     f"<td>{bar(float(r['gap_pct']), 110, PAL[3])} {float(r['gap_pct']):.1f}%</td>"
                     f"<td>{bnd:.1f}</td>"
                     f"<td>{eff_pct:.1f}%</td>"
                     f"<td>{int(r['b0_n']):,} / {b[0]:.2f}</td>"
                     f"<td>{int(r['b1_n']):,} / {b[1]:.2f}</td>"
                     f"<td>{int(r['b2_n']):,} / {b[2]:.2f}</td>"
                     f"<td>{int(r['b3_n']):,} / {b[3]:.2f}</td>"
                     f"<td>{ex:.1f} / {ex / cm * 100.0 if cm else 0:.0f}%</td>"
                     f"<td>{note}</td></tr>")

        def _tot(col):
            return sum(float(r[col]) for r in sched_rows if col in r)

        comm_tot = _tot("comm_ms")
        # 算力 busy = 全部 GPU busy − 通信 (busy 含通信/memcpy, 整窗口径)
        comp_tot = max(sum(busy_ms_map.values()) - comm_tot, 0.0)
        comm_ratio = comm_tot / comp_tot if comp_tot else 0.0
        ex_tot = _tot("comm_exposed_ms")
        hid_tot = _tot("comm_hidden_ms")
        comm_rows = "".join(
            f"<tr><td>{lab}</td><td>{val:.1f}</td>"
            f"<td>{bar(val / max(comm_tot, 1e-9) * 100.0, 200, PAL[i % len(PAL)])} "
            f"{val / max(comm_tot, 1e-9) * 100.0:.1f}%</td></tr>"
            for i, (lab, val) in enumerate((
                    ("deep_ep dispatch", _tot("comm_dispatch_ms")),
                    ("deep_ep combine", _tot("comm_combine_ms")),
                    ("deep_ep notify / cached_notify", _tot("comm_notify_ms")),
                    ("集合通信 (allreduce/allgather 等)", _tot("comm_other_ms")))))
        if has_active:
            comm_rows += (
                f"<tr><td>├ 暴露 (未与算力/memcpy 重叠)</td><td>{ex_tot:.1f}</td>"
                f"<td>{bar(ex_tot / max(comm_tot, 1e-9) * 100.0, 200, PAL[3])} "
                f"{ex_tot / max(comm_tot, 1e-9) * 100.0:.1f}%</td></tr>"
                f"<tr><td>└ 已隐藏 (与算力重叠)</td><td>{hid_tot:.1f}</td>"
                f"<td>{bar(hid_tot / max(comm_tot, 1e-9) * 100.0, 200, PAL[1])} "
                f"{hid_tot / max(comm_tot, 1e-9) * 100.0:.1f}%</td></tr>")
        grows = ""
        for r in [g for g in sorted(gap_rows, key=lambda x: -float(x["dur_ms"]))
                  if float(g["dur_ms"]) >= 0.5][:12]:
            band = r.get("band") or ""
            if has_band:
                adj = float(r.get("comm_adj_us") or -1)
                if band == "B":
                    flag = "<span style='color:#94a3b8'>窗口边界 (起停/残留)</span>"
                elif band == "C":
                    flag = ("<b style='color:#fca5a5'>⚠ 紧邻通信 → 疑同步等待</b>"
                            + (f" (相距 {adj:.0f}µs)" if adj >= 0 else ""))
                else:
                    flag = "launch gap / 调度抖动"
                cols = (f"<td><b>{band}</b></td><td>{adj:.0f}</td><td>{flag}</td>")
            else:  # 旧产物: 无 band 列, 退回旧的通信重叠口径
                ov = float(r["comm_overlap_pct"])
                cols = (f"<td>—</td><td>—</td><td>{ov:.0f}% 重叠"
                        + (" <b style='color:#fca5a5'>⚠ 通信相邻</b>" if ov >= 50 else "") + "</td>")
            grows += (f"<tr><td>{short(r['file'], 36)}</td>"
                      f"<td>{float(r['rel_ms']):.3f}</td><td>{float(r['dur_ms']):.3f}</td>"
                      + cols + "</tr>")
        if not grows:
            grows = "<tr><td colspan='6' style='color:#94a3b8'>窗口内无 ≥0.5ms 级大空档</td></tr>"
        sched_panel = (
            '<div class="panel"><table>'
            '<tr><th>rank 文件</th><th>空档占 span</th><th>边界空档 ms</th><th>有效空档率</th>'
            '<th>&lt;10µs n/ms</th><th>10~100µs n/ms</th><th>0.1~1ms n/ms</th><th>≥1ms n/ms</th>'
            '<th>通信暴露 ms / 占comm</th><th>形态判读</th></tr>' + rows + "</table>"
            '<table style="margin-top:10px"><tr><th>通信子类 (全 rank 合计)</th><th>ms</th><th>占通信</th></tr>'
            + comm_rows
            + f'<tr><td><b>通信合计</b></td><td>{comm_tot:.0f}</td>'
              f"<td><b>通信/算力比 {comm_ratio:.1%}</b> (算力 = busy − comm)</td></tr></table>"
            + '<table style="margin-top:10px"><tr><th>Top 大空档 (全 rank)</th><th>rel_ms (距窗口起点)</th>'
              '<th>时长 ms</th><th>band</th><th>距通信 µs</th><th>判定</th></tr>' + grows + "</table>"
              '<div class="note">rel_ms: Perfetto 加载对应 rank 的 trace 后从时间 0 平移即达该空档。'
              "band: B=窗口边界 (profiler 起停 + 调度残留, 已从每步耗时扣除); "
              "C=窗口内部且距通信 ≤100µs (跨 rank 同步等待线索); G=其他 (launch gap / 调度抖动)。</div>"
            + '<div class="note"><ul style="margin:4px 0 0 18px">'
              "<li>通信暴露 = 通信 kernel 中未与算力/memcpy 重叠的时长。串行流下暴露≈100% (通算零重叠), "
              f"全 rank 合计暴露 {ex_tot:.0f} ms"
              + (f" 占有效窗口 {ex_tot / _tot('span_active_ms') * 100.0:.1f}%, 即 tbo 可覆盖上限。" if _tot("span_active_ms") else "。")
              + "</li>"
              "<li>通信 kernel 时长含「早到者」在集体通信上的自旋等待, comm_ms 长不等于该 rank 慢; "
              "慢 rank 以 busy 最低 + 内部空档 (band=C/G) 最多识别。</li>"
              "<li>band=C 空档与 deep_ep/all-reduce 相邻 → tbo / 专家负载均衡; "
              "&lt;10µs 密集短空档属 launch gap → 开 CUDA Graph / torch.compile 削减。</li>"
              "</ul></div></div>")

    b_total = ktotal
    # 并行方式: 从 trace 文件名 TP-x-DP-y-EP-z 自动解析各维大小
    def _dim_size(tag):
        vals = set()
        for r in rrows:
            m = re.search(rf"{tag}-(\d+)", os.path.basename(r["file"]))
            if m:
                vals.add(int(m.group(1)))
        return max(vals) + 1 if vals else 0

    par_parts = []
    for t in ("TP", "DP", "EP"):
        n = _dim_size(t)
        if n:
            par_parts.append(f"{t}{n}")
    par = "/".join(par_parts)
    par_chip = f"<span>并行 {par}</span>" if par else ""

    # 运行配置: MTP/spec 解码 + CUDA Graph + tbo, 从 prof_capture 自动保存的
    # server_info.json (POST /server_info 服务端参数快照) 读取; accept 率取 --accept 或 driver 实测
    conf_parts = []
    sa = None
    si_path = os.path.join(d, "server_info.json")
    if os.path.isfile(si_path):
        try:
            with open(si_path, encoding="utf-8-sig") as f:
                si = json.load(f)
            if isinstance(si, dict):
                sa = si.get("server_args") if isinstance(si.get("server_args"), dict) else si
        except (OSError, ValueError):
            sa = None
    if sa:
        alg = str(sa.get("speculative_algorithm") or "")
        nspec = sa.get("speculative_num_steps")
        if alg:
            au = alg.upper()
            if "MTP" in au or "NEXTN" in au:
                conf_parts.append("MTP" + (str(nspec) if nspec else ""))
            else:
                conf_parts.append(alg + (str(nspec) if nspec else ""))
        else:
            conf_parts.append("MTP 关")
        dis = sa.get("disable_cuda_graph")
        en = sa.get("enable_cuda_graph")
        if dis is not None:
            conf_parts.append("CUDA Graph 关(抓取态)" if dis else "CUDA Graph 开")
        elif en is not None:
            conf_parts.append("CUDA Graph 开" if en else "CUDA Graph 关(抓取态)")
        if sa.get("enable_two_batch_overlap"):
            conf_parts.append("tbo 开")
        if sa.get("disable_overlap_schedule"):
            conf_parts.append("overlap调度 关")
    else:
        conf_parts.append("配置未知(server_info.json 缺失)")
    if acc:
        conf_parts.append(f"accept≈{acc:g}")
    conf_chip = "".join(f"<span>{p}</span>" for p in conf_parts)
    html = HTML_FMT.format(
        mode=args.mode or "profile",
        dirname=os.path.basename(d),
        chips=(f"<span>{len(rrows)} rank</span>{par_chip}{conf_chip}<span>steps={steps}</span>"
               + (f"<span>bs={args.bs}</span>" if args.bs else "")
               + (f"<span>case={args.case}</span>" if args.case else "")),
        cards_row=cards([
            (f"{span_avg:.3f} s", "每 rank span"),
            (f"{busy_avg:.3f} s", f"每 rank GPU busy ({busy_pct_avg:.1f}%, 含通信/memcpy)"),
            (f"{b_total:.2f} s", "kernel 累计 (全 rank)"),
            (f"{(span_avg - busy_avg) * len(rrows):.3f} s", "GPU 空档 bubble (合计)"),
            (f"{bnd_avg * 1000:.1f} ms", "窗口边界空档 / rank (已扣除)"),
            (f"{act_step:.1f} ms", f"有效步长 / step ({steps} 步)"),
            (f"{wall_step:.1f} ms", "整窗 wall / step (对账口径)"),
            (f"{busy_step:.1f} ms", "GPU busy / step"),
            (f"{len(krows):,}", "kernel 种类数"),
        ]),
        cli_rows=cli_rows + deriv_rows,
        legend_rows=cat_rows if True else "",
        donut=donut(cat_pie, f"{b_total:.1f} s", "GPU kernel 累计"),
        rank_rows=rank_rows,
        bubble_panel=bubble_panel,
        top_rows=top_rows,
        sched_panel=sched_panel,
    )
    outp = args.out or os.path.join(d, "prof_dashboard.html")
    try:
        with open(outp, "w", encoding="utf-8") as f:
            f.write(html)
    except PermissionError:  # 结果目录常由容器内 root 创建, 回退写到当前目录
        outp = os.path.join(os.getcwd(),
                            f"prof_dashboard_{os.path.basename(d)}.html")
        print(f"[WARN] 目录无写权限, 看板改写到当前目录: {outp}", file=sys.stderr)
        with open(outp, "w", encoding="utf-8") as f:
            f.write(html)
    print(f"[INFO] 看板: {outp}")
    return 0


HTML_FMT = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>Prof 看板 — {dirname}</title>
<style>
:root{{--bg:#0f172a;--card:#1e293b;--ink:#e2e8f0;--mut:#94a3b8;--line:#334155;--acc:#38bdf8}}
*{{box-sizing:border-box;margin:0}}
body{{background:var(--bg);color:var(--ink);font:14px/1.6 "Segoe UI",system-ui,sans-serif;padding:28px}}
h1{{font-size:20px;margin-bottom:2px}}
h2{{font-size:15px;margin:26px 0 10px;color:var(--acc)}}
.sub{{color:var(--mut);font-size:12.5px;margin-bottom:12px}}
.chips span{{display:inline-block;background:var(--card);border:1px solid var(--line);border-radius:4px;padding:2px 10px;margin:0 6px 6px 0;font-size:12px;color:var(--mut)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}}
.card b{{display:block;font-size:22px;color:#fff}}
.card i{{font-style:normal;color:var(--mut);font-size:12px}}
.duo{{display:grid;grid-template-columns:300px 1fr;gap:22px}}
@media(max-width:900px){{.duo{{grid-template-columns:1fr}}}}
.panel{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th{{color:var(--mut);text-align:left;font-weight:normal;border-bottom:1px solid var(--line);padding:4px 8px}}
td{{border-bottom:1px solid #26334a;padding:4px 8px;font-family:Consolas,monospace}}
td:first-child{{font-family:inherit}}
.bar{{height:16px;background:#0b1220;border-radius:3px;overflow:hidden;display:inline-block;vertical-align:middle}}
.bar>span{{display:block;height:100%}}
.note{{color:var(--mut);font-size:12px;margin-top:8px}}
</style></head><body>
<h1>Prof 可视化看板 ({mode})</h1>
<div class="sub">{dirname}</div>
<div class="chips">{chips}</div>
{cards_row}
<h2>① 端到端对账</h2>
<div class="panel"><table>
<tr><th>口径</th><th>值</th><th></th></tr>
{cli_rows}
</table>
<div class="note">经验签名: prof wall ≈ E2E 推算 +12~17%, busy ≈ +3~8% (profiler 固定开销);
Wall 显著低于推算值说明抓取窗口落在 GPU 快速暂态 (如 4/DP acc4.5 的 55ms 快段), 以第一轮 perf 为准绳。</div></div>
<h2>② GPU kernel 类目构成</h2>
<div class="duo"><div class="panel" style="text-align:center">{donut}</div>
<div class="panel"><table><tr><th></th><th>类目</th><th>占比</th><th>时长</th></tr>{legend_rows}</table></div></div>
<h2>③ GPU kernel Top10 (按累计时长)</h2>
<div class="panel"><table>
<tr><th>#</th><th>kernel</th><th></th><th>count</th><th>mean</th><th>share</th></tr>
{top_rows}</table></div>
<h2>④ 各 rank 对称性</h2>
<div class="panel"><table>
<tr><th>文件</th><th>span(s)</th><th>busy(s)</th><th style="width:35%">busy%</th><th>events</th></tr>
{rank_rows}</table>
<div class="note">⚠ = busy 显著低于其他 rank (差距 &gt;1.5pp), kernel 数相同则为同步/通信等待, 建议在 Perfetto 中定位。</div></div>
<h2>⑤ 空泡 (GPU idle) 诊断与改进建议</h2>
{bubble_panel}
<h2>⑥ 交互式时间线</h2>
<div class="panel"><div class="note" style="font-size:13px">
将目录内 <code>*.trace.json.gz</code> 拖入 <b>https://ui.perfetto.dev</b> (或 chrome://tracing)
可缩放查看每个 kernel 的精确时序; {dirname} 数据源为本目录 prof_kernels.csv / prof_ranks.csv。</div></div>
<h2>⑦ 调度与通信 (空档分桶 / 通信子类)</h2>
{sched_panel}
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description="prof 分析产物 → HTML 看板")
    ap.add_argument("path", help="结果目录 (含 prof_kernels.csv / prof_ranks.csv)")
    ap.add_argument("--mode", default="", help="prefill / decode (标题与对账公式)")
    ap.add_argument("--num-steps", type=int, default=0, help="抓取步数 (默认 prefill 5 / decode 10)")
    ap.add_argument("--bs", default="", help="总并发 (展示 + decode 推导吞吐)")
    ap.add_argument("--case", default="", help="case 标签 (展示 + prefill 取输入长度推导吞吐)")
    ap.add_argument("--accept", type=float, default=0,
                    help="decode accept_length (默认取 driver.jsonl 实测值)")
    ap.add_argument("--out", default="", help="输出 HTML 路径 (默认 <dir>/prof_dashboard.html)")
    args = ap.parse_args()
    sys.exit(build(args))


if __name__ == "__main__":
    main()
