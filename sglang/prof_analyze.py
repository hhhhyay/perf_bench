#!/usr/bin/env python3
"""Chrome trace (torch profiler / kineto) 汇总分析。

用法:
    python3 sglang/prof_analyze.py <trace_dir_or_file> [--top 20] [--mode decode] [--jobs 8]

输出:
    stdout                  Markdown 摘要 (GPU busy / kernel TopN / CPU 顶层 op TopN)
    <dir>/prof_summary.md   同上, 落盘
    <dir>/prof_kernels.csv  按 kernel 聚合 (count / total_ms / mean_us / share)
    <dir>/prof_ranks.csv    按 trace 文件 (rank) 聚合
    <dir>/prof_sched.csv    每 rank GPU 空档分桶 (<10us/10~100us/0.1~1ms/>=1ms)
                            + 通信聚合 (deep_ep dispatch/combine/notify、集合通信)
                            + 去窗口边界有效窗口 (head/tail_gap、span_active)
                            + 通信暴露/隐藏等待 (comm_exposed/hidden、bcc_*)
    <dir>/prof_gaps_top.csv 每 rank Top10 大空档 (rel_ms + 归因 band B/C/G + 通信距离 µs)

只依赖标准库, 可在没有 torch 的客户端容器直接跑。

解析策略 (针对 decode 10 步 × 8 rank 这类数十 MB gzip 的 trace):
  - 流式读取, 每 4 MB 解压文本作为一批, 用一次 json.loads 整体反序列化
    (比逐事件 raw_decode 快, 且不会把整个 trace 读进内存);
  - 解析期间关闭 GC, 避免千万级临时对象触发多次分代回收;
  - 批次解析失败时回退到逐事件 raw_decode, 保证鲁棒性;
  - 多个 trace 文件默认并行解析 (--jobs), 每个文件一个进程。
CPU op 事件数量与 torch.compile 相关, 默认每文件只分析前 50 万条,
可用 --cpu-ops-limit 调整 (0 = 不限)。
"""
import argparse
import bisect
import csv
import gc
import glob
import gzip
import json
import os
import re
import sys
import time

_READ_CHUNK = 1 << 22  # 4 MB
_SEP = re.compile(r"\}\s*,\s*\{")
_INTERESTING = ("kernel", "gpu_memcpy", "gpu_memset", "cpu_op")


def _rank_key(path):
    name = os.path.basename(path)
    m = re.search(r"-TP-(\d+)", name)
    if m:
        return (0, int(m.group(1)), name)
    return (1, 0, name)


def iter_trace_files(path):
    if os.path.isfile(path):
        return [path]
    pats = ["*.trace.json.gz", "*.trace.json", "*.json.gz"]
    files = []
    for p in pats:
        files.extend(glob.glob(os.path.join(path, p)))
    return sorted(set(files), key=_rank_key)


def iter_event_chunks(path, chunk=_READ_CHUNK):
    """流式产出 traceEvents 数组中一段完整事件文本, 形如 '{...},{...},...'。

    每段都在事件边界上切分, 外层补上 '[' ']' 即为合法 JSON 数组。
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        buf = ""
        pos = 0
        started = False
        while True:
            more = f.read(chunk)
            if not more and not buf:
                return
            if more:
                buf += more
            if not started:
                i = buf.find('"traceEvents"')
                if i < 0:
                    if not more:
                        return
                    buf = buf[-64:]
                    continue
                a = buf.find("[", i)
                if a < 0:
                    if not more:
                        return
                    continue
                pos = a + 1
                started = True
            ms = list(_SEP.finditer(buf, pos))
            if ms:
                last = ms[-1]
                end = last.start() + 1  # 含最后一个事件的 '}'
                if end > pos:
                    yield buf[pos:end]
                pos = last.end() - 1  # 保留下一个事件的 '{'
                buf = buf[pos:]
                pos = 0
                continue
            if not more:
                tail = buf[pos:].strip()
                if tail.startswith("{"):
                    yield tail
                return


def iter_events(path):
    """逐条产出事件 dict; 批次解析失败时回退到逐事件解析。"""
    dec = json.JSONDecoder()
    for body in iter_event_chunks(path):
        try:
            for o in json.loads("[" + body + "]"):
                yield o
            continue
        except json.JSONDecodeError:
            pass
        i = 0
        n = len(body)
        while i < n:
            while i < n and body[i] in " \t\r\n,":
                i += 1
            if i >= n:
                break
            try:
                o, i = dec.raw_decode(body, i)
            except json.JSONDecodeError:
                break
            yield o


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort()
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    return merged


_COMM_KW = ("deep_ep", "deepep", "all_reduce", "allreduce", "allgather", "reduce_scatter",
            "all_to_all", "alltoall", "cross_device_reduce", "nccl")


def is_comm(name):
    n = name.lower()
    return any(k in n for k in _COMM_KW)


def _uncovered(iv_a, iv_b):
    """iv_a 中未被 iv_b 覆盖的总时长 (两者均已合并、按起点排序)。"""
    if not iv_a:
        return 0.0
    if not iv_b:
        return sum(e - s for s, e in iv_a)
    total = 0.0
    j = 0
    nb = len(iv_b)
    for s, e in iv_a:
        while j < nb and iv_b[j][1] <= s:
            j += 1
        cur = s
        k = j
        while k < nb and iv_b[k][0] < e:
            bs, be2 = iv_b[k]
            if bs > cur:
                total += bs - cur
            if be2 > cur:
                cur = be2
            if cur >= e:
                break
            k += 1
        if cur < e:
            total += e - cur
    return total


def _active_bounds(busy_iv, split_gap=1000.0, min_block=2000.0):
    """有效窗口边界: 剔除窗口首尾的碎片块 (采集起停 + 上一批调度残留)。

    把 busy 时间线按 >split_gap 的空档切块, 首末的孤立小块 (时长 < min_block)
    视为窗口边界残留, 真正的窗口从第一个有实质长度的块开始。
    """
    if not busy_iv:
        return None, None
    blocks = []
    bs, be2 = busy_iv[0][0], busy_iv[0][1]
    for s, e in busy_iv[1:]:
        if s - be2 > split_gap:
            blocks.append((bs, be2))
            bs, be2 = s, e
        elif e > be2:
            be2 = e
    blocks.append((bs, be2))
    solid = [b for b in blocks if b[1] - b[0] >= min_block]
    if not solid:
        return busy_iv[0][0], busy_iv[-1][1]
    return solid[0][0], solid[-1][1]


def classify_comm(name):
    """通信 kernel 子类: dispatch / combine / notify / 其他集合通信。"""
    n = name.lower()
    if "dispatch" in n:
        return "dispatch"
    if "combine" in n:
        return "combine"
    if "notify" in n:
        return "notify"
    return "other"


def cpu_exclusive_time(by_tid):
    """按 (pid, tid) 栈式解析嵌套, 返回 ({op: 独占us}, {op: 次数})。"""
    excl = {}
    cnt = {}
    for evs in by_tid.values():
        evs.sort(key=lambda x: (x[0], -x[1]))
        stack = []  # [end_ts, name, dur, child_sum]
        for ts, dur, name in evs:
            end = ts + dur
            while stack and ts >= stack[-1][0]:
                s = stack.pop()
                excl[s[1]] = excl.get(s[1], 0.0) + (s[2] - s[3] if s[2] > s[3] else 0.0)
                if stack:
                    stack[-1][3] += s[2]
                cnt[s[1]] = cnt.get(s[1], 0) + 1
            stack.append([end, name, dur, 0.0])
        while stack:
            s = stack.pop()
            excl[s[1]] = excl.get(s[1], 0.0) + (s[2] - s[3] if s[2] > s[3] else 0.0)
            if stack:
                stack[-1][3] += s[2]
            cnt[s[1]] = cnt.get(s[1], 0) + 1
    return excl, cnt


def analyze_file(path, cpu_ops_limit=500000):
    span_lo = span_hi = None
    gpu_iv = []
    comp_iv = []
    comm_iv = []
    comm_sub = {}
    kernels = {}
    gpu_other = {}
    cpu_by_tid = {}
    cpu_seen = 0
    n_events = 0

    gc_was = gc.isenabled()
    gc.disable()
    try:
        for e in iter_events(path):
            if e.get("ph") != "X":
                continue
            cat = e.get("cat")
            if cat not in _INTERESTING:
                continue
            ts = e.get("ts")
            if ts is None:
                continue
            n_events += 1
            dur = e.get("dur") or 0.0
            end = ts + dur
            if span_lo is None or ts < span_lo:
                span_lo = ts
            if span_hi is None or end > span_hi:
                span_hi = end
            name = e.get("name", "?")
            if cat == "cpu_op":
                cpu_seen += 1
                if cpu_ops_limit <= 0 or cpu_seen <= cpu_ops_limit:
                    key = e.get("tid")
                    lst = cpu_by_tid.get(key)
                    if lst is None:
                        cpu_by_tid[key] = lst = []
                    lst.append((ts, dur, name))
                continue
            gpu_iv.append((ts, end))
            if cat == "kernel":
                item = kernels.get(name)
                if item is None:
                    kernels[name] = item = [0, 0.0]
            else:
                item = gpu_other.get(name)
                if item is None:
                    gpu_other[name] = item = [0, 0.0]
            item[0] += 1
            item[1] += dur
            if cat == "kernel" and is_comm(name):
                comm_iv.append((ts, end))
                sub = classify_comm(name)
                sitem = comm_sub.get(sub)
                if sitem is None:
                    comm_sub[sub] = sitem = [0, 0.0]
                sitem[0] += 1
                sitem[1] += dur
            else:
                # 非通信 GPU 事件 (算力 kernel / memcpy / memset) 单独建时间线
                comp_iv.append((ts, end))
    finally:
        if gc_was:
            gc.enable()

    cpu_excl, cpu_cnt = cpu_exclusive_time(cpu_by_tid) if cpu_by_tid else ({}, {})
    span_us = 0.0 if span_lo is None else (span_hi - span_lo)
    busy_iv = merge_intervals(gpu_iv)
    # 契约: busy 含通信/memcpy, 整窗口径 (下游对账签名以此为准, 勿改)
    busy_us = sum(e - s for s, e in busy_iv)
    comm_merged = merge_intervals(comm_iv)
    comm_us = sum(e - s for s, e in comm_merged)
    comp_merged = merge_intervals(comp_iv)
    comp_us = sum(e - s for s, e in comp_merged)

    # 有效窗口 (去窗口边界空档): 头部常先有一段"零星小 kernel + 长空档"的起停残留,
    # 直接按首末事件界定会把残留算进每步耗时。
    k_lo, k_hi = _active_bounds(busy_iv)
    head_gap_us = 0.0 if k_lo is None else max(k_lo - span_lo, 0.0)
    tail_gap_us = 0.0 if k_hi is None else max(span_hi - k_hi, 0.0)
    bnd_us = head_gap_us + tail_gap_us
    span_active_us = max(span_us - bnd_us, 0.0)

    # 通信暴露: comm 区间中不与任何算力/memcpy 事件重叠的部分。
    # 旧 comm_overlap 恒 0% 的原因: busy 含 comm → 空档=b busy 补集 → 与 comm 永不相交。
    # 串行流 (未开 tbo) 时暴露≈100%, 即"通算完全无重叠"; 该值可作为 tbo 可覆盖上限。
    comm_exposed_us = _uncovered(comm_merged, comp_merged)
    comm_hidden_us = max(comm_us - comm_exposed_us, 0.0)

    # GPU 空档 = busy 合并区间的补集, 分桶 (<10us / 10~100us / 0.1~1ms / >=1ms) + Top10
    # 归因 band: B=窗口边界空档 (采集起停 + 调度残留)
    #            C=窗口内部且紧邻通信区间 (<=100us, 跨 rank 同步等待线索)
    #            G=其他 (launch gap / 调度抖动)
    buckets = [[0, 0.0], [0, 0.0], [0, 0.0], [0, 0.0]]
    gaps_us = 0.0
    gaps_n = 0
    bcc_n = 0
    bcc_ms = 0.0
    top_gaps = []
    gaps = []
    if busy_iv:
        prev = span_lo
        for s, e in busy_iv:
            if s > prev:
                gaps.append((prev, s - prev))
            prev = e
        if prev < span_hi:
            gaps.append((prev, span_hi - prev))

    comm_e = [c[1] for c in comm_merged]

    def _comm_adj(gs, ge):
        """空档与最近通信区间的距离 (µs); 无通信区间时返回 None。"""
        if not comm_merged:
            return None
        i = bisect.bisect_right(comm_e, gs)
        best = None
        for k in range(max(i - 2, 0), min(i + 2, len(comm_merged))):
            cs, ce = comm_merged[k]
            d = 0.0 if (cs <= ge and ce >= gs) else max(cs - ge, gs - ce)
            if best is None or d < best:
                best = d
        return best

    for gts, gdur in gaps:
        gaps_n += 1
        gaps_us += gdur
        i = 0 if gdur < 10 else (1 if gdur < 100 else (2 if gdur < 1000 else 3))
        buckets[i][0] += 1
        buckets[i][1] += gdur
        if k_lo is None or gts + gdur <= k_lo + 1.0 or gts >= k_hi - 1.0:
            band, adj = "B", None
        else:
            adj = _comm_adj(gts, gts + gdur)
            if adj is not None and adj <= 100.0:
                band = "C"
                bcc_n += 1
                bcc_ms += gdur / 1000.0
            else:
                band = "G"
        if len(top_gaps) < 10 or gdur > top_gaps[-1][1]:
            top_gaps.append((gts, gdur, band, adj))
            top_gaps.sort(key=lambda x: -x[1])
            if len(top_gaps) > 10:
                top_gaps.pop()

    gaps_top = []
    for gts, gdur, band, adj in sorted(top_gaps, key=lambda x: x[0]):
        gaps_top.append({
            "rel_us": gts - span_lo,
            "dur_us": gdur,
            "band": band,
            "comm_adj_us": -1.0 if adj is None else adj,
        })
    return {
        "file": path,
        "span_us": span_us,
        "busy_us": busy_us,
        "kernels": kernels,
        "gpu_other": gpu_other,
        "cpu_excl": cpu_excl,
        "cpu_cnt": cpu_cnt,
        "cpu_seen": cpu_seen,
        "cpu_limit": cpu_ops_limit,
        "events": n_events,
        "gap_us": gaps_us,
        "gap_n": gaps_n,
        "buckets": buckets,
        "comm_us": comm_us,
        "comm_sub": comm_sub,
        "comp_us": comp_us,
        "head_gap_us": head_gap_us,
        "tail_gap_us": tail_gap_us,
        "span_active_us": span_active_us,
        "comm_exposed_us": comm_exposed_us,
        "comm_hidden_us": comm_hidden_us,
        "bcc_n": bcc_n,
        "bcc_ms": bcc_ms,
        "top_gaps": gaps_top,
    }


def _worker(task):
    path, cpu_ops_limit = task
    t0 = time.time()
    r = analyze_file(path, cpu_ops_limit)
    r["elapsed"] = time.time() - t0
    return r


def fmt_ms(us):
    return f"{us / 1000.0:.3f}"


def main():
    ap = argparse.ArgumentParser(description="torch profiler trace 汇总分析")
    ap.add_argument("path", help="trace 目录或单个 .trace.json.gz")
    ap.add_argument("--top", type=int, default=20, help="TopN 条数 (默认 20)")
    ap.add_argument("--mode", default="", help="仅用于摘要标题: prefill/decode")
    ap.add_argument("--out-dir", default="", help="摘要输出目录 (默认=输入目录)")
    ap.add_argument("--jobs", type=int, default=0,
                    help="并行解析的进程数 (默认 min(8, 文件数, CPU 核数); 1=串行)")
    ap.add_argument("--cpu-ops-limit", type=int, default=500000,
                    help="每个 trace 参与 CPU 分析的 cpu_op 上限 (默认 50 万, 0=不限)")
    args = ap.parse_args()

    files = iter_trace_files(args.path)
    if not files:
        print(f"[ERROR] 未找到 trace 文件: {args.path}", file=sys.stderr)
        return 2

    out_dir = args.out_dir or (args.path if os.path.isdir(args.path) else os.path.dirname(args.path))
    out_dir = os.path.abspath(out_dir or ".")

    jobs = args.jobs
    if jobs <= 0:
        jobs = min(8, len(files), os.cpu_count() or 1)
    jobs = max(1, min(jobs, len(files)))

    t_start = time.time()
    tasks = [(f, args.cpu_ops_limit) for f in files]
    results = []
    if jobs == 1:
        for t in tasks:
            print(f"[INFO] 解析 {t[0]}", file=sys.stderr)
            try:
                r = _worker(t)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] 解析失败 {t[0]}: {exc}", file=sys.stderr)
                continue
            results.append(r)
            print(f"[INFO]   完成 {os.path.basename(t[0])} "
                  f"({r['events']} events, {r['elapsed']:.1f}s)", file=sys.stderr)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"[INFO] 并行解析 {len(files)} 个 trace, jobs={jobs}", file=sys.stderr)
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(_worker, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                src = futs[fut]
                try:
                    r = fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] 解析失败 {src}: {exc}", file=sys.stderr)
                    continue
                results.append(r)
                print(f"[INFO]   完成 {os.path.basename(src)} "
                      f"({r['events']} events, {r['elapsed']:.1f}s)", file=sys.stderr)

    if not results:
        print("[ERROR] 所有 trace 解析失败", file=sys.stderr)
        return 2
    results.sort(key=lambda r: _rank_key(r["file"]))
    parse_s = time.time() - t_start

    agg_kernels = {}
    agg_other = {}
    agg_cpu = {}
    agg_cpu_cnt = {}
    tot_span = tot_busy = 0.0
    tot_bnd = tot_active = tot_comm = tot_exposed = 0.0
    for r in results:
        tot_span += r["span_us"]
        tot_busy += r["busy_us"]
        tot_bnd += r["head_gap_us"] + r["tail_gap_us"]
        tot_active += r["span_active_us"]
        tot_comm += r["comm_us"]
        tot_exposed += r["comm_exposed_us"]
        for name, (c, d) in r["kernels"].items():
            item = agg_kernels.get(name)
            if item is None:
                agg_kernels[name] = item = [0, 0.0]
            item[0] += c
            item[1] += d
        for name, (c, d) in r["gpu_other"].items():
            item = agg_other.get(name)
            if item is None:
                agg_other[name] = item = [0, 0.0]
            item[0] += c
            item[1] += d
        for name, d in r["cpu_excl"].items():
            agg_cpu[name] = agg_cpu.get(name, 0.0) + d
        for name, c in r["cpu_cnt"].items():
            agg_cpu_cnt[name] = agg_cpu_cnt.get(name, 0) + c

    kernel_total = sum(d for _, d in agg_kernels.values())
    title = f"Profile 摘要{f' ({args.mode})' if args.mode else ''}"
    lines = [f"# {title}", ""]
    lines.append("## 概览")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| trace 文件数 | {len(results)} |")
    lines.append(f"| 事件数合计 | {sum(r['events'] for r in results)} |")
    lines.append(f"| 时间跨度合计 | {tot_span / 1e6:.3f} s |")
    lines.append(f"| GPU busy 合计 | {tot_busy / 1e6:.3f} s |")
    ratio = (tot_busy / tot_span * 100.0) if tot_span else 0.0
    lines.append(f"| GPU busy 占比 | {ratio:.1f}% |")
    lines.append(f"| GPU 空档 (bubble) | {(tot_span - tot_busy) / 1e6:.3f} s |")
    lines.append(f"| 窗口边界空档 (起停) | {tot_bnd / 1e6:.3f} s |")
    lines.append(f"| 有效窗口 (去边界) | {tot_active / 1e6:.3f} s |")
    lines.append(f"| 通信暴露 (未与算力重叠) | {tot_exposed / 1e6:.3f} s "
                 f"({tot_exposed / tot_comm * 100.0 if tot_comm else 0:.1f}% of comm) |")
    lines.append(f"| kernel 累计时长 | {kernel_total / 1e6:.3f} s |")
    lines.append(f"| 分析耗时 | {parse_s:.1f} s |")
    lines.append("")
    lines.append("> 口径: `GPU busy` 含通信/memcpy, 为整窗 span 口径 (对账基线); "
                 "`有效窗口` = span − 窗口首尾空档, 用于每步 TPOT/吞吐推导; "
                 "`通信暴露` = 通信 kernel 中未与算力/memcpy 重叠的时长 (串行流≈100%, 即可被 tbo 覆盖上限)。")
    lines.append("")

    lines.append("## 各 rank / 文件")
    lines.append("")
    lines.append("| 文件 | span(s) | GPU busy(s) | busy% | 有效窗口(s) | 通信暴露(ms) | kernels |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        s = r["span_us"] / 1e6
        b = r["busy_us"] / 1e6
        p = (r["busy_us"] / r["span_us"] * 100.0) if r["span_us"] else 0.0
        lines.append(
            f"| {os.path.basename(r['file'])} | {s:.3f} | {b:.3f} | {p:.1f}% | "
            f"{r['span_active_us'] / 1e6:.3f} | {r['comm_exposed_us'] / 1000.0:.1f} | "
            f"{sum(c for c, _ in r['kernels'].values())} |"
        )
    lines.append("")

    top = sorted(agg_kernels.items(), key=lambda kv: kv[1][1], reverse=True)[: args.top]
    share_base = kernel_total or 1.0
    lines.append(f"## GPU kernel Top{len(top)} (按累计时长)")
    lines.append("")
    lines.append("| # | kernel | count | total(ms) | mean(us) | share |")
    lines.append("|---|---|---|---|---|---|")
    for i, (name, (c, d)) in enumerate(top, 1):
        lines.append(
            f"| {i} | `{name}` | {c} | {fmt_ms(d)} | {d / c if c else 0:.2f} | {d / share_base * 100.0:.1f}% |"
        )
    lines.append("")

    if agg_other:
        other = sorted(agg_other.items(), key=lambda kv: kv[1][1], reverse=True)[: args.top]
        lines.append("## GPU memcpy/memset")
        lines.append("")
        lines.append("| 操作 | count | total(ms) |")
        lines.append("|---|---|---|")
        for name, (c, d) in other:
            lines.append(f"| `{name}` | {c} | {fmt_ms(d)} |")
        lines.append("")

    truncated = [r for r in results if r["cpu_limit"] > 0 and r["cpu_seen"] > r["cpu_limit"]]
    if truncated:
        lines.append(
            f"> CPU op 分析按每 trace 前 {args.cpu_ops_limit} 条采样 "
            f"({len(truncated)} 个 rank 被截断), 仅供参考; 需要完整结果时加 `--cpu-ops-limit 0`。"
        )
        lines.append("")

    cpu_top = sorted(agg_cpu.items(), key=lambda kv: kv[1], reverse=True)[: args.top]
    cpu_total = sum(agg_cpu.values()) or 1.0
    lines.append(f"## CPU 顶层 op Top{len(cpu_top)} (独占时长)")
    lines.append("")
    lines.append("| # | op | count | total(ms) | mean(us) | share |")
    lines.append("|---|---|---|---|---|---|")
    for i, (name, d) in enumerate(cpu_top, 1):
        c = agg_cpu_cnt.get(name, 0)
        lines.append(
            f"| {i} | `{name}` | {c} | {fmt_ms(d)} | {d / c if c else 0:.2f} | {d / cpu_total * 100.0:.1f}% |"
        )
    lines.append("")

    text = "\n".join(lines)
    print(text)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "prof_summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    with open(os.path.join(out_dir, "prof_kernels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "count", "total_ms", "mean_us", "share_pct"])
        for name, (c, d) in sorted(agg_kernels.items(), key=lambda kv: kv[1][1], reverse=True):
            w.writerow([name, c, f"{d / 1000.0:.4f}", f"{d / c if c else 0:.2f}", f"{d / share_base * 100.0:.2f}"])

    with open(os.path.join(out_dir, "prof_ranks.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "span_s", "gpu_busy_s", "busy_pct", "events", "parse_s"])
        for r in results:
            p = (r["busy_us"] / r["span_us"] * 100.0) if r["span_us"] else 0.0
            w.writerow([
                os.path.basename(r["file"]),
                f"{r['span_us'] / 1e6:.4f}",
                f"{r['busy_us'] / 1e6:.4f}",
                f"{p:.2f}",
                r["events"],
                f"{r.get('elapsed', 0.0):.1f}",
            ])

    with open(os.path.join(out_dir, "prof_sched.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "gaps_total_ms", "gap_pct",
                    "b0_n", "b0_ms", "b1_n", "b1_ms", "b2_n", "b2_ms", "b3_n", "b3_ms",
                    "comm_ms", "comm_pct",
                    "comm_dispatch_ms", "comm_combine_ms", "comm_notify_ms", "comm_other_ms",
                    "head_gap_ms", "tail_gap_ms", "span_active_ms",
                    "comm_exposed_ms", "comm_hidden_ms", "bcc_n", "bcc_ms"])
        for r in results:
            span = r["span_us"] or 1.0
            b = r["buckets"]
            sub = r["comm_sub"]
            w.writerow([
                os.path.basename(r["file"]),
                f"{r['gap_us'] / 1000.0:.4f}", f"{r['gap_us'] / span * 100.0:.2f}",
                b[0][0], f"{b[0][1] / 1000.0:.4f}",
                b[1][0], f"{b[1][1] / 1000.0:.4f}",
                b[2][0], f"{b[2][1] / 1000.0:.4f}",
                b[3][0], f"{b[3][1] / 1000.0:.4f}",
                f"{r['comm_us'] / 1000.0:.4f}", f"{r['comm_us'] / span * 100.0:.2f}",
                f"{sub.get('dispatch', (0, 0.0))[1] / 1000.0:.4f}",
                f"{sub.get('combine', (0, 0.0))[1] / 1000.0:.4f}",
                f"{sub.get('notify', (0, 0.0))[1] / 1000.0:.4f}",
                f"{sub.get('other', (0, 0.0))[1] / 1000.0:.4f}",
                f"{r['head_gap_us'] / 1000.0:.4f}",
                f"{r['tail_gap_us'] / 1000.0:.4f}",
                f"{r['span_active_us'] / 1000.0:.4f}",
                f"{r['comm_exposed_us'] / 1000.0:.4f}",
                f"{r['comm_hidden_us'] / 1000.0:.4f}",
                r["bcc_n"], f"{r['bcc_ms']:.4f}",
            ])

    with open(os.path.join(out_dir, "prof_gaps_top.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "rel_ms", "dur_ms", "band", "comm_adj_us"])
        for r in results:
            for g in r["top_gaps"]:
                w.writerow([os.path.basename(r["file"]),
                            f"{g['rel_us'] / 1000.0:.4f}",
                            f"{g['dur_us'] / 1000.0:.4f}",
                            g["band"],
                            f"{g['comm_adj_us']:.1f}"])

    print(f"[INFO] 摘要: {os.path.join(out_dir, 'prof_summary.md')} (总耗时 {parse_s:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
