#!/usr/bin/env bash
# ============================================================
# gsp.sh — GSP Prefill 组批估算 + 快速饱和验证 (纯客户端压测脚本)
# ============================================================
#
# 定位: 只做客户端构造 GSP 数据集、发压与校验, 不负责服务生命周期。
#       服务由使用者在任意机器自行起好 (sglang 等兼容 /generate 与
#       /get_server_info 的服务), 客户端在本机或远程执行均可。
#
# 理论（详见仓库 README 或团队 wiki）：
#   服务端每轮 prefill 计算量被 --chunked-prefill-size (C) 锁死，
#   客户端请求超过饱和数后，加量只加队列、不加稳态吞吐。
#
#   记:  L     = 单条输入 token 数
#        H     = 目标前缀命中率（GSP: H = P/L, P 为组内共享前缀长度）
#        S     = L - P                    （每条请求真正要算的 token）
#        l_eff = L * (1 - H)  == S        （首条 miss 由 warmup 消化）
#        n*    = min(max_running_requests, floor(C / l_eff))
#                若 l_eff > C: 单条被切块, n* = 1
#        N_req = 2 * n*  （n*==1 时取 4） → 一个 batch 在算 + 一个在排队
#        总轮数 ≈ ceil(warmup L / C) + ceil(N_req * l_eff / C)
#
#   验证口径（全部来自服务端/客户端实测, 不依赖估算）：
#     1. 命中率:  client meta_info.cached_tokens 之和 / prompt_tokens 之和 ≈ H (±2pt)
#     2. 组批数:  服务端日志稳态 #new-seq 众数 ≈ n*        （需 --server-log）
#     3. 预算:    稳态每轮 #new-token ≈ min(C, n*·S)
#     4. 纯吞吐:  Σnew-token / 主测时长 = new-token 口径（不含命中, 上报用）
#
# 参数获取规则:
#   C: 显式 --C, 否则自动读服务端 chunked_prefill_size
#   L: 显式 --L, 否则自动读服务端 context_length (L=ctx 最饱和; 贴近 ctx 时
#      自动钳到 ctx-16 给输出/预留 token 留位; 想测短些可显式 --L)
#   H: 必须显式 --H —— 压测场景设计参数, 服务端无从获取
#
# 用法:
#   # 纯估算（不碰服务器, C/L/H 全部显式给）
#   ./gsp.sh estimate --C 32768 --L 65536 --H 0.9
#
#   # 客户端压测（在能访问到服务的机器上执行; C/L 自动探测）
#   ./gsp.sh run --base-url http://127.0.0.1:31888 --H 0.9
#   ./gsp.sh run --base-url http://server:31888 --C 32768 --L 65536 --H 0.9 \
#                [--server-log /path/to/server.log] [--out-dir ./all_result]
#
# 环境变量:
#   PYTHON_BIN=python3 SEED=42 TIMEOUT=1800 PAGE=64 HIT_TOL_PT=2.0
#   逐请求 CSV 边跑边存: $OUT_DIR/gsp_bench_<port>_<stamp>.csv
set -uo pipefail

PY="${PYTHON_BIN:-python3}"
SEED="${SEED:-42}"
TIMEOUT="${TIMEOUT:-1800}"   # 单请求 HTTP 超时; L=ctx 级长 prefill 勿低于 600
PAGE="${PAGE:-64}"
HIT_TOL_PT="${HIT_TOL_PT:-2.0}"

usage() { sed -n '2,51p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

die() { echo "[ERROR] $1" >&2; exit "$2"; }

# ---------- 参数解析 ----------
parse_kv() { # --k v -> K=v
  case "$1" in
    --C) C_VAL="$2" ;;        --L) L_VAL="$2" ;;
    --H) H_VAL="$2" ;;
    --base-url) BASE_URL="$2" ;;
    --server-log) SERVER_LOG="$2" ;;
    --out-dir) OUT_DIR="$2" ;;
    --port) PORT="$2" ;;
    --max-running) MAXRUN="$2" ;;
    *) ARGS="$ARGS $1" ;;
  esac
}

CMD="${1:-}"; shift || true
C_VAL="" L_VAL="" H_VAL="" BASE_URL="" SERVER_LOG="" PORT="31888" MAXRUN=""
OUT_DIR="${OUT_DIR:-./all_result}"
while (( $# >= 2 )); do parse_kv "$1" "$2"; shift 2; done
(( $# > 0 )) && parse_kv "$1" ""

# ============================================================
# 子命令: estimate — 纯估算
# ============================================================
do_estimate() {
  [[ -n "$C_VAL" && -n "$L_VAL" && -n "$H_VAL" ]] || die "estimate 需要 --C --L --H" 1
  PAGE="$PAGE" MAXRUN="$MAXRUN" "$PY" - "$C_VAL" "$L_VAL" "$H_VAL" <<'PY'
import os, math, sys
C, L, H = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
page = int(os.environ.get("PAGE", "64"))
maxrun = os.environ.get("MAXRUN", "")
assert 0 <= H < 1, "H 应在 [0,1)"
assert C > 0 and L > 0

# GSP 切分: 前缀对齐 KV page（尾部请求 suffix 未对齐部分会少量 miss）
Hs = round(L * H)
S = L - Hs
P = (Hs // page) * page          # 组内共享前缀按 page 对齐
S_real = L - P
eff = L * (1 - H)
chunked = eff > C

n_pack = C // S_real if S_real > 0 else 0
n_star = max(1, n_pack)
if maxrun:
    n_star = min(n_star, int(maxrun))
n_req = 4 if n_star == 1 else 2 * n_star
if maxrun and int(maxrun) < n_req:
    n_req = int(maxrun)

w_r = math.ceil(L / C)
m_r = math.ceil(n_req * S_real / C) if not chunked else math.ceil(n_req * eff / C)
total_r = w_r + m_r
budget_use = 1.0 if chunked else (min(1.0, n_pack * S_real / C) if S_real > 0 else 0.0)

print(f"输入:  C(chunked_prefill_size)={C}  L={L}  H={H:.0%}  page={page}")
print(f"GSP:   共享前缀 P={P}  后缀 S={S_real}   (P 按 page 对齐, 理论 H'={P/L:.0%})")
print(f"估算:")
print(f"  l_eff = L×(1-H)        = {eff:.0f} tok")
print(f"  n*    = ⌊C/l_eff⌋      = {n_star}   {'(切块: l_eff>C)' if chunked else ''}")
print(f"  N_req = 2×n*{'→凑4' if n_star==1 else ''}    = {n_req}")
print(f"  预算占用 = n*×S/C       = {budget_use:.0%}   {'(吃满)' if budget_use>0.99 else '(不满: 有空转, 或受 max_running 限制)'}")
print(f"  轮数  ≈ warmup({w_r}) + 主测({m_r}) = {total_r}")
print(f"  耗时  ≈ total_r × t_round;  t_round 从 warmup 实测标定 (≈C/纯算力吞吐)")
print(f"执行提示: 请求数={n_req} 足以饱和; 加到 4×n* 吞吐不变、时长翻倍")
PY
}

# ============================================================
# 服务探测
# ============================================================
probe_server() {
  "$PY" - "$BASE_URL" <<'PY'
import json, sys, urllib.request
base = sys.argv[1].rstrip('/')
def get(path):
    try:
        with urllib.request.urlopen(base + path, timeout=10) as r:
            return json.load(r)
    except Exception:
        return None
info = get("/get_server_info") or get("/server_info") or {}
sa = info.get("server_args", info) or {}
out = {
    "version": info.get("version"),
    "chunked_prefill_size": sa.get("chunked_prefill_size"),
    "max_running_requests": sa.get("max_running_requests"),
    "context_length": sa.get("context_length"),
    "page_size": sa.get("page_size"),
    "model_path": sa.get("model_path"),
    "tp_size": sa.get("tp_size"),
}
print(json.dumps(out))
PY
}

# ============================================================
# 压测客户端: warmup 1 条 + 主测 N_req 条并发（GSP: 同一前缀, 唯一后缀）
# ============================================================
run_bench() {
  "$PY" - "$BASE_URL" "$C_VAL" "$L_VAL" "$H_VAL" "$SEED" "$MAXRUN" <<'PY'
import json, sys, time, random, urllib.request, os, csv, threading
from concurrent.futures import ThreadPoolExecutor

base, C, L, H, seed, maxrun_s = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
                                  float(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
base = base.rstrip('/')
maxrun = int(maxrun_s) if maxrun_s else None
CSV_PATH = os.environ.get("CSV_PATH", "")       # 逐请求 CSV, 每条返回即落盘

# ---- 估算（与 estimate 完全一致的口径）----
Hs = round(L * H)
P = (Hs // 64) * 64
S = L - P
eff = L * (1 - H)
n_star = max(1, C // S)
if maxrun:
    n_star = min(n_star, maxrun)
n_req = 4 if n_star == 1 else 2 * n_star
if maxrun and maxrun < n_req:
    n_req = maxrun

# ---- 构造 token ids（确定性 LCG, 纯 prefill, 逐 token 随机保证无意外命中）----
class LCG:
    def __init__(self, s): self.x = s & 0x7fffffff
    def next(self):
        self.x = (self.x * 48271) % 0x7fffffff
        return self.x

def gen(lo, n, salt):
    g = LCG(seed * 1000003 + salt)
    return [1000 + g.next() % 20000 for _ in range(n)]

PREFIX = gen(0, P, 0)                      # 全组共享, warmup 也用它

def req_idx(i):
    ids = PREFIX + gen(1, S, i + 1)        # 唯一后缀
    assert len(ids) == L
    body = json.dumps({
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True},
    }).encode()
    r = urllib.request.Request(base + "/generate", data=body,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=int(__import__('os').environ.get('TIMEOUT', '600'))) as resp:
        return json.load(resp)

print(json.dumps({"P": P, "S": S, "n_star": n_star, "n_req": n_req}), flush=True)

# ---- 逐请求 CSV: 每条返回(或失败)立即 append+flush, 边跑边保存 ----
_csv_f = csv_lock = None
if CSV_PATH:
    _csv_f = open(CSV_PATH, "a", newline="", encoding="utf-8")
    _csv_w = csv.writer(_csv_f)
    csv_lock = threading.Lock()
    if _csv_f.tell() == 0:
        _csv_w.writerow(["phase", "req_idx", "send_ts", "recv_ts",
                         "latency_s", "prompt_tokens", "cached_tokens",
                         "new_tokens", "error"])
    def csv_row(phase, i, st, en, resp):
        mi = resp.get("meta_info", {}) if "error" not in resp else {}
        pt, ct = mi.get("prompt_tokens", 0), mi.get("cached_tokens", 0)
        with csv_lock:
            _csv_w.writerow([phase, i, f"{st:.3f}", f"{en:.3f}", round(en - st, 3),
                             pt, ct, pt - ct, resp.get("error", "")])
            _csv_f.flush()
    print(f"[csv] {CSV_PATH}", flush=True)
else:
    def csv_row(phase, i, st, en, resp): pass

# ---- flush radix cache: 残留前缀会使命中率/t_warm 失真(重复压测时) ----
try:
    urllib.request.urlopen(urllib.request.Request(base + "/flush_cache", data=b"", method="POST"), timeout=15).read()
    print("[flush_cache] OK", flush=True)
except Exception as e:
    print(f"[flush_cache] FAIL {e} (继续, 命中率/t_warm 可能受残留缓存影响)", flush=True)

# ---- warmup: 1 条填满前缀 radix 树 ----
t0 = time.time(); wst = t0; r = req_idx(999999); t_warm = time.time() - t0
mi = r.get("meta_info", {})
csv_row("warmup", 999999, wst, time.time(), r)
print(json.dumps({"warmup": {"t": round(t_warm, 3), "prompt_tokens": mi.get("prompt_tokens"),
                             "cached_tokens": mi.get("cached_tokens")}}), flush=True)

# ---- 主测: n_req 条并发打出; 2 趟, 趟1 仅补齐前缀链(fill), 趟2 计入统计 ----
def phase(tag):
    results = [None] * n_req
    def run_one(i):
        t = time.time()
        try:
            r = req_idx(i); en = time.time()
            results[i] = (t, en, r)
            csv_row(tag, i, t, en, r)
        except Exception as e:
            results[i] = (t, time.time(), {"error": str(e)})
            csv_row(tag, i, t, time.time(), {"error": str(e)})

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=n_req) as ex:
        list(ex.map(run_one, range(n_req)))
    t_main = time.time() - t_start

    agg_prompt = agg_cached = 0
    errs = 0
    for st, en, r in results:
        if "error" in r:
            errs += 1; continue
        mi = r.get("meta_info", {})
        agg_prompt += mi.get("prompt_tokens", 0)
        agg_cached += mi.get("cached_tokens", 0)

    new_sum = agg_prompt - agg_cached
    return {"t_main_s": round(t_main, 3), "n_req": n_req, "errors": errs,
            "prompt_tokens_sum": agg_prompt, "cached_tokens_sum": agg_cached,
            "hit_rate_pct": round(100.0 * agg_cached / agg_prompt, 2) if agg_prompt else 0.0,
            "new_tokens_sum": new_sum,
            "input_throughput_tok_s": round(new_sum / t_main, 1) if t_main > 0 else 0}

# 趟 1 (fill): hybrid/mamba 模型单条 warmup 常补不满 radix 链, 先打一遍前缀到稳态, 结果不计入
s_fill = phase("fill")
print(json.dumps({"fill": {"hit_rate_pct": s_fill["hit_rate_pct"],
                           "new_tokens_sum": s_fill["new_tokens_sum"]}}), flush=True)
if s_fill["errors"]:
    print(f"[warn] fill 趟有 {s_fill['errors']} 条错误", flush=True)

s_main = phase("main")
print(json.dumps({
    "main": s_main,
    "theory": {"H_pct": round(H * 100, 2), "n_star": n_star,
               "expect_main_rounds": -(-n_req * S // C),
               "t_main_over_warm": round(n_req * S / L, 3)},
}), flush=True)
PY
}

# ============================================================
# 服务端日志分析: 稳态 #new-seq / #new-token
# ============================================================
analyze_server_log() {
  [[ -n "$SERVER_LOG" && -f "$SERVER_LOG" ]] || { echo "[analyze] 无服务端日志可分析 (可 --server-log 指定; 不传则跳过组批数校验)"; return 0; }
  START_EPOCH="$BENCH_START_EPOCH" "$PY" - "$SERVER_LOG" "$C_VAL" "$L_VAL" "$H_VAL" <<'PY'
import re, sys, os, statistics as st, time
from datetime import datetime, timezone
path, C, L, H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
start_ep = float(os.environ.get("START_EPOCH", "0"))
S = L - (round(L * H) // 64) * 64
pat = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[^\]]*\]\s*Prefill batch[,.] #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)')
rows_all, rows = [], []
n_raw = 0
for line in open(path, encoding='utf-8', errors='replace'):
    m = pat.search(line)
    if not m:
        continue
    n_raw += 1
    ts = datetime.strptime(m.group(1)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    rows_all.append((ts, int(m.group(2)), int(m.group(3)), int(m.group(4))))
    if start_ep and ts < start_ep - 5:      # 只统计压测窗口, 排除 warmup 与启动自检
        continue
    rows.append(rows_all[-1])
if not rows and rows_all:
    f, l = datetime.fromtimestamp(rows_all[0][0]), datetime.fromtimestamp(rows_all[-1][0])
    print(f"[analyze] 时间窗过滤后 0 行, 但日志共 {n_raw} 条 Prefill 行 "
          f"(时间戳 {f} ~ {l}), 疑似容器/宿主机时区错位, 退化为全量统计")
    rows = rows_all
if not rows:
    print(f'server_log: 未匹配到 Prefill batch 行 (共扫描 {n_raw} 行), '
          f'请核对 sglang 版本日志格式')
    sys.exit(0)
new_seq = [r[1] for r in rows]
new_tok = [r[2] for r in rows]
cached  = [r[3] for r in rows]
mode = st.mode(new_seq)
steady_tok = st.median(new_tok)
print(f"server_log: rounds={len(rows)}  #new-seq mode={mode} median={st.median(new_seq)}  "
      f"稳态轮#new-token中位={steady_tok:.0f}  Σnew={sum(new_tok)} Σcached={sum(cached)}")
PY
}

# ============================================================
# 子命令: run — 估算 + 压测 + 校验
# ============================================================
do_run() {
  [[ -n "$H_VAL" ]] || die "run 需要 --H (命中率; C/L 缺省自动从服务端探测)" 1
  [[ -n "$BASE_URL" ]] || BASE_URL="http://127.0.0.1:${PORT:-31888}"
  mkdir -p "$OUT_DIR"
  local info
  info=$(probe_server) || die "无法探测 $BASE_URL (服务未起?)" 1
  echo "[probe] $info"

  # C 缺省: 服务端 chunked_prefill_size
  if [[ -z "$C_VAL" ]]; then
    C_VAL=$("$PY" -c "import json,sys; print(json.loads(sys.argv[1]).get('chunked_prefill_size') or '')" "$info")
    [[ -n "$C_VAL" ]] || die "服务端未暴露 chunked_prefill_size, 请显式 --C" 1
  fi
  # L 缺省: 服务端 context_length (L=ctx 为最饱和场景, 显存紧张请显式 --L)
  if [[ -z "$L_VAL" ]]; then
    L_VAL=$("$PY" -c "import json,sys; print(json.loads(sys.argv[1]).get('context_length') or '')" "$info")
    [[ -n "$L_VAL" ]] || die "服务端未暴露 context_length, 请显式 --L" 1
  fi
  # 满长输入会被服务端按 ctx 减预留量拒绝 (v0.5.18 实测留给 6 token), 钳到 ctx-16
  local ctx
  ctx=$("$PY" -c "import json,sys; print(json.loads(sys.argv[1]).get('context_length') or '')" "$info")
  if [[ -n "$ctx" && "$L_VAL" -ge $((ctx - 16)) ]]; then
    if [[ "$L_VAL" -gt "$ctx" ]]; then
      echo "[probe] L($L_VAL) > context_length($ctx): 钳到 ctx-16 (满长输入被服务端 400 拒绝)"
    else
      echo "[probe] L($L_VAL) 贴近 context_length($ctx): 钳到 ctx-16 给输出/预留 token 留位"
    fi
    L_VAL=$((ctx - 16))
  fi

  echo "================ ESTIMATE ================"
  do_estimate
  echo "================ BENCH ================"
  local bench_start stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  local bench_out_file="$OUT_DIR/gsp_run_${stamp}.log"
  local csv_file="$OUT_DIR/gsp_bench_${PORT}_${stamp}.csv"
  bench_start=$(date +%s)
  CSV_PATH="$csv_file" TIMEOUT="$TIMEOUT" run_bench | tee "$bench_out_file"
  echo "================ SERVER LOG ================"
  BENCH_START_EPOCH="$bench_start" analyze_server_log | tee -a "$bench_out_file"

  # ---- 结果校验 ----
  "$PY" - "$bench_out_file" "$H_VAL" "$HIT_TOL_PT" "$C_VAL" "$L_VAL" <<'PY'
import json, sys
path, H, tol = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
C, L = int(sys.argv[4]), int(sys.argv[5])
warm = main = theory = slog = None
for line in open(path, encoding='utf-8', errors='replace'):
    line = line.strip()
    if line.startswith('{"warmup"'): warm = json.loads(line)["warmup"]
    elif line.startswith('{"main"'): main = json.loads(line)["main"]; theory = json.loads(line)["theory"]
    elif line.startswith('server_log: rounds='):
        parts = dict(kv.split('=') for kv in line.split()[1:] if '=' in kv); slog = parts

print("======== VERDICT CHECKLIST ========")
ok = True
def check(name, passed, detail):
    global ok
    ok = ok and passed
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

if not main:
    print("  [FAIL] 主测结果缺失"); sys.exit(1)

hit = main["hit_rate_pct"]
if hit >= theory["H_pct"] - tol:
    check("命中下限", True,
          f"实测 {hit:.2f}% >= 理论 {theory['H_pct']:.2f}%-{tol}pt"
          + (" (100%=radix 连后缀也全命中, dense 模型典型; 偏低才是构造/丢前缀问题)" if hit > theory["H_pct"] + tol else ""))
else:
    check("命中下限", False,
          f"实测 {hit:.2f}% < 理论 {theory['H_pct']:.2f}%-{tol}pt: 前缀构造错误或服务端丢缓存")

if warm:
    ratio_theory = theory["t_main_over_warm"]
    ratio_real = main["t_main_s"] / max(warm["t"], 1e-6)
    # 只判"过慢": 前缀没省到才 FAIL; 偏快(全命中省更多, dense 可到 ~0)是正常
    check("时长比 t_main/t_warm", ratio_real <= ratio_theory * 1.8,
          f"实测 {ratio_real:.3f} vs 理论上限 {ratio_theory * 1.8:.2f} (理论 {ratio_theory:.2f}=N_req×S/L; 全命中远低于理论亦 PASS)")

if slog:
    mode = int(slog.get("mode", -1))
    S_ = L - (round(L * H) // 64) * 64
    exp_tok = min(C, theory["n_star"] * S_)
    med = float(slog.get("稳态轮#new-token中位", "0"))
    ok_mode = mode == theory["n_star"]
    ok_tok = abs(med - exp_tok) <= 0.1 * exp_tok
    print(f"  [{'PASS' if ok_mode else 'WARN'}] 组批数 #new-seq 众数 {mode} vs 估算 n*={theory['n_star']}"
          + ('' if ok_mode else ' (budget 上限口径; mamba/hybrid 模型受 state slot 限制常低于 n*, 不判 FAIL)'))
    print(f"  [{'PASS' if ok_tok else 'WARN'}] 稳态轮#new-token 中位 {med:.0f} vs 期望 min(C,n*×S)={exp_tok} (±10%)"
          + ('' if ok_tok else ' (调度细节差异, 仅提示)'))
else:
    print("  [SKIP] 组批数校验: 未提供 --server-log 且无容器快照可取")

print(f"  input吞吐(new-token口径, t_main): {main['input_throughput_tok_s']:.0f} tok/s")
print(f"  错误请求: {main['errors']}/{main['n_req']}")
ok = ok and main["errors"] == 0
print(f"======== 总体: {'PASS ✅ 理论与实现一致' if ok else 'FAIL ❌ 存在偏差, 检查上方 FAIL 项'} ========")
sys.exit(0 if ok else 2)
PY
}

case "$CMD" in
  estimate) do_estimate ;;
  run)      do_run ;;
  help|-h|--help|'') usage ;;
  *) die "未知子命令: $CMD (可选: estimate|run|help)" 1 ;;
esac
