#!/usr/bin/env bash
# ============================================================
# gsp.sh v2 — GSP 命中率修正 + SLO 回测最大 RPS (纯客户端, gsp_prefix 格式)
# ============================================================
#
# 定位: 纯客户端压测脚本(clH 客户端的 gsp_prefix 化改造)。
#       服务由使用者自行起好 (sglang, 兼容 /get_server_info 与 OpenAI 风格端口),
#       客户端在本机或免密可达的机器上执行。
#
# 与 v1(手搓 /generate burst)的差异 —— 改成 gsp_prefix 格式:
#   1. 数据集: sglang.bench_serving --dataset-name generated-shared-prefix
#      (--gsp-num-groups/--gsp-prompts-per-group/--gsp-system-prompt-len/
#       --gsp-question-len/--gsp-output-len/--gsp-ordered),
#      速率受控发压(--request-rate), 指标口径与 perf_bench/gsp_prefix.sh 一致。
#   2. 命中率修正: 识别服务端 context_length(CL)/page_size/dp_size 与客户端
#      传入 H, 推导 GSP 参数:  P = round(H × L × Q/(Q−dp)) 再按 KV page 对齐,
#      theory% = (Q−dp)×P /(Q×L);  L 贴近 ctx 时自动钳到 ctx−out−8。
#      (cache-report 在 chunked-prefill+radix 下会重复计数(实测>100%),
#       因此判定使用理论修正值; 给 --server-log 可按 trial 时间窗从服务端
#       Prefill batch 行统计真实命中做二次验证。)
#   3. SLO 回测: case 传入 TTFT/TPOT SLO, 求满足 SLO 的最大 RPS:
#      - bisect 模式(默认): inf 标定上限 → 在 [0, 上限] 二分, 找到最大
#        MEET 速率 → 全尺寸 confirm。trial 数 ≈ log2, 总时长最短。
#      - ladder 模式: 与 perf_bench 兼容(inf baseline + offset 下扫),
#        可 --ladder-stop-on-meet 提前止步。
#
# SLO 回测规则 (与 perf_bench/gsp_prefix 口径一致):
#   整体 inf 标定只做一次 (workload 级), 回测一律有限请求率降序网格:
#   ./gsp.sh calib --case "..."            → 打印 CALIB_RESULT inf_rps=X rate_hi=Y
#   ./gsp.sh run --mode grid --rate-hi <inf_rps> --case "...,3|4|5|6,..."
#     → 从 inf_rps 起每次减 RATE_STEP(0.1) 生成降序网格(下限 0.1),
#       逐点实测, 一点同时判定所有 TTFT 档, 全档锁定即停 (gsp_prefix 同构).
#     (bisect/ladder 模式保留兼容: --rate-hi 共用上界, 不再各自 inf 标定)
#
# 用法:
#   ./gsp.sh estimate --L 32768 --H 0.9 [--CTX 131072 --PAGE 1 --QPP 16 --DP 1 --OUT 512]
#   ./gsp.sh run --base-url http://127.0.0.1:31889 \
#              --case "32k,0.5k,90,mean_ttft_ms,5,mean_tpot_ms,50" [--mode grid|bisect|ladder] ...
#
# case 语法(与 perf_bench/gsp_prefix.sh 一致):
#   INPUT,OUTPUT,HIT_RATE,TTFT_METRIC,TTFT_TARGETS_S,TPOT_METRIC,TPOT_TARGET_MS[,MAX_CONCURRENCY]
#   例: '32k,0.5k,90,mean_ttft_ms,5,mean_tpot_ms,50'
#   TTFT_TARGETS_S 支持多档 '3|4|5|6'; grid 全档同点判定, bisect 以最严档回测。
#
# run 关键选项:
#   --mode grid|bisect|ladder 回测策略, 默认 bisect
#                             grid=gsp_prefix 口径: calib 的 inf_rps 起降序 step 网格
#   --groups N --qpp N        GSP 组数/每组请求数 (默认 8/16, 官方默认 32/16)
#   --auto                    groups/qpp 自适应档: qpp = max(16, ceil(100/(100-hit))),
#                             groups = 2 if (input>=128k 且 output>=1k) else 4
#                             (qpp 保证 cap=(qpp-dp)/qpp>=hit; 长输入x长输出压单点
#                              耗时, 其余保统计稳定; 显式 --groups/--qpp 优先,
#                              不加 --auto 时行为与旧版完全一致)
#   --search-groups N         bisect 搜索阶段用小组数提速(默认 groups/2, ≥2)
#   --skip-confirm            跳过全尺寸 confirm(更快, 精度略降)
#   --rate-hi R --rate-lo R   手动指定上下界(给 --rate-hi 可跳过 inf 标定)
#   --rate-step F             bisect 收敛精度/ladder 最小步长(默认 0.01)
#   --max-trials N            bisect 最多 trial 数(默认 6, 不含标定/confirm)
#   --rates R1,R2,...         ladder 显式速率点
#   --server-log FILE         服务端日志路径(trial 时间窗内测真实命中率)
#   --trial-timeout SEC       单 trial 超时(0=不限, 默认 0)
#   --plot                    生成容量矩阵热力图(需 matplotlib)
#   --out-dir DIR --run-id X  结果目录(默认 ./all_result/gsp_<run_id>)
#
# 退出码: 0=有结论(MEET/INF_OK)  3=NO_MEET  4=trial 失败  1=用法/环境错误
set -uo pipefail

PY="${PYTHON_BIN:-python3}"
SEED="${SEED:-42}"
TOKEN_UNIT="${TOKEN_UNIT:-1024}"
BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
MODEL="${MODEL:-}"
MODE="bisect"
N_GROUPS="${GSP_NUM_GROUPS:-8}"
QPP="${GSP_PROMPTS_PER_GROUP:-16}"
SEARCH_GROUPS=""
AUTO_MODE="${GSP_AUTO:-0}"
EXPLICIT_G=0
EXPLICIT_Q=0
SKIP_CONFIRM=0
RATE_HI="${RATE_HI:-}"
RATE_LO="0.005"
RATE_STEP="${GSP_RATE_STEP:-0.01}"
MAX_TRIALS=6
EXPLICIT_RATES="${GSP_REQUEST_RATES:-}"
RATE_OFFSETS="${GSP_RATE_OFFSETS:-0,-0.1,-0.1,-0.1,-0.2,-0.2,-0.3,-0.5}"
LADDER_STOP=1
TRIAL_TIMEOUT=0
SERVER_LOG=""
SERVER_LOG_EPOCH_START=""
SERVER_LOG_EPOCH_END=""
OUT_DIR=""
RUN_ID="$(date +%Y%m%d_%H%M%S)"
PLOT=0
CASES=()
DP_SIZE=""
ANSWER=""            # 回测结论: 最大 MEET RPS / INF_OK / ""(NO_MEET)
MEET=0               # 最近一次 trial 的 MEET 判定
LO_EXPLICIT=0        # 用户显式给了 --rate-lo 时需验证下界
LOG_DIR=""
TRIALS_CSV=""
RUN_FAILURE=0

usage() { sed -n '2,64p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
die() { echo "[ERROR] $1" >&2; exit "${2:-1}"; }

# ---------------- 参数解析 ----------------
CMD="${1:-}"; shift || true
while (( $# > 0 )); do
  case "$1" in
    --case)          CASES+=("$2"); shift 2 ;;
    --base-url)      BASE_URL="$2"; shift 2 ;;
    --model)         MODEL="$2"; shift 2 ;;
    --mode)          MODE="$2"; shift 2 ;;
    --groups)        N_GROUPS="$2"; EXPLICIT_G=1; shift 2 ;;
    --qpp)           QPP="$2"; EXPLICIT_Q=1; shift 2 ;;
    --auto)          AUTO_MODE=1; shift ;;
    --search-groups) SEARCH_GROUPS="$2"; shift 2 ;;
    --skip-confirm)  SKIP_CONFIRM=1; shift ;;
    --rate-hi)       RATE_HI="$2"; shift 2 ;;
    --rate-lo)       RATE_LO="$2"; LO_EXPLICIT=1; shift 2 ;;
    --rate-step)     RATE_STEP="$2"; shift 2 ;;
    --max-trials)    MAX_TRIALS="$2"; shift 2 ;;
    --rates)         EXPLICIT_RATES="$2"; shift 2 ;;
    --rate-offsets)  RATE_OFFSETS="$2"; shift 2 ;;
    --no-ladder-stop) LADDER_STOP=0; shift ;;
    --trial-timeout) TRIAL_TIMEOUT="$2"; shift 2 ;;
    --server-log)    SERVER_LOG="$2"; shift 2 ;;
    --out-dir)       OUT_DIR="$2"; shift 2 ;;
    --run-id)        RUN_ID="$2"; shift 2 ;;
    --plot)          PLOT=1; shift ;;
    --L)  L_EST="$2"; shift 2 ;;
    --H)  H_EST="$2"; shift 2 ;;
    --C)  C_EST="$2"; shift 2 ;;
    --CTX) CTX_EST="$2"; shift 2 ;;
    --OUT) OUT_EST="$2"; shift 2 ;;
    --PAGE) PAGE_EST="$2"; shift 2 ;;
    --QPP) QPP_EST="$2"; shift 2 ;;
    --DP)  DP_EST="$2"; shift 2 ;;
    -h|--help|help) usage ;;
    *) die "未知参数: $1 (help 查看用法)" 1 ;;
  esac
done

is_decimal() { [[ "$1" =~ ^[0-9]+([.][0-9]*)?$ || "$1" =~ ^[.][0-9]+$ ]]; }
pos_dec()    { is_decimal "$1" && awk -v v="$1" 'BEGIN{exit !(v>0)}'; }
nn_dec()     { is_decimal "$1" && awk -v v="$1" 'BEGIN{exit !(v>=0)}'; }
f_now() { date +%s; }
slug() { printf '%s' "$1" | sed -E 's/[^A-Za-z0-9_.-]+/_/g; s/^_+//; s/_+$//'; }
fmt2() { awk -v v="$1" 'BEGIN{printf "%.2f", v}'; }
clamp2() { awk -v v="$1" 'BEGIN{printf "%.2f", (v<0?0:v)}'; }

tokens_from_k() { # "32k" → tokens (非法输入返回非 0)
  "$PY" - "$1" "${TOKEN_UNIT}" <<'PY' | tr -d '\r'
import math, re, sys
raw, unit = sys.argv[1], int(sys.argv[2])
m = re.fullmatch(r"([0-9]+(?:\.[0-9]*)?|\.[0-9]+)([kK])?", raw.strip())
if not m: raise SystemExit(1)
v = float(m.group(1))
if m.group(2): v *= unit
if not (math.isfinite(v) and v > 0): raise SystemExit(1)
print(max(1, int(round(v))))
PY
}

# ---------------- 命中率修正推导(核心) ----------------
# 输入: L, H(%), qpp, page, dp, ctx(0=无), out_len
# 输出: L_used system_len question_len theory% req_theory% notes
# P = round(H × L × Q/(Q−dp)) → page 对齐取离目标最近 → theory=(Q−dp)×P/(Q×L)
auto_select_gq() { # IN_LEN OUT_LEN HIT%
  [[ "$AUTO_MODE" == "1" ]] || return 0
  local ilen="$1" olen="$2" hit="$3" q g
  awk -v h="$hit" 'BEGIN{exit !(h>=0 && h<100)}' || die "auto: HIT_RATE need in [0,100), got ${hit}" 1
  if [[ "$EXPLICIT_Q" == "0" ]]; then
    q="$(awk -v h="$hit" 'BEGIN{q=100.0/(100.0-h); n=int(q); if(n<q) n++; printf "%d", (n<16?16:n)}')"
    QPP="$q"
  fi
  if [[ "$EXPLICIT_G" == "0" ]]; then
    if (( ilen >= 131072 && olen >= 1024 )); then g=2; else g=4; fi
    N_GROUPS="$g"
  fi
  echo "[auto] groups=${N_GROUPS} qpp=${QPP} (in=${ilen} out=${olen} hit=${hit}%, explicitG=${EXPLICIT_G} explicitQ=${EXPLICIT_Q})"
}

derive_lengths() {
  "$PY" - "$1" "$2" "$3" "$4" "$5" "${6:-0}" "${7:-0}" <<'PY' | tr -d '\r'
import math, sys
L, hit, qpp, page, dp = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
ctx, out = int(float(sys.argv[6] or 0)), int(float(sys.argv[7] or 0))
notes = []
l_used = L
if ctx > 0 and l_used + out + 8 > ctx:
    l_used = ctx - out - 8
    notes.append(f"ctx_clamp:{L}->{l_used}(ctx={ctx},reserve={out}+8)")
hit_req = qpp - dp
if hit_req <= 0:
    raise SystemExit(f"qpp({qpp}) must be > dp({dp})")
cap = 100.0 * hit_req / qpp
if hit > cap:
    qpp_min = int(math.ceil(dp / (1.0 - hit / 100.0)))
    raise SystemExit(f"H={hit}% unreachable at qpp={qpp},dp={dp}: max theory={cap:.2f}%; use qpp >= {qpp_min} (or lower H)")
desired = (hit / 100.0) * qpp * l_used / hit_req
rounded = min(int(math.floor(desired + 0.5)), max(1, l_used - 1))
sys_len = rounded
if page > 1:
    cands = {max(1, (rounded // page) * page), min(l_used - 1, ((rounded + page - 1) // page) * page)}
    cands = [x for x in cands if 0 < x < l_used]
    if cands:
        sys_len = min(cands, key=lambda x: abs(100.0 * hit_req * x / (qpp * l_used) - hit))
q_len = l_used - sys_len
if not (0 < sys_len < l_used and q_len > 0):
    raise SystemExit(f"invalid split: P={sys_len} Ql={q_len} (L={l_used}, H={hit}%)")
theory = 100.0 * hit_req * sys_len / (qpp * l_used)
req_theory = 100.0 * hit_req * rounded / (qpp * l_used)
print(l_used, sys_len, q_len, f"{theory:.4f}", f"{req_theory:.4f}", ";".join(notes) or "-")
PY
}

# ---------------- 服务探测: 识别服务端 CL/page/dp ----------------
probe_server() {
  "$PY" - "$BASE_URL" <<'PY' | tr -d '\r'
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
def iv(v):
    try: return int(v)
    except (TypeError, ValueError): return 0
model = ""
v1m = get("/v1/models")
if v1m and isinstance(v1m.get("data"), list) and v1m["data"]:
    model = v1m["data"][0].get("id", "")
print("\t".join(str(x) for x in (
    iv(sa.get("context_length")), iv(sa.get("page_size")) or 1,
    iv(sa.get("dp_size")) or 1, iv(sa.get("chunked_prefill_size")),
    iv(sa.get("max_running_requests")), info.get("version") or "", model)))
PY
}

# ---------------- bench_serving console → metrics JSON ----------------
parse_benchmark_log() {
  "$PY" - "$1" <<'PY'
import json, math, re, sys
aliases = {
    "Traffic request rate": "traffic_request_rate",
    "Max request concurrency": "max_request_concurrency",
    "Request throughput (req/s)": "request_throughput",
    "Input token throughput (tok/s)": "input_throughput",
    "Output token throughput (tok/s)": "output_throughput",
    "Total token throughput (tok/s)": "total_throughput",
    "Peak concurrency": "peak_concurrency",
    "Mean TTFT (ms)": "mean_ttft_ms", "Median TTFT (ms)": "p50_ttft_ms",
    "P50 TTFT (ms)": "p50_ttft_ms", "P90 TTFT (ms)": "p90_ttft_ms",
    "P95 TTFT (ms)": "p95_ttft_ms", "P99 TTFT (ms)": "p99_ttft_ms",
    "Max TTFT (ms)": "max_ttft_ms",
    "Mean TPOT (ms)": "mean_tpot_ms", "Median TPOT (ms)": "p50_tpot_ms",
    "P50 TPOT (ms)": "p50_tpot_ms", "P90 TPOT (ms)": "p90_tpot_ms",
    "P95 TPOT (ms)": "p95_tpot_ms", "P99 TPOT (ms)": "p99_tpot_ms",
    "Mean ITL (ms)": "mean_itl_ms",
    "Total prompt tokens": "total_prompt_tokens",
    "Total cached tokens": "total_cached_tokens",
    "Total input tokens": "total_input_tokens",
    "Total generated tokens": "total_generated_tokens",
    "Cache hit rate": "cache_hit_rate_pct",
    "Successful requests": "successful_requests",
    "Benchmark duration (s)": "benchmark_duration_s",
}
number = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?$")
def convert(v):
    v = v.strip()
    if v.endswith("%"): v = v[:-1].strip()
    if number.fullmatch(v):
        p = float(v)
        if math.isfinite(p): return int(p) if p.is_integer() else p
    return v
values = {}
try:
    lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
except OSError:
    print("{}"); raise SystemExit(0)
for line in lines:
    if ":" not in line: continue
    label, raw = line.split(":", 1)
    if label.strip() in aliases:
        values[aliases[label.strip()]] = convert(raw)
if "cache_hit_rate_pct" not in values:
    p, c = values.get("total_prompt_tokens"), values.get("total_cached_tokens")
    if isinstance(p, (int, float)) and p > 0 and isinstance(c, (int, float)):
        values["cache_hit_rate_pct"] = 100.0 * c / p
print(json.dumps(values, separators=(",", ":")))
PY
}

# 判定 MEET: exit==0 && TTFT<=target && TPOT<=target (成功率/命中口径另行注记)
meet_check() { # metrics_json ttft_metric ttft_target_ms tpot_metric tpot_target_ms num_prompts
  "$PY" - "$1" "$2" "$3" "$4" "$5" "${6:-0}" <<'PY' | tr -d '\r'
import json, sys
m = json.loads(sys.argv[1])
ttft_m, ttft_t = sys.argv[2], float(sys.argv[3])
tpot_m, tpot_t = sys.argv[4], float(sys.argv[5])
n, ok = int(float(sys.argv[6] or 0)), True
v1, v2 = m.get(ttft_m), m.get(tpot_m)
if v1 is None or v2 is None: ok = False
if ok and not (float(v1) <= ttft_t and float(v2) <= tpot_t): ok = False
succ, errs = m.get("successful_requests"), m.get("num_errored_reqs")
if ok and succ is not None and n > 0 and int(float(succ)) < n: ok = False
print("1" if ok else "0")
PY
}

# 时间窗内服务端真实命中率(可选 --server-log; 只统计本 trial 窗口)
window_hit() {
  [[ -n "$SERVER_LOG" && -f "$SERVER_LOG" && -n "$SERVER_LOG_EPOCH_START" && -n "$SERVER_LOG_EPOCH_END" ]] || return 0
  SERVER_LOG_EPOCH_START="$SERVER_LOG_EPOCH_START" SERVER_LOG_EPOCH_END="$SERVER_LOG_EPOCH_END" \
  "$PY" - "$SERVER_LOG" <<'PY' | tr -d '\r'
import os, re, sys
from datetime import datetime, timezone
path, ep0, ep1 = sys.argv[1], float(os.environ["SERVER_LOG_EPOCH_START"]), float(os.environ["SERVER_LOG_EPOCH_END"])
pat = re.compile(r'\[([0-9\-: ]{19})[^\]]*\]\s*Prefill batch[,.] #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)')
new = cached = 0
for line in open(path, encoding="utf-8", errors="replace"):
    m = pat.search(line)
    if not m: continue
    ts = datetime.strptime(m.group(1)[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    if ep0 - 5 <= ts <= ep1 + 5:
        new += int(m.group(3)); cached += int(m.group(4))
tot = new + cached
print(f"{100.0 * cached / tot:.4f}" if tot > 0 else "")
PY
}

# ---------------- append trials.csv + 打印 trial 速览行 ----------------
TRIAL_FIELDS="case_id,input,output,hit_target_pct,hit_req_theory_pct,hit_theory_pct,system_len,question_len,page_size,groups,qpp,dp,num_prompts,phase,configured_rate,actual_rps,ttft_metric,ttft_value_ms,tpot_metric,tpot_value_ms,exit_code,meet,successful_requests,benchmark_duration_s,input_throughput,output_throughput,cache_report_hit_pct,server_hit_pct,p50_ttft_ms,p99_ttft_ms,p99_tpot_ms,elapsed_wall_s,log_file"

append_and_report() { # rate phase exit_code elapsed metrics_json log_file
  local rate="$1" phase="$2" ec="$3" elapsed="$4" mjson="$5" logfile="$6"
  MEET="$(meet_check "$mjson" "$CASE_TTFT_METRIC" "$STRICT_TTFT_MS" "$CASE_TPOT_METRIC" "$CASE_TPOT_TARGET_MS" "${CUR_NUM_PROMPTS:-$CASE_NUM_PROMPTS}")"
  TTFT_METRIC="$CASE_TTFT_METRIC" TPOT_METRIC="$CASE_TPOT_METRIC" "$PY" - "$TRIALS_CSV" "$TRIAL_FIELDS" "${CSV_ARGS[@]}" "$rate" "$phase" "$ec" "$elapsed" "$MEET" "$mjson" "$logfile" <<'PY'
import csv, json, os, sys
path, fields_s, *rest = sys.argv[1:]  # argv[0]="-" 占位, 需跳过
fields = fields_s.split(",")
(rate, phase, ec, elapsed, meet, mjson, logfile) = rest[-7:]
keys = ["case_id","input","output","hit_target_pct","hit_req_theory_pct","hit_theory_pct",
        "system_len","question_len","page_size","groups","qpp","dp","num_prompts"]
vals = rest[:-7]
row = dict(zip(keys, vals))
row.update({"phase": phase, "configured_rate": rate, "exit_code": ec,
            "meet": meet, "elapsed_wall_s": elapsed, "log_file": logfile})
m = {}
try: m = json.loads(mjson)
except Exception: pass
for k in ("successful_requests","benchmark_duration_s",
          "input_throughput","output_throughput",
          "p50_ttft_ms","p99_ttft_ms","p99_tpot_ms"):
    if k in m: row[k] = m[k]
row["actual_rps"] = m.get("request_throughput", "")
row["ttft_metric"] = os.environ.get("TTFT_METRIC", "")
row["tpot_metric"] = os.environ.get("TPOT_METRIC", "")
row["ttft_value_ms"] = m.get(row["ttft_metric"], "")
row["tpot_value_ms"] = m.get(row["tpot_metric"], "")
row["cache_report_hit_pct"] = m.get("cache_hit_rate_pct", "")
sh = os.environ.get("WINDOW_HIT_PCT", "")
if sh: row["server_hit_pct"] = sh
rows = []
if os.path.exists(path):
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        rows = []
rows = [r for r in rows if not (r.get("case_id")==row["case_id"] and r.get("configured_rate")==rate and r.get("phase")==phase)]
rows.append(row)
with open(path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"{row.get('actual_rps','')}\t{row['ttft_value_ms']}\t{row['tpot_value_ms']}\t{sh if sh else ''}")
PY
}

# ---------------- 单 trial ----------------
run_trial() { # rate phase
  local rate="$1" phase="$2"
  local lfile efile ec t0 t1 elapsed mjson wh
  lfile="${LOG_DIR}/${CASE_ID}__rate_$(slug "$rate").log"
  efile="${lfile}.exitcode"
  t0=$(f_now)
  if [[ -s "$lfile" && -f "$efile" && "$(<"$efile")" == "0" ]]; then
    mjson="$(parse_benchmark_log "$lfile")"
    append_and_report "$rate" "$phase" 0 0 "$mjson" "$lfile"
    echo "[SKIP-RESUME] rate=$rate (已有成功日志)" >&2
    return 0
  fi
  echo "--------------------------------------------------------------------"
  echo "[trial] case=${CASE_ID} rate=${rate} phase=${phase} prompts=${CASE_NUM_PROMPTS} (${GROUPS_STD}x${QPP}, P=${CASE_SYSTEM_LEN}/Ql=${CASE_QUESTION_LEN})"
  local cmd=()
  cmd=("$PY" -m sglang.bench_serving --backend sglang --base-url "$BASE_URL"
       --model "$MODEL" --dataset-name generated-shared-prefix
       --gsp-num-groups "$CUR_GROUPS" --gsp-prompts-per-group "$QPP"
       --gsp-system-prompt-len "$CUR_SYSTEM_LEN" --gsp-question-len "$CUR_QUESTION_LEN"
       --gsp-output-len "$CASE_OUTPUT_LEN" --gsp-ordered --num-prompts "$CUR_NUM_PROMPTS"
       --flush-cache --seed "$SEED" --cache-report
       --request-rate "$rate" --disable-tqdm --output-file /dev/null)
  [[ -n "${CASE_CONCURRENCY:-}" ]] && cmd+=(--max-concurrency "$CASE_CONCURRENCY")
  printf '[CMD]'; printf ' %q' "${cmd[@]}"; printf '\n'
  if [[ "$TRIAL_TIMEOUT" != "0" ]]; then
    timeout "$TRIAL_TIMEOUT" "${cmd[@]}" 2>&1 | tee "$lfile"; ec="${PIPESTATUS[0]}"
  else
    "${cmd[@]}" 2>&1 | tee "$lfile"; ec="${PIPESTATUS[0]}"
  fi
  printf '%s\n' "$ec" > "$efile"
  t1=$(f_now); elapsed=$((t1 - t0))
  mjson="$(parse_benchmark_log "$lfile")"
  SERVER_LOG_EPOCH_START="$t0" SERVER_LOG_EPOCH_END="$t1"
  wh="$(window_hit)"
  WINDOW_HIT_PCT="$wh" append_and_report "$rate" "$phase" "$ec" "$elapsed" "$mjson" "$lfile"
  echo "[trial] rate=$rate → exit=$ec meet=${MEET} rps=$(metrics_get "$mjson" request_throughput) ttft=$(metrics_get "$mjson" "$CASE_TTFT_METRIC")ms tpot=$(metrics_get "$mjson" "$CASE_TPOT_METRIC")ms wall=${elapsed}s"
  [[ "$ec" == "0" ]] || RUN_FAILURE=1
  return 0
}

metrics_get() { printf '%s' "$1" | "$PY" -c 'import json,sys
try:
    d=json.load(sys.stdin); v=d.get(sys.argv[1],"")
    print(v if v!="" else "-")
except Exception: print("-")' "$2" 2>/dev/null; }

# ---------------- 搜索: bisect / ladder ----------------
search_bisect() {
  local lo="$RATE_LO" hi="$RATE_HI" step_eff mid mw
  # 1) 标定: inf 上限 (除非用户给了 --rate-hi)
  if [[ -z "$hi" ]]; then
    CUR_GROUPS="$SEARCH_GROUPS"; CUR_NUM_PROMPTS=$((SEARCH_GROUPS * QPP)); GROUPS_STD="search"
    echo "======== [1/3] inf 标定 (search 尺寸 ${SEARCH_GROUPS}x${QPP}) ========"
    run_trial inf calib
    local crps; crps="$(metrics_get "$(parse_benchmark_log "${LOG_DIR}/${CASE_ID}__rate_inf.log")" request_throughput)"
    if [[ -z "$crps" || "$crps" == "-" ]]; then die "inf 标定未产出 request_throughput, 无法定上界; 请 --rate-hi 手动指定" 1; fi
    [[ "$MEET" == "1" ]] && { ANSWER="INF_OK"; return 0; }
    hi="$(awk -v r="$crps" 'BEGIN{printf "%.2f", r*1.05}')"
    echo "[calib] inf 实际 rps=${crps} → 回测上界 hi=${hi}"
  fi
  # 1.5) 显式下界需先验证 (默认 0.005 近串行, 视为必然 MEET)
  if [[ "$LO_EXPLICIT" == "1" ]]; then
    run_trial "$lo" bisect
    if [[ "$MEET" == "1" ]]; then
      echo "[lo] 显式下界 ${lo} 验证 MEET"
    else
      echo "[lo] 显式下界 ${lo} 未满足 SLO → 下探重锚 lo=0.005"
      lo="0.005"
    fi
  fi
  # 2) 在 [lo, hi] 二分 (不变式: lo=MEET / hi=FAIL); 每个 lo 更新均来自实测 MEET
  ANSWER=""
  step_eff="$(awk -v s="$RATE_STEP" -v h="$hi" 'BEGIN{printf "%.4f", (s>0.03*h? s : 0.03*h)}')"
  echo "======== [2/3] bisect 回测: [${lo}, ${hi}] step_eff=${step_eff} ========"
  local t=0
  while :; do
    mw="$(awk -v l="$lo" -v h="$hi" 'BEGIN{printf "%.4f", h-l}')"
    awk -v w="$mw" -v s="$step_eff" 'BEGIN{exit !(w<=s)}' && break
    mid="$(awk -v l="$lo" -v h="$hi" 'BEGIN{mid=(l+h)/2; printf "%.2f", mid}')"
    awk -v a="$mid" -v l="$lo" -v h="$hi" 'BEGIN{exit !(a<=l+0.004 || a>=h-0.004)}' && break
    t=$((t+1)); (( t > MAX_TRIALS )) && { echo "[bisect] 达 --max-trials=$MAX_TRIALS 上限, 停止"; break; }
    echo "-------- bisect ${t}/${MAX_TRIALS}: rate=${mid} (lo=${lo} MEET / hi=${hi} FAIL) --------"
    run_trial "$mid" bisect
    if [[ "$MEET" == "1" ]]; then lo="$mid"; else hi="$mid"; fi
  done
  ANSWER="$lo"
}

# ---------------- 搜索: grid (gsp_prefix 口径: calib inf_rps 起降序 step 网格) ----------------
# 规则与 perf_bench/gsp_prefix_bench.sh finite_rate_candidates 一致:
#   网格 = [start, start-STEP, start-2*STEP, ...] (下限 0.1, start 默认 calib 的 inf 实际 rps)
#   降序逐点实测; 每个 rate 点同时对所有未锁定 TTFT 档判定 MEET
#   (ttft<=t*1000ms 且 tpot<=TPOT_TARGET_MS), 首个 MEET 点即该档最大 RPS;
#   全档锁定或网格耗尽即停。结果行: GRID_RESULT ttft_s=<t> meet=1 at_rate=R actual_rps=A ...
search_grid() {
  local start="$RATE_HI"
  if [[ -z "$start" ]]; then
    CUR_GROUPS="$SEARCH_GROUPS"; CUR_NUM_PROMPTS=$((SEARCH_GROUPS * QPP)); GROUPS_STD="search"
    echo "======== [1/2] inf 标定 (search 尺寸 ${SEARCH_GROUPS}x${QPP}) ========"
    run_trial inf calib
    local crps; crps="$(metrics_get "$(parse_benchmark_log "${LOG_DIR}/${CASE_ID}__rate_inf.log")" request_throughput)"
    if [[ -z "$crps" || "$crps" == "-" ]]; then die "inf 标定未产出 request_throughput; 请 --rate-hi 指定网格起点" 1; fi
    start="$crps"
    echo "[grid] inf 实际 rps=${crps} → 网格起点=${start}"
  fi
  local rates=() r
  while IFS= read -r r; do [[ -n "$r" ]] && rates+=("$r"); done < <(
    awk -v s="$start" -v st="$RATE_STEP" 'BEGIN{
      rate=s; while (rate >= 0.1-1e-9){ printf "%.2f\n", rate
      nr=rate-st; if (nr < 0.1-1e-9) break; rate=nr } }')
  [[ ${#rates[@]} -gt 0 ]] || rates=("0.10")
  echo "======== [2/2] grid 降序回测 (gsp_prefix 口径): ${rates[*]} ========"
  local mjson act ttft_ms tpot_ms line locked i
  local old_ifs="$IFS"; IFS='|'; local GRID_TTFTS=($CASE_TTFT_TARGETS); IFS="$old_ifs"
  GRID_LOCK=(); GRID_RATE=(); GRID_RPS=(); GRID_TTFTV=(); GRID_TPOTV=()
  for i in "${!GRID_TTFTS[@]}"; do GRID_LOCK[i]=0; done
  for r in "${rates[@]}"; do
    run_trial "$r" grid
    if [[ "${RUN_FAILURE:-0}" == "1" ]]; then echo "[grid] trial 失败, 中止扫描"; break; fi
    mjson="$(parse_benchmark_log "${LOG_DIR}/${CASE_ID}__rate_$(slug "$r").log")"
    act="$(metrics_get "$mjson" request_throughput)"
    ttft_ms="$(metrics_get "$mjson" "$CASE_TTFT_METRIC")"
    tpot_ms="$(metrics_get "$mjson" "$CASE_TPOT_METRIC")"
    line=""
    for i in "${!GRID_TTFTS[@]}"; do
      [[ "${GRID_LOCK[i]}" == "1" ]] && { line+=" t${GRID_TTFTS[i]}:LOCKED"; continue; }
      if awk -v t="$ttft_ms" -v tt="${GRID_TTFTS[i]}" -v p="$tpot_ms" -v pl="$CASE_TPOT_TARGET_MS" \
           'BEGIN{ lim=tt*1000; exit !(t!="-" && t!="" && t<=lim && p!="-" && p!="" && p<=pl) }'; then
        GRID_LOCK[i]=1; GRID_RATE[i]="$r"; GRID_RPS[i]="$act"
        GRID_TTFTV[i]="$ttft_ms"; GRID_TPOTV[i]="$tpot_ms"
        line+=" t${GRID_TTFTS[i]}:MEET"
      else
        line+=" t${GRID_TTFTS[i]}:FAIL"
      fi
    done
    echo "[grid] rate=${r} actual_rps=${act} ttft=${ttft_ms}ms tpot=${tpot_ms}ms →${line}"
    locked=0
    for i in "${!GRID_LOCK[@]}"; do [[ "${GRID_LOCK[i]}" == "1" ]] && locked=$((locked+1)); done
    (( locked == ${#GRID_TTFTS[@]} )) && { echo "[grid] 所有 TTFT 档均已锁定, 停止降序扫描"; break; }
  done
  echo "======== grid 结论 (各 TTFT 档最大 RPS) ========"
  for i in "${!GRID_TTFTS[@]}"; do
    if [[ "${GRID_LOCK[i]}" == "1" ]]; then
      echo "GRID_RESULT ttft_s=${GRID_TTFTS[i]} meet=1 at_rate=${GRID_RATE[i]} actual_rps=${GRID_RPS[i]} ttft_ms=${GRID_TTFTV[i]} tpot_ms=${GRID_TPOTV[i]}"
    else
      echo "GRID_RESULT ttft_s=${GRID_TTFTS[i]} meet=0 at_rate=- actual_rps=- ttft_ms=- tpot_ms=-"
    fi
  done
  ANSWER="GRID"
}

search_ladder() {
  local rates_list=() r
  echo "======== [1/2] inf baseline ========"
  run_trial inf baseline
  if [[ -n "$EXPLICIT_RATES" ]]; then
    IFS=',' read -ra rates_list <<< "$EXPLICIT_RATES"
  else
    local crps cur off cand prev=""
    crps="$(metrics_get "$(parse_benchmark_log "${LOG_DIR}/${CASE_ID}__rate_inf.log")" request_throughput)"
    [[ -z "$crps" ]] && crps="$RATE_HI"
    [[ -z "$crps" ]] && die "无法确定梯子锚点(读取 inf rps 失败), 请 --rates 或 --rate-hi" 1
    cur="$crps"
    IFS=',' read -ra _offs <<< "$RATE_OFFSETS"
    for off in "${_offs[@]}"; do
      cur="$(awk -v c="$cur" -v o="$off" 'BEGIN{printf "%.6f", c+o}')"
      cand="$(fmt2 "$cur")"
      if awk -v c="$cand" 'BEGIN{exit !(c>0)}' && [[ "$cand" != "$prev" ]]; then
        rates_list+=("$cand"); prev="$cand"
      fi
    done
  fi
  echo "======== [2/2] ladder 回测: ${rates_list[*]} ========"
  for r in "${rates_list[@]}"; do
    run_trial "$r" ladder
    if [[ "$MEET" == "1" && "$LADDER_STOP" == "1" ]]; then
      echo "[ladder] 首个 MEET(最严档)=${r}, 降序扫描止步(倍速模式 --no-ladder-stop 可全扫)"
      break
    fi
  done
}

# ---------------- summary + verdict ----------------
write_summary() {
  "$PY" - "$TRIALS_CSV" "$OUT_DIR" "$MODE" <<'PY'
import csv, os, sys
trials, out_dir, mode = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
if os.path.exists(trials):
    with open(trials, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
def fl(r, k):
    try:
        v = float(r.get(k, ""))
        return v
    except Exception:
        return None
meets = [r for r in rows if r.get("meet") == "1" and r.get("exit_code") == "0" and r.get("phase") not in ("calib",)]
best = max(meets, key=lambda r: fl(r, "actual_rps") or 0) if meets else None
fails = [r for r in rows if r.get("exit_code") == "0" and r.get("phase") not in ("calib", "baseline") and r.get("meet") != "1"]
near = max(fails, key=lambda r: fl(r, "actual_rps") or 0) if fails else None
total_wall = sum(fl(r, "elapsed_wall_s") or 0 for r in rows)
status = "NO_MEET"
if best is not None:
    status = "INF_OK" if best.get("configured_rate") == "inf" else "MEET"
elif os.environ.get("ANSWER") == "INF_OK":
    status = "INF_OK"   # bisect: inf 标定即满足 SLO, 无后续 trial
elif any(r.get("exit_code") != "0" for r in rows):
    status = "FAILED"
# grid 模式: 从 trials.csv 逐档二次判定 (任一档有达标点即 MEET)
grid_details = []
if mode == "grid":
    targets = [float(x) for x in (os.environ.get("TTFT_TARGETS_S","") or "").split("|") if x.strip()]
    try: plimit = float(os.environ.get("TPOT_TARGET_MS","") or 999999)
    except Exception: plimit = 999999
    g = [r for r in rows if r.get("phase") == "grid" and r.get("exit_code") == "0"]
    for t in sorted(targets):
        cand = [r for r in g if fl(r,"ttft_value_ms") is not None and fl(r,"ttft_value_ms") <= t*1000
                and fl(r,"tpot_value_ms") is not None and fl(r,"tpot_value_ms") <= plimit]
        b = max(cand, key=lambda r: fl(r,"actual_rps") or 0) if cand else None
        grid_details.append((t, b))
    if any(b is not None for _, b in grid_details):
        status = "MEET"
fields = ["case_id","input","output","hit_target_pct","hit_theory_pct","system_len","question_len",
          "num_prompts","ttft_metric","ttft_targets_s","tpot_metric","tpot_target_ms",
          "status","max_rps","at_rate","ttft_ms","tpot_ms","near_miss_rps","near_miss_ttft_ms",
          "near_miss_tpot_ms","total_trials","total_wall_s"]
b = rows[0] if rows else {}
summary = {
  "case_id": b.get("case_id",""), "input": b.get("input",""), "output": b.get("output",""),
  "hit_target_pct": b.get("hit_target_pct",""), "hit_theory_pct": b.get("hit_theory_pct",""),
  "system_len": b.get("system_len",""), "question_len": b.get("question_len",""),
  "num_prompts": b.get("num_prompts",""),
  "ttft_metric": b.get("ttft_metric",""), "ttft_targets_s": os.environ.get("TTFT_TARGETS_S",""),
  "tpot_metric": b.get("tpot_metric",""), "tpot_target_ms": b.get("tpot_target_ms", os.environ.get("TPOT_TARGET_MS","")),
  "status": status,
  "max_rps": fl(best, "actual_rps") if best else "",
  "at_rate": best.get("configured_rate","") if best else "",
  "ttft_ms": fl(best, "ttft_value_ms") if best else "",
  "tpot_ms": fl(best, "tpot_value_ms") if best else "",
  "near_miss_rps": fl(near, "actual_rps") if near else "",
  "near_miss_ttft_ms": fl(near, "ttft_value_ms") if near else "",
  "near_miss_tpot_ms": fl(near, "tpot_value_ms") if near else "",
  "total_trials": len(rows), "total_wall_s": round(total_wall, 1),
}
spath = os.path.join(out_dir, "summary.csv")
with open(spath, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows([summary])
def show(v): return "-" if v == "" or v is None else v
print("======== 结论 (SLO 回测) ========")
print(f"case            : {summary['case_id']}")
print(f"命中率(修正)     : 目标 {summary['hit_target_pct']}% → 理论 {summary['hit_theory_pct']}%  (P={summary['system_len']}/Ql={summary['question_len']}, num_prompts={summary['num_prompts']})")
print(f"SLO             : {summary['ttft_metric']}<={summary['ttft_targets_s']}s & {summary['tpot_metric']}<={summary['tpot_target_ms']}ms")
print(f"最大 RPS        : {show(summary['max_rps'])} (status={status}, at_rate={show(summary['at_rate'])})")
if grid_details:
    print("各 TTFT 档 (grid 降序扫描, gsp_prefix 口径):")
    for t, b in grid_details:
        if b is not None:
            print(f"  TTFT<={t:g}s : at_rate={b.get('configured_rate','')} actual_rps={fl(b,'actual_rps')} ttft={fl(b,'ttft_value_ms')}ms tpot={fl(b,'tpot_value_ms')}ms")
        else:
            print(f"  TTFT<={t:g}s : 未满足 (网格内无达标点)")
if best:
    print(f"  该点指标       : TTFT={show(summary['ttft_ms'])}ms TPOT={show(summary['tpot_ms'])}ms")
if near:
    print(f"  最近失败点     : rps={show(summary['near_miss_rps'])} TTFT={show(summary['near_miss_ttft_ms'])}ms TPOT={show(summary['near_miss_tpot_ms'])}ms")
print(f"trials          : {summary['total_trials']} 个, 合计 wall {summary['total_wall_s']}s ({summary['total_wall_s']/60:.1f} min)")
if best and best.get("configured_rate") == "inf":
    print("注: inf(突发全速)下即满足 SLO → 无需限速, 最大 RPS 受吞吐上限约束 (取 inf 实测值)")
if near and status == "NO_MEET":
    print("注: 所有已测速率均未满足 SLO; 可降低 SLO 或检查 decode/并行配置")
with open(os.path.join(out_dir, "VERDICT.txt"), "w", encoding="utf-8") as f:
    f.write(status + "\n")
PY
}

# ---------------- estimate 子命令(修正公式预览) ----------------
do_estimate() {
  : "${L_EST:?estimate 需要 --L}" ; : "${H_EST:?estimate 需要 --H (0-1 或百分数块如 90? 需 0<=H<1)}"
  local h="$H_EST"
  awk -v h="$h" 'BEGIN{exit !(h>=1)}' && { echo "[i] H 按百分数解释: ${h}%"; h="$(awk -v h="$h" 'BEGIN{printf "%.4f", h/100}')"; }
  local out="${OUT_EST:-0}" page="${PAGE_EST:-1}" qpp="${QPP_EST:-16}" dp="${DP_EST:-1}" ctx="${CTX_EST:-0}"
  read -r l_used sl ql theory reqth notes < <(derive_lengths "$L_EST" "$(awk -v h="$h" 'BEGIN{printf "%.2f", h*100}')" "$qpp" "$page" "$dp" "$ctx" "$out")
  echo "================ GSP 命中率修正推导 ================"
  echo "输入          : L=${L_EST}  H=${H_EST}  qpp=${qpp}  dp=${dp}  page=${page}  ctx=${ctx:-NA}  out=${out}"
  echo "L 修正        : ${L_EST} → ${l_used}  [${notes}]"
  echo "GSP 切分      : system_len(P)=${sl}  question_len=${ql}"
  echo "命中率修正    : 目标 ${H_EST} → 请求理论 ${reqth}% → 页对齐理论 ${theory}%"
  echo "提示          : 该 theory% 即数据集构造保证的命中上界(cache-report 在"
  echo "                chunked-prefill+radix 下会重复计数, 实测>100% 属 known-issue)"
}

# ============================================================
# 子命令: calib — 仅整体 inf 标定一次, 打印 CALIB_RESULT rate_hi
#   (workload 级饱和上界; 各 TTFT 档回测 run --rate-hi <hi> 共用,
#    回测请求率恒为有限值, 与 perf_bench/gsp_prefix 口径一致)
# ============================================================
do_calib() {
  (( ${#CASES[@]} == 1 )) || die "calib 需要 --case (1 个 case)" 1
  [[ -n "$MODEL" ]] || die "calib 需要 --model" 1

  # ---- case 解析 (SLO 字段仅用于履历, calib 不做 MEET 判定) ----
  local old_ifs="$IFS" case_str="${CASES[0]}"
  IFS=',' read -r CASE_INPUT_RAW CASE_OUTPUT_RAW CASE_HIT_PCT CASE_TTFT_METRIC CASE_TTFT_TARGETS CASE_TPOT_METRIC CASE_TPOT_TARGET_MS CASE_CONCURRENCY _rest <<< "$case_str"
  IFS="$old_ifs"
  [[ -n "$CASE_INPUT_RAW" && -n "$CASE_OUTPUT_RAW" && -n "$CASE_HIT_PCT" && -n "$CASE_TTFT_METRIC" \
     && -n "$CASE_TTFT_TARGETS" && -n "$CASE_TPOT_METRIC" && -n "$CASE_TPOT_TARGET_MS" ]] \
    || die "case 字段不足: INPUT,OUTPUT,HIT_RATE,TTFT_METRIC,TTFT_TARGETS_S,TPOT_METRIC,TPOT_TARGET_MS" 1
  nn_dec "$CASE_HIT_PCT"        || die "HIT_RATE 非法: $CASE_HIT_PCT" 1
  pos_dec "$CASE_TPOT_TARGET_MS" || die "TPOT_TARGET_MS 非法: $CASE_TPOT_TARGET_MS" 1
  CASE_INPUT_LEN="$(tokens_from_k "$CASE_INPUT_RAW")"   || die "INPUT 非法: $CASE_INPUT_RAW" 1
  CASE_OUTPUT_LEN="$(tokens_from_k "$CASE_OUTPUT_RAW")" || die "OUTPUT 非法: $CASE_OUTPUT_RAW" 1
  auto_select_gq "$CASE_INPUT_LEN" "$CASE_OUTPUT_LEN" "$CASE_HIT_PCT"
  STRICT_TTFT_MS="$("$PY" - "$CASE_TTFT_TARGETS" <<'PY' | tr -d '\r'
import sys
vals = [float(x) for x in sys.argv[1].split("|") if x.strip()]
print(int(round(min(vals) * 1000)))
PY
)"

  # ---- 服务端探测 + 命中率推导 ----
  local probe_out ctx page dp chunked maxrun sver s_model
  probe_out="$(probe_server)"
  read -r ctx page dp chunked maxrun sver s_model <<< "$probe_out"
  [[ -n "$ctx" ]] || die "无法探测服务端 (${BASE_URL}); 请确认服务已启动" 1
  dp="${DP_SIZE:-$dp}"; [[ -n "$dp" ]] || dp=1
  local derive_out l_used
  derive_out="$(derive_lengths "$CASE_INPUT_LEN" "$CASE_HIT_PCT" "$QPP" "$page" "$dp" "$ctx" "$CASE_OUTPUT_LEN")" \
    || die "推导失败: L=$CASE_INPUT_LEN H=$CASE_HIT_PCT% page=$page dp=$dp ctx=$ctx out=$CASE_OUTPUT_LEN" 1
  read -r l_used CASE_SYSTEM_LEN CASE_QUESTION_LEN CASE_HIT_REQ_THEORY CASE_HIT_THEORY DERIVE_NOTES <<< "$derive_out"

  CASE_ID="$(slug "${CASE_INPUT_RAW}_${CASE_OUTPUT_RAW}_${CASE_HIT_PCT}")"
  CASE_NUM_PROMPTS=$((N_GROUPS * QPP))
  [[ -n "$SEARCH_GROUPS" ]] || SEARCH_GROUPS=$((N_GROUPS / 2))
  (( SEARCH_GROUPS >= 2 )) || SEARCH_GROUPS=2

  OUT_DIR="${OUT_DIR:-./all_result/gsp_${RUN_ID}}"
  LOG_DIR="${OUT_DIR}/logs"; TRIALS_CSV="${OUT_DIR}/trials.csv"
  mkdir -p "$LOG_DIR"
  [[ -f "$TRIALS_CSV" ]] || printf '%s\n' "$TRIAL_FIELDS" > "$TRIALS_CSV"
  CSV_ARGS=("$CASE_ID" "$CASE_INPUT_RAW" "$CASE_OUTPUT_RAW" "$CASE_HIT_PCT" \
            "$CASE_HIT_REQ_THEORY" "$CASE_HIT_THEORY" "$CASE_SYSTEM_LEN" "$CASE_QUESTION_LEN" \
            "$page" "$N_GROUPS" "$QPP" "$dp" "$CASE_NUM_PROMPTS")

  echo "================ GSP calib (整体 inf 标定, 一次) ================"
  echo "server         : ctx=${ctx} page=${page} dp=${dp} ver=${sver:-NA}"
  echo "case           : id=${CASE_ID} input=${CASE_INPUT_LEN} output=${CASE_OUTPUT_LEN} hit=${CASE_HIT_PCT}%"
  echo "GSP 参数       : search=${SEARCH_GROUPS}x${QPP} prompts; P=${CASE_SYSTEM_LEN}/Ql=${CASE_QUESTION_LEN} [${DERIVE_NOTES}]"

  CUR_GROUPS="$SEARCH_GROUPS"; CUR_NUM_PROMPTS=$((SEARCH_GROUPS * QPP)); GROUPS_STD="search"
  CUR_SYSTEM_LEN="$CASE_SYSTEM_LEN"; CUR_QUESTION_LEN="$CASE_QUESTION_LEN"
  ANSWER=""
  run_trial inf calib
  local crps hi
  crps="$(metrics_get "$(parse_benchmark_log "${LOG_DIR}/${CASE_ID}__rate_inf.log")" request_throughput)"
  if [[ -z "$crps" || "$crps" == "-" ]]; then
    die "inf 标定未产出 request_throughput" 1
  fi
  hi="$(awk -v r="$crps" 'BEGIN{printf "%.2f", r*1.05}')"
  echo "[calib] inf 实际 rps=${crps} → 回测上界 hi=${hi}"
  echo "CALIB_RESULT inf_rps=${crps} rate_hi=${hi}"
  exit 0
}

# ============================================================
# 子命令: run — probe → derive → 标定 → 回测 → confirm → summary
# ============================================================
do_run() {
  (( ${#CASES[@]} == 1 )) || die "run 需要 --case (1 个 case)" 1
  [[ -n "$MODEL" ]] || die "run 需要 --model 或 MODEL env (bench_serving --model)" 1
  [[ "$MODE" == "bisect" || "$MODE" == "ladder" || "$MODE" == "grid" ]] || die "--mode 仅支持 bisect|ladder|grid" 1

  # ---- case 解析: INPUT,OUTPUT,HIT_RATE,TTFT_METRIC,TTFT_TARGETS_S,TPOT_METRIC,TPOT_TARGET_MS[,MAX_CONCURRENCY] ----
  local old_ifs="$IFS" case_str="${CASES[0]}"
  IFS=',' read -r CASE_INPUT_RAW CASE_OUTPUT_RAW CASE_HIT_PCT CASE_TTFT_METRIC CASE_TTFT_TARGETS CASE_TPOT_METRIC CASE_TPOT_TARGET_MS CASE_CONCURRENCY _rest <<< "$case_str"
  IFS="$old_ifs"
  [[ -n "$CASE_INPUT_RAW" && -n "$CASE_OUTPUT_RAW" && -n "$CASE_HIT_PCT" && -n "$CASE_TTFT_METRIC" \
     && -n "$CASE_TTFT_TARGETS" && -n "$CASE_TPOT_METRIC" && -n "$CASE_TPOT_TARGET_MS" ]] \
    || die "case 字段不足: INPUT,OUTPUT,HIT_RATE,TTFT_METRIC,TTFT_TARGETS_S,TPOT_METRIC,TPOT_TARGET_MS[,MAX_CONCURRENCY]" 1
  nn_dec "$CASE_HIT_PCT"        || die "HIT_RATE 非法: $CASE_HIT_PCT (0-100 百分数)" 1
  pos_dec "$CASE_TPOT_TARGET_MS" || die "TPOT_TARGET_MS 非法: $CASE_TPOT_TARGET_MS" 1

  CASE_INPUT_LEN="$(tokens_from_k "$CASE_INPUT_RAW")"   || die "INPUT 非法: $CASE_INPUT_RAW (如 32k / 4096)" 1
  CASE_OUTPUT_LEN="$(tokens_from_k "$CASE_OUTPUT_RAW")" || die "OUTPUT 非法: $CASE_OUTPUT_RAW (如 0.5k / 512)" 1
  auto_select_gq "$CASE_INPUT_LEN" "$CASE_OUTPUT_LEN" "$CASE_HIT_PCT"

  # 多档 TTFT SLO ('5|7') → bisect 判定取最严档 (min×1000 → ms)
  STRICT_TTFT_MS="$("$PY" - "$CASE_TTFT_TARGETS" <<'PY' | tr -d '\r'
import sys
vals = [float(x) for x in sys.argv[1].split("|") if x.strip()]
print(int(round(min(vals) * 1000)))
PY
)"
  [[ -n "$STRICT_TTFT_MS" ]] || die "TTFT_TARGETS_S 非法: $CASE_TTFT_TARGETS (如 5 或 5|7)" 1

  # ---- 服务端探测: ctx / page / dp ----
  local probe_out ctx page dp chunked maxrun sver s_model
  probe_out="$(probe_server)"
  read -r ctx page dp chunked maxrun sver s_model <<< "$probe_out"
  [[ -n "$ctx" ]] || die "无法探测服务端 (${BASE_URL} /get_server_info); 请确认服务已启动" 1
  dp="${DP_SIZE:-$dp}"; [[ -n "$dp" ]] || dp=1

  # ---- 命中率修正推导 ----
  local derive_out l_used
  derive_out="$(derive_lengths "$CASE_INPUT_LEN" "$CASE_HIT_PCT" "$QPP" "$page" "$dp" "$ctx" "$CASE_OUTPUT_LEN")" \
    || die "推导失败: L=$CASE_INPUT_LEN H=$CASE_HIT_PCT% page=$page dp=$dp ctx=$ctx out=$CASE_OUTPUT_LEN" 1
  read -r l_used CASE_SYSTEM_LEN CASE_QUESTION_LEN CASE_HIT_REQ_THEORY CASE_HIT_THEORY DERIVE_NOTES <<< "$derive_out"

  # ---- 运行参数 ----
  CASE_ID="$(slug "${CASE_INPUT_RAW}_${CASE_OUTPUT_RAW}_${CASE_HIT_PCT}")"
  CASE_NUM_PROMPTS=$((N_GROUPS * QPP))
  CUR_GROUPS="$N_GROUPS"; CUR_NUM_PROMPTS="$CASE_NUM_PROMPTS"
  CUR_SYSTEM_LEN="$CASE_SYSTEM_LEN"; CUR_QUESTION_LEN="$CASE_QUESTION_LEN"
  GROUPS_STD="full"
  [[ -n "$SEARCH_GROUPS" ]] || SEARCH_GROUPS=$((N_GROUPS / 2))
  (( SEARCH_GROUPS >= 2 )) || SEARCH_GROUPS=2

  OUT_DIR="${OUT_DIR:-./all_result/gsp_${RUN_ID}}"
  LOG_DIR="${OUT_DIR}/logs"; TRIALS_CSV="${OUT_DIR}/trials.csv"
  mkdir -p "$LOG_DIR"
  [[ -f "$TRIALS_CSV" ]] || printf '%s\n' "$TRIAL_FIELDS" > "$TRIALS_CSV"

  CSV_ARGS=("$CASE_ID" "$CASE_INPUT_RAW" "$CASE_OUTPUT_RAW" "$CASE_HIT_PCT" \
            "$CASE_HIT_REQ_THEORY" "$CASE_HIT_THEORY" "$CASE_SYSTEM_LEN" "$CASE_QUESTION_LEN" \
            "$page" "$N_GROUPS" "$QPP" "$dp" "$CASE_NUM_PROMPTS")
  export TTFT_TARGETS_S="$CASE_TTFT_TARGETS" TPOT_TARGET_MS="$CASE_TPOT_TARGET_MS" ANSWER

  echo "================ GSP 推导 (服务端感知) ================"
  echo "server         : ctx=${ctx} page=${page} dp=${dp} chunked=${chunked:-NA} max_run=${maxrun:-NA} ver=${sver:-NA}"
  echo "model          : ${s_model:-$MODEL}"
  echo "case           : id=${CASE_ID} input=${CASE_INPUT_LEN} output=${CASE_OUTPUT_LEN} hit=${CASE_HIT_PCT}%"
  echo "GSP 参数       : groups=${N_GROUPS}xqpp=${QPP}=${CASE_NUM_PROMPTS} prompts; P=${CASE_SYSTEM_LEN}/Ql=${CASE_QUESTION_LEN} [${DERIVE_NOTES}]"
  echo "命中率修正     : 目标 ${CASE_HIT_PCT}% → 请求理论 ${CASE_HIT_REQ_THEORY}% → 页对齐理论 ${CASE_HIT_THEORY}%"
  echo "SLO            : ${CASE_TTFT_METRIC}<=${CASE_TTFT_TARGETS}s (strict ${STRICT_TTFT_MS}ms) & ${CASE_TPOT_METRIC}<=${CASE_TPOT_TARGET_MS}ms"
  echo "回测           : mode=${MODE} rate=[${RATE_LO}, ${RATE_HI:-auto}] step=${RATE_STEP} out_dir=${OUT_DIR}"

  if [[ "$MODE" == "grid" ]]; then
    search_grid
    # grid 模式 (gsp_prefix 口径): 降序实测点即结论, 无 bisect/confirm 语义
  elif [[ "$MODE" == "bisect" ]]; then
    search_bisect
    # [3/3] 全尺寸 confirm: 校验答案在完整规模下仍满足 SLO (可用 --skip-confirm 跳过)
    if [[ "$SKIP_CONFIRM" != "1" && -n "$ANSWER" && "$ANSWER" != "INF_OK" ]]; then
      echo "======== [3/3] 全尺寸 confirm: rate=${ANSWER} ========"
      CUR_GROUPS="$N_GROUPS"; CUR_NUM_PROMPTS="$CASE_NUM_PROMPTS"; GROUPS_STD="full"
      run_trial "$ANSWER" confirm
    fi
  else
    search_ladder
  fi

  write_summary
  local verdict
  verdict="$(cat "${OUT_DIR}/VERDICT.txt" 2>/dev/null || echo FAILED)"
  [[ "$RUN_FAILURE" == "1" ]] && exit 4
  [[ "$verdict" == "NO_MEET" || "$verdict" == "FAILED" ]] && exit 3
  exit 0
}

case "$CMD" in
  estimate) do_estimate ;;
  calib)    do_calib ;;
  run)      do_run ;;
  -h|--help|help|"") usage ;;
  *) die "未知子命令 (estimate|calib|run; -h 查看用法)" 1 ;;
esac
