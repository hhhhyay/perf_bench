#!/usr/bin/env bash
# ============================================================
# SGLang 在线抓取 torch profiler (CPU/GPU) + 自动分析
#
# 流程:
#   1) 用 sla_bench/prefix_hit.py 起后台负载 (与容量测试同一数据集口径)
#   2) 等负载进入稳定态 (--wait-sec) 后 POST /start_profile
#   3) 轮询 output_dir 的 *.trace.json.gz 落盘 (num_steps 到点后服务端自动 stop)
#   4) 结束负载, 调 prof_analyze.py 产出 prof_summary.md / prof_kernels.csv,
#      再由 prof_dashboard.py 自动生成 prof_dashboard.html 可视化看板
#      (对账卡片: prof wall/busy 每 step  vs  客户端 TPOT×accept (decode) 或 TTFT (prefill))
#
# 用法:
#   bash sglang/prof_capture.sh --mode decode  --case '4k,1024,0' --bs 32
#   bash sglang/prof_capture.sh --mode prefill --case '4096,1,0'
#   bash sglang/prof_capture.sh --mode decode  --base-url http://10.243.1.5:32024 \
#        --output-dir /mnt/gbj/53/decode-in4k-out1k-bs32-profiles
#
# 参数:
#   --mode prefill|decode   必填。决定默认 case/步数/预热等待
#   --case IN,OUT,HIT       默认 prefill: 4096,1,0   decode: 4096,1024,0
#                           IN 支持 k/K/m/M 后缀; HIT=prefix cache 命中率(0-100)
#   --bs N                  decode 并发 (--max-concurrency), 默认 32; prefill 忽略
#   --num-steps N           抓取步数, 默认 prefill 5 / decode 10
#   --start-step N          对应 /start_profile 的 start_step, 默认 5
#   --activities A,B        profiler activities, 默认 CPU,GPU (可选 MEM/RPD/CUDA_PROFILER)
#   --wait-sec N            负载就绪后再稳定多少秒才抓, 默认 prefill 8 / decode 20
#   --load-need N           判定负载就绪的并发增量, 默认 decode=BS / prefill=4
#                           (从 /v1/loads 的 num_running_reqs 取, 相对起压前基线)
#   --ready-timeout N       等负载就绪的超时秒数, 默认 240 (驱动要先 import 模型栈)
#   --num-prompts N         驱动请求数, 默认 512 (输入 >=32k 时 128; 抓完即杀负载)
#   --output-dir DIR        trace 落盘目录 (服务端路径, 默认 ./all_result/prof_<...>_<ts>)
#   --local-trace-dir DIR   同一目录在客户端侧的路径 (共享存储路径映射不同时用)
#   --capture-timeout N     等 trace 落盘的超时秒数, 默认 1800
#                           (num_steps 到点只是停止采集; 高负载节点写完 8 个几十 MB
#                            的 gzip 可能还要十几分钟, 不要按采集时长估) 
#   --expect-ranks N        期望 trace 文件数, 默认自动取 /server_info 的 tp_size
#   --fake-prefill          decode 专用: 给 bench_serving 传 --fake-prefill,
#                           客户端在请求体注入 bootstrap_host/bootstrap_room。
#                           仅对 PD 分离且服务端带 --disaggregation-transfer-backend fake
#                           的 decode 实例有效 (跳过真实 prefill KV 传输, 压纯 decode)。
#                           普通单机服务传了也没用, 仍会走真实 prefill。
#   --flush-cache           每次驱动启动前 POST /flush_cache 清空 radix/prefix cache
#                           (SEED 固定时重复跑同一批随机 prompt 会命中上一轮 cache,
#                            造成第二轮吞吐虚高), 性能基准轮和抓取轮各清一次。
#   --perf-first            抓 prof 前先完整跑一遍同样的驱动负载作为客户端性能基准
#                           (结果在 <local>/perf_pass/), 再清 cache 进入抓取轮;
#                           对账关系: 客户端 Mean TPOT ≈ prof 每步耗时 / accept_length。
#   --no-analyze            只抓不分析
#   --no-driver             不启动内置负载 (已有外部压测流量时用)
#   --keep-load             抓完不杀驱动负载
#   --dry-run               只打印将要执行的命令
#
# 环境变量: BASE_URL MODEL PYTHON_BIN SEED PREFIX_SEED FAKE_PREFILL FLUSH_CACHE PERF_FIRST
# 说明: 分析需要能读到 output_dir (客户端与服务端共享挂载时直接可读;
#       否则请把目录拷回客户端后执行 python3 sglang/prof_analyze.py <dir>)。
# ============================================================

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(dirname -- "$script_dir")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
MODEL="${MODEL:-}"
SEED="${SEED:-42}"
PREFIX_SEED="${PREFIX_SEED:-$SEED}"

MODE=""
CASE=""
BS="32"
NUM_STEPS=""
START_STEP="5"
ACTIVITIES="CPU,GPU"
WAIT_SEC=""
LOAD_NEED=""
READY_TIMEOUT="240"
NUM_PROMPTS=""
OUTPUT_DIR=""
LOCAL_TRACE_DIR=""
CAPTURE_TIMEOUT="1800"
EXPECT_RANKS=""
DRY_RUN=0
ANALYZE=1
KEEP_LOAD=0
NO_DRIVER=0
FAKE_PREFILL="${FAKE_PREFILL:-0}"
FLUSH_CACHE="${FLUSH_CACHE:-0}"
PERF_FIRST="${PERF_FIRST:-0}"

log()  { echo "[INFO]  $(date '+%F %T') $*"; }
warn() { echo "[WARN]  $(date '+%F %T') $*"; }
die()  { echo "[ERROR] $(date '+%F %T') $*" >&2; exit "${2:-1}"; }

usage() { awk '/^# ={10,}/{c++; next} c==1{sub(/^# ?/,""); print} c>1{exit}' "$0"; exit 0; }

parse_len() {
  local v="${1,,}"
  case "$v" in
    *k) echo $(( ${v%?} * 1024 )) ;;
    *m) echo $(( ${v%?} * 1024 * 1024 )) ;;
    ''|*[!0-9]*) die "非法长度: $1" ;;
    *) echo "$v" ;;
  esac
}

while (( $# > 0 )); do
  case "$1" in
    --mode)            MODE="$2"; shift 2 ;;
    --case)            CASE="$2"; shift 2 ;;
    --bs|--max-concurrency) BS="$2"; shift 2 ;;
    --num-steps)       NUM_STEPS="$2"; shift 2 ;;
    --start-step)      START_STEP="$2"; shift 2 ;;
    --activities)      ACTIVITIES="$2"; shift 2 ;;
    --wait-sec)        WAIT_SEC="$2"; shift 2 ;;
    --load-need)       LOAD_NEED="$2"; shift 2 ;;
    --ready-timeout)   READY_TIMEOUT="$2"; shift 2 ;;
    --num-prompts)     NUM_PROMPTS="$2"; shift 2 ;;
    --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
    --local-trace-dir) LOCAL_TRACE_DIR="$2"; shift 2 ;;
    --capture-timeout) CAPTURE_TIMEOUT="$2"; shift 2 ;;
    --expect-ranks)    EXPECT_RANKS="$2"; shift 2 ;;
    --base-url)        BASE_URL="$2"; shift 2 ;;
    --model)           MODEL="$2"; shift 2 ;;
    --no-analyze)      ANALYZE=0; shift ;;
    --analyze)         ANALYZE=1; shift ;;
    --no-driver)       NO_DRIVER=1; shift ;;
    --fake-prefill)    FAKE_PREFILL=1; shift ;;
    --flush-cache)     FLUSH_CACHE=1; shift ;;
    --perf-first)      PERF_FIRST=1; shift ;;
    --keep-load)       KEEP_LOAD=1; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;
    -h|--help|help)    usage ;;
    *) die "未知参数: $1 (用 --help 查看)" 2 ;;
  esac
done

[[ "$MODE" == "prefill" || "$MODE" == "decode" ]] || die "--mode 必须为 prefill 或 decode"
[[ -n "$CASE" ]] || { if [[ "$MODE" == "prefill" ]]; then CASE="4096,1,0"; else CASE="4096,1024,0"; fi; }
[[ -n "$NUM_STEPS" ]] || { if [[ "$MODE" == "prefill" ]]; then NUM_STEPS=5; else NUM_STEPS=10; fi; }
[[ -n "$WAIT_SEC" ]] || { if [[ "$MODE" == "prefill" ]]; then WAIT_SEC=8; else WAIT_SEC=20; fi; }
[[ -n "$LOAD_NEED" ]] || { if [[ "$MODE" == "decode" ]]; then LOAD_NEED="$BS"; else LOAD_NEED=4; fi; }
[[ "$LOAD_NEED" =~ ^[1-9][0-9]*$ ]] || die "--load-need 必须是正整数: $LOAD_NEED"
[[ "$READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "--ready-timeout 必须是正整数: $READY_TIMEOUT"

IFS=',' read -r in_s out_s hit_s <<< "$CASE"
hit_s="${hit_s:-0}"
[[ "$hit_s" =~ ^[0-9]+$ ]] || die "case 的 HIT 必须是百分数: $CASE"
(( hit_s >= 0 && hit_s <= 100 )) || die "hit 须为 0..100: $CASE"
IN_LEN=$(parse_len "$in_s")
OUT_LEN=$(parse_len "$out_s")
[[ -n "$NUM_PROMPTS" ]] || { if (( IN_LEN >= 32768 )); then NUM_PROMPTS=128; else NUM_PROMPTS=512; fi; }
tag="${in_s}o${out_s}h${hit_s}"
[[ -n "${OUTPUT_DIR}" ]] || OUTPUT_DIR="$(pwd)/all_result/prof_${MODE}_${tag}_$(date +%Y%m%d_%H%M%S)"

# output_dir 是"服务端可见"的路径; 客户端侧可能用不同路径访问同一份共享存储,
# 此时用 --local-trace-dir 指向客户端视图。本地日志统一写 LOCAL_DIR。
[[ -n "${LOCAL_TRACE_DIR}" ]] || LOCAL_TRACE_DIR="$OUTPUT_DIR"
LOCAL_DIR="$LOCAL_TRACE_DIR"
if ! mkdir -p "$LOCAL_DIR" 2>/dev/null || [[ ! -w "$LOCAL_DIR" ]]; then
  LOCAL_DIR="$(pwd)/all_result/prof_${MODE}_${tag}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$LOCAL_DIR"
  warn "output_dir/local-trace-dir 本地不可写, 本地日志改写到 $LOCAL_DIR"
fi
log "mode=$MODE case=$tag in=$IN_LEN out=$OUT_LEN bs=$BS steps=$NUM_STEPS start_step=$START_STEP wait=${WAIT_SEC}s"
log "output_dir(服务端)=$OUTPUT_DIR"
log "local_trace_dir(客户端)=$LOCAL_TRACE_DIR"

ACT_JSON=$("$PYTHON_BIN" - "$ACTIVITIES" <<'PY'
import json, sys
print(json.dumps([a.strip() for a in sys.argv[1].split(",") if a.strip()]))
PY
)
PAYLOAD=$("$PYTHON_BIN" - "$OUTPUT_DIR" "$START_STEP" "$NUM_STEPS" "$ACT_JSON" <<'PY'
import json, sys
out, start, steps, acts = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), json.loads(sys.argv[4])
print(json.dumps({"output_dir": out, "start_step": start, "num_steps": steps, "activities": acts}))
PY
)

# decode 没有 prefix cache 命中率概念, 用官方客户端 python3 -m sglang.bench_serving
# (与 sglc.sh 完全一致); prefill 需要可控命中率, 用 sla_bench/prefix_hit.py。
if [[ "$MODE" == "decode" ]]; then
  driver_cmd=(
    "$PYTHON_BIN" -m sglang.bench_serving
    --backend sglang --base-url "$BASE_URL"
    --dataset-name random-ids --tokenize-prompt
    --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN"
    --random-range-ratio 1.0 --request-rate inf
    --max-concurrency "$BS"
    --num-prompts "$NUM_PROMPTS" --warmup-requests 0 --disable-tqdm
    --output-file "$LOCAL_DIR/driver.jsonl"
  )
  DRIVER_ENVS=()
  # --fake-prefill 只在 decode 有意义 (跳过真实 prefill, 压纯 decode);
  # 需要服务端是 PD 分离 decode 实例且带 --disaggregation-transfer-backend fake。
  if (( FAKE_PREFILL )); then
    driver_cmd+=(--fake-prefill)
    log "decode 客户端: python3 -m sglang.bench_serving + --fake-prefill (需 PD decode 服务端)"
  else
    log "decode 客户端: python3 -m sglang.bench_serving (与 sglc.sh 一致)"
  fi
else
  driver_cmd=(
    "$PYTHON_BIN" "$repo_root/sla_bench/prefix_hit.py"
    --backend sglang --base-url "$BASE_URL"
    --dataset-name random-ids --tokenize-prompt
    --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN"
    --random-range-ratio 1.0 --request-rate inf
    --num-prompts "$NUM_PROMPTS" --warmup-requests 0 --disable-tqdm
    --output-file "$LOCAL_DIR/driver.jsonl"
  )
  DRIVER_ENVS=(PREFIX_HIT="$hit_s" PREFIX_SEED="$PREFIX_SEED")
  log "prefill 客户端: sla_bench/prefix_hit.py (PREFIX_HIT=$hit_s)"
fi
[[ -n "$MODEL" ]] && driver_cmd+=(--model "$MODEL")
(( FAKE_PREFILL )) && [[ "$MODE" != "decode" ]] \
  && die "--fake-prefill 只适用于 decode 模式 (prefill 本身就是真实 prefill)"

if (( DRY_RUN )); then
  if (( NO_DRIVER )); then
    log "[dry-run] --no-driver: 不启动内置负载"
  else
    log "[dry-run] 驱动命令:"
    printf '  cd %q' "$repo_root"
    (( ${#DRIVER_ENVS[@]} > 0 )) && printf ' %q' "${DRIVER_ENVS[@]}"
    printf ' exec'
    printf ' %q' "${driver_cmd[@]}"; printf '\n'
  fi
  log "[dry-run] start_profile payload: $PAYLOAD"
  (( PERF_FIRST )) && log "[dry-run] --perf-first: 抓取前先完整跑一遍驱动作为性能基准 (输出 $LOCAL_DIR/perf_pass/)"
  (( FLUSH_CACHE )) && log "[dry-run] 每次驱动启动前将 POST $BASE_URL/flush_cache"
  log "[dry-run] 分析: $PYTHON_BIN $script_dir/prof_analyze.py $OUTPUT_DIR --mode $MODE"
  exit 0
fi

# ---- 1. 健康检查 ----
curl -s --max-time 5 -o /dev/null "$BASE_URL/health" || die "服务不可达: $BASE_URL/health"

# 期望 trace 文件数: 默认取 /server_info 的 tp_size, 取不到按 1 处理
si_tmp="$LOCAL_DIR/server_info.json"
curl -s --max-time 5 "$BASE_URL/server_info" > "$si_tmp" 2>/dev/null || true
if [[ -z "$EXPECT_RANKS" ]]; then
  EXPECT_RANKS=$("$PYTHON_BIN" - "$si_tmp" <<'PY' 2>/dev/null || echo 1
import json, sys

def dig(o, key):
    if isinstance(o, dict):
        if isinstance(o.get(key), int) and o[key] > 0:
            return o[key]
        for v in o.values():
            r = dig(v, key)
            if r:
                return r
    elif isinstance(o, list):
        for v in o:
            r = dig(v, key)
            if r:
                return r
    return 0

try:
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    print(1)
else:
    print(dig(data, "tp_size") or 1)
PY
)
fi
[[ "$EXPECT_RANKS" =~ ^[1-9][0-9]*$ ]] || EXPECT_RANKS=1
log "期望 trace 文件数: $EXPECT_RANKS"

# ---- 2. 起后台负载 ----
# 读取服务端当前并发请求数: 优先 /v1/loads 的 num_running_reqs 之和,
# 回退 /metrics 的 sglang:num_running_reqs; 两者都不可用时输出空串。
probe_running() {
  local body
  body=$(curl -s --max-time 3 "$BASE_URL/v1/loads" 2>/dev/null)
  case "$body" in
    \{*)
      printf '%s' "$body" | "$PYTHON_BIN" -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(sum(int(x.get("num_running_reqs") or 0) for x in (d.get("loads") or [])))
except Exception:
    pass
' 2>/dev/null
      return 0 ;;
  esac
  body=$(curl -s --max-time 3 "$BASE_URL/metrics" 2>/dev/null)
  case "$body" in
    *sglang:num_running_reqs*)
      printf '%s\n' "$body" | "$PYTHON_BIN" -c '
import sys
tot = 0.0
for line in sys.stdin:
    if line.startswith("sglang:num_running_reqs"):
        try:
            tot += float(line.rsplit(" ", 1)[1])
        except Exception:
            pass
print(int(tot))
' 2>/dev/null
      return 0 ;;
  esac
  return 0
}

driver_log="$LOCAL_DIR/driver.log"
driver_pid=""
: > "$driver_log"

flush_cache() {
  log "POST $BASE_URL/flush_cache ..."
  local resp
  if resp=$(curl -sS --max-time 30 -X POST "$BASE_URL/flush_cache" 2>&1); then
    log "flush_cache -> ${resp:-OK}"
  else
    warn "flush_cache 请求失败: ${resp}; 忽略并继续"
  fi
}

# 只做轻量的 spec 查找, 不真正 import (import sglang 会拉起 torch,
# 在负载高/共享存储繁忙的节点上可能卡住很久)
# 按模式检查具体子模块: decode 用 sglang.bench_serving, prefill 用 sglang.benchmark
# (prefix_hit.py 里 from sglang.benchmark import serving)。
# 只查顶层 'sglang' 会漏过"装了 sglang 但没装 benchmark"的情况。
check_driver_python() {
  local need_mod
  if [[ "$MODE" == "decode" ]]; then need_mod="sglang.bench_serving"; else need_mod="sglang.benchmark"; fi
  timeout 60 "$PYTHON_BIN" -c \
    "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$need_mod') else 1)" 2>/dev/null \
    || die "当前 PYTHON_BIN 找不到 $need_mod, 请在带完整 sglang(含 benchmark) 的客户端容器运行, 或用 --no-driver 配外部负载"
}

# 客户端性能基准轮 (--perf-first): 用同样的驱动负载完整跑一遍, 不抓 prof。
# 目的是拿到可直接对账的客户端指标 (TPOT/吞吐), perf 结果落在 <LOCAL_DIR>/perf_pass/。
run_perf_pass() {
  local perf_dir="$LOCAL_DIR/perf_pass"
  mkdir -p "$perf_dir"
  local perf_cmd=("${driver_cmd[@]}") i
  for i in "${!perf_cmd[@]}"; do
    [[ "${perf_cmd[$i]}" == "--output-file" ]] && perf_cmd[$((i+1))]="$perf_dir/driver.jsonl"
  done
  check_driver_python
  (( FLUSH_CACHE )) && flush_cache
  log "[perf基准] 开始完整性能基准轮 (num_prompts=$NUM_PROMPTS, 同抓取轮负载), 输出 $perf_dir/"
  ( cd "$repo_root" && exec env "${DRIVER_ENVS[@]+"${DRIVER_ENVS[@]}"}" "${perf_cmd[@]}" ) > "$perf_dir/driver.log" 2>&1
  local rc=$?
  if (( rc != 0 )); then
    tail -n 20 "$perf_dir/driver.log" >&2 || true
    die "客户端性能基准轮异常退出 rc=$rc, 详见 $perf_dir/driver.log"
  fi
  log "[perf基准] 完成, 客户端指标:"
  "$PYTHON_BIN" - "$perf_dir/driver.jsonl" <<'PY' 2>/dev/null || true
import json, sys
lines = [ln for ln in open(sys.argv[1], encoding="utf-8") if ln.strip()]
rows = None
for ln in reversed(lines):
    try:
        rows = json.loads(ln)
        break
    except Exception:
        continue
if isinstance(rows, list) and rows:
    rows = rows[-1]
keys = ["duration", "num_prompts", "request_throughput", "output_throughput",
        "total_token_throughput", "mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms",
        "mean_e2e_latency_ms", "accept_length", "mean_accept_length"]
if isinstance(rows, dict):
    hits = {k: rows[k] for k in keys if k in rows}
    if not hits:
        print(f"  (jsonl 顶层键: {sorted(rows)[:12]}...)")
    for k, v in hits.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, (int, float)) else f"  {k}: {v}")
PY
  log "[perf基准] 对账公式: 客户端 Mean TPOT ≈ prof 每步耗时 / accept_length"
}

start_driver() {
  check_driver_python
  # exec 让子 shell 直接被 python 替换, 否则 driver_pid 只是子 shell,
  # kill 它不会杀掉真正的 benchmark 进程 (会残留继续压测)
  # DRIVER_ENVS: decode 模式为空 (bench_serving 不读 PREFIX_HIT),
  #             prefill 模式为 PREFIX_HIT/PREFIX_SEED
  ( cd "$repo_root" && exec env "${DRIVER_ENVS[@]+"${DRIVER_ENVS[@]}"}" \
      "${driver_cmd[@]}" ) >> "$driver_log" 2>&1 &
  driver_pid=$!
  log "驱动负载 pid=$driver_pid, 日志=$driver_log"
}
# ---- 客户端性能基准轮 (可选): 抓 prof 前先完整跑一遍同负载并记录客户端指标 ----
if (( PERF_FIRST )) && (( ! NO_DRIVER )); then
  run_perf_pass
elif (( PERF_FIRST )) && (( NO_DRIVER )); then
  warn "--perf-first 与 --no-driver 同用时没有内置负载可跑, 跳过性能基准轮"
fi

if (( NO_DRIVER )); then
  log "--no-driver: 跳过内置负载, 请确保已有外部流量"
else
  base_running=$(probe_running)
  base_running=${base_running:-0}
  log "起压前 num_running_reqs 基线: ${base_running:-未知}"
  (( FLUSH_CACHE )) && flush_cache
  start_driver
fi

# 等到真实负载跑起来: num_running_reqs 达到 --load-need 且连续 3 次采样(约 6s)
# 保持, 才算稳定。避免"驱动还在 import 模型栈"时就发 /start_profile 抓到空闲步。
POLL_SEC=2
wait_load_ready() {
  local base="$1" need="$2"
  # 目标只取 need: 起压前若已有别人的/上一轮的请求在跑, base+need 往往永远达不到
  # (那些请求会先排空, 服务端也不会让总数超过 max_running_requests)。
  local target="$need"
  local deadline=$(( $(date +%s) + READY_TIMEOUT ))
  local cur="" hits=0
  log "等待负载就绪: num_running_reqs 需 ≥ $target (起压前基线 $base)"
  while (( $(date +%s) < deadline )); do
    if [[ -n "$driver_pid" ]] && ! kill -0 "$driver_pid" 2>/dev/null; then
      warn "驱动负载已退出, 未达到就绪条件 (见 $driver_log)"
      return 1
    fi
    cur=$(probe_running)
    if [[ -z "$cur" ]]; then
      warn "/v1/loads 与 /metrics 均不可用, 退回固定等待 ${WAIT_SEC}s"
      sleep "$WAIT_SEC"
      return 0
    fi
    if (( cur >= target )); then
      (( hits++ ))
      if (( hits >= 3 )); then
        log "负载就绪: num_running_reqs=$cur (连续 ${hits} 次采样)"
        return 0
      fi
    else
      hits=0
    fi
    sleep "$POLL_SEC"
  done
  warn "等待负载就绪超时 ${READY_TIMEOUT}s: 当前 ${cur:-?} / 目标 $target (基线 $base)"
  return 1
}

cleanup() {
  [[ -z "$driver_pid" ]] && return 0
  if (( KEEP_LOAD == 0 )) && kill -0 "$driver_pid" 2>/dev/null; then
    kill "$driver_pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do kill -0 "$driver_pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$driver_pid" 2>/dev/null || true
    log "已结束驱动负载 pid=$driver_pid"
  elif (( KEEP_LOAD == 1 )); then
    log "--keep-load: 保留驱动负载 pid=$driver_pid"
  fi
}
trap cleanup EXIT

# ---- 3. 等负载真正跑起来并稳定后再抓 ----
if (( NO_DRIVER )); then
  wait_load_ready 0 1 || warn "外部流量未确认, 继续按固定等待抓取"
  log "再等 ${WAIT_SEC}s 让负载进入稳定态..."
  sleep "$WAIT_SEC"
else
  if ! wait_load_ready "$base_running" "$LOAD_NEED"; then
    if [[ -n "$driver_pid" ]] && kill -0 "$driver_pid" 2>/dev/null; then
      die "负载未达到就绪条件 (num_running_reqs 增量 < $LOAD_NEED) 且进程仍在跑, 见 $driver_log; 可加大 --ready-timeout 或用 --no-driver"
    fi
    warn "驱动负载已退出, 重新拉起一次 (见 $driver_log)"
    start_driver
    base_running=$(probe_running)
    base_running=${base_running:-0}
    wait_load_ready "$base_running" "$LOAD_NEED" \
      || die "二次拉起后仍未就绪, 检查 $driver_log; 可加大 --ready-timeout 或用 --no-driver"
  fi
  log "负载已稳定, 再等 ${WAIT_SEC}s 后开始抓取..."
  sleep "$WAIT_SEC"
fi

log "POST $BASE_URL/start_profile"
resp=$(curl -sS --max-time 30 -X POST "$BASE_URL/start_profile" \
  -H "Content-Type: application/json" -d "$PAYLOAD" 2>&1) || die "start_profile 请求失败: $resp"
printf '%s\n' "$resp" > "$LOCAL_DIR/start_profile_resp.json"
log "start_profile -> $resp"

# ---- 4. 等 trace 落盘 (num_steps 到点后服务端自动 stop) ----
# 注意: num_steps 到点只是"停止采集", 落盘 (gzip 写 8 个几十 MB 的文件) 在高负载
# 节点上可能再花十几分钟, 所以这里要一直等到文件数齐 + 大小稳定, 不能按秒数提前放弃。
deadline=$(( $(date +%s) + CAPTURE_TIMEOUT ))
count=0
last_report=$(( $(date +%s) ))

count_traces() {
  find "$LOCAL_TRACE_DIR" -maxdepth 1 -name '*.trace.json.gz' -size +0c 2>/dev/null | wc -l
}

while (( $(date +%s) < deadline )); do
  count=$(count_traces)
  (( count >= EXPECT_RANKS )) && break
  now=$(date +%s)
  if (( now - last_report >= 30 )); then
    last_report=$now
    zero=$(find "$LOCAL_TRACE_DIR" -maxdepth 1 -name '*.trace.json.gz' -size 0c 2>/dev/null | wc -l)
    if [[ -n "$driver_pid" ]] && ! kill -0 "$driver_pid" 2>/dev/null; then
      warn "驱动负载已结束, 仍在等服务端落盘 ($count 个有内容, $zero 个为空) ..."
    else
      log "等待 trace 落盘: 有内容 $count / 期望 $EXPECT_RANKS, 空文件 $zero"
    fi
  fi
  sleep 3
done

if (( count < EXPECT_RANKS )); then
  warn "trace 文件不足 ($count/$EXPECT_RANKS), 尝试 /stop_profile 强制落盘"
  curl -sS --max-time 30 -X POST "$BASE_URL/stop_profile" >/dev/null 2>&1 || true
  for _ in $(seq 1 10); do
    sleep 2
    count=$(find "$LOCAL_TRACE_DIR" -maxdepth 1 -name '*.trace.json.gz' -size +0c 2>/dev/null | wc -l)
    (( count >= EXPECT_RANKS )) && break
  done
fi
(( count > 0 )) || die "未生成 trace 文件: 确认 output_dir 在服务端可写; 若客户端路径不同请加 --local-trace-dir"

# 抓完立即结束负载
cleanup
trap - EXIT

log "trace 文件: $count 个"
find "$LOCAL_TRACE_DIR" -maxdepth 1 -name '*.trace.json.gz' -size +0c -printf '  %f  %s bytes\n' 2>/dev/null || true

# 等文件写完: 大小连续两次不变(间隔 3s) 且 gzip 完整性校验通过
# (服务端导出 70MB+ 的大 trace 时, 中途可能出现数秒的大小停滞, 只看大小会误判)
snap() {
  find "$LOCAL_TRACE_DIR" -maxdepth 1 -name '*.trace.json.gz' -size +0c -printf '%f:%s\n' 2>/dev/null | sort | md5sum
}
gz_ok() {
  local f
  for f in "$LOCAL_TRACE_DIR"/*.trace.json.gz; do
    [[ -e "$f" ]] || return 1
    gzip -t "$f" 2>/dev/null || return 1
  done
  return 0
}
prev=""
same=0
ready=0
for _ in $(seq 1 60); do
  cur=$(snap)
  cur_cnt=$(find "$LOCAL_TRACE_DIR" -maxdepth 1 -name '*.trace.json.gz' -size +0c 2>/dev/null | wc -l)
  if [[ "$cur" == "$prev" ]]; then same=$((same + 1)); else same=0; fi
  if (( cur_cnt >= EXPECT_RANKS && same >= 2 )) && gz_ok; then
    ready=1
    break
  fi
  prev="$cur"
  sleep 3
done
(( ready )) || warn "trace 可能在写入中 (180s 内大小仍变化或 gzip 校验未通过), 分析结果可能不完整"

# 保存一份服务端参数快照, 便于回溯
curl -s --max-time 5 "$BASE_URL/server_info" > "$LOCAL_DIR/server_info.json" 2>/dev/null || true

# ---- 5. 自动分析 ----
if (( ANALYZE )); then
  if "$PYTHON_BIN" "$script_dir/prof_analyze.py" "$LOCAL_TRACE_DIR" --mode "$MODE" --top 20; then
    log "分析完成: $LOCAL_TRACE_DIR/prof_summary.md"
    if "$PYTHON_BIN" "$script_dir/prof_dashboard.py" "$LOCAL_TRACE_DIR" --mode "$MODE" \
         --num-steps "$NUM_STEPS" --bs "$BS" --case "$tag"; then
      log "可视化看板: $LOCAL_TRACE_DIR/prof_dashboard.html"
    else
      warn "可视化看板生成失败 (prof_summary.md 不受影响)"
    fi
  else
    warn "分析失败 (trace 可能不可读, 详见上面的错误)"
  fi
fi

log "完成。服务端目录: $OUTPUT_DIR  客户端目录: $LOCAL_TRACE_DIR"
