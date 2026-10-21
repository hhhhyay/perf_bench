#!/usr/bin/env bash
# ============================================================
# SGLang 单侧 Prefill TTFT 矩阵压测
#
# 使用与 sglc.sh 相同的服务配置和 bench_serving 参数，
# 只改变输入长度、per-DP BS、输出 1 token 和 request-rate 搜索。
# ============================================================

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
out_dir="./all_result/bench_prefill_ttft_${run_id}"
log_dir="${out_dir}/logs"
trials_csv="${out_dir}/prefill_ttft_trials.csv"
matrix_csv="${out_dir}/prefill_ttft_matrix.csv"

NODE_NAME="${NODE_NAME:-$(hostname -s 2>/dev/null || hostname 2>/dev/null || printf 'unknown')}"
CONTAINER_NAME="${CONTAINER_NAME:-${HOSTNAME:-unknown}}"
RUN_STARTED_AT="${RUN_STARTED_AT:-$(date +%Y-%m-%dT%H:%M:%S%z)}"
SCRIPT_COMMIT="$(git -C "${script_dir}/.." rev-parse --short HEAD 2>/dev/null || true)"

BASE_URL="${BASE_URL:-http://12.12.12.37:10015}"
MODEL="${MODEL:-/public4/opendas/DL_DATA/llm-models/DeepSeek-V4-Flash-0731-FP8-Channel}"
DP_SIZE="${DP_SIZE:-}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"

TP_SIZE="${TP_SIZE:-}"
PP_SIZE="${PP_SIZE:-}"
EP_SIZE="${EP_SIZE:-}"
ATTN_CP_SIZE="${ATTN_CP_SIZE:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-}"
DP_ATTENTION="${DP_ATTENTION:-}"
CUDA_GRAPH_DECODE_BS="${CUDA_GRAPH_DECODE_BS:-}"
PARALLEL_CONFIG="${PARALLEL_CONFIG:-}"

USE_FAKE_PREFILL="${USE_FAKE_PREFILL:-0}"
NUM_PROMPTS_MULTIPLIER="${NUM_PROMPTS_MULTIPLIER:-1}"
TOKEN_UNIT="${TOKEN_UNIT:-1024}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PREFILL_TTFT_INPUTS="${PREFILL_TTFT_INPUTS:-32 64 128}"
PREFILL_SLA_RULES="${PREFILL_SLA_RULES:-}"
PREFILL_TTFT_METRIC="${PREFILL_TTFT_METRIC:-mean}"
PREFILL_TTFT_TARGETS_S="${PREFILL_TTFT_TARGETS_S:-3 5 10}"
PREFILL_TTFT_SEARCH_ITERS="${PREFILL_TTFT_SEARCH_ITERS:-7}"
PREFILL_TTFT_RATE_STEP="${PREFILL_TTFT_RATE_STEP:-0.1}"
PREFILL_TTFT_INF_MAX_TRIALS="${PREFILL_TTFT_INF_MAX_TRIALS:-5}"
PREFILL_TTFT_INF_NO_IMPROVEMENT_LIMIT="${PREFILL_TTFT_INF_NO_IMPROVEMENT_LIMIT:-2}"
PREFILL_TTFT_BS_RULES="${PREFILL_TTFT_BS_RULES:-8:256,16:128,32:64,64:32,128:16}"
PREFILL_TTFT_TEST_MODE="${PREFILL_TTFT_TEST_MODE:-}"
PLOT_REQUESTS=()
PLOT_CELL="${PLOT_CELL:-rps}"

write_run_metadata() {
  local metadata_path="${out_dir}/run_metadata.json"
  "${PYTHON_BIN}" - "${metadata_path}" \
    "${RUN_STARTED_AT}" "${script_dir}/$(basename "${BASH_SOURCE[0]}")" \
    "${SCRIPT_COMMIT}" "${run_id}" "${BASE_URL}" "${MODEL}" \
    "${NODE_NAME}" "${CONTAINER_NAME}" "$(hostname 2>/dev/null || true)" \
    "${out_dir}" "${DP_SIZE}" "${PARALLEL_CONFIG}" <<'PY_METADATA'
import json
import os
import sys
import tempfile

(path, started_at, script, script_commit, run_id, base_url, model, node,
 container, container_hostname, output_dir, dp_size, parallel_config) = sys.argv[1:]
prefixes = ("PREFILL_", "TOKEN_UNIT", "DP_SIZE", "TP_SIZE", "PP_SIZE",
            "EP_SIZE", "ATTN_CP_SIZE", "KV_CACHE_DTYPE", "MEM_FRACTION_STATIC",
            "DP_ATTENTION", "USE_FAKE_PREFILL", "NUM_PROMPTS_MULTIPLIER")
environment = {
    key: value for key, value in os.environ.items()
    if key.startswith(prefixes)
}
metadata = {
    "started_at": started_at,
    "script": script,
    "script_commit": script_commit,
    "run_id": run_id,
    "base_url": base_url,
    "model": model,
    "node": node,
    "container": container,
    "container_hostname": container_hostname,
    "output_dir": output_dir,
    "dp_size": dp_size,
    "parallel_config": parallel_config,
    "environment": environment,
}
parent = os.path.dirname(os.path.abspath(path))
os.makedirs(parent, exist_ok=True)
fd, tmp = tempfile.mkstemp(prefix=".run_metadata.", dir=parent, text=True)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
os.replace(tmp, path)
PY_METADATA
}

write_log_header() {
  local log_file="$1"
  {
    printf '# perf_bench run metadata: %s\n' "${out_dir}/run_metadata.json"
    printf '# model: %s\n' "${MODEL}"
    printf '# node: %s\n' "${NODE_NAME}"
    printf '# container: %s\n' "${CONTAINER_NAME}"
    printf '# base_url: %s\n' "${BASE_URL}"
    printf '# run_id: %s\n' "${run_id}"
    printf '# started_at: %s\n' "${RUN_STARTED_AT}"
    printf '#\n'
  } > "${log_file}"
}

usage() {
  cat <<'USAGE'
Usage: bash sglang/prefill_ttft.sh [--plot MODE]

MODE:
  mean       生成 Mean TTFT 容量矩阵
  p99        生成 P99 TTFT 容量矩阵
  sla        按每个输入 case 自己的 target_metric/target_value 生成容量矩阵
  line       生成 TTFT 随请求率变化曲线
  heatmap    生成原始 TTFT 热力图

格子维度：--cell rps|input_throughput（简写 input_tp）。列始终按输入 token/BS 展示。
有限请求率默认保留两位小数，搜索最小区间为 0.1 req/s（可用 PREFILL_TTFT_RATE_STEP 调大）。
未指定 --plot 时只生成 CSV。--plot 可重复指定多个 MODE。
USAGE
}

while (( $# > 0 )); do
  case "$1" in
    --plot)
      [[ $# -ge 2 ]] || { echo "[ERROR] --plot 需要一个模式" >&2; usage >&2; exit 1; }
      case "$2" in
        mean|p99|sla|line|heatmap) PLOT_REQUESTS+=("$2") ;;
        *) echo "[ERROR] 不支持的 --plot 模式: $2" >&2; usage >&2; exit 1 ;;
      esac
      shift 2
      ;;
    --plot=*)
      plot_mode="${1#*=}"
      case "${plot_mode}" in
        mean|p99|sla|line|heatmap) PLOT_REQUESTS+=("${plot_mode}") ;;
        *) echo "[ERROR] 不支持的 --plot 模式: ${plot_mode}" >&2; usage >&2; exit 1 ;;
      esac
      shift
      ;;
    --cell)
      [[ $# -ge 2 ]] || { echo "[ERROR] --cell 需要 rps 或 input_throughput" >&2; usage >&2; exit 1; }
      case "$2" in
        rps) PLOT_CELL="rps" ;;
        input_throughput|input_tp) PLOT_CELL="input_throughput" ;;
        *) echo "[ERROR] 不支持的 --cell 值: $2" >&2; usage >&2; exit 1 ;;
      esac
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] 未知参数: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! "${TOKEN_UNIT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] TOKEN_UNIT 必须是正整数，当前为: ${TOKEN_UNIT}" >&2
  exit 1
fi

if [[ ! "${PREFILL_TTFT_SEARCH_ITERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] PREFILL_TTFT_SEARCH_ITERS 必须是正整数" >&2
  exit 1
fi

if ! awk -v value="${PREFILL_TTFT_RATE_STEP}" 'BEGIN {
  if (value !~ /^[0-9]+([.][0-9]*)?$/ && value !~ /^[.][0-9]+$/) exit 1
  exit !(value >= 0.1)
}' 2>/dev/null; then
  echo "[ERROR] PREFILL_TTFT_RATE_STEP 必须是不小于 0.1 的数字，当前为: ${PREFILL_TTFT_RATE_STEP}" >&2
  exit 1
fi

for value_name in PREFILL_TTFT_INF_MAX_TRIALS PREFILL_TTFT_INF_NO_IMPROVEMENT_LIMIT; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] ${value_name} 必须是正整数，当前为: ${value}" >&2
    exit 1
  fi
done

case "${PREFILL_TTFT_METRIC}" in
  mean|mean_ttft_ms) PREFILL_TTFT_METRIC="mean_ttft_ms" ;;
  p90|p90_ttft_ms) PREFILL_TTFT_METRIC="p90_ttft_ms" ;;
  p95|p95_ttft_ms) PREFILL_TTFT_METRIC="p95_ttft_ms" ;;
  p99|p99_ttft_ms) PREFILL_TTFT_METRIC="p99_ttft_ms" ;;
  *)
    echo "[ERROR] PREFILL_TTFT_METRIC 仅支持 mean/p90/p95/p99，当前为: ${PREFILL_TTFT_METRIC}" >&2
    exit 1
    ;;
esac

prefill_sla_rules_text() {
  if [[ -n "${PREFILL_SLA_RULES}" ]]; then
    printf "%s\n" "${PREFILL_SLA_RULES}"
    return 0
  fi

  local target_list="${PREFILL_TTFT_TARGETS_S}"
  local input_list="${PREFILL_TTFT_INPUTS//,/ }"
  local input_k target
  for input_k in ${input_list}; do
    for target in ${target_list//,/ }; do
      printf "%s:%s:%s:0," "${input_k}" "${PREFILL_TTFT_METRIC}" "${target}"
    done
  done
  printf "\n"
}

seconds_to_ms() {
  local seconds="$1"
  LC_ALL=C awk -v value="${seconds}" 'BEGIN { printf "%.6f", value * 1000 }' \
    | sed -E 's/0+$//; s/[.]$//'
}

prefill_sla_rules_for_input() {
  local wanted_input="$1"
  local rules_text rule input_k metric target_seconds near_seconds extra
  rules_text="$(prefill_sla_rules_text)"
  IFS=',' read -ra rules <<< "${rules_text}"
  for rule in "${rules[@]}"; do
    rule="${rule//[[:space:]]/}"
    [[ -n "${rule}" ]] || continue
    IFS=':' read -r input_k metric target_seconds near_seconds extra <<< "${rule}"
    [[ "${input_k}" == "${wanted_input}" ]] || continue
    printf "%s|%s|%s\n" "${metric}" "$(seconds_to_ms "${target_seconds}")" "$(seconds_to_ms "${near_seconds}")"
  done
}

decimal_value() {
  [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]
}

positive_decimal_value() {
  decimal_value "$1" && awk -v value="$1" 'BEGIN { exit !(value > 0) }'
}

nonnegative_decimal_value() {
  decimal_value "$1" && awk -v value="$1" 'BEGIN { exit !(value >= 0) }'
}

prefill_metric_supported() {
  case "$1" in
    mean_ttft_ms|p50_ttft_ms|std_ttft_ms|p75_ttft_ms|p90_ttft_ms|p95_ttft_ms|p99_ttft_ms|p999_ttft_ms|max_ttft_ms)
      return 0 ;;
    *)  return 1 ;;
  esac
}

validate_prefill_sla_rules() {
  local rules_text rule input_k metric target_seconds near_seconds extra
  rules_text="$(prefill_sla_rules_text)"
  [[ -n "${rules_text//[[:space:],]/}" ]] || {
    echo "[ERROR] PREFILL_SLA_RULES 不能为空" >&2
    return 1
  }

  IFS=',' read -ra rules <<< "${rules_text}"
  for rule in "${rules[@]}"; do
    rule="${rule//[[:space:]]/}"
    [[ -n "${rule}" ]] || continue
    IFS=':' read -r input_k metric target_seconds near_seconds extra <<< "${rule}"
    if [[ -n "${extra:-}" ]] || ! decimal_value "${input_k}" \
      || ! positive_decimal_value "${input_k}" \
      || ! prefill_metric_supported "${metric}" \
      || ! positive_decimal_value "${target_seconds}" \
      || ! nonnegative_decimal_value "${near_seconds}"; then
      echo "[ERROR] 非法 PREFILL_SLA_RULES 规则: ${rule}" >&2
      echo "[ERROR] 支持格式: input_k:metric:target_seconds:near_seconds" >&2
      echo "[ERROR] target/near 使用秒，可填写小数；metric 必须是已提取的 TTFT 指标" >&2
      return 1
    fi
  done
}

load_capacity_from_loads_api() {
  local loads_url="${BASE_URL%/}/v1/loads?include=all"
  local values

  values="$(
    curl -fsS "${loads_url}" 2>/dev/null \
    | "${PYTHON_BIN}" -c 'import json,sys
d=json.load(sys.stdin)
l=d.get("data") or d.get("loads") or []
if isinstance(l,dict): l=list(l.values())
if not isinstance(l,list) or not l: raise SystemExit(1)
def val(x,k):
    try: return int(float(x[k]))
    except (KeyError,TypeError,ValueError): return None
dp=d.get("dp_size") or d.get("dp") or len(l)
tokens=[val(x,"max_total_num_tokens") for x in l if isinstance(x,dict)]
reqs=[val(x,"max_running_requests") for x in l if isinstance(x,dict)]
tokens=[x for x in tokens if x is not None]
reqs=[x for x in reqs if x is not None]
if not tokens or not reqs: raise SystemExit(1)
print(int(dp), min(tokens), min(reqs))' 2>/dev/null \
    || true
  )"

  if [[ -n "${values}" ]]; then
    read -r DP_SIZE MAX_TOTAL_TOKENS MAX_RUNNING_REQUESTS <<< "${values}"
    echo "[INFO] capacity: DP_SIZE=${DP_SIZE} MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS} MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}"
    return 0
  fi
  return 1
}

parse_metric_value() {
  local metric_name="$1"
  local metrics_text="$2"
  printf "%s\n" "${metrics_text}" \
    | grep -E "^${metric_name}([{ ]|$)" \
    | head -n 1 \
    | awk '{
        for (i = NF; i >= 1; i--) {
          if ($i ~ /^-?[0-9]+(\.[0-9]+)?$/) {
            printf "%.0f\n", $i + 0
            exit
          }
        }
      }'
}

parse_dp_size_from_metrics() {
  local metrics_text="$1"
  local rank_count explicit_dp
  rank_count="$(
    printf "%s\n" "${metrics_text}" \
      | grep -oE 'dp_rank="[0-9]+"' \
      | sed -E 's/.*"([0-9]+)"/\1/' \
      | sort -nu | wc -l | awk '{print $1}'
  )"
  if [[ "${rank_count:-0}" =~ ^[0-9]+$ ]] && (( rank_count > 0 )); then
    printf "%s\n" "${rank_count}"
    return
  fi
  explicit_dp="$(parse_metric_value "sglang:dp_size" "${metrics_text}")"
  if [[ -n "${explicit_dp}" ]] && (( explicit_dp > 0 )); then
    printf "%s\n" "${explicit_dp}"
  fi
}

fetch_model_from_metrics() {
  local metrics_url="$1"
  curl -s --connect-timeout 5 --max-time 30 "${metrics_url}" 2>/dev/null \
    | sed -n 's/.*model_name="\([^"]*\)".*/\1/p' | head -n 1
}

load_server_info() {
  local info_url="${BASE_URL%/}/server_info"
  local values

  values="$(
    curl -fsS "${info_url}" 2>/dev/null \
    | "${PYTHON_BIN}" -c 'import sys,json
d=json.load(sys.stdin)
tp=d.get("tp_size"); dp=d.get("dp_size"); pp=d.get("pp_size"); ep=d.get("ep_size")
acp=d.get("attn_cp_size"); dpa=d.get("enable_dp_attention")
kv=d.get("kv_cache_dtype"); mem=d.get("mem_fraction_static")
if mem is None and isinstance(d.get("server_args"), dict):
    mem=d["server_args"].get("mem_fraction_static")
cfg=d.get("cuda_graph_config") or {}; dec=cfg.get("decode") or {}; bs=dec.get("bs") or []
def sv(v):
    if isinstance(v,bool): return "true" if v else "false"
    return "" if v is None else str(v)
print(sv(tp),sv(dp),sv(pp),sv(ep),sv(acp),sv(dpa),sv(kv),sv(mem),",".join(str(int(x)) for x in bs if isinstance(x,(int,float))))' 2>/dev/null \
    || true
  )"

  if [[ -n "${values}" ]]; then
    read -r _tp _dp _pp _ep _acp _dpa _kv _mem _cudabs <<< "${values}"
    [[ -n "${DP_SIZE}" ]] || DP_SIZE="${_dp}"
    [[ -n "${TP_SIZE}" ]] || TP_SIZE="${_tp}"
    [[ -n "${PP_SIZE}" ]] || PP_SIZE="${_pp}"
    [[ -n "${EP_SIZE}" ]] || EP_SIZE="${_ep}"
    [[ -n "${ATTN_CP_SIZE}" ]] || ATTN_CP_SIZE="${_acp}"
    [[ -n "${DP_ATTENTION}" ]] || DP_ATTENTION="${_dpa}"
    [[ -n "${KV_CACHE_DTYPE}" ]] || KV_CACHE_DTYPE="${_kv}"
    [[ -n "${MEM_FRACTION_STATIC}" ]] || MEM_FRACTION_STATIC="${_mem}"
    [[ -n "${CUDA_GRAPH_DECODE_BS}" ]] || CUDA_GRAPH_DECODE_BS="${_cudabs}"

    local attn_tag
    if [[ "${DP_ATTENTION}" == "true" ]]; then attn_tag="DPA"; else attn_tag="TPA"; fi
    PARALLEL_CONFIG="TP${TP_SIZE:-NA}-DP${DP_SIZE:-NA}(${attn_tag})-PP${PP_SIZE:-NA}-EP${EP_SIZE:-NA}-ACP${ATTN_CP_SIZE:-NA}-KV${KV_CACHE_DTYPE:-NA}-MEM${MEM_FRACTION_STATIC:-NA}"
    echo "[INFO] server_info: ${PARALLEL_CONFIG}"
    return 0
  fi

  echo "[WARN] /server_info 不可用，使用容量接口和环境变量回退"
  return 1
}

load_server_capacity() {
  load_capacity_from_loads_api && return 0

  echo "[WARN] /v1/loads 不可用，回退 /metrics"
  local metrics_url="${BASE_URL%/}/metrics"
  local raw_metrics max_tokens detected_dp
  raw_metrics="$(curl -fsS "${metrics_url}" 2>/dev/null || true)"

  if [[ -n "${raw_metrics}" ]]; then
    max_tokens="$(
      printf "%s\n" "${raw_metrics}" \
        | awk '/^sglang:max_total_num_tokens([{ ]|$)/ {print $NF}' \
        | sort -n | head -n 1
    )"
    detected_dp="$(parse_dp_size_from_metrics "${raw_metrics}")"
    [[ -n "${DP_SIZE}" ]] || DP_SIZE="${detected_dp}"
    [[ -n "${MAX_TOTAL_TOKENS}" ]] || MAX_TOTAL_TOKENS="${max_tokens%.*}"
  fi

  DP_SIZE="${DP_SIZE:-1}"
  MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-0}"
  if [[ -z "${MAX_TOTAL_TOKENS}" ]]; then
    echo "[ERROR] 无法自动获取 MAX_TOTAL_NUM_TOKENS，请手动设置 MAX_TOTAL_TOKENS" >&2
    return 1
  fi
  echo "[INFO] fallback capacity: DP_SIZE=${DP_SIZE} MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS} MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}"
}

tokens_from_k() {
  local value="$1"
  awk -v value="${value}" -v unit="${TOKEN_UNIT}" 'BEGIN {
    tokens = value * unit
    if (tokens < 1) exit 1
    printf "%.0f", tokens
  }'
}

format_k_name() {
  local value="$1"
  awk -v value="${value}" -v unit="${TOKEN_UNIT}" 'BEGIN {
    scaled = value / unit
    rounded = int(scaled + 0.5)
    if (value >= unit && scaled == rounded) printf "%gm", scaled
    else printf "%gk", value
  }'
}

format_request_rate() {
  local value="$1"
  if [[ "${value}" == "inf" ]]; then
    printf "inf\n"
    return 0
  fi
  LC_ALL=C awk -v value="${value}" 'BEGIN {
    if (value <= 0) exit 1
    printf "%.2f\n", value
  }'
}

select_prefill_bs() {
  local input_k="$1"
  local raw rule upper bs
  IFS=',' read -ra rules <<< "${PREFILL_TTFT_BS_RULES}"
  for rule in "${rules[@]}"; do
    rule="${rule//[[:space:]]/}"
    [[ -n "${rule}" ]] || continue
    upper="${rule%%:*}"
    bs="${rule##*:}"
    if [[ "${upper}" =~ ^[0-9]+([.][0-9]+)?$ ]] && [[ "${bs}" =~ ^[1-9][0-9]*$ ]]; then
      if awk -v input="${input_k}" -v upper="${upper}" 'BEGIN { exit !(input <= upper) }'; then
        printf "%s\n" "${bs}"
        return 0
      fi
    fi
  done
  echo "[ERROR] PREFILL_TTFT_BS_RULES 未覆盖输入 ${input_k}K: ${PREFILL_TTFT_BS_RULES}" >&2
  return 1
}

build_bench_command() {
  local input_len="$1"
  local global_concurrency="$2"
  local num_prompts="$3"
  local warmup_requests="$4"
  local request_rate="$5"
  cmd=(
    python3 -m sglang.bench_serving
    --backend sglang
    --base-url "${BASE_URL}"
    --model "${MODEL}"
    --dataset-name random-ids
    --tokenize-prompt
    --num-prompts "${num_prompts}"
    --random-input-len "${input_len}"
    --random-output-len 1
    --random-range-ratio 1.0
    --request-rate "${request_rate}"
    --warmup-requests "${warmup_requests}"
    --flush-cache
    --disable-tqdm
  )
  if (( USE_FAKE_PREFILL == 1 )); then
    cmd+=(--fake-prefill)
  fi
}

parse_benchmark_log() {
  local log_file="$1"
  "${PYTHON_BIN}" - "${log_file}" <<'PY_PARSE'
import json
import re
import sys

path = sys.argv[1]
prefix = {
    "Backend": "backend",
    "Traffic request rate": "traffic_request_rate",
    "Max request concurrency": "max_request_concurrency",
    "Successful requests": "successful_requests",
    "Benchmark duration (s)": "benchmark_duration_s",
    "Total input tokens": "total_input_tokens",
    "Total input text tokens": "total_input_text_tokens",
    "Total generated tokens": "total_generated_tokens",
    "Total generated tokens (retokenized)": "total_generated_tokens_retokenized",
    "Request throughput (req/s)": "request_throughput",
    "Input token throughput (tok/s)": "input_throughput",
    "Output token throughput (tok/s)": "output_throughput",
    "Peak output token throughput (tok/s)": "peak_output_throughput",
    "Peak concurrency": "peak_concurrency",
    "Total token throughput (tok/s)": "total_throughput",
    "Concurrency": "concurrency",
    "Accept length": "accept_length",
    "Mean E2E Latency (ms)": "mean_e2e_latency_ms",
}
ttft = {
    "Std TTFT (ms)": "std_ttft_ms",
    "Mean TTFT (ms)": "mean_ttft_ms",
    "P75 TTFT (ms)": "p75_ttft_ms",
    "P90 TTFT (ms)": "p90_ttft_ms",
    "P50 TTFT (ms)": "p50_ttft_ms",
    "P95 TTFT (ms)": "p95_ttft_ms",
    "P99 TTFT (ms)": "p99_ttft_ms",
    "P999 TTFT (ms)": "p999_ttft_ms",
    "Max TTFT (ms)": "max_ttft_ms",
}
client_latency = {
    "Mean TPOT (ms)": "mean_tpot_ms",
    "P95 TPOT (ms)": "p95_tpot_ms",
    "P99 TPOT (ms)": "p99_tpot_ms",
    "Mean ITL (ms)": "mean_itl_ms",
    "P95 ITL (ms)": "p95_itl_ms",
    "P99 ITL (ms)": "p99_itl_ms",
}
labels = {**prefix, **ttft, **client_latency}
values = {}
try:
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
except OSError:
    print("{}")
    raise SystemExit(0)

number = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")
def convert(raw):
    raw = raw.strip()
    if number.fullmatch(raw):
        try:
            return int(raw) if re.fullmatch(r"[-+]?\d+", raw) else float(raw)
        except ValueError:
            pass
    return raw

for line in lines:
    if ":" not in line:
        continue
    label, raw = line.split(":", 1)
    label = label.strip()
    if label in labels:
        values[labels[label]] = convert(raw)
print(json.dumps(values, ensure_ascii=False, separators=(",", ":")))
PY_PARSE
}

extract_metric() {
  local log_file="$1"
  local metric="$2"
  parse_benchmark_log "${log_file}" \
    | "${PYTHON_BIN}" -c '
import json
import sys
metric = sys.argv[1]
try:
    data = json.load(sys.stdin)
except json.JSONDecodeError:
    raise SystemExit(1)
value = data.get(metric)
if value is None or value == "":
    raise SystemExit(1)
print(value)
' "${metric}"
}

csv_upsert_trial() {
  local input_k="$1"
  local input_len="$2"
  local per_dp_bs="$3"
  local global_concurrency="$4"
  local target_metric="$5"
  local target_value_ms="$6"
  local target_near_ms="$7"
  local trial="$8"
  local phase="$9"
  local input_rate="${10}"
  local case_status="${11}"
  local log_file="${12}"
  local metrics_json
  metrics_json="$(parse_benchmark_log "${log_file}")"

  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" - "${trials_csv}" "${input_k}" "${input_len}" "${per_dp_bs}" "${global_concurrency}" "${target_metric}" "${target_value_ms}" "${target_near_ms}" "${trial}" "${phase}" "${input_rate}" "${case_status}" "${log_file}" "${metrics_json}" <<'PY_CSV'
import csv
import json
import os
import sys

path = sys.argv[1]
input_k, input_len, per_dp_bs, global_concurrency, target_metric, target_value_ms, target_near_ms, trial, phase, input_rate, case_status, log_file = sys.argv[2:14]
metrics = json.loads(sys.argv[14])
fieldnames = [
    "input_k", "input_len", "output_len", "per_dp_bs", "global_concurrency",
    "target_metric", "target_value_ms", "target_near_ms", "trial", "search_phase", "input_request_rate",
    "actual_rps", "backend", "traffic_request_rate", "max_request_concurrency",
    "successful_requests", "benchmark_duration_s", "total_input_tokens",
    "total_input_text_tokens", "total_generated_tokens",
    "total_generated_tokens_retokenized", "request_throughput",
    "input_throughput", "output_throughput", "peak_output_throughput",
    "peak_concurrency", "total_throughput", "concurrency", "accept_length",
    "mean_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "p95_itl_ms", "p99_itl_ms", "mean_e2e_latency_ms",
    "p50_ttft_ms", "std_ttft_ms", "p75_ttft_ms", "p90_ttft_ms", "p999_ttft_ms",
    "max_ttft_ms", "case_status", "target_status", "log_file",
]
row = {
    "input_k": input_k, "input_len": input_len, "output_len": "1",
    "per_dp_bs": per_dp_bs, "global_concurrency": global_concurrency,
    "target_metric": target_metric, "target_value_ms": target_value_ms,
    "target_near_ms": target_near_ms, "trial": trial, "search_phase": phase,
    "input_request_rate": input_rate, "case_status": case_status,
    "log_file": log_file,
}
row.update(metrics)
row["actual_rps"] = metrics.get("request_throughput", "")
if input_rate != "inf":
    row["traffic_request_rate"] = input_rate
target_value = metrics.get(target_metric) if target_metric else None
if not target_metric or not target_value_ms:
    row["target_status"] = "BASELINE"
elif isinstance(target_value, (int, float)):
    row["target_status"] = "MEET" if target_value <= float(target_value_ms) else "EXCEED"
else:
    row["target_status"] = "NO_METRIC"

rows = []
if os.path.exists(path):
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        rows = []
key = (input_k, target_metric, target_value_ms, trial, input_rate, phase)
rows = [
    old for old in rows
    if (old.get("input_k", ""), old.get("target_metric", ""),
        old.get("target_value_ms", ""), old.get("trial", ""),
        old.get("input_request_rate", ""),
        old.get("search_phase", "")) != key
]
rows.append({name: row.get(name, "") for name in fieldnames})
tmp = path + ".tmp"
with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
os.replace(tmp, path)
PY_CSV
}

run_probe() {
  local input_k="$1"
  local input_len="$2"
  local per_dp_bs="$3"
  local target_metric="$4"
  local target_value_ms="$5"
  local target_near_ms="$6"
  local trial="$7"
  local phase="$8"
  local request_rate
  request_rate="$(format_request_rate "$9")" || return 1
  local global_concurrency=$((DP_SIZE * per_dp_bs))
  local num_prompts=$((global_concurrency * NUM_PROMPTS_MULTIPLIER))
  local warmup_requests=0
  local target_tag rate_tag base_name log_file exit_file exit_code

  if (( per_dp_bs == 1 )); then
    warmup_requests=$((DP_SIZE * 16))
  fi
  target_tag="${target_metric:-baseline}_${target_value_ms:-baseline}_${target_near_ms:-0}"
  rate_tag="${request_rate//./p}"
  rate_tag="${rate_tag//-/m}"
  base_name="input${input_k}k_perdp${per_dp_bs}_target${target_tag}_trial${trial}_rate${rate_tag}"
  log_file="${log_dir}/${base_name}.log"
  exit_file="${log_file}.exitcode"

  if [[ -s "${log_file}" && -f "${exit_file}" ]] && [[ "$(cat "${exit_file}" 2>/dev/null)" == "0" ]]; then
    echo "[SKIP] input=${input_k}K target=${target_metric:-baseline}<=${target_value_ms:-inf} rate=${request_rate}"
  else
    build_bench_command "${input_len}" "${global_concurrency}" "${num_prompts}" "${warmup_requests}" "${request_rate}"
    echo "[RUN] input=${input_k}K per_dp_bs=${per_dp_bs} target=${target_metric:-baseline}<=${target_value_ms:-inf} phase=${phase} rate=${request_rate}"
    if [[ "${PREFILL_TTFT_TEST_MODE}" == "command" ]]; then
      printf '%q ' "${cmd[@]}"
      echo
      return 0
    fi
    mkdir -p "${log_dir}"
    write_log_header "${log_file}"
    "${cmd[@]}" 2>&1 | tee -a "${log_file}"
    exit_code="${PIPESTATUS[0]}"
    printf "%s\n" "${exit_code}" > "${exit_file}"
  fi

  exit_code="$(cat "${exit_file}" 2>/dev/null || echo 1)"
  local metrics_json actual_rps target_value
  metrics_json="$(parse_benchmark_log "${log_file}")"
  actual_rps="$(printf "%s" "${metrics_json}" | "${PYTHON_BIN}" -c 'import json,sys
d=json.load(sys.stdin); v=d.get("request_throughput")
print(v if v is not None else "")')"
  target_value="$(printf "%s" "${metrics_json}" | "${PYTHON_BIN}" -c 'import json,sys
d=json.load(sys.stdin); metric=sys.argv[1]; v=d.get(metric) if metric else None
print(v if v is not None else "")' "${target_metric}")"

  local status="FAIL"
  [[ "${exit_code}" == "0" ]] && status="PASS"
  csv_upsert_trial "${input_k}" "${input_len}" "${per_dp_bs}" "${global_concurrency}" "${target_metric}" "${target_value_ms}" "${target_near_ms}" "${trial}" "${phase}" "${request_rate}" "${status}" "${log_file}"

  LAST_ACTUAL_RPS="${actual_rps}"
  LAST_TARGET_VALUE="${target_value}"
  LAST_CASE_STATUS="${status}"
  LAST_LOG_FILE="${log_file}"
}

run_inf_until_input_throughput_stable() {
  local input_k="$1" input_len="$2" per_dp_bs="$3"
  local trial=0 no_improvement=0 phase log_file input_throughput
  local best_input_throughput="" best_log="" best_rps=""

  while (( trial < PREFILL_TTFT_INF_MAX_TRIALS )); do
    if (( trial == 0 )); then
      phase="baseline"
    else
      phase="inf_throughput${trial}"
    fi
    run_probe "${input_k}" "${input_len}" "${per_dp_bs}" "" "" "" "${trial}" "${phase}" "inf"
    [[ "${LAST_CASE_STATUS}" == "PASS" ]] || return 1
    log_file="${LAST_LOG_FILE}"
    input_throughput="$(extract_metric "${log_file}" input_throughput 2>/dev/null || true)"
    [[ -n "${input_throughput}" ]] || {
      echo "[ERROR] input=${input_k}K 的 inf 缺少 input throughput，无法判断是否稳定" >&2
      return 1
    }
    if [[ -z "${best_input_throughput}" ]] || awk -v current="${input_throughput}" -v best="${best_input_throughput}" 'BEGIN { exit !(current > best + 1e-9) }'; then
      best_input_throughput="${input_throughput}"
      best_log="${log_file}"
      best_rps="${LAST_ACTUAL_RPS}"
      no_improvement=0
      echo "[INF] input=${input_k}K trial=$((trial + 1)) input_throughput=${input_throughput} (new best)"
    else
      ((no_improvement++))
      echo "[INF] input=${input_k}K trial=$((trial + 1)) input_throughput=${input_throughput} best=${best_input_throughput} no_improvement=${no_improvement}/${PREFILL_TTFT_INF_NO_IMPROVEMENT_LIMIT}"
    fi
    ((trial++))
    if (( no_improvement >= PREFILL_TTFT_INF_NO_IMPROVEMENT_LIMIT )); then
      break
    fi
  done

  [[ -n "${best_log}" ]] || return 1
  LAST_INF_LOG="${best_log}"
  LAST_LOG_FILE="${best_log}"
  LAST_ACTUAL_RPS="${best_rps}"
  echo "[INF] input=${input_k}K selected input_throughput=${best_input_throughput} log=${best_log}"
}

search_target() {
  local input_k="$1"
  local input_len="$2"
  local per_dp_bs="$3"
  local target_metric="$4"
  local target_value_ms="$5"
  local target_near_ms="$6"
  local baseline_rps="$7"
  local baseline_value="$8"
  local baseline_log="$9"
  local high_rate
  high_rate="$(format_request_rate "${baseline_rps}")"
  local low_rate=""
  local low_rps=""
  local low_value=""
  local trial=0
  local probe_rate
  local target_met

  while (( trial < PREFILL_TTFT_SEARCH_ITERS )); do
    trial=$((trial + 1))
    if [[ -z "${low_rate}" ]]; then
      probe_rate="$(format_request_rate "$(awk -v high="${high_rate}" 'BEGIN { print high / 2 }')")"
      phase="descending"
    else
      probe_rate="$(format_request_rate "$(awk -v low="${low_rate}" -v high="${high_rate}" 'BEGIN { print (low + high) / 2 }')")"
      phase="binary"
    fi
    if awk -v rate="${probe_rate}" 'BEGIN { exit !(rate > 0) }' 2>/dev/null; then :; else break; fi
    if [[ "${probe_rate}" == "${low_rate}" || "${probe_rate}" == "${high_rate}" ]]; then
      break
    fi

    run_probe "${input_k}" "${input_len}" "${per_dp_bs}" "${target_metric}" "${target_value_ms}" "${target_near_ms}" "${trial}" "${phase}" "${probe_rate}"
    target_met=1
    if [[ "${LAST_CASE_STATUS}" != "PASS" || -z "${LAST_TARGET_VALUE}" ]]; then
      target_met=0
    elif ! awk -v value="${LAST_TARGET_VALUE}" -v target="${target_value_ms}" 'BEGIN { exit !(value <= target) }' 2>/dev/null; then
      target_met=0
    fi

    if (( target_met == 1 )); then
      low_rate="${probe_rate}"
      low_rps="${LAST_ACTUAL_RPS}"
      low_value="${LAST_TARGET_VALUE}"
      if [[ "${phase}" == "descending" ]]; then
        echo "[BRACKET] input=${input_k}K ${target_metric}<=${target_value_ms}ms upper_rate=${high_rate} pass_rate=${low_rate}"
      fi
    else
      high_rate="${probe_rate}"
      if [[ -n "${low_rate}" ]] && awk -v low="${low_rate}" -v high="${high_rate}" 'BEGIN { exit !(high > low) }' 2>/dev/null; then :; fi
    fi

    if [[ -n "${low_rate}" ]] && awk -v low="${low_rate}" -v high="${high_rate}" -v step="${PREFILL_TTFT_RATE_STEP}" 'BEGIN { exit !((high - low) <= step) }' 2>/dev/null; then
      break
    fi

    [[ -n "${low_rate}" ]] || continue
    if [[ "${phase}" == "binary" ]] && [[ "${low_rate}" == "${high_rate}" ]]; then
      break
    fi
  done

  if [[ -n "${low_rate}" ]]; then
    echo "[TARGET-PASS] input=${input_k}K ${target_metric}<=${target_value_ms}ms candidate_rps=${low_rps} rate=${low_rate} value=${low_value}"
  else
    echo "[TARGET-FAIL] input=${input_k}K ${target_metric}<=${target_value_ms}ms no passing request rate"
  fi
}

write_matrix_summary() {
  "${PYTHON_BIN}" - "${trials_csv}" "${matrix_csv}" <<'PY_SUMMARY'
import csv
import os
import sys

trials_path, matrix_path = sys.argv[1:]
fields = [
    "input_k", "input_len", "output_len", "per_dp_bs", "global_concurrency",
    "target_metric", "target_value_ms", "target_near_ms", "trial", "search_phase", "input_request_rate",
    "actual_rps", "backend", "traffic_request_rate", "max_request_concurrency",
    "successful_requests", "benchmark_duration_s", "total_input_tokens",
    "total_input_text_tokens", "total_generated_tokens",
    "total_generated_tokens_retokenized", "request_throughput",
    "input_throughput", "output_throughput", "peak_output_throughput",
    "peak_concurrency", "total_throughput", "concurrency", "accept_length",
    "mean_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "p95_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "p95_itl_ms", "p99_itl_ms", "mean_e2e_latency_ms",
    "p50_ttft_ms", "std_ttft_ms", "p75_ttft_ms", "p90_ttft_ms", "p999_ttft_ms",
    "max_ttft_ms", "case_status", "target_status", "log_file",
    "selection_status",
]
try:
    with open(trials_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
except (OSError, csv.Error):
    rows = []

def num(row, key):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None

inputs = sorted({row.get("input_k", "") for row in rows if row.get("input_k")})
targets = sorted({
    (row.get("target_metric", ""), row.get("target_value_ms", ""), row.get("target_near_ms", ""))
    for row in rows
    if row.get("target_metric", "") and row.get("target_value_ms", "")
}, key=lambda x: (x[0], float(x[1]), float(x[2] or 0)))
out = []
for input_k in inputs:
    baseline_candidates = [
        row for row in rows
        if row.get("input_k") == input_k
        and row.get("target_metric", "") == ""
        and row.get("input_request_rate", "") == "inf"
        and row.get("case_status") == "PASS"
    ]
    baseline = max(
        baseline_candidates,
        key=lambda row: num(row, "input_throughput") or float("-inf"),
        default=None,
    )
    target_values = [t for t in targets if any(
        row.get("input_k") == input_k
        and row.get("target_metric") == t[0]
        and row.get("target_value_ms") == t[1]
        and row.get("target_near_ms", "") == t[2]
        for row in rows
    )]
    for target_metric, target_value, target_near in target_values:
        candidates = [
            row for row in rows
            if row.get("input_k") == input_k
            and row.get("target_metric") == target_metric
            and row.get("target_value_ms") == target_value
            and row.get("target_near_ms", "") == target_near
            and row.get("case_status") == "PASS"
            and row.get("target_status") == "MEET"
            and num(row, "actual_rps") is not None
        ]
        baseline_value = num(baseline, target_metric) if baseline else None
        if baseline and baseline_value is not None and baseline_value <= float(target_value):
            base = dict(baseline)
            base["target_metric"] = target_metric
            base["target_value_ms"] = target_value
            base["target_near_ms"] = target_near
            base["target_status"] = "MEET"
            candidates.append(base)
        if candidates:
            best = max(candidates, key=lambda row: (num(row, "actual_rps"), -num(row, target_metric)))
            best["selection_status"] = "BEST"
            out.append({field: best.get(field, "") for field in fields})
        else:
            out.append({
                "input_k": input_k, "target_metric": target_metric,
                "target_value_ms": target_value, "target_near_ms": target_near,
                "case_status": "NO_PASS", "target_status": "NO_PASS",
                "selection_status": "NO_PASS",
            })

os.makedirs(os.path.dirname(matrix_path) or ".", exist_ok=True)
tmp = matrix_path + ".tmp"
with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(out)
os.replace(tmp, matrix_path)
PY_SUMMARY
}

run_input_matrix() {
  local input_k="$1"
  local input_len per_dp_bs baseline_rps baseline_log
  local target_metric target_value_ms target_near_ms baseline_value
  local matched_rules=0

  input_len="$(tokens_from_k "${input_k}")" || {
    echo "[ERROR] 无法转换输入长度 ${input_k}K" >&2
    return 1
  }
  per_dp_bs="$(select_prefill_bs "${input_k}")" || return 1
  echo
  echo "============================================================"
  echo "PREFILL INPUT       : ${input_k}K = ${input_len} tokens"
  echo "PREFILL OUTPUT      : 1 token"
  echo "PREFILL PER-DP BS   : ${per_dp_bs}"
  echo "PREFILL GLOBAL CONC : $((DP_SIZE * per_dp_bs))"
  echo "============================================================"

  run_inf_until_input_throughput_stable "${input_k}" "${input_len}" "${per_dp_bs}" || {
    echo "[ERROR] input=${input_k}K 的 inf 基准未能稳定测得 input throughput，跳过矩阵" >&2
    return 1
  }
  baseline_rps="${LAST_ACTUAL_RPS}"
  baseline_log="${LAST_LOG_FILE}"
  if [[ "${LAST_CASE_STATUS}" != "PASS" || -z "${baseline_rps}" ]]; then
    echo "[ERROR] input=${input_k}K 的 inf 基准缺少实际 RPS，跳过矩阵" >&2
    return 1
  fi

  while IFS='|' read -r target_metric target_value_ms target_near_ms; do
    [[ -n "${target_metric}" ]] || continue
    matched_rules=1
    baseline_value="$(extract_metric "${baseline_log}" "${target_metric}" 2>/dev/null || true)"
    if [[ -z "${baseline_value}" ]]; then
      echo "[ERROR] input=${input_k}K 的基准日志缺少 ${target_metric}，跳过该规则" >&2
      continue
    fi
    search_target "${input_k}" "${input_len}" "${per_dp_bs}" "${target_metric}" "${target_value_ms}" "${target_near_ms}" "${baseline_rps}" "${baseline_value}" "${baseline_log}"
  done < <(prefill_sla_rules_for_input "${input_k}")

  if (( matched_rules == 0 )); then
    echo "[ERROR] input=${input_k}K 没有匹配的 PREFILL_SLA_RULES 规则" >&2
    return 1
  fi
}

render_plots() {
  (( ${#PLOT_REQUESTS[@]} > 0 )) || return 0
  [[ -s "${trials_csv}" ]] || {
    echo "[ERROR] 没有 Prefill CSV，无法作图: ${trials_csv}" >&2
    return 1
  }
  [[ -f "${script_dir}/plot_prefill_ttft.py" ]] || {
    echo "[ERROR] 找不到绘图脚本: ${script_dir}/plot_prefill_ttft.py" >&2
    return 1
  }

  local mode kind metric suffix output
  for mode in "${PLOT_REQUESTS[@]}"; do
    kind="${mode}"
    metric=""
    suffix="${mode}"
    case "${mode}" in
      mean)
        kind="capacity"
        metric="mean_ttft_ms"
        suffix="mean_capacity_matrix_${PLOT_CELL}"
        ;;
      p99)
        kind="capacity"
        metric="p99_ttft_ms"
        suffix="p99_capacity_matrix_${PLOT_CELL}"
        ;;
      sla)
        kind="capacity"
        metric="auto"
        suffix="sla_capacity_matrix_${PLOT_CELL}"
        ;;
    esac
    output="${out_dir}/prefill_ttft_${suffix}.png"
    echo "[INFO] 绘制 ${mode}: ${output}"
    if [[ "${kind}" == "capacity" ]]; then
      "${PYTHON_BIN}" "${script_dir}/plot_prefill_ttft.py" "${trials_csv}" \
        --kind "${kind}" --metric "${metric}" --targets auto \
        --cell "${PLOT_CELL}" -o "${output}" || return 1
    else
      "${PYTHON_BIN}" "${script_dir}/plot_prefill_ttft.py" "${trials_csv}" \
        --kind "${kind}" -o "${output}" || return 1
    fi
  done
}

main() {
  mkdir -p "${log_dir}"
  validate_prefill_sla_rules || return 1
  if [[ "${PREFILL_TTFT_TEST_MODE}" == "command" ]]; then
    DP_SIZE="${DP_SIZE:-2}"
    local input_len=32768
    local global_concurrency=128
    local num_prompts=128
    local warmup_requests=0
    build_bench_command "${input_len}" "${global_concurrency}" "${num_prompts}" "${warmup_requests}" "100"
    printf '%q ' "${cmd[@]}"
    echo
    return 0
  fi

  load_server_info || true
  load_server_capacity || return 1
  [[ -n "${MODEL}" ]] || MODEL="$(fetch_model_from_metrics "${BASE_URL%/}/metrics")"
  [[ -n "${MODEL}" ]] || {
    echo "[ERROR] MODEL 为空且无法从 metrics 识别" >&2
    return 1
  }
  write_run_metadata

  echo "[INFO] Prefill TTFT matrix"
  echo "[INFO] BASE_URL=${BASE_URL}"
  echo "[INFO] MODEL=${MODEL}"
  echo "[INFO] DP_SIZE=${DP_SIZE}"
  echo "[INFO] TOKEN_UNIT=${TOKEN_UNIT}"
  echo "[INFO] INPUTS=${PREFILL_TTFT_INPUTS}"
  echo "[INFO] SLA_RULES=$(prefill_sla_rules_text)"
  echo "[INFO] BS_RULES=${PREFILL_TTFT_BS_RULES}"
  echo "[INFO] OUTPUT_DIR=${out_dir}"

  local input_list="${PREFILL_TTFT_INPUTS//,/ }"
  for input_k in ${input_list}; do
    if [[ ! "${input_k}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
      echo "[ERROR] 非法输入长度: ${input_k}" >&2
      return 1
    fi
    run_input_matrix "${input_k}" || true
  done
  write_matrix_summary
  render_plots || return 1
  echo "[DONE] trials=${trials_csv}"
  echo "[DONE] matrix=${matrix_csv}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
