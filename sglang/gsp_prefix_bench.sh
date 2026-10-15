#!/usr/bin/env bash
# ============================================================
# SGLang generated-shared-prefix prefix 命中率容量测试
# ============================================================

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
MODEL="${MODEL:-}"
TOKEN_UNIT="${TOKEN_UNIT:-1024}"
GSP_NUM_GROUPS="${GSP_NUM_GROUPS:-32}"
GSP_PROMPTS_PER_GROUP="${GSP_PROMPTS_PER_GROUP:-16}"
GSP_MAX_CONCURRENCY="${GSP_MAX_CONCURRENCY:-}"
GSP_PAGE_SIZE="${GSP_PAGE_SIZE:-}"
GSP_PAGE_ALIGN="${GSP_PAGE_ALIGN:-1}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_PLOT_BIN="${PYTHON_PLOT_BIN:-${PYTHON_BIN}}"
RATE_OFFSETS="${GSP_RATE_OFFSETS:-0,-0.1,-0.1,-0.1,-0.2,-0.2,-0.3,-0.5}"
RATE_STEP="${GSP_RATE_STEP:-0.1}"
HIT_RATE_TOLERANCE="${GSP_HIT_RATE_TOLERANCE_PCT:-2.0}"
EXPLICIT_RATES="${GSP_REQUEST_RATES:-}"

out_dir="./all_result/bench_gsp_prefix_${RUN_ID}"
log_dir="${out_dir}/logs"
trials_csv="${out_dir}/gsp_prefix_trials.csv"
summary_csv="${out_dir}/gsp_prefix_summary.csv"
plot_requested=0
plot_cell="rps"
dry_run=0
run_failure=0

declare -a CASES=()

usage() {
  cat <<'USAGE'
Usage:
  bash sglang/gsp_prefix_bench.sh [options]

Options:
  --case CASE       Repeatable case:
                    INPUT,OUTPUT,HIT_RATE,TTFT_METRIC,TTFT_TARGETS_S,TPOT_METRIC,TPOT_TARGET_MS[,MAX_CONCURRENCY]
  --plot            Generate a heatmap after CSV summary is written
  --cell CELL       Heatmap cell: rps|input_throughput (alias: input_tp)
  --dry-run         Validate and print official commands without calling the server
  -h, --help        Show this help

Example:
  MODEL=/models/foo BASE_URL=http://127.0.0.1:30000 \
  bash sglang/gsp_prefix_bench.sh \
    --case '16k,1k,80,mean_ttft_ms,3|5,mean_tpot_ms,30' \
    --case '32k,0.5k,80,mean_ttft_ms,5|7,mean_tpot_ms,50' \
    --plot --cell rps

Environment:
  TOKEN_UNIT=1024                 Default token unit; set 1000 explicitly for decimal k
  GSP_NUM_GROUPS=32               Official GSP group count
  GSP_PROMPTS_PER_GROUP=16        Official prompts per group
  GSP_MAX_CONCURRENCY=            Optional global max concurrency
  GSP_PAGE_SIZE=                  KV cache page size; auto-read from /server_info, fallback 64
  GSP_PAGE_ALIGN=1                Align shared prefix to a whole KV page; set 0 to disable
  GSP_REQUEST_RATES=              Optional comma-separated finite rate list
  GSP_RATE_OFFSETS=0,-0.1,...     Cumulative high-to-low offsets after inf baseline
  GSP_RATE_STEP=0.1               Minimum finite-rate step
  GSP_HIT_RATE_TOLERANCE_PCT=2.0  Allowed measured/target hit-rate difference
  RUN_ID=...                       Result directory suffix

The hit rate is token-level: total_cached_tokens / total_prompt_tokens.
The first request in each ordered prefix group is treated as a cache miss.
USAGE
}

while (( $# > 0 )); do
  case "$1" in
    --case)
      [[ $# -ge 2 ]] || { echo "[ERROR] --case 需要一个配置" >&2; exit 1; }
      CASES+=("$2")
      shift 2
      ;;
    --case=*)
      CASES+=("${1#*=}")
      shift
      ;;
    --plot)
      plot_requested=1
      shift
      ;;
    --cell)
      [[ $# -ge 2 ]] || { echo "[ERROR] --cell 需要 rps 或 input_throughput" >&2; exit 1; }
      plot_cell="$2"
      shift 2
      ;;
    --cell=*)
      plot_cell="${1#*=}"
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
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

case "${plot_cell}" in
  rps) ;;
  input_throughput|input_tp) plot_cell="input_throughput" ;;
  *) echo "[ERROR] --cell 仅支持 rps|input_throughput" >&2; exit 1 ;;
esac

if (( ${#CASES[@]} == 0 )); then
  CASES+=(
    '16k,1k,80,mean_ttft_ms,3|5,mean_tpot_ms,30'
    '32k,0.5k,80,mean_ttft_ms,5|7,mean_tpot_ms,50'
  )
fi

is_decimal() {
  [[ "$1" =~ ^[0-9]+([.][0-9]*)?$ || "$1" =~ ^[.][0-9]+$ ]]
}

is_positive_decimal() {
  is_decimal "$1" && awk -v v="$1" 'BEGIN { exit !(v > 0) }'
}

is_nonnegative_decimal() {
  is_decimal "$1" && awk -v v="$1" 'BEGIN { exit !(v >= 0) }'
}

tokens_from_k() {
  "${PYTHON_BIN}" - "$1" "${TOKEN_UNIT}" <<'PY' | tr -d '\r'
import math
import re
import sys

raw, unit_s = sys.argv[1:]
unit = int(unit_s)
match = re.fullmatch(r"([0-9]+(?:\.[0-9]*)?|\.[0-9]+)([kK])?", raw)
if not match:
    raise SystemExit(1)
value = float(match.group(1))
if match.group(2):
    value *= unit
if not math.isfinite(value) or value <= 0:
    raise SystemExit(1)
print(max(1, int(round(value))))
PY
}

format_decimal() {
  "${PYTHON_BIN}" - "$1" <<'PY' | tr -d '\r'
import math
import sys
value = float(sys.argv[1])
if not math.isfinite(value):
    raise SystemExit(1)
print(f"{value:.2f}")
PY
}

sanitize_slug() {
  printf '%s' "$1" | sed -E 's/[^A-Za-z0-9_.-]+/_/g; s/^_+//; s/_+$//'
}

derive_gsp_lengths() {
  local input_len="$1" hit_pct="$2" prompts="$3" page_size="$4" page_align="$5"
  "${PYTHON_BIN}" - "$input_len" "$hit_pct" "$prompts" "$page_size" "$page_align" <<'PY' | tr -d '\r'
import math
import sys

length, hit_pct, prompts = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
page_size, page_align = int(sys.argv[4]), sys.argv[5] == "1"
if prompts <= 1:
    raise SystemExit("prompts_per_group must be greater than 1")
desired_system_len = (hit_pct / 100.0) * prompts * length / (prompts - 1)
rounded_system_len = int(math.floor(desired_system_len + 0.5))
system_len = rounded_system_len
if page_align and page_size > 1:
    candidates = {
        max(1, (rounded_system_len // page_size) * page_size),
        min(length - 1, ((rounded_system_len + page_size - 1) // page_size) * page_size),
    }
    candidates = [x for x in candidates if 0 < x < length]
    if candidates:
        system_len = min(
            candidates,
            key=lambda x: abs(100.0 * (prompts - 1) * x / (prompts * length) - hit_pct),
        )
question_len = length - system_len
if system_len <= 0 or question_len <= 0 or system_len >= length:
    raise SystemExit("target hit rate cannot produce positive system/question lengths")
theory = 100.0 * (prompts - 1) * system_len / (prompts * length)
requested_theory = 100.0 * (prompts - 1) * rounded_system_len / (prompts * length)
print(system_len, question_len, f"{theory:.6f}", f"{requested_theory:.6f}")
PY
}

resolve_page_size() {
  if [[ -n "${GSP_PAGE_SIZE}" ]]; then
    [[ "${GSP_PAGE_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
      echo "[ERROR] GSP_PAGE_SIZE 必须是正整数" >&2
      return 1
    }
    return 0
  fi

  GSP_PAGE_SIZE=""
  if (( dry_run == 0 )); then
    GSP_PAGE_SIZE="$(${PYTHON_BIN} - "${BASE_URL%/}/server_info" <<'PY'
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
        obj = json.load(response)
except Exception:
    obj = {}

def positive_int(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return ""
    return str(value) if value > 0 else ""

if isinstance(obj, dict):
    page = positive_int(obj.get("page_size"))
    if not page and isinstance(obj.get("server_args"), dict):
        page = positive_int(obj["server_args"].get("page_size"))
    if not page and isinstance(obj.get("internal_states"), list):
        for state in obj["internal_states"]:
            if isinstance(state, dict):
                page = positive_int(state.get("page_size"))
                if page:
                    break
    print(page)
PY
    )"
  fi
  [[ "${GSP_PAGE_SIZE}" =~ ^[1-9][0-9]*$ ]] || GSP_PAGE_SIZE=64
}

parse_case() {
  local raw="$1"
  local old_ifs="$IFS"
  IFS=',' read -r CASE_INPUT_RAW CASE_OUTPUT_RAW CASE_HIT_PCT CASE_TTFT_METRIC \
    CASE_TTFT_TARGETS CASE_TPOT_METRIC CASE_TPOT_TARGET_MS CASE_CONCURRENCY extra <<< "$raw"
  IFS="$old_ifs"

  [[ -z "${extra:-}" ]] || {
    echo "[ERROR] case 字段过多: ${raw}" >&2
    return 1
  }
  [[ -n "${CASE_INPUT_RAW:-}" && -n "${CASE_OUTPUT_RAW:-}" && -n "${CASE_HIT_PCT:-}" \
    && -n "${CASE_TTFT_METRIC:-}" && -n "${CASE_TTFT_TARGETS:-}" \
    && -n "${CASE_TPOT_METRIC:-}" && -n "${CASE_TPOT_TARGET_MS:-}" ]] || {
    echo "[ERROR] 非法 case: ${raw}" >&2
    return 1
  }
  is_positive_decimal "${CASE_HIT_PCT}" || {
    echo "[ERROR] HIT_RATE 必须是正数: ${raw}" >&2
    return 1
  }
  awk -v v="${CASE_HIT_PCT}" 'BEGIN { exit !(v < 100) }' || {
    echo "[ERROR] HIT_RATE 必须小于 100，且需要保留 question token: ${raw}" >&2
    return 1
  }
  if [[ -n "${CASE_CONCURRENCY:-}" ]]; then
    [[ "${CASE_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || {
      echo "[ERROR] MAX_CONCURRENCY 必须是正整数: ${raw}" >&2
      return 1
    }
  else
    CASE_CONCURRENCY="${GSP_MAX_CONCURRENCY}"
  fi

  case "${CASE_TTFT_METRIC}" in
    mean_ttft_ms|p50_ttft_ms|p90_ttft_ms|p95_ttft_ms|p99_ttft_ms|max_ttft_ms) ;;
    *) echo "[ERROR] 不支持的 TTFT 指标: ${CASE_TTFT_METRIC}" >&2; return 1 ;;
  esac
  case "${CASE_TPOT_METRIC}" in
    mean_tpot_ms|p50_tpot_ms|p90_tpot_ms|p95_tpot_ms|p99_tpot_ms) ;;
    *) echo "[ERROR] 不支持的 TPOT 指标: ${CASE_TPOT_METRIC}" >&2; return 1 ;
  esac

  CASE_TTFT_TARGET_MS_LIST=()
  local target
  IFS='|' read -ra _targets <<< "${CASE_TTFT_TARGETS}"
  for target in "${_targets[@]}"; do
    is_positive_decimal "$target" || {
      echo "[ERROR] TTFT SLA 必须是正数秒: ${raw}" >&2
      return 1
    }
    CASE_TTFT_TARGET_MS_LIST+=("$(awk -v v="$target" 'BEGIN { printf "%.6f", v * 1000 }' | sed -E 's/0+$//; s/[.]$//')")
  done
  is_positive_decimal "${CASE_TPOT_TARGET_MS}" || {
    echo "[ERROR] TPOT SLA 必须是正数毫秒: ${raw}" >&2
    return 1
  }

  CASE_INPUT_LEN="$(tokens_from_k "${CASE_INPUT_RAW}")" || {
    echo "[ERROR] 无法解析输入长度: ${raw}" >&2
    return 1
  }
  CASE_OUTPUT_LEN="$(tokens_from_k "${CASE_OUTPUT_RAW}")" || {
    echo "[ERROR] 无法解析输出长度: ${raw}" >&2
    return 1
  }
  read -r CASE_SYSTEM_LEN CASE_QUESTION_LEN CASE_THEORY_HIT_PCT CASE_REQUESTED_THEORY_HIT_PCT < <(
    derive_gsp_lengths "${CASE_INPUT_LEN}" "${CASE_HIT_PCT}" "${GSP_PROMPTS_PER_GROUP}" \
      "${GSP_PAGE_SIZE}" "${GSP_PAGE_ALIGN}"
  ) || {
    echo "[ERROR] 无法根据命中率生成 GSP system/question 长度: ${raw}" >&2
    return 1
  }
  CASE_NUM_PROMPTS=$((GSP_NUM_GROUPS * GSP_PROMPTS_PER_GROUP))
  CASE_ID="$(sanitize_slug "${CASE_INPUT_RAW}_${CASE_OUTPUT_RAW}_hit${CASE_HIT_PCT}_${CASE_TTFT_METRIC}")"
  return 0
}

validate_global_config() {
  [[ "${TOKEN_UNIT}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] TOKEN_UNIT 必须是正整数" >&2; return 1; }
  [[ "${GSP_NUM_GROUPS}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] GSP_NUM_GROUPS 必须是正整数" >&2; return 1; }
  [[ "${GSP_PROMPTS_PER_GROUP}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] GSP_PROMPTS_PER_GROUP 必须是正整数" >&2; return 1; }
  (( GSP_PROMPTS_PER_GROUP >= 2 )) || { echo "[ERROR] GSP_PROMPTS_PER_GROUP 必须不小于 2" >&2; return 1; }
  [[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "[ERROR] SEED 必须是非负整数" >&2; return 1; }
  [[ "${GSP_PAGE_ALIGN}" =~ ^[01]$ ]] || { echo "[ERROR] GSP_PAGE_ALIGN 必须是 0 或 1" >&2; return 1; }
  is_positive_decimal "${RATE_STEP}" || { echo "[ERROR] GSP_RATE_STEP 必须是正数" >&2; return 1; }
  awk -v v="${RATE_STEP}" 'BEGIN { exit !(v >= 0.1) }' || { echo "[ERROR] GSP_RATE_STEP 不能小于 0.1" >&2; return 1; }
  is_positive_decimal "${HIT_RATE_TOLERANCE}" || { echo "[ERROR] GSP_HIT_RATE_TOLERANCE_PCT 必须是正数" >&2; return 1; }
  if [[ -n "${GSP_MAX_CONCURRENCY}" ]]; then
    [[ "${GSP_MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] GSP_MAX_CONCURRENCY 必须是正整数" >&2; return 1; }
  fi
}

validate_global_config || exit 1
resolve_page_size || exit 1

print_case() {
  echo "CASE                 : ${CASE_ID}"
  echo "Input                : ${CASE_INPUT_RAW} = ${CASE_INPUT_LEN} tokens"
  echo "Output               : ${CASE_OUTPUT_RAW} = ${CASE_OUTPUT_LEN} tokens"
  echo "Target hit rate      : ${CASE_HIT_PCT}%"
  echo "Target theory (raw)  : ${CASE_REQUESTED_THEORY_HIT_PCT}%"
  echo "Theoretical hit rate : ${CASE_THEORY_HIT_PCT}% (page size ${GSP_PAGE_SIZE})"
  echo "System/question      : ${CASE_SYSTEM_LEN}/${CASE_QUESTION_LEN}"
  echo "Groups/prompts       : ${GSP_NUM_GROUPS}/${GSP_PROMPTS_PER_GROUP}"
  echo "Num prompts          : ${CASE_NUM_PROMPTS}"
  echo "TTFT                 : ${CASE_TTFT_METRIC} < ${CASE_TTFT_TARGETS} s"
  echo "TPOT                 : ${CASE_TPOT_METRIC} < ${CASE_TPOT_TARGET_MS} ms"
  echo "Max concurrency      : ${CASE_CONCURRENCY:-UNLIMITED}"
}

print_dry_run_case() {
  local request_rate="${1:-inf}"
  print_case
  echo "Official command:"
  printf '  %q' "${PYTHON_BIN}" -m sglang.bench_serving
  printf ' %q' --backend sglang --base-url "${BASE_URL}" --model "${MODEL:-<model>}"
  printf ' %q' --dataset-name generated-shared-prefix
  printf ' %q' --gsp-num-groups "${GSP_NUM_GROUPS}"
  printf ' %q' --gsp-prompts-per-group "${GSP_PROMPTS_PER_GROUP}"
  printf ' %q' --gsp-system-prompt-len "${CASE_SYSTEM_LEN}"
  printf ' %q' --gsp-question-len "${CASE_QUESTION_LEN}"
  printf ' %q' --gsp-output-len "${CASE_OUTPUT_LEN}"
  printf ' %q' --gsp-ordered --num-prompts "${CASE_NUM_PROMPTS}"
  printf ' %q' --flush-cache --seed "${SEED}" --cache-report
  printf ' %q' --request-rate "${request_rate}" --disable-tqdm --output-file /dev/null
  if [[ -n "${CASE_CONCURRENCY:-}" ]]; then
    printf ' %q' --max-concurrency "${CASE_CONCURRENCY}"
  fi
  printf '\n\n'
}

if (( dry_run == 1 )); then
  echo "[DRY-RUN] BASE_URL=${BASE_URL} MODEL=${MODEL:-<model>} TOKEN_UNIT=${TOKEN_UNIT}"
  for raw_case in "${CASES[@]}"; do
    parse_case "${raw_case}" || exit 1
    print_dry_run_case inf
  done
  exit 0
fi

[[ -n "${MODEL}" ]] || { echo "[ERROR] 真实执行必须设置 MODEL" >&2; exit 1; }

check_bench_flags() {
  local help_text missing=0 flag
  help_text="$(${PYTHON_BIN} -m sglang.bench_serving --help 2>&1)" || {
    echo "[ERROR] 无法加载 ${PYTHON_BIN} -m sglang.bench_serving" >&2
    printf '%s\n' "${help_text}" >&2
    return 1
  }
  for flag in \
    generated-shared-prefix \
    --gsp-num-groups --gsp-prompts-per-group --gsp-system-prompt-len \
    --gsp-question-len --gsp-output-len --gsp-ordered --flush-cache \
    --seed --cache-report --output-file; do
    if ! grep -Fq -- "${flag}" <<< "${help_text}"; then
      echo "[ERROR] 当前 bench_serving 缺少官方参数/数据集: ${flag}" >&2
      missing=1
    fi
  done
  (( missing == 0 ))
}

parse_benchmark_log() {
  local log_file="$1"
  "${PYTHON_BIN}" - "${log_file}" <<'PY'
import json
import math
import re
import sys

path = sys.argv[1]
aliases = {
    "Backend": "backend",
    "Traffic request rate": "traffic_request_rate",
    "Max request concurrency": "max_request_concurrency",
    "Max concurrent requests": "max_request_concurrency",
    "Request throughput (req/s)": "request_throughput",
    "Input token throughput (tok/s)": "input_throughput",
    "Output token throughput (tok/s)": "output_throughput",
    "Peak output token throughput (tok/s)": "peak_output_throughput",
    "Total token throughput (tok/s)": "total_throughput",
    "Peak concurrency": "peak_concurrency",
    "Concurrency": "concurrency",
    "Accept length": "accept_length",
    "Mean E2E Latency (ms)": "mean_e2e_latency_ms",
    "Mean TTFT (ms)": "mean_ttft_ms",
    "Median TTFT (ms)": "p50_ttft_ms",
    "P50 TTFT (ms)": "p50_ttft_ms",
    "P90 TTFT (ms)": "p90_ttft_ms",
    "P95 TTFT (ms)": "p95_ttft_ms",
    "P99 TTFT (ms)": "p99_ttft_ms",
    "Max TTFT (ms)": "max_ttft_ms",
    "Mean TPOT (ms)": "mean_tpot_ms",
    "Median TPOT (ms)": "p50_tpot_ms",
    "P50 TPOT (ms)": "p50_tpot_ms",
    "P90 TPOT (ms)": "p90_tpot_ms",
    "P95 TPOT (ms)": "p95_tpot_ms",
    "P99 TPOT (ms)": "p99_tpot_ms",
    "Mean ITL (ms)": "mean_itl_ms",
    "Median ITL (ms)": "p50_itl_ms",
    "P50 ITL (ms)": "p50_itl_ms",
    "P90 ITL (ms)": "p90_itl_ms",
    "P95 ITL (ms)": "p95_itl_ms",
    "P99 ITL (ms)": "p99_itl_ms",
    "Max ITL (ms)": "max_itl_ms",
    "Total prompt tokens": "total_prompt_tokens",
    "Total cached tokens": "total_cached_tokens",
    "Total input tokens": "total_input_tokens",
    "Total input text tokens": "total_input_text_tokens",
    "Total generated tokens": "total_generated_tokens",
    "Total generated tokens (retokenized)": "total_generated_tokens_retokenized",
    "Cache hit rate": "cache_hit_rate_pct",
    "Successful requests": "successful_requests",
    "Benchmark duration (s)": "benchmark_duration_s",
}

number = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?$")

def convert(value):
    value = value.strip()
    if value.endswith("%"):
        value = value[:-1].strip()
    if number.fullmatch(value):
        parsed = float(value)
        if math.isfinite(parsed):
            return int(parsed) if parsed.is_integer() else parsed
    return value

values = {}
try:
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
except OSError:
    print("{}")
    raise SystemExit(0)
for line in lines:
    if ":" not in line:
        continue
    label, raw = line.split(":", 1)
    label = label.strip()
    if label in aliases:
        values[aliases[label]] = convert(raw)
if "cache_hit_rate_pct" not in values:
    prompt = values.get("total_prompt_tokens")
    cached = values.get("total_cached_tokens")
    if isinstance(prompt, (int, float)) and isinstance(cached, (int, float)) and prompt > 0:
        values["cache_hit_rate_pct"] = 100.0 * cached / prompt
print(json.dumps(values, ensure_ascii=False, separators=(",", ":")))
PY
}

build_bench_command() {
  local request_rate="$1"
  cmd=(
    "${PYTHON_BIN}" -m sglang.bench_serving
    --backend sglang
    --base-url "${BASE_URL}"
    --model "${MODEL}"
    --dataset-name generated-shared-prefix
    --gsp-num-groups "${GSP_NUM_GROUPS}"
    --gsp-prompts-per-group "${GSP_PROMPTS_PER_GROUP}"
    --gsp-system-prompt-len "${CASE_SYSTEM_LEN}"
    --gsp-question-len "${CASE_QUESTION_LEN}"
    --gsp-output-len "${CASE_OUTPUT_LEN}"
    --gsp-ordered
    --num-prompts "${CASE_NUM_PROMPTS}"
    --flush-cache
    --seed "${SEED}"
    --cache-report
    --request-rate "${request_rate}"
    --disable-tqdm
    --output-file /dev/null
  )
  if [[ -n "${CASE_CONCURRENCY:-}" ]]; then
    cmd+=(--max-concurrency "${CASE_CONCURRENCY}")
  fi
}

append_trial_row() {
  local request_rate="$1" phase="$2" log_file="$3" exit_code="$4" metrics_json="$5"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" - "${trials_csv}" "${CASE_ID}" "${CASE_INPUT_RAW}" "${CASE_OUTPUT_RAW}" \
    "${CASE_HIT_PCT}" "${CASE_SYSTEM_LEN}" "${CASE_QUESTION_LEN}" "${CASE_THEORY_HIT_PCT}" \
    "${CASE_REQUESTED_THEORY_HIT_PCT}" \
    "${GSP_PAGE_SIZE}" "${GSP_PAGE_ALIGN}" \
    "${GSP_NUM_GROUPS}" "${GSP_PROMPTS_PER_GROUP}" "${CASE_NUM_PROMPTS}" "${CASE_TTFT_METRIC}" \
    "${CASE_TTFT_TARGETS}" "${CASE_TPOT_METRIC}" "${CASE_TPOT_TARGET_MS}" "${request_rate}" \
    "${phase}" "${CASE_CONCURRENCY:-}" "${log_file}" "${exit_code}" "${metrics_json}" <<'PY'
import csv
import json
import os
import sys

(path, case_id, input_raw, output_raw, target_hit, system_len, question_len,
 theory_hit, requested_theory_hit, page_size, page_aligned, groups, prompts_per_group, num_prompts, ttft_metric, ttft_targets,
 tpot_metric, tpot_target, request_rate, phase, max_concurrency, log_file,
 exit_code, metrics_raw) = sys.argv[1:]
try:
    metrics = json.loads(metrics_raw)
except json.JSONDecodeError:
    metrics = {}
fields = [
    "case_id", "input", "output", "specified_cache_hit_rate_pct", "system_prompt_len",
    "question_len", "theoretical_cache_hit_rate_pct", "requested_theoretical_cache_hit_rate_pct",
    "gsp_page_size", "page_aligned",
    "gsp_num_groups",
    "gsp_prompts_per_group", "num_prompts", "ttft_metric", "ttft_targets_s",
    "tpot_metric", "tpot_target_ms", "configured_request_rate", "search_phase",
    "exit_code", "actual_rps", "backend", "traffic_request_rate", "max_request_concurrency",
    "successful_requests", "benchmark_duration_s", "total_input_tokens",
    "total_input_text_tokens", "total_generated_tokens", "total_generated_tokens_retokenized",
    "request_throughput", "input_throughput", "output_throughput", "total_throughput",
    "peak_output_throughput", "peak_concurrency", "max_concurrency", "concurrency",
    "mean_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p95_tpot_ms",
    "p99_tpot_ms", "mean_itl_ms", "p95_itl_ms", "p99_itl_ms", "accept_length",
    "mean_e2e_latency_ms", "actual_cache_hit_rate_pct", "total_prompt_tokens",
    "total_cached_tokens", "p50_ttft_ms", "p90_ttft_ms", "max_ttft_ms",
    "p50_tpot_ms", "p90_tpot_ms", "p50_itl_ms", "p90_itl_ms", "max_itl_ms",
    "duration_s", "log_file",
]
row = {key: "" for key in fields}
row.update({
    "case_id": case_id, "input": input_raw, "output": output_raw,
    "specified_cache_hit_rate_pct": target_hit, "system_prompt_len": system_len,
    "question_len": question_len, "theoretical_cache_hit_rate_pct": theory_hit,
    "requested_theoretical_cache_hit_rate_pct": requested_theory_hit,
    "gsp_page_size": page_size, "page_aligned": page_aligned,
    "gsp_num_groups": groups, "gsp_prompts_per_group": prompts_per_group,
    "num_prompts": num_prompts, "ttft_metric": ttft_metric,
    "ttft_targets_s": ttft_targets, "tpot_metric": tpot_metric,
    "tpot_target_ms": tpot_target, "configured_request_rate": request_rate,
    "search_phase": phase, "max_concurrency": max_concurrency,
    "exit_code": exit_code, "log_file": log_file,
})
row.update({k: v for k, v in metrics.items() if k in fields})
row["actual_rps"] = metrics.get("request_throughput", "")
row["request_throughput"] = metrics.get("request_throughput", "")
row["actual_cache_hit_rate_pct"] = metrics.get("cache_hit_rate_pct", "")

rows = []
if os.path.exists(path):
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        rows = []
key = (case_id, request_rate)
rows = [r for r in rows if (r.get("case_id", ""), r.get("configured_request_rate", "")) != key]
rows.append(row)
with open(path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
PY
}

run_trial() {
  local request_rate="$1" phase="$2"
  local rate_slug log_file exit_file exit_code metrics_json
  rate_slug="$(sanitize_slug "${request_rate}")"
  [[ -n "${rate_slug}" ]] || rate_slug="zero"
  log_file="${log_dir}/${CASE_ID}__rate_${rate_slug}.log"
  exit_file="${log_file}.exitcode"
  mkdir -p "${log_dir}"

  if [[ -s "${log_file}" && -f "${exit_file}" && "$(<"${exit_file}")" == "0" ]]; then
    echo "[SKIP] ${CASE_ID} request_rate=${request_rate}"
    metrics_json="$(parse_benchmark_log "${log_file}")"
    append_trial_row "${request_rate}" "${phase}" "${log_file}" 0 "${metrics_json}"
    return 0
  fi

  build_bench_command "${request_rate}"
  echo
  echo "============================================================"
  echo "CASE         : ${CASE_ID}"
  echo "RATE         : ${request_rate} (${phase})"
  echo "LOG          : ${log_file}"
  echo "============================================================"
  printf '[CMD]'; printf ' %q' "${cmd[@]}"; printf '\n'

  "${cmd[@]}" 2>&1 | tee "${log_file}"
  exit_code="${PIPESTATUS[0]}"
  if [[ "${exit_code}" != "0" ]]; then
    run_failure=1
  fi
  printf '%s\n' "${exit_code}" > "${exit_file}"
  metrics_json="$(parse_benchmark_log "${log_file}")"
  append_trial_row "${request_rate}" "${phase}" "${log_file}" "${exit_code}" "${metrics_json}"
  return 0
}

read_actual_rps() {
  local log_file="$1"
  parse_benchmark_log "${log_file}" | "${PYTHON_BIN}" -c '
import json,sys
try:
    data=json.load(sys.stdin)
    value=float(data["request_throughput"])
    if value <= 0: raise ValueError
except Exception:
    raise SystemExit(1)
print(f"{value:.2f}")
'
}

finite_rate_candidates() {
  local baseline="$1"
  if [[ -n "${EXPLICIT_RATES}" ]]; then
    local rate
    IFS=',' read -ra rates <<< "${EXPLICIT_RATES}"
    for rate in "${rates[@]}"; do
      is_positive_decimal "${rate}" || { echo "[ERROR] GSP_REQUEST_RATES 含非法值: ${rate}" >&2; return 1; }
      format_decimal "${rate}"
    done
    return 0
  fi
  local current="${baseline}" offset candidate previous=""
  IFS=',' read -ra offsets <<< "${RATE_OFFSETS}"
  for offset in "${offsets[@]}"; do
    is_decimal "${offset#-}" || { echo "[ERROR] GSP_RATE_OFFSETS 含非法值: ${offset}" >&2; return 1; }
    current="$(awk -v c="${current}" -v o="${offset}" 'BEGIN { printf "%.6f", c + o }')"
    candidate="$(format_decimal "${current}")"
    if awk -v c="${candidate}" 'BEGIN { exit !(c > 0) }' && [[ "${candidate}" != "${previous}" ]]; then
      printf '%s\n' "${candidate}"
      previous="${candidate}"
    fi
  done
}

write_summary() {
  "${PYTHON_BIN}" - "${trials_csv}" "${summary_csv}" "${HIT_RATE_TOLERANCE}" <<'PY'
import csv
import math
import os
import sys

trials_path, summary_path, tolerance_s = sys.argv[1:]
tolerance = float(tolerance_s)
fields = [
    "case_id", "input", "output", "specified_cache_hit_rate_pct", "theoretical_cache_hit_rate_pct",
    "requested_theoretical_cache_hit_rate_pct", "gsp_page_size", "page_aligned",
    "measured_hit_rate_pct", "hit_rate_delta_pct", "ttft_metric", "ttft_target_s",
    "selected_actual_rps", "selected_input_throughput", "selected_ttft_ms",
    "selected_tpot_ms", "selected_configured_request_rate", "status", "log_file",
]
rows = []
if os.path.exists(trials_path):
    with open(trials_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

def num(row, key):
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None

summary = []
groups = {}
for row in rows:
    groups.setdefault(row.get("case_id", ""), []).append(row)

for case_id, case_rows in groups.items():
    if not case_rows:
        continue
    base = case_rows[0]
    targets = [x for x in base.get("ttft_targets_s", "").split("|") if x]
    for target_s in targets:
        try:
            target_ms = float(target_s) * 1000.0
        except ValueError:
            continue
        candidates = []
        for row in case_rows:
            if row.get("exit_code", "") != "0":
                continue
            actual_rps = num(row, "request_throughput")
            measured_hit = num(row, "actual_cache_hit_rate_pct")
            target_hit = num(row, "specified_cache_hit_rate_pct")
            selected_ttft = num(row, row.get("ttft_metric", ""))
            selected_tpot = num(row, row.get("tpot_metric", ""))
            tpot_limit = num(row, "tpot_target_ms")
            if None in (actual_rps, measured_hit, target_hit, selected_ttft, selected_tpot, tpot_limit):
                continue
            delta = abs(measured_hit - target_hit)
            if delta <= tolerance and selected_ttft <= target_ms and selected_tpot <= tpot_limit:
                candidates.append((actual_rps, row, selected_ttft, selected_tpot, delta))
        if candidates:
            actual_rps, row, ttft, tpot, delta = max(candidates, key=lambda x: x[0])
            measured = num(row, "actual_cache_hit_rate_pct")
            summary.append({
                "case_id": case_id, "input": base.get("input", ""), "output": base.get("output", ""),
                "specified_cache_hit_rate_pct": base.get("specified_cache_hit_rate_pct", ""),
                "theoretical_cache_hit_rate_pct": base.get("theoretical_cache_hit_rate_pct", ""),
                "requested_theoretical_cache_hit_rate_pct": base.get("requested_theoretical_cache_hit_rate_pct", ""),
                "gsp_page_size": base.get("gsp_page_size", ""),
                "page_aligned": base.get("page_aligned", ""),
                "measured_hit_rate_pct": f"{measured:.4f}", "hit_rate_delta_pct": f"{delta:.4f}",
                "ttft_metric": base.get("ttft_metric", ""), "ttft_target_s": target_s,
                "selected_actual_rps": f"{actual_rps:.2f}",
                "selected_input_throughput": row.get("input_throughput", ""),
                "selected_ttft_ms": f"{ttft:.4f}", "selected_tpot_ms": f"{tpot:.4f}",
                "selected_configured_request_rate": row.get("configured_request_rate", ""),
                "status": "MEET", "log_file": row.get("log_file", ""),
            })
            continue

        measured_values = [num(r, "actual_cache_hit_rate_pct") for r in case_rows]
        measured_values = [x for x in measured_values if x is not None]
        target_hit = num(base, "specified_cache_hit_rate_pct")
        status = "NO_MEET"
        if not measured_values:
            status = "NO_CACHE_REPORT"
        elif target_hit is not None and all(abs(x - target_hit) > tolerance for x in measured_values):
            status = "HIT_RATE_MISMATCH"
        summary.append({
            "case_id": case_id, "input": base.get("input", ""), "output": base.get("output", ""),
            "specified_cache_hit_rate_pct": base.get("specified_cache_hit_rate_pct", ""),
            "theoretical_cache_hit_rate_pct": base.get("theoretical_cache_hit_rate_pct", ""),
            "requested_theoretical_cache_hit_rate_pct": base.get("requested_theoretical_cache_hit_rate_pct", ""),
            "gsp_page_size": base.get("gsp_page_size", ""),
            "page_aligned": base.get("page_aligned", ""),
            "measured_hit_rate_pct": f"{max(measured_values):.4f}" if measured_values else "",
            "hit_rate_delta_pct": "", "ttft_metric": base.get("ttft_metric", ""),
            "ttft_target_s": target_s, "selected_actual_rps": "",
            "selected_input_throughput": "", "selected_ttft_ms": "", "selected_tpot_ms": "",
            "selected_configured_request_rate": "", "status": status, "log_file": "",
        })

with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary)
PY
}

plot_summary() {
  [[ "${plot_requested}" == "1" ]] || return 0
  "${PYTHON_PLOT_BIN}" - "${summary_csv}" "${out_dir}/gsp_prefix_matrix.png" "${plot_cell}" <<'PY'
import csv
import math
import sys
from pathlib import Path

summary_path, output_path, cell = sys.argv[1:]
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    raise SystemExit(f"matplotlib/numpy 不可用，无法作图: {exc}")

with open(summary_path, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit("summary CSV 没有可作图数据")

columns = []
for row in rows:
    label = f"{row.get('input','')} / {row.get('output','')}\nhit {row.get('specified_cache_hit_rate_pct','')}%"
    if label not in columns:
        columns.append(label)
targets = []
for row in rows:
    label = f"{row.get('ttft_metric','')} < {row.get('ttft_target_s','')} s"
    if label not in targets:
        targets.append(label)

matrix = np.full((len(targets), len(columns)), np.nan, dtype=float)
annotations = [["NO\nMEET" for _ in columns] for _ in targets]
for row in rows:
    col = f"{row.get('input','')} / {row.get('output','')}\nhit {row.get('specified_cache_hit_rate_pct','')}%"
    target = f"{row.get('ttft_metric','')} < {row.get('ttft_target_s','')} s"
    i, j = targets.index(target), columns.index(col)
    value_key = "selected_actual_rps" if cell == "rps" else "selected_input_throughput"
    try:
        value = float(row.get(value_key, ""))
        if not math.isfinite(value):
            raise ValueError
        matrix[i, j] = value
        annotations[i][j] = f"{value:.2f}"
    except (TypeError, ValueError):
        annotations[i][j] = row.get("status", "NO_MEET")

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
PY
}

mkdir -p "${out_dir}" "${log_dir}"
check_bench_flags || exit 1

for raw_case in "${CASES[@]}"; do
  parse_case "${raw_case}" || exit 1
  print_case

  inf_slug="$(sanitize_slug inf)"
  inf_log="${log_dir}/${CASE_ID}__rate_${inf_slug}.log"
  run_trial inf baseline
  if [[ ! -s "${inf_log}" ]]; then
    echo "[WARN] inf baseline 没有生成日志，跳过后续 rate: ${CASE_ID}" >&2
    run_failure=1
    continue
  fi
  baseline_rps="$(read_actual_rps "${inf_log}" 2>/dev/null || true)"
  if [[ -z "${baseline_rps}" ]]; then
    echo "[WARN] 无法从 inf baseline 读取实际 RPS，跳过后续 rate: ${CASE_ID}" >&2
    run_failure=1
    continue
  fi

  while IFS= read -r finite_rate; do
    [[ -n "${finite_rate}" ]] || continue
    run_trial "${finite_rate}" descending
  done < <(finite_rate_candidates "${baseline_rps}")
done

write_summary
if ! plot_summary; then
  if (( plot_requested == 1 )); then
    echo "[ERROR] --plot requested but the matrix image was not generated" >&2
    exit 1
  fi
fi

echo
echo "============================================================"
echo "GSP prefix benchmark finished"
echo "Result directory : ${out_dir}"
echo "Trials CSV       : ${trials_csv}"
echo "Summary CSV      : ${summary_csv}"
if (( plot_requested == 1 )); then
  echo "Plot             : ${out_dir}/gsp_prefix_matrix.png"
fi
echo "============================================================"
if (( run_failure != 0 )); then
  echo "[ERROR] 至少一个官方 benchmark trial 失败，详见日志和 exitcode 文件" >&2
  exit 1
fi
