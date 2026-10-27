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
GSP_NUM_GROUPS_WAS_SET="${GSP_NUM_GROUPS+x}"
GSP_NUM_GROUPS="${GSP_NUM_GROUPS:-2}"
GSP_NUM_GROUPS_CANDIDATES="${GSP_NUM_GROUPS_CANDIDATES:-2,4,8}"
GSP_PROMPTS_PER_GROUP_WAS_SET="${GSP_PROMPTS_PER_GROUP+x}"
GSP_PROMPTS_PER_GROUP="${GSP_PROMPTS_PER_GROUP:-16}"
GSP_PROMPTS_PER_GROUP_CANDIDATES="${GSP_PROMPTS_PER_GROUP_CANDIDATES:-16,32,64,96,128,256}"
GSP_CACHE_MISS_REQUESTS_PER_GROUP="${GSP_CACHE_MISS_REQUESTS_PER_GROUP:-4}"
GSP_AUTO_CALIBRATE="${GSP_AUTO_CALIBRATE:-1}"
GSP_CALIBRATION_MAX_ITERS="${GSP_CALIBRATION_MAX_ITERS:-1}"
GSP_CALIBRATION_STEP_RATIO="${GSP_CALIBRATION_STEP_RATIO:-0.5}"
GSP_AUTO_LAYOUT="${GSP_AUTO_LAYOUT:-1}"
GSP_AUTO_DURATION="${GSP_AUTO_DURATION:-1}"
GSP_TARGET_DURATION_S="${GSP_TARGET_DURATION_S:-60}"
GSP_PILOT_INPUT_TOKEN_BUDGET="${GSP_PILOT_INPUT_TOKEN_BUDGET:-1048576}"
GSP_DURATION_FIRST_PASS_S="${GSP_DURATION_FIRST_PASS_S:-30}"
GSP_DURATION_ADJUST_MAX_ITERS="${GSP_DURATION_ADJUST_MAX_ITERS:-2}"
GSP_DURATION_TOLERANCE_PCT="${GSP_DURATION_TOLERANCE_PCT:-25}"
GSP_STOP_ON_SLA="${GSP_STOP_ON_SLA:-1}"
GSP_MAX_CONCURRENCY="${GSP_MAX_CONCURRENCY:-}"
GSP_INPUT_IDS="${GSP_INPUT_IDS:-1}"
GSP_INF_MAX_TRIALS="${GSP_INF_MAX_TRIALS:-5}"
GSP_INF_NO_IMPROVEMENT_LIMIT="${GSP_INF_NO_IMPROVEMENT_LIMIT:-2}"
GSP_PAGE_SIZE="${GSP_PAGE_SIZE:-}"
PARALLEL_CONFIG="${PARALLEL_CONFIG:-}"
GSP_PAGE_ALIGN="${GSP_PAGE_ALIGN:-1}"
GSP_SYSTEM_LEN_EXTRA_TOKENS="${GSP_SYSTEM_LEN_EXTRA_TOKENS:-0}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
NODE_HOSTNAME="${NODE_HOSTNAME:-$(hostname 2>/dev/null || printf 'unknown')}"
RUN_STARTED_AT="${RUN_STARTED_AT:-$(date +%Y-%m-%dT%H:%M:%S%z)}"
SCRIPT_COMMIT="$(git -C "${script_dir}/.." rev-parse --short HEAD 2>/dev/null || true)"
PYTHON_PLOT_BIN="${PYTHON_PLOT_BIN:-${PYTHON_BIN}}"
RATE_STEP="${GSP_RATE_STEP:-0.1}"
HIT_RATE_TOLERANCE="${GSP_HIT_RATE_TOLERANCE_PCT:-0.5}"
BASELINE_HIT_RATE_TOLERANCE="${GSP_BASELINE_HIT_RATE_TOLERANCE_PCT:-0.5}"
FINITE_HIT_RATE_TOLERANCE="${GSP_FINITE_HIT_RATE_TOLERANCE_PCT:-1.5}"
EXPLICIT_RATES="${GSP_REQUEST_RATES:-}"

out_dir="./all_result/bench_gsp_prefix_${RUN_ID}"
log_dir="${out_dir}/logs"
trials_csv="${out_dir}/gsp_prefix_trials.csv"
summary_tsv="${out_dir}/gsp_prefix_summary.tsv"
plot_requested="${GSP_PLOT:-1}"
plot_cell="rps"
dry_run=0
run_failure=0

declare -a CASES=()
CASE_NO_SHARED_PREFIX=0

write_run_metadata() {
  local metadata_path="${out_dir}/run_metadata.json"
  "${PYTHON_BIN}" - "${metadata_path}" \
    "${RUN_STARTED_AT}" "${script_dir}/$(basename "${BASH_SOURCE[0]}")" \
    "${SCRIPT_COMMIT}" "${RUN_ID}" "${BASE_URL}" "${MODEL}" \
    "${NODE_HOSTNAME}" \
    "${out_dir}" "${GSP_PAGE_SIZE}" "${PARALLEL_CONFIG}" <<'PY_METADATA'
import json
import os
import sys
import tempfile

(path, started_at, script, script_commit, run_id, base_url, model,
 node_hostname, output_dir, page_size, parallel_config) = sys.argv[1:]
prefixes = ("GSP_", "TOKEN_UNIT", "DP_SIZE", "TP_SIZE", "PP_SIZE",
            "EP_SIZE", "ATTN_CP_SIZE", "KV_CACHE_DTYPE", "MEM_FRACTION_STATIC",
            "DP_ATTENTION", "SEED")
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
    "node_hostname": node_hostname,
    "output_dir": output_dir,
    "gsp_page_size": page_size,
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
    printf '# node_hostname: %s\n' "${NODE_HOSTNAME}"
    printf '# base_url: %s\n' "${BASE_URL}"
    printf '# run_id: %s\n' "${RUN_ID}"
    printf '# started_at: %s\n' "${RUN_STARTED_AT}"
    printf '#\n'
  } > "${log_file}"
}

usage() {
  cat <<'USAGE'
Usage:
  bash sglang/gsp_prefix_bench.sh [options]

Options:
  --case CASE       Repeatable case:
                    INPUT,OUTPUT,HIT_RATE,TTFT_METRIC,TTFT_TARGETS_S[,TPOT_METRIC,TPOT_TARGET_MS[,MAX_CONCURRENCY]]
  --plot            Generate a heatmap after CSV summary is written (default)
  --no-plot         Disable the default heatmap generation
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
  GSP_NUM_GROUPS=2                Preferred group count when explicitly set
  GSP_NUM_GROUPS_CANDIDATES=2,4,8 inf fallback candidates
  GSP_PROMPTS_PER_GROUP=16        Preferred prompts/group when explicitly set
  GSP_PROMPTS_PER_GROUP_CANDIDATES=16,32,64,96,128,256  inf candidates
  GSP_CACHE_MISS_REQUESTS_PER_GROUP=4  Concurrent cache misses per group used for calibration
  GSP_AUTO_CALIBRATE=1                Auto-correct shared prefix after measured hit-rate drift
  GSP_CALIBRATION_MAX_ITERS=1         Maximum corrective full workloads per layout
  GSP_CALIBRATION_STEP_RATIO=0.5      Move this fraction toward the measured correction each time
  GSP_AUTO_LAYOUT=1                   Try alternate groups/prompts layouts after hit-rate calibration fails
  GSP_AUTO_DURATION=1             Resize each case to the target benchmark duration
  GSP_TARGET_DURATION_S=60        Target main benchmark duration, allowed range 60-120s
  GSP_PILOT_INPUT_TOKEN_BUDGET=1048576  Max input tokens for an automatic pilot probe
  GSP_DURATION_FIRST_PASS_S=30    Short first duration pass before target-duration adjustment
  GSP_DURATION_ADJUST_MAX_ITERS=2 Maximum duration adjustment passes per case
  GSP_DURATION_TOLERANCE_PCT=25  Acceptable duration error percentage around target
  GSP_STOP_ON_SLA=1              Stop descending backtests after all TTFT targets meet
  GSP_MAX_CONCURRENCY=            Optional global max concurrency
  GSP_INPUT_IDS=1                 Pass raw token IDs with official --gsp-input-ids
  GSP_INF_MAX_TRIALS=5            Maximum repeated inf probes before finite-rate backtests
  GSP_INF_NO_IMPROVEMENT_LIMIT=2 Stop inf probes after this many non-improving input-throughput trials
  GSP_PAGE_SIZE=                  KV cache page size; auto-read from /server_info, fallback 64
  PARALLEL_CONFIG=                Parallel layout; auto-read from /server_info
  GSP_PAGE_ALIGN=1                Align shared prefix to a whole KV page; set 0 to disable
  GSP_SYSTEM_LEN_EXTRA_TOKENS=0   Optional page-aligned cache-report calibration tokens
  GSP_REQUEST_RATES=              Optional comma-separated finite rate list
  GSP_PLOT=1                      Generate the Prefill/GSP heatmap by default; set 0 to disable
  GSP_RATE_STEP=0.1               Uniform descending finite-rate step; minimum 0.1
  GSP_HIT_RATE_TOLERANCE_PCT=0.5  Workload/page-alignment tolerance fallback
  GSP_BASELINE_HIT_RATE_TOLERANCE_PCT=0.5  Strict inf hit-rate tolerance
  GSP_FINITE_HIT_RATE_TOLERANCE_PCT=1.5  Finite-rate hit-rate tolerance for concurrency drift
  RUN_ID=...                       Result directory suffix

The hit rate is token-level: total_cached_tokens / total_prompt_tokens.
  HIT_RATE=0 uses the official random-ids workload with independent token ids, so it
  is a true no-prefix-cache control without downloading ShareGPT. Multiple --case values can be run
  together; each case is calibrated and measured independently.
  The default calibration assumes four concurrent first requests per ordered prefix group can miss;
  the script adjusts system/question length with the official full workload and limits correction
  to one additional full workload by default. The default inf tolerance is 0.5 percentage points to
  avoid repeating a full run for normal page-alignment drift. Set GSP_BASELINE_HIT_RATE_TOLERANCE_PCT
  or GSP_CALIBRATION_MAX_ITERS explicitly for stricter or more exhaustive calibration. Set
  GSP_AUTO_LAYOUT=0 to keep explicit layouts fixed.
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
    --no-plot)
      plot_requested=0
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

if [[ "${plot_requested}" != "0" && "${plot_requested}" != "1" ]]; then
  echo "[ERROR] GSP_PLOT/plot_requested 必须是 0 或 1，当前为: ${plot_requested}" >&2
  exit 1
fi

for value_name in GSP_INF_MAX_TRIALS GSP_INF_NO_IMPROVEMENT_LIMIT; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] ${value_name} 必须是正整数，当前为: ${value}" >&2
    exit 1
  fi
done

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

validate_rate_spacing() {
  local previous="" rate offset
  if [[ -n "${EXPLICIT_RATES}" ]]; then
    IFS=',' read -ra _explicit_rates <<< "${EXPLICIT_RATES}"
    for rate in "${_explicit_rates[@]}"; do
      is_positive_decimal "${rate}" || {
        echo "[ERROR] GSP_REQUEST_RATES 含非法值: ${rate}" >&2
        return 1
      }
      if [[ -n "${previous}" ]] && ! awk -v prev="${previous}" -v curr="${rate}" -v step="${RATE_STEP}" \
        'BEGIN { exit !(prev > curr && prev - curr + 1e-9 >= step) }'; then
        echo "[ERROR] GSP_REQUEST_RATES 必须按高到低排列，且相邻请求率间隔不小于 ${RATE_STEP}" >&2
        return 1
      fi
      previous="${rate}"
    done
    return 0
  fi

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

declare -a GROUP_CANDIDATES=()
declare -a PROMPT_CANDIDATES=()

resolve_group_candidates() {
  local raw="${GSP_NUM_GROUPS_CANDIDATES}"
  local group existing
  if [[ "${GSP_NUM_GROUPS_WAS_SET}" == "x" && "${GSP_AUTO_LAYOUT}" == "0" ]]; then
    raw="${GSP_NUM_GROUPS}"
  elif [[ "${GSP_NUM_GROUPS_WAS_SET}" == "x" ]]; then
    raw="${GSP_NUM_GROUPS},${raw}"
  fi
  IFS=',' read -ra _group_values <<< "${raw}"
  GROUP_CANDIDATES=()
  for group in "${_group_values[@]}"; do
    [[ "${group}" =~ ^[1-9][0-9]*$ ]] || {
      echo "[ERROR] GSP_NUM_GROUPS_CANDIDATES 含非法值: ${group}" >&2
      return 1
    }
    existing=0
    for existing_group in "${GROUP_CANDIDATES[@]}"; do
      [[ "${existing_group}" == "${group}" ]] && existing=1
    done
    (( existing == 0 )) && GROUP_CANDIDATES+=("${group}")
  done
  (( ${#GROUP_CANDIDATES[@]} > 0 )) || {
    echo "[ERROR] GSP_NUM_GROUPS_CANDIDATES 不能为空" >&2
    return 1
  }
}

resolve_prompt_candidates() {
  local raw="${GSP_PROMPTS_PER_GROUP_CANDIDATES}"
  local prompts existing_prompt existing_prompt_value
  if [[ "${GSP_PROMPTS_PER_GROUP_WAS_SET}" == "x" && "${GSP_AUTO_LAYOUT}" == "0" ]]; then
    raw="${GSP_PROMPTS_PER_GROUP}"
  elif [[ "${GSP_PROMPTS_PER_GROUP_WAS_SET}" == "x" ]]; then
    raw="${GSP_PROMPTS_PER_GROUP},${raw}"
  fi
  IFS=',' read -ra _prompt_values <<< "${raw}"
  PROMPT_CANDIDATES=()
  for prompts in "${_prompt_values[@]}"; do
    [[ "${prompts}" =~ ^[1-9][0-9]*$ ]] || {
      echo "[ERROR] GSP_PROMPTS_PER_GROUP_CANDIDATES 含非法值: ${prompts}" >&2
      return 1
    }
    (( prompts >= 2 )) || {
      echo "[ERROR] GSP_PROMPTS_PER_GROUP_CANDIDATES 中的值必须不小于 2: ${prompts}" >&2
      return 1
    }
    existing_prompt=0
    for existing_prompt_value in "${PROMPT_CANDIDATES[@]}"; do
      [[ "${existing_prompt_value}" == "${prompts}" ]] && existing_prompt=1
    done
    (( existing_prompt == 0 )) && PROMPT_CANDIDATES+=("${prompts}")
  done
  (( ${#PROMPT_CANDIDATES[@]} > 0 )) || {
    echo "[ERROR] GSP_PROMPTS_PER_GROUP_CANDIDATES 不能为空" >&2
    return 1
  }
}

order_layout_candidates_for_case() {
  [[ "${GSP_AUTO_LAYOUT}" == "1" ]] || return 0

  local value
  local -a ordered=() rest=()
  # Probe every automatic case from the smallest layout. The measured RPS
  # then determines the request count for the target duration; this avoids
  # making one input-length threshold special.

  # Keep the familiar fixed layout first around the 90% workload. Otherwise,
  # try smaller prompt groups first so the first probe stays bounded for every
  # input length and the measured RPS can drive the later request-count resize.
  if [[ "${GSP_PROMPTS_PER_GROUP_WAS_SET}" == "x" ]] &&
     awk -v target="${CASE_HIT_PCT}" 'BEGIN { exit !(target >= 85 && target < 92) }'; then
    ordered+=("${GSP_PROMPTS_PER_GROUP}")
    for value in "${PROMPT_CANDIDATES[@]}"; do
      [[ "${value}" == "${GSP_PROMPTS_PER_GROUP}" ]] || rest+=("${value}")
    done
  else
    rest=("${PROMPT_CANDIDATES[@]}")
  fi

  if (( ${#rest[@]} > 0 )); then
    # Values are numeric and contain no whitespace; avoid a persistent IFS
    # assignment leaking into the next repeated --case parse.
    while read -r value; do [[ -n "${value}" ]] && ordered+=("${value}"); done < <(printf '%s\n' "${rest[@]}" | sort -n)
  fi
  PROMPT_CANDIDATES=("${ordered[@]}")
}

cap_initial_probe_groups() {
  [[ "${GSP_AUTO_DURATION}" == "1" ]] || return 0
  [[ "${GSP_NUM_GROUPS_WAS_SET}" != "x" ]] || return 0
  [[ "${GSP_PILOT_INPUT_TOKEN_BUDGET}" =~ ^[1-9][0-9]*$ ]] || return 0
  [[ "${CASE_INPUT_LEN}" =~ ^[1-9][0-9]*$ ]] || return 0
  [[ "${GSP_PROMPTS_PER_GROUP}" =~ ^[1-9][0-9]*$ ]] || return 0

  local max_groups
  max_groups=$((GSP_PILOT_INPUT_TOKEN_BUDGET / CASE_INPUT_LEN / GSP_PROMPTS_PER_GROUP))
  (( max_groups >= 1 )) || max_groups=1
  if (( GSP_NUM_GROUPS > max_groups )); then
    echo "[INFO] 自动首轮 probe 限制总输入 token: groups=${GSP_NUM_GROUPS}->${max_groups}, prompts/group=${GSP_PROMPTS_PER_GROUP}, budget=${GSP_PILOT_INPUT_TOKEN_BUDGET}" >&2
    GSP_NUM_GROUPS="${max_groups}"
  fi
}

derive_gsp_lengths() {
  local input_len="$1" hit_pct="$2" prompts="$3" page_size="$4" page_align="$5" extra_tokens="$6" misses="$7" tolerance="$8"
  "${PYTHON_BIN}" - "$input_len" "$hit_pct" "$prompts" "$page_size" "$page_align" "$extra_tokens" "$misses" "$tolerance" <<'PY' | tr -d '\r'
import math
import sys

length, hit_pct, prompts, misses = (
    int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[7])
)
tolerance = float(sys.argv[8])
page_size, page_align, extra_tokens = int(sys.argv[4]), sys.argv[5] == "1", int(sys.argv[6])
if abs(hit_pct) < 1e-12:
    print(0, length, "0.000000", "0.000000")
    raise SystemExit(0)
if prompts <= misses:
    raise SystemExit("prompts_per_group must be greater than cache misses per group")
if extra_tokens < 0:
    raise SystemExit("extra system tokens must be non-negative")
desired_system_len = (hit_pct / 100.0) * prompts * length / (prompts - misses)
if desired_system_len >= length:
    raise SystemExit("target hit rate cannot fit with this prompts_per_group and cache misses")
rounded_system_len = int(math.floor(desired_system_len + 0.5))
system_len = rounded_system_len
if page_align and page_size > 1:
    floor_len = max(1, (rounded_system_len // page_size) * page_size)
    ceil_len = min(length - 1, ((rounded_system_len + page_size - 1) // page_size) * page_size)
    floor_hit = 100.0 * (prompts - misses) * floor_len / (prompts * length)
    ceil_hit = 100.0 * (prompts - misses) * ceil_len / (prompts * length)
    system_len = floor_len
    if abs(floor_hit - hit_pct) > tolerance and 0 < ceil_len < length:
        system_len = ceil_len
system_len += extra_tokens
rounded_system_len += extra_tokens
question_len = length - system_len
if system_len <= 0 or question_len <= 0 or system_len >= length:
    raise SystemExit("target hit rate cannot produce positive system/question lengths")
theory = 100.0 * (prompts - misses) * system_len / (prompts * length)
requested_theory = 100.0 * (prompts - misses) * rounded_system_len / (prompts * length)
print(system_len, question_len, f"{theory:.6f}", f"{requested_theory:.6f}")
PY
}

set_case_prompt_config() {
  local prompts="$1" lengths
  lengths="$(derive_gsp_lengths "${CASE_INPUT_LEN}" "${CASE_HIT_PCT}" "${prompts}" \
    "${GSP_PAGE_SIZE}" "${GSP_PAGE_ALIGN}" "${GSP_SYSTEM_LEN_EXTRA_TOKENS}" \
    "${GSP_CACHE_MISS_REQUESTS_PER_GROUP}" "${HIT_RATE_TOLERANCE}")" || return 1
  read -r CASE_SYSTEM_LEN CASE_QUESTION_LEN CASE_THEORY_HIT_PCT CASE_REQUESTED_THEORY_HIT_PCT <<< "${lengths}"
  GSP_PROMPTS_PER_GROUP="${prompts}"
  CASE_NUM_PROMPTS=$((GSP_NUM_GROUPS * GSP_PROMPTS_PER_GROUP))
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

resolve_parallel_config() {
  [[ -n "${PARALLEL_CONFIG}" ]] && return 0
  if (( dry_run == 1 )); then
    PARALLEL_CONFIG="UNKNOWN"
    return 0
  fi
  PARALLEL_CONFIG="$(${PYTHON_BIN} - "${BASE_URL%/}/server_info" <<'PY'
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
        obj = json.load(response)
except Exception:
    obj = {}

def value(name, *fallbacks):
    current = obj.get(name)
    if current is not None:
        return current
    for parent in fallbacks:
        if isinstance(obj.get(parent), dict) and obj[parent].get(name) is not None:
            return obj[parent][name]
    return "NA"

def text(item):
    if isinstance(item, bool):
        return "true" if item else "false"
    return str(item) if item not in (None, "") else "NA"

tp = text(value("tp_size", "server_args"))
dp = text(value("dp_size", "server_args"))
pp = text(value("pp_size", "server_args"))
ep = text(value("ep_size", "server_args"))
acp = text(value("attn_cp_size", "server_args"))
dpa = value("enable_dp_attention", "server_args")
attn = "DPA" if dpa is True or str(dpa).lower() == "true" else "TPA"
kv = text(value("kv_cache_dtype", "server_args"))
mem = text(value("mem_fraction_static", "server_args"))
print(f"TP{tp}-DP{dp}({attn})-PP{pp}-EP{ep}-ACP{acp}-KV{kv}-MEM{mem}")
PY
  )"
  [[ -n "${PARALLEL_CONFIG}" ]] || PARALLEL_CONFIG="UNKNOWN"
}

parse_case() {
  local raw="${1//$'\r'/}"
  local input_raw="" output_raw="" hit_pct="" ttft_metric="" ttft_targets=""
  local tpot_metric="" tpot_target_ms="" concurrency="" extra=""
  local IFS=','
  read -r input_raw output_raw hit_pct ttft_metric ttft_targets \
    tpot_metric tpot_target_ms concurrency extra <<< "$raw"

  # Keep parsing isolated between repeated --case values.  The benchmark mutates
  # global case fields while calibrating layouts and duration, so stale fields
  # here can make a later case's SLA appear invalid.
  CASE_INPUT_RAW="${input_raw}"
  CASE_OUTPUT_RAW="${output_raw}"
  CASE_HIT_PCT="${hit_pct}"
  CASE_TTFT_METRIC="${ttft_metric}"
  CASE_TTFT_TARGETS="${ttft_targets}"
  CASE_TPOT_METRIC="${tpot_metric}"
  CASE_TPOT_TARGET_MS="${tpot_target_ms}"
  CASE_CONCURRENCY="${concurrency}"
  extra="${extra}"

  [[ -z "${extra:-}" ]] || {
    echo "[ERROR] case 字段过多: ${raw}" >&2
    return 1
  }
  [[ -n "${CASE_INPUT_RAW:-}" && -n "${CASE_OUTPUT_RAW:-}" && -n "${CASE_HIT_PCT:-}" \
    && -n "${CASE_TTFT_METRIC:-}" && -n "${CASE_TTFT_TARGETS:-}" ]] || {
    echo "[ERROR] 非法 case: ${raw}" >&2
    return 1
  }
  if [[ -n "${CASE_TPOT_METRIC:-}" || -n "${CASE_TPOT_TARGET_MS:-}" ]]; then
    [[ -n "${CASE_TPOT_METRIC:-}" && -n "${CASE_TPOT_TARGET_MS:-}" ]] || {
      echo "[ERROR] TPOT 指标和阈值必须同时设置: ${raw}" >&2
      return 1
    }
  fi
  is_nonnegative_decimal "${CASE_HIT_PCT}" || {
    echo "[ERROR] HIT_RATE 必须是非负数: ${raw}" >&2
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
  CASE_NO_SHARED_PREFIX=0
  if awk -v v="${CASE_HIT_PCT}" 'BEGIN { exit !(v == 0) }'; then
    CASE_NO_SHARED_PREFIX=1
  fi

  case "${CASE_TTFT_METRIC}" in
    mean_ttft_ms|p50_ttft_ms|p90_ttft_ms|p95_ttft_ms|p99_ttft_ms|max_ttft_ms) ;;
    *) echo "[ERROR] 不支持的 TTFT 指标: ${CASE_TTFT_METRIC}" >&2; return 1 ;;
  esac
  if [[ -n "${CASE_TPOT_METRIC}" ]]; then
    case "${CASE_TPOT_METRIC}" in
      mean_tpot_ms|p50_tpot_ms|p90_tpot_ms|p95_tpot_ms|p99_tpot_ms) ;;
      *) echo "[ERROR] 不支持的 TPOT 指标: ${CASE_TPOT_METRIC}" >&2; return 1 ;;
    esac
  fi

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
  if [[ -n "${CASE_TPOT_METRIC}" ]]; then
    is_positive_decimal "${CASE_TPOT_TARGET_MS}" || {
      echo "[ERROR] TPOT SLA 必须是正数毫秒: ${raw}" >&2
      return 1
    }
  fi

  CASE_INPUT_LEN="$(tokens_from_k "${CASE_INPUT_RAW}")" || {
    echo "[ERROR] 无法解析输入长度: ${raw}" >&2
    return 1
  }
  CASE_OUTPUT_LEN="$(tokens_from_k "${CASE_OUTPUT_RAW}")" || {
    echo "[ERROR] 无法解析输出长度: ${raw}" >&2
    return 1
  }
  GSP_PROMPTS_PER_GROUP="${PROMPT_CANDIDATES[0]}"
  CASE_SYSTEM_LEN=""
  CASE_QUESTION_LEN=""
  CASE_THEORY_HIT_PCT=""
  CASE_REQUESTED_THEORY_HIT_PCT=""
  CASE_NUM_PROMPTS=$((GSP_NUM_GROUPS * PROMPT_CANDIDATES[0]))
  CASE_ID="$(sanitize_slug "${CASE_INPUT_RAW}_${CASE_OUTPUT_RAW}_hit${CASE_HIT_PCT}_${CASE_TTFT_METRIC}")"
  return 0
}

validate_global_config() {
  [[ "${TOKEN_UNIT}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] TOKEN_UNIT 必须是正整数" >&2; return 1; }
  [[ "${GSP_NUM_GROUPS}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] GSP_NUM_GROUPS 必须是正整数" >&2; return 1; }
  [[ "${GSP_PROMPTS_PER_GROUP}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] GSP_PROMPTS_PER_GROUP 必须是正整数" >&2; return 1; }
  [[ "${GSP_CACHE_MISS_REQUESTS_PER_GROUP}" =~ ^[0-9]+$ ]] || { echo "[ERROR] GSP_CACHE_MISS_REQUESTS_PER_GROUP 必须是非负整数" >&2; return 1; }
  [[ "${GSP_AUTO_CALIBRATE}" =~ ^[01]$ ]] || { echo "[ERROR] GSP_AUTO_CALIBRATE 必须是 0 或 1" >&2; return 1; }
  [[ "${GSP_CALIBRATION_MAX_ITERS}" =~ ^[0-9]+$ ]] || { echo "[ERROR] GSP_CALIBRATION_MAX_ITERS 必须是非负整数" >&2; return 1; }
  is_positive_decimal "${GSP_CALIBRATION_STEP_RATIO}" || { echo "[ERROR] GSP_CALIBRATION_STEP_RATIO 必须是正数" >&2; return 1; }
  awk -v v="${GSP_CALIBRATION_STEP_RATIO}" 'BEGIN { exit !(v <= 1) }' || {
    echo "[ERROR] GSP_CALIBRATION_STEP_RATIO 不能大于 1" >&2
    return 1
  }
  [[ "${GSP_AUTO_LAYOUT}" =~ ^[01]$ ]] || { echo "[ERROR] GSP_AUTO_LAYOUT 必须是 0 或 1" >&2; return 1; }
  [[ "${GSP_AUTO_DURATION}" =~ ^[01]$ ]] || { echo "[ERROR] GSP_AUTO_DURATION 必须是 0 或 1" >&2; return 1; }
  [[ "${GSP_TARGET_DURATION_S}" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "[ERROR] GSP_TARGET_DURATION_S 必须是数字" >&2; return 1; }
  awk -v v="${GSP_TARGET_DURATION_S}" 'BEGIN { exit !(v >= 60 && v <= 120) }' || {
    echo "[ERROR] GSP_TARGET_DURATION_S 必须在 60-120 秒之间" >&2
    return 1
  }
  [[ "${GSP_PILOT_INPUT_TOKEN_BUDGET}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] GSP_PILOT_INPUT_TOKEN_BUDGET 必须是正整数" >&2; return 1; }
  [[ "${GSP_DURATION_FIRST_PASS_S}" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "[ERROR] GSP_DURATION_FIRST_PASS_S 必须是数字" >&2; return 1; }
  awk -v v="${GSP_DURATION_FIRST_PASS_S}" -v target="${GSP_TARGET_DURATION_S}" 'BEGIN { exit !(v > 0 && v <= target) }' || {
    echo "[ERROR] GSP_DURATION_FIRST_PASS_S 必须大于 0 且不超过 GSP_TARGET_DURATION_S" >&2
    return 1
  }
  [[ "${GSP_DURATION_ADJUST_MAX_ITERS}" =~ ^[0-9]+$ ]] || { echo "[ERROR] GSP_DURATION_ADJUST_MAX_ITERS 必须是非负整数" >&2; return 1; }
  [[ "${GSP_DURATION_TOLERANCE_PCT}" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "[ERROR] GSP_DURATION_TOLERANCE_PCT 必须是非负数字" >&2; return 1; }
  awk -v v="${GSP_DURATION_TOLERANCE_PCT}" 'BEGIN { exit !(v < 100) }' || {
    echo "[ERROR] GSP_DURATION_TOLERANCE_PCT 必须小于 100" >&2
    return 1
  }
  [[ "${GSP_STOP_ON_SLA}" =~ ^[01]$ ]] || { echo "[ERROR] GSP_STOP_ON_SLA 必须是 0 或 1" >&2; return 1; }
  [[ "${GSP_INPUT_IDS}" =~ ^[01]$ ]] || { echo "[ERROR] GSP_INPUT_IDS 必须是 0 或 1" >&2; return 1; }
  (( GSP_PROMPTS_PER_GROUP >= 2 )) || { echo "[ERROR] GSP_PROMPTS_PER_GROUP 必须不小于 2" >&2; return 1; }
  [[ "${SEED}" =~ ^[0-9]+$ ]] || { echo "[ERROR] SEED 必须是非负整数" >&2; return 1; }
  [[ "${GSP_PAGE_ALIGN}" =~ ^[01]$ ]] || { echo "[ERROR] GSP_PAGE_ALIGN 必须是 0 或 1" >&2; return 1; }
  [[ "${GSP_SYSTEM_LEN_EXTRA_TOKENS}" =~ ^[0-9]+$ ]] || { echo "[ERROR] GSP_SYSTEM_LEN_EXTRA_TOKENS 必须是非负整数" >&2; return 1; }
  is_positive_decimal "${RATE_STEP}" || { echo "[ERROR] GSP_RATE_STEP 必须是正数" >&2; return 1; }
  awk -v v="${RATE_STEP}" 'BEGIN { exit !(v >= 0.1) }' || { echo "[ERROR] GSP_RATE_STEP 不能小于 0.1" >&2; return 1; }
  validate_rate_spacing || return 1
  is_positive_decimal "${HIT_RATE_TOLERANCE}" || { echo "[ERROR] GSP_HIT_RATE_TOLERANCE_PCT 必须是正数" >&2; return 1; }
  is_positive_decimal "${BASELINE_HIT_RATE_TOLERANCE}" || { echo "[ERROR] GSP_BASELINE_HIT_RATE_TOLERANCE_PCT 必须是正数" >&2; return 1; }
  is_positive_decimal "${FINITE_HIT_RATE_TOLERANCE}" || { echo "[ERROR] GSP_FINITE_HIT_RATE_TOLERANCE_PCT 必须是正数" >&2; return 1; }
  if [[ -n "${GSP_MAX_CONCURRENCY}" ]]; then
    [[ "${GSP_MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] GSP_MAX_CONCURRENCY 必须是正整数" >&2; return 1; }
  fi
  resolve_group_candidates || return 1
  resolve_prompt_candidates || return 1
}

validate_global_config || exit 1
resolve_page_size || exit 1
resolve_parallel_config || exit 1

print_case() {
  echo "CASE                 : ${CASE_ID}"
  echo "Input                : ${CASE_INPUT_RAW} = ${CASE_INPUT_LEN} tokens"
  echo "Output               : ${CASE_OUTPUT_RAW} = ${CASE_OUTPUT_LEN} tokens"
  echo "Target hit rate      : ${CASE_HIT_PCT}%"
  echo "Target theory (raw)  : ${CASE_REQUESTED_THEORY_HIT_PCT}%"
  echo "Theoretical hit rate : ${CASE_THEORY_HIT_PCT}% (page size ${GSP_PAGE_SIZE})"
  echo "System/question      : ${CASE_SYSTEM_LEN}/${CASE_QUESTION_LEN}"
  if (( CASE_NO_SHARED_PREFIX == 1 )); then
    echo "Workload             : random independent prompts (no shared prefix)"
  fi
  echo "Groups/prompts       : ${GSP_NUM_GROUPS}/${GSP_PROMPTS_PER_GROUP}"
  echo "Num prompts          : ${CASE_NUM_PROMPTS}"
  echo "TTFT                 : ${CASE_TTFT_METRIC} < ${CASE_TTFT_TARGETS} s"
  if [[ -n "${CASE_TPOT_METRIC}" ]]; then
    echo "TPOT                 : ${CASE_TPOT_METRIC} < ${CASE_TPOT_TARGET_MS} ms"
  else
    echo "TPOT                 : not constrained"
  fi
  echo "Max concurrency      : ${CASE_CONCURRENCY:-UNLIMITED}"
}

print_dry_run_case() {
  local request_rate="${1:-inf}"
  print_case
  echo "Official command:"
  printf '  %q' "${PYTHON_BIN}" -m sglang.bench_serving
  printf ' %q' --backend sglang --base-url "${BASE_URL}" --model "${MODEL:-<model>}"
  if (( CASE_NO_SHARED_PREFIX == 1 )); then
    printf ' %q' --dataset-name random-ids
    printf ' %q' --random-input-len "${CASE_INPUT_LEN}"
    printf ' %q' --random-output-len "${CASE_OUTPUT_LEN}"
    printf ' %q' --tokenize-prompt
    printf ' %q' --num-prompts "${CASE_NUM_PROMPTS}"
  else
    printf ' %q' --dataset-name generated-shared-prefix
    printf ' %q' --gsp-num-groups "${GSP_NUM_GROUPS}"
    printf ' %q' --gsp-prompts-per-group "${GSP_PROMPTS_PER_GROUP}"
    printf ' %q' --gsp-system-prompt-len "${CASE_SYSTEM_LEN}"
    printf ' %q' --gsp-question-len "${CASE_QUESTION_LEN}"
    printf ' %q' --gsp-output-len "${CASE_OUTPUT_LEN}"
    printf ' %q' --gsp-ordered --num-prompts "${CASE_NUM_PROMPTS}"
  fi
  printf ' %q' --flush-cache --warmup-requests 0 --seed "${SEED}" --cache-report
  if (( CASE_NO_SHARED_PREFIX == 0 )) && [[ "${GSP_INPUT_IDS}" == "1" ]]; then
    printf ' %q' --gsp-input-ids
  fi
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
    order_layout_candidates_for_case
    for candidate_prompts in "${PROMPT_CANDIDATES[@]}"; do
      if ! set_case_prompt_config "${candidate_prompts}"; then
        echo "[DRY-RUN] prompts/group=${candidate_prompts} 无法满足目标命中率，跳过"
        continue
      fi
      for candidate_groups in "${GROUP_CANDIDATES[@]}"; do
        GSP_NUM_GROUPS="${candidate_groups}"
        cap_initial_probe_groups
        CASE_NUM_PROMPTS=$((GSP_NUM_GROUPS * GSP_PROMPTS_PER_GROUP))
        print_dry_run_case inf
      done
    done
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
    random random-ids --random-input-len --random-output-len --tokenize-prompt \
    --gsp-num-groups --gsp-prompts-per-group --gsp-system-prompt-len \
    --gsp-question-len --gsp-output-len --gsp-ordered --flush-cache \
    --seed --cache-report --output-file; do
    if ! grep -Fq -- "${flag}" <<< "${help_text}"; then
      echo "[ERROR] 当前 bench_serving 缺少官方参数/数据集: ${flag}" >&2
      missing=1
    fi
  done
  if [[ "${GSP_INPUT_IDS}" == "1" ]] && ! grep -Fq -- "--gsp-input-ids" <<< "${help_text}"; then
    echo "[ERROR] 当前 bench_serving 缺少官方参数: --gsp-input-ids" >&2
    missing=1
  fi
  (( missing == 0 ))
}

parse_benchmark_log() {
  "${PYTHON_BIN}" "${script_dir}/gsp_prefix_metrics.py" parse-log "$1"
}
build_bench_command() {
  local request_rate="$1"
  cmd=(
    "${PYTHON_BIN}" -m sglang.bench_serving
    --backend sglang
    --base-url "${BASE_URL}"
    --model "${MODEL}"
    --flush-cache
    --warmup-requests 0
    --seed "${SEED}"
    --cache-report
    --request-rate "${request_rate}"
    --disable-tqdm
    --output-file /dev/null
  )
  if (( CASE_NO_SHARED_PREFIX == 1 )); then
    cmd+=(
      --dataset-name random-ids
      --random-input-len "${CASE_INPUT_LEN}"
      --random-output-len "${CASE_OUTPUT_LEN}"
      --tokenize-prompt
      --num-prompts "${CASE_NUM_PROMPTS}"
    )
  else
    cmd+=(
      --dataset-name generated-shared-prefix
      --gsp-num-groups "${GSP_NUM_GROUPS}"
      --gsp-prompts-per-group "${GSP_PROMPTS_PER_GROUP}"
      --gsp-system-prompt-len "${CASE_SYSTEM_LEN}"
      --gsp-question-len "${CASE_QUESTION_LEN}"
      --gsp-output-len "${CASE_OUTPUT_LEN}"
      --gsp-ordered
      --num-prompts "${CASE_NUM_PROMPTS}"
    )
  fi
  if (( CASE_NO_SHARED_PREFIX == 0 )) && [[ "${GSP_INPUT_IDS}" == "1" ]]; then
    cmd+=(--gsp-input-ids)
  fi
  if [[ -n "${CASE_CONCURRENCY:-}" ]]; then
    cmd+=(--max-concurrency "${CASE_CONCURRENCY}")
  fi
}

trial_log_path() {
  local request_rate="$1" phase="$2" rate_slug suffix=""
  rate_slug="$(sanitize_slug "${request_rate}")"
  [[ -n "${rate_slug}" ]] || rate_slug="zero"
  case "${phase}" in
    baseline|duration_baseline|descending) ;;
    *) suffix="__${phase}" ;;
  esac
  printf '%s/%s__g%s__p%s__rate_%s%s.log\n' \
    "${log_dir}" "${CASE_ID}" "${GSP_NUM_GROUPS}" "${GSP_PROMPTS_PER_GROUP}" "${rate_slug}" "${suffix}"
}

append_trial_row() {
  local request_rate="$1" phase="$2" log_file="$3" exit_code="$4" metrics_json="$5"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" "${script_dir}/gsp_prefix_metrics.py" append-trial \
    "${trials_csv}" "${CASE_ID}" "${CASE_INPUT_RAW}" "${CASE_OUTPUT_RAW}" \
    "${CASE_HIT_PCT}" "${CASE_SYSTEM_LEN}" "${CASE_QUESTION_LEN}" "${CASE_THEORY_HIT_PCT}" \
    "${CASE_REQUESTED_THEORY_HIT_PCT}" \
    "${GSP_PAGE_SIZE}" "${PARALLEL_CONFIG}" "${GSP_PAGE_ALIGN}" \
    "${GSP_NUM_GROUPS}" "${GSP_PROMPTS_PER_GROUP}" "${CASE_NUM_PROMPTS}" "${CASE_TTFT_METRIC}" \
    "${CASE_TTFT_TARGETS}" "${CASE_TPOT_METRIC}" "${CASE_TPOT_TARGET_MS}" "${request_rate}" \
    "${phase}" "${CASE_CONCURRENCY:-}" "${log_file}" "${exit_code}" "${metrics_json}"
}
run_trial() {
  local request_rate="$1" phase="$2"
  local log_file exit_file exit_code metrics_json
  log_file="$(trial_log_path "${request_rate}" "${phase}")"
  exit_file="${log_file}.exitcode"
  mkdir -p "${log_dir}"

  if [[ -s "${log_file}" && -f "${exit_file}" && "$(<"${exit_file}")" == "0" ]]; then
    echo "[SKIP] ${CASE_ID} request_rate=${request_rate}"
    metrics_json="$(parse_benchmark_log "${log_file}")"
    append_trial_row "${request_rate}" "${phase}" "${log_file}" 0 "${metrics_json}"
    refresh_summary_checkpoint
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

  write_log_header "${log_file}"
  "${cmd[@]}" 2>&1 | tee -a "${log_file}"
  exit_code="${PIPESTATUS[0]}"
  if [[ "${exit_code}" != "0" ]]; then
    run_failure=1
  fi
  printf '%s\n' "${exit_code}" > "${exit_file}"
  metrics_json="$(parse_benchmark_log "${log_file}")"
  append_trial_row "${request_rate}" "${phase}" "${log_file}" "${exit_code}" "${metrics_json}"
  refresh_summary_checkpoint
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

read_input_throughput() {
  local log_file="$1"
  parse_benchmark_log "${log_file}" | "${PYTHON_BIN}" -c '
import json,sys
try:
    data=json.load(sys.stdin)
    value=float(data["input_throughput"])
    if value <= 0: raise ValueError
except Exception:
    raise SystemExit(1)
print(f"{value:.6f}")
'
}

read_benchmark_duration() {
  local log_file="$1"
  parse_benchmark_log "${log_file}" | "${PYTHON_BIN}" -c '
import json,sys
try:
    data=json.load(sys.stdin)
    value=data.get("benchmark_duration_s", data.get("duration_s"))
    value=float(value)
    if value <= 0: raise ValueError
except Exception:
    raise SystemExit(1)
print(f"{value:.3f}")
'
}

trial_meets_sla() {
  local log_file="$1" target_s="$2"
  [[ -s "${log_file}" ]] || return 1
  [[ -f "${log_file}.exitcode" && "$(<"${log_file}.exitcode")" == "0" ]] || return 1
  parse_benchmark_log "${log_file}" | "${PYTHON_BIN}" -c '
import json,math,sys
try:
    data=json.load(sys.stdin)
    ttft_metric, target_s, tpot_metric, tpot_target, hit_target, hit_tolerance = sys.argv[1:]
    ttft=float(data[ttft_metric])
    target_ms=float(target_s)*1000.0
    if not math.isfinite(ttft) or ttft > target_ms:
        raise SystemExit(1)
    if tpot_metric:
        tpot=float(data[tpot_metric])
        if not math.isfinite(tpot) or tpot > float(tpot_target):
            raise SystemExit(1)
    actual_hit=float(data["cache_hit_rate_pct"])
    if not math.isfinite(actual_hit) or abs(actual_hit - float(hit_target)) > float(hit_tolerance):
        raise SystemExit(1)
except (KeyError, TypeError, ValueError, json.JSONDecodeError):
    raise SystemExit(1)
print("1")
' "${CASE_TTFT_METRIC}" "${target_s}" "${CASE_TPOT_METRIC:-}" "${CASE_TPOT_TARGET_MS:-}" \
    "${CASE_HIT_PCT}" "${FINITE_HIT_RATE_TOLERANCE}" 2>/dev/null | grep -qx '1'
}

read_actual_hit_rate() {
  local log_file="$1"
  parse_benchmark_log "${log_file}" | "${PYTHON_BIN}" -c '
import json,sys
try:
    data=json.load(sys.stdin)
    value=float(data["cache_hit_rate_pct"])
    if value < 0 or value > 100: raise ValueError
except Exception:
    raise SystemExit(1)
print(f"{value:.4f}")
'
}

baseline_hit_rate_passes() {
  local log_file="$1"
  local measured
  measured="$(read_actual_hit_rate "${log_file}" 2>/dev/null || true)"
  [[ -n "${measured}" ]] || return 1
  awk -v actual="${measured}" -v target="${CASE_HIT_PCT}" -v tolerance="${BASELINE_HIT_RATE_TOLERANCE}" \
    'BEGIN { exit !(actual >= target - tolerance && actual <= target + tolerance) }'
}

adjust_case_prompt_config_from_hit() {
  local measured="$1" lengths
  lengths="$("${PYTHON_BIN}" - "${CASE_SYSTEM_LEN}" "${measured}" "${CASE_HIT_PCT}" \
    "${CASE_INPUT_LEN}" "${GSP_PAGE_SIZE}" "${GSP_PAGE_ALIGN}" \
    "${GSP_PROMPTS_PER_GROUP}" "${GSP_CACHE_MISS_REQUESTS_PER_GROUP}" \
    "${GSP_CALIBRATION_STEP_RATIO}" <<'PY' | tr -d '\r'
import math
import sys

current, actual, target, total, page, align, prompts, misses, step_ratio = sys.argv[1:]
current = int(current)
actual = float(actual)
target = float(target)
total = int(total)
page = int(page)
align = align == "1"
prompts = int(prompts)
misses = int(misses)
step_ratio = float(step_ratio)
if actual <= 0 or current <= 0 or current >= total:
    raise SystemExit("invalid measured hit-rate correction inputs")

raw = current * target / actual
corrected_raw = current + step_ratio * (raw - current)
if align and page > 1:
    corrected = int(math.floor(corrected_raw / page + 0.5)) * page
    if corrected == current:
        corrected += page if target > actual else -page
else:
    corrected = int(math.floor(corrected_raw + 0.5))
corrected = max(1, min(total - 1, corrected))
if corrected == current:
    raise SystemExit("page alignment cannot produce a different correction")
question = total - corrected
effective_misses = min(max(misses, 0), prompts - 1)
theory = 100.0 * (prompts - effective_misses) * corrected / (prompts * total)
print(corrected, question, f"{theory:.6f}", f"{theory:.6f}")
PY
  )" || return 1
  read -r CASE_SYSTEM_LEN CASE_QUESTION_LEN CASE_THEORY_HIT_PCT CASE_REQUESTED_THEORY_HIT_PCT <<< "${lengths}"
  echo "[CALIBRATE] 实测命中率=${measured}%，修正 system/question=${CASE_SYSTEM_LEN}/${CASE_QUESTION_LEN}，理论命中率=${CASE_THEORY_HIT_PCT}%" >&2
}

run_inf_with_auto_calibration() {
  local base_phase="$1" attempt trial_phase log_file measured
  local inf_trial=0 no_improvement=0 input_throughput best_input_throughput=""
  local best_log=""
  LAST_INF_LOG=""
  while (( inf_trial < GSP_INF_MAX_TRIALS )); do
    attempt=0
    if (( inf_trial == 0 )); then
      trial_phase="${base_phase}"
    else
      trial_phase="${base_phase}_throughput${inf_trial}"
    fi
    while :; do
      if (( attempt == 0 )); then
        run_trial inf "${trial_phase}"
      else
        run_trial inf "${trial_phase}_calibration${attempt}"
      fi
      if (( attempt == 0 )); then
        log_file="$(trial_log_path inf "${trial_phase}")"
      else
        log_file="$(trial_log_path inf "${trial_phase}_calibration${attempt}")"
      fi
      LAST_INF_LOG="${log_file}"
      if [[ -s "${log_file}" ]] && baseline_hit_rate_passes "${log_file}"; then
        break
      fi
      (( GSP_AUTO_CALIBRATE == 1 && attempt < GSP_CALIBRATION_MAX_ITERS )) || return 1
      measured="$(read_actual_hit_rate "${log_file}" 2>/dev/null || true)"
      [[ -n "${measured}" ]] || return 1
      adjust_case_prompt_config_from_hit "${measured}" || return 1
      ((attempt++))
      print_case
    done

    input_throughput="$(read_input_throughput "${log_file}" 2>/dev/null || true)"
    [[ -n "${input_throughput}" ]] || {
      echo "[ERROR] inf 缺少 input throughput，无法判断是否稳定: ${log_file}" >&2
      return 1
    }
    if [[ -z "${best_input_throughput}" ]] || awk -v current="${input_throughput}" -v best="${best_input_throughput}" 'BEGIN { exit !(current > best + 1e-9) }'; then
      best_input_throughput="${input_throughput}"
      best_log="${log_file}"
      no_improvement=0
      echo "[INF] ${base_phase} trial=$((inf_trial + 1)) input_throughput=${input_throughput} (new best)"
    else
      ((no_improvement++))
      echo "[INF] ${base_phase} trial=$((inf_trial + 1)) input_throughput=${input_throughput} best=${best_input_throughput} no_improvement=${no_improvement}/${GSP_INF_NO_IMPROVEMENT_LIMIT}"
    fi
    ((inf_trial++))
    if (( no_improvement >= GSP_INF_NO_IMPROVEMENT_LIMIT )); then
      break
    fi
  done
  [[ -n "${best_log}" ]] || return 1
  LAST_INF_LOG="${best_log}"
  echo "[INF] ${base_phase} selected input_throughput=${best_input_throughput} log=${best_log}"
  return 0
}

resize_case_for_duration() {
  local baseline_rps="$1" requested_duration="${2:-${GSP_TARGET_DURATION_S}}" target_prompts target_groups
  [[ "${GSP_AUTO_DURATION}" == "1" ]] || return 1
  [[ "${GSP_NUM_GROUPS_WAS_SET}" == "x" ]] && return 1
  target_prompts="$(awk -v rps="${baseline_rps}" -v duration="${requested_duration}" \
    -v prompts="${GSP_PROMPTS_PER_GROUP}" \
    'BEGIN { n = int(rps * duration + 0.999999); n = int((n + prompts - 1) / prompts) * prompts; print n }')"
  target_groups=$((target_prompts / GSP_PROMPTS_PER_GROUP))
  (( target_groups >= 1 )) || target_groups=1
  GSP_NUM_GROUPS="${target_groups}"
  CASE_NUM_PROMPTS=$((GSP_NUM_GROUPS * GSP_PROMPTS_PER_GROUP))
  echo "[INFO] 按实际 RPS=${baseline_rps} 和目标时长=${requested_duration}s 调整 case: groups=${GSP_NUM_GROUPS} prompts/group=${GSP_PROMPTS_PER_GROUP} num_prompts=${CASE_NUM_PROMPTS}"
  return 0
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
  # Use a uniform descending grid.  The previous cumulative offset list
  # skipped valid points such as 0.95 when the baseline was 1.55.
  awk -v start="${baseline}" -v step="${RATE_STEP}" '
    BEGIN {
      rate = start
      while (rate >= 0.1 - 1e-9) {
        printf "%.2f\n", rate
        next_rate = rate - step
        if (next_rate < 0.1 - 1e-9) {
          break
        }
        rate = next_rate
      }
    }
  '
}

write_summary() {
  "${PYTHON_BIN}" "${script_dir}/gsp_prefix_summary.py" summary \
    "${trials_csv}" "${summary_tsv}" \
    "${BASELINE_HIT_RATE_TOLERANCE}" "${TOKEN_UNIT}"
}
refresh_summary_checkpoint() {
  [[ -s "${trials_csv}" ]] || return 0
  write_summary || echo "[WARN] 无法刷新 summary checkpoint: ${summary_tsv}" >&2
}

handle_benchmark_signal() {
  local exit_code="$1"
  refresh_summary_checkpoint
  trap - INT TERM
  exit "${exit_code}"
}

trap 'handle_benchmark_signal 130' INT
trap 'handle_benchmark_signal 143' TERM

prune_confirmed_trials() {
  "${PYTHON_BIN}" "${script_dir}/gsp_prefix_summary.py" prune \
    "${trials_csv}" "${summary_tsv}"
}
plot_summary() {
  [[ "${plot_requested}" == "1" ]] || return 0
  "${PYTHON_PLOT_BIN}" "${script_dir}/gsp_prefix_plot.py" \
    "${summary_tsv}" "${trials_csv}" "${out_dir}/gsp_prefix_matrix.png" \
    "${plot_cell}" "${FINITE_HIT_RATE_TOLERANCE}"
}

mkdir -p "${out_dir}" "${log_dir}"
check_bench_flags || exit 1
write_run_metadata

for raw_case in "${CASES[@]}"; do
  parse_case "${raw_case}" || exit 1
  order_layout_candidates_for_case
  baseline_rps=""
  baseline_passed=0
  selected_groups=""
  for candidate_prompts in "${PROMPT_CANDIDATES[@]}"; do
    if ! set_case_prompt_config "${candidate_prompts}"; then
      echo "[INFO] 当前 prompts/group=${candidate_prompts} 无法在输入长度内满足目标命中率，继续尝试下一候选" >&2
      continue
    fi
    for candidate_groups in "${GROUP_CANDIDATES[@]}"; do
      GSP_NUM_GROUPS="${candidate_groups}"
      cap_initial_probe_groups
      CASE_NUM_PROMPTS=$((GSP_NUM_GROUPS * GSP_PROMPTS_PER_GROUP))
      print_case
      if ! run_inf_with_auto_calibration baseline; then
        measured_hit="$(read_actual_hit_rate "${LAST_INF_LOG}" 2>/dev/null || printf 'unavailable')"
        echo "[WARN] inf cache 命中率校准未通过: ${CASE_ID} groups=${GSP_NUM_GROUPS} prompts=${GSP_PROMPTS_PER_GROUP} actual=${measured_hit}% target=${CASE_HIT_PCT}%" >&2
        continue
      fi
      inf_log="${LAST_INF_LOG}"
      baseline_rps="$(read_actual_rps "${inf_log}" 2>/dev/null || true)"
      if [[ -z "${baseline_rps}" ]]; then
        echo "[WARN] 无法从 inf baseline 读取实际 RPS，继续尝试下一组数: ${CASE_ID} groups=${GSP_NUM_GROUPS} prompts=${GSP_PROMPTS_PER_GROUP}" >&2
        continue
      fi
      baseline_passed=1
      selected_groups="${GSP_NUM_GROUPS}"
      echo "[MEET] inf cache 命中率通过: ${CASE_ID} groups=${selected_groups} prompts=${GSP_PROMPTS_PER_GROUP} actual_rps=${baseline_rps}"
      break 2
    done
  done
  if (( baseline_passed == 0 )); then
    echo "[ERROR] 所有 prompts/groups 候选的 inf cache 命中率都未达标: ${CASE_ID}" >&2
    run_failure=1
    continue
  fi

  if [[ "${GSP_AUTO_DURATION}" == "1" && "${GSP_NUM_GROUPS_WAS_SET}" != "x" ]]; then
    duration_pass=0
    if resize_case_for_duration "${baseline_rps}" "${GSP_DURATION_FIRST_PASS_S}"; then
      cap_initial_probe_groups
      CASE_NUM_PROMPTS=$((GSP_NUM_GROUPS * GSP_PROMPTS_PER_GROUP))
      while :; do
        duration_phase="duration_baseline"
        (( duration_pass > 0 )) && duration_phase="duration_adjust${duration_pass}"
        print_case
        if ! run_inf_with_auto_calibration "${duration_phase}"; then
          measured_hit="$(read_actual_hit_rate "${LAST_INF_LOG}" 2>/dev/null || printf 'unavailable')"
          echo "[ERROR] 动态请求数 baseline 命中率不达标: ${CASE_ID} groups=${GSP_NUM_GROUPS} actual=${measured_hit}% target=${CASE_HIT_PCT}%" >&2
          run_failure=1
          continue 2
        fi
        inf_log="${LAST_INF_LOG}"
        baseline_rps="$(read_actual_rps "${inf_log}" 2>/dev/null || true)"
        actual_duration="$(read_benchmark_duration "${inf_log}" 2>/dev/null || true)"
        [[ -n "${baseline_rps}" ]] || {
          echo "[ERROR] 动态请求数 baseline 无法读取实际 RPS: ${CASE_ID}" >&2
          run_failure=1
          continue 2
        }
        selected_groups="${GSP_NUM_GROUPS}"
        echo "[MEET] 动态请求数 baseline 通过: ${CASE_ID} groups=${selected_groups} num_prompts=${CASE_NUM_PROMPTS} actual_rps=${baseline_rps} duration=${actual_duration:-unknown}s"

        [[ -n "${actual_duration}" ]] || break
        within_target="$(awk -v actual="${actual_duration}" -v target="${GSP_TARGET_DURATION_S}" -v tolerance="${GSP_DURATION_TOLERANCE_PCT}" \
          'BEGIN { low = target * (1 - tolerance / 100); high = target * (1 + tolerance / 100); print (actual >= low && actual <= high) ? 1 : 0 }')"
        [[ "${within_target}" == "1" ]] && break
        if (( duration_pass >= GSP_DURATION_ADJUST_MAX_ITERS )); then
          echo "[WARN] ${CASE_ID} 动态时长在 ${GSP_DURATION_ADJUST_MAX_ITERS} 次调整后仍为 ${actual_duration}s，目标=${GSP_TARGET_DURATION_S}s" >&2
          break
        fi
        previous_groups="${GSP_NUM_GROUPS}"
        resize_case_for_duration "${baseline_rps}" "${GSP_TARGET_DURATION_S}" || break
        (( duration_pass++ ))
        [[ "${GSP_NUM_GROUPS}" != "${previous_groups}" ]] || {
          echo "[INFO] ${CASE_ID} 按 prompts/group 取整后请求数不再变化，停止动态时长调整"
          break
        }
      done
    fi
  fi

  IFS='|' read -ra sla_targets <<< "${CASE_TTFT_TARGETS}"
  sla_target_met=()
  for sla_index in "${!sla_targets[@]}"; do
    sla_target_met[${sla_index}]=0
  done
  sla_targets_remaining="${#sla_targets[@]}"
  while read -r finite_rate; do
    [[ -n "${finite_rate}" ]] || continue
    run_trial "${finite_rate}" descending
    finite_log="$(trial_log_path "${finite_rate}" descending)"
    if [[ "${GSP_STOP_ON_SLA}" == "1" && "${sla_targets_remaining}" -gt 0 ]]; then
      for sla_index in "${!sla_targets[@]}"; do
        [[ "${sla_target_met[$sla_index]}" == "1" ]] && continue
        if trial_meets_sla "${finite_log}" "${sla_targets[$sla_index]}"; then
          sla_target_met[${sla_index}]=1
          (( sla_targets_remaining-- ))
          echo "[MEET] ${CASE_ID} request_rate=${finite_rate} actual_rps=$(read_actual_rps "${finite_log}" 2>/dev/null || printf 'unknown') ${CASE_TTFT_METRIC}<${sla_targets[$sla_index]}s"
        fi
      done
      if (( sla_targets_remaining == 0 )); then
        echo "[STOP] ${CASE_ID} 所有 TTFT 目标均已找到满足点，停止继续降低请求率"
        break
      fi
    fi
  done < <(finite_rate_candidates "${baseline_rps}")
done

write_summary
if ! plot_summary; then
  echo "[WARN] Prefill/GSP 容量矩阵未生成；不影响 CSV/TSV 和 benchmark 结果" >&2
fi
prune_confirmed_trials

echo
echo "============================================================"
echo "GSP prefix benchmark finished"
echo "Result directory : ${out_dir}"
echo "Trials CSV       : ${trials_csv}"
echo "Summary TSV      : ${summary_tsv}  (cat 后可直接复制到表格)"
if [[ "${plot_requested}" == "1" && -f "${out_dir}/gsp_prefix_matrix.png" ]]; then
  echo "Plot             : ${out_dir}/gsp_prefix_matrix.png"
fi
echo "============================================================"
if (( run_failure != 0 )); then
  echo "[ERROR] 至少一个官方 benchmark trial 失败，详见日志和 exitcode 文件" >&2
  exit 1
fi
