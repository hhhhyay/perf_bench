#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
metrics_py="$script_dir/vllm_bench_metrics.py"
plot_py="$script_dir/vllm_prefill_plot.py"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
fi

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-}"
TOKENIZER="${TOKENIZER:-}"
TOKEN_UNIT="${TOKEN_UNIT:-1024}"
PREFILL_TARGET_DURATION_S="${PREFILL_TARGET_DURATION_S:-60}"
PREFILL_RATE_STEP="${PREFILL_RATE_STEP:-0.1}"
PREFILL_INF_MAX_TRIALS="${PREFILL_INF_MAX_TRIALS:-5}"
PREFILL_INF_NO_IMPROVEMENT_LIMIT="${PREFILL_INF_NO_IMPROVEMENT_LIMIT:-2}"
PREFILL_DURATION_ADJUST_MAX_ITERS="${PREFILL_DURATION_ADJUST_MAX_ITERS:-2}"
PREFILL_INITIAL_NUM_PROMPTS="${PREFILL_INITIAL_NUM_PROMPTS:-8}"
PREFILL_NUM_PROMPTS="${PREFILL_NUM_PROMPTS:-}"
PREFILL_WARMUP_REQUESTS="${PREFILL_WARMUP_REQUESTS:-0}"
FIXED_CONCURRENCY="${FIXED_CONCURRENCY:-}"
PREFILL_PLOT="${PREFILL_PLOT:-1}"
PREFILL_PLOT_CELL="${PREFILL_PLOT_CELL:-rps}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
NODE_HOSTNAME="${NODE_HOSTNAME:-$(hostname 2>/dev/null || printf unknown)}"
RUN_STARTED_AT="${RUN_STARTED_AT:-$(date +%Y-%m-%dT%H:%M:%S%z)}"
SCRIPT_COMMIT="$(git -C "$repo_root" rev-parse --short HEAD 2>/dev/null || true)"
PARALLEL_CONFIG="${PARALLEL_CONFIG:-}"

out_dir="./all_result/bench_vllm_prefill_$RUN_ID"
log_dir="$out_dir/logs"
trials_csv="$out_dir/vllm_prefill_trials.csv"
summary_tsv="$out_dir/vllm_prefill_summary.tsv"
cases_json="$out_dir/cases.json"
mkdir -p "$log_dir"

DRY_RUN=0
PLOT_REQUESTED="$PREFILL_PLOT"
CASE_SPECS=()
CASE_JSONS=()
LAST_LOG_FILE=""
LAST_EXIT_CODE=1
LAST_NUM_PROMPTS=0
BEST_INF_RPS=""

usage() {
  cat <<USAGE
Usage:
  BASE_URL=http://127.0.0.1:8000 MODEL=/path/to/model bash vllm/vllm_prefill_bench.sh \
    --case '4k,mean_ttft_ms,3|4' --case '64k,p99_ttft_ms,5|6'

Case format: INPUT,TTFT_METRIC,TTFT_TARGETS_S
Output length is fixed to 1. Default TOKEN_UNIT is 1024; use TOKEN_UNIT=1000 for decimal K.
USAGE
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

json_get() {
  local json="$1"
  local key="$2"
  printf "%s\n" "$json" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get(sys.argv[1], ""))' "$key"
}

validate_config() {
  [[ "$TOKEN_UNIT" =~ ^[1-9][0-9]*$ ]] || die "TOKEN_UNIT 必须是正整数"
  [[ "$PREFILL_TARGET_DURATION_S" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "PREFILL_TARGET_DURATION_S 必须是正数"
  [[ "$PREFILL_INF_MAX_TRIALS" =~ ^[1-9][0-9]*$ ]] || die "PREFILL_INF_MAX_TRIALS 必须是正整数"
  [[ "$PREFILL_INF_NO_IMPROVEMENT_LIMIT" =~ ^[1-9][0-9]*$ ]] || die "PREFILL_INF_NO_IMPROVEMENT_LIMIT 必须是正整数"
  [[ "$PREFILL_DURATION_ADJUST_MAX_ITERS" =~ ^[0-9]+$ ]] || die "PREFILL_DURATION_ADJUST_MAX_ITERS 必须是非负整数"
  [[ "$PREFILL_INITIAL_NUM_PROMPTS" =~ ^[1-9][0-9]*$ ]] || die "PREFILL_INITIAL_NUM_PROMPTS 必须是正整数"
  [[ "$PREFILL_WARMUP_REQUESTS" =~ ^[0-9]+$ ]] || die "PREFILL_WARMUP_REQUESTS 必须是非负整数"
  [[ "$PREFILL_PLOT" =~ ^[01]$ ]] || die "PREFILL_PLOT 必须是 0 或 1"
  [[ "$PREFILL_PLOT_CELL" == "rps" || "$PREFILL_PLOT_CELL" == "input_throughput" ]] || die "PREFILL_PLOT_CELL 只能是 rps 或 input_throughput"
  [[ "$PREFILL_RATE_STEP" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "PREFILL_RATE_STEP 必须是数字"
  "$PYTHON_BIN" "$metrics_py" rate-list 1 "$PREFILL_RATE_STEP" >/dev/null 2>&1 || die "PREFILL_RATE_STEP 不能小于 0.1"
  if [[ -n "$PREFILL_NUM_PROMPTS" ]]; then
    [[ "$PREFILL_NUM_PROMPTS" =~ ^[1-9][0-9]*$ ]] || die "PREFILL_NUM_PROMPTS 必须是正整数"
  fi
  if [[ -n "$FIXED_CONCURRENCY" ]]; then
    [[ "$FIXED_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || die "FIXED_CONCURRENCY 必须是正整数"
  fi
  [[ -n "$MODEL" ]] || die "MODEL 不能为空"
  [[ ${#CASE_SPECS[@]} -gt 0 ]] || die "至少需要一个 --case"
}

load_parallel_config() {
  [[ -n "$PARALLEL_CONFIG" || "$DRY_RUN" == 1 ]] && { PARALLEL_CONFIG="${PARALLEL_CONFIG:-UNKNOWN}"; return; }
  local info detected
  info="$(curl -fsS --connect-timeout 3 --max-time 5 "$BASE_URL/server_info" 2>/dev/null || true)"
  if [[ -n "$info" ]]; then
    detected="$("$PYTHON_BIN" - "$info" <<PY
import json
import re
import sys
try:
    data = json.loads(sys.argv[1])
except Exception:
    data = {}
config = data.get("vllm_config", "")
if not isinstance(config, str):
    config = str(config)
parts = []
for name, prefix in (("tensor_parallel_size", "TP"), ("data_parallel_size", "DP"), ("pipeline_parallel_size", "PP"), ("expert_parallel_size", "EP")):
    match = re.search(r"\\b" + name + r"=([0-9]+)", config)
    if match:
        parts.append(prefix + match.group(1))
print("-".join(parts))
PY
    )"
    [[ -n "$detected" ]] && PARALLEL_CONFIG="$detected"
  fi
  PARALLEL_CONFIG="${PARALLEL_CONFIG:-UNKNOWN}"
}

write_run_metadata() {
  "$PYTHON_BIN" - "$out_dir/run_metadata.json" "$RUN_STARTED_AT" "$MODEL" "$TOKENIZER" "$NODE_HOSTNAME" "$BASE_URL" "$RUN_ID" "$SCRIPT_COMMIT" "$PARALLEL_CONFIG" <<PY
import json
import os
import sys
path, started, model, tokenizer, node, base_url, run_id, commit, parallel = sys.argv[1:]
payload = {
    "started_at": started,
    "model": model,
    "tokenizer": tokenizer,
    "node_hostname": node,
    "base_url": base_url,
    "run_id": run_id,
    "script_commit": commit,
    "parallel_config": parallel,
    "environment": {k: v for k, v in os.environ.items() if k.startswith(("PREFILL_", "TOKEN_", "FIXED_CONCURRENCY"))},
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
}

write_log_header() {
  local log_file="$1"
  {
    printf "# perf_bench run metadata: %s\n" "$out_dir/run_metadata.json"
    printf "# model: %s\n" "$MODEL"
    printf "# tokenizer: %s\n" "${TOKENIZER:-<client default>}"
    printf "# node_hostname: %s\n" "$NODE_HOSTNAME"
    printf "# base_url: %s\n" "$BASE_URL"
    printf "# run_id: %s\n" "$RUN_ID"
    printf "# started_at: %s\n#\n" "$RUN_STARTED_AT"
  } > "$log_file"
}

write_cases_file() {
  local lines_file="$out_dir/cases.lines"
  printf "%s\n" "${CASE_JSONS[@]}" > "$lines_file"
  "$PYTHON_BIN" - "$lines_file" "$cases_json" <<PY
import json
import sys
lines_path, output_path = sys.argv[1:]
with open(lines_path, encoding="utf-8") as handle:
    payload = [json.loads(line) for line in handle if line.strip()]
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
PY
}

refresh_summary() {
  [[ -f "$cases_json" ]] || return 0
  "$PYTHON_BIN" "$metrics_py" summary "$trials_csv" "$summary_tsv" "$cases_json" >/dev/null || echo "[WARN] summary 刷新失败" >&2
}

build_bench_args() {
  local input_tokens="$1"
  local rate="$2"
  local num_prompts="$3"
  BENCH_ARGS=(
    vllm bench serve
    --backend openai
    --base-url "$BASE_URL"
    --model "$MODEL"
    --dataset-name random
    --random-input-len "$input_tokens"
    --random-output-len 1
    --random-range-ratio 0.0
    --num-prompts "$num_prompts"
    --request-rate "$rate"
    --num-warmups "$PREFILL_WARMUP_REQUESTS"
    --ignore-eos
    --percentile-metrics ttft,tpot,itl,e2el
    --metric-percentiles 95,99
    --disable-tqdm
  )
  [[ -n "$TOKENIZER" ]] && BENCH_ARGS+=(--tokenizer "$TOKENIZER")
  [[ -n "$FIXED_CONCURRENCY" ]] && BENCH_ARGS+=(--max-concurrency "$FIXED_CONCURRENCY")
}

print_bench_command() {
  printf "[DRY-RUN]"
  printf " %q" "${BENCH_ARGS[@]}"
  printf "\n"
}

append_trial() {
  local case_json="$1"
  local log_file="$2"
  local rate="$3"
  local num_prompts="$4"
  local phase="$5"
  "$PYTHON_BIN" "$metrics_py" append-trial "$trials_csv" "$log_file" "$case_json" "$rate" "$num_prompts" "$FIXED_CONCURRENCY" "$PARALLEL_CONFIG" "$LAST_EXIT_CODE" "$phase" || echo "[WARN] trial 记录失败: $log_file" >&2
}

run_trial() {
  local case_json="$1"
  local rate="$2"
  local num_prompts="$3"
  local phase="$4"
  local case_id input_tokens rate_tag log_file exit_file start_epoch end_epoch elapsed
  case_id="$(json_get "$case_json" case_id)"
  input_tokens="$(json_get "$case_json" input_tokens)"
  rate_tag="${rate//./_}"
  log_file="$log_dir/${case_id}_${phase}_rate_${rate_tag}.log"
  exit_file="$log_file.exitcode"
  LAST_LOG_FILE="$log_file"
  LAST_EXIT_CODE=1
  build_bench_args "$input_tokens" "$rate" "$num_prompts"

  if [[ -s "$log_file" && -f "$exit_file" ]]; then
    LAST_EXIT_CODE="$(tr -d "[:space:]" < "$exit_file")"
    echo "[SKIP] $case_id rate=$rate phase=$phase exit=$LAST_EXIT_CODE" >&2
  elif [[ "$DRY_RUN" == 1 ]]; then
    print_bench_command
    return 0
  else
    start_epoch="$(date +%s)"
    write_log_header "$log_file"
    echo "[CMD] $(printf "%q " "${BENCH_ARGS[@]}")" | tee -a "$log_file"
    set +e
    "${BENCH_ARGS[@]}" 2>&1 | tee -a "$log_file"
    LAST_EXIT_CODE="${PIPESTATUS[0]}"
    set -e
    printf "%s\n" "$LAST_EXIT_CODE" > "$exit_file"
    end_epoch="$(date +%s)"
    elapsed=$((end_epoch - start_epoch))
    printf "start_epoch=%s\nend_epoch=%s\nelapsed_s=%s\n" "$start_epoch" "$end_epoch" "$elapsed" > "$log_file.time"
    echo "[DONE] $case_id rate=$rate phase=$phase exit=$LAST_EXIT_CODE duration=${elapsed}s" >&2
  fi
  append_trial "$case_json" "$log_file" "$rate" "$num_prompts" "$phase"
  refresh_summary
}

compute_prompt_count() {
  "$PYTHON_BIN" - "$1" "$2" "$3" <<PY
import math
import sys
num, target, duration = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
if duration <= 0:
    print(num)
else:
    print(max(1, math.ceil(num * target / duration)))
PY
}

duration_in_target() {
  "$PYTHON_BIN" - "$1" "$2" <<PY
import sys
duration, target = float(sys.argv[1]), float(sys.argv[2])
raise SystemExit(0 if target * 0.75 <= duration <= target * 1.25 else 1)
PY
}

metric_from_log() {
  local log_file="$1"
  local key="$2"
  local parsed
  [[ -f "$log_file" ]] || return 0
  parsed="$("$PYTHON_BIN" "$metrics_py" parse-log "$log_file")"
  json_get "$parsed" "$key"
}

prepare_case_prompts() {
  local case_json="$1"
  local num="${PREFILL_NUM_PROMPTS:-$PREFILL_INITIAL_NUM_PROMPTS}"
  LAST_NUM_PROMPTS="$num"
  [[ -n "$PREFILL_NUM_PROMPTS" || "$DRY_RUN" == 1 ]] && return 0
  local pass duration next
  for ((pass=1; pass<=PREFILL_DURATION_ADJUST_MAX_ITERS; pass++)); do
    run_trial "$case_json" inf "$num" "inf_pilot_$pass"
    [[ "$LAST_EXIT_CODE" == 0 ]] || break
    duration="$(metric_from_log "$LAST_LOG_FILE" duration_s)"
    [[ "$duration" =~ ^[0-9]+([.][0-9]+)?$ ]] || break
    if duration_in_target "$duration" "$PREFILL_TARGET_DURATION_S"; then
      break
    fi
    next="$(compute_prompt_count "$num" "$PREFILL_TARGET_DURATION_S" "$duration")"
    [[ "$next" == "$num" ]] && break
    num="$next"
    LAST_NUM_PROMPTS="$num"
  done
}

run_inf_stability() {
  local case_json="$1"
  local num="$2"
  local trial input_tp rps best_input="-inf" no_improvement=0
  BEST_INF_RPS=""
  for ((trial=1; trial<=PREFILL_INF_MAX_TRIALS; trial++)); do
    run_trial "$case_json" inf "$num" "inf_trial_$trial"
    [[ "$LAST_EXIT_CODE" == 0 ]] || continue
    input_tp="$(metric_from_log "$LAST_LOG_FILE" input_throughput_tok_s)"
    rps="$(metric_from_log "$LAST_LOG_FILE" rps)"
    [[ "$input_tp" =~ ^[0-9]+([.][0-9]+)?$ && "$rps" =~ ^[0-9]+([.][0-9]+)?$ ]] || continue
    if "$PYTHON_BIN" - "$input_tp" "$best_input" <<PY
import math
import sys
value, best = float(sys.argv[1]), float(sys.argv[2])
raise SystemExit(0 if math.isfinite(value) and value > best else 1)
PY
    then
      best_input="$input_tp"
      BEST_INF_RPS="$rps"
      no_improvement=0
    else
      no_improvement=$((no_improvement + 1))
    fi
    (( no_improvement >= PREFILL_INF_NO_IMPROVEMENT_LIMIT )) && break
  done
  [[ -n "$BEST_INF_RPS" ]] || return 1
}

run_case() {
  local case_json="$1"
  local num_prompts rate rates_json met
  local -a rates
  prepare_case_prompts "$case_json"
  num_prompts="$LAST_NUM_PROMPTS"
  run_inf_stability "$case_json" "$num_prompts" || { echo "[ERROR] inf 基线失败: $(json_get "$case_json" case_id)" >&2; return 1; }
  echo "[INFO] inf baseline actual_rps=$BEST_INF_RPS case=$(json_get "$case_json" case_id)"
  rates_json="$("$PYTHON_BIN" "$metrics_py" rate-list "$BEST_INF_RPS" "$PREFILL_RATE_STEP")"
  while IFS= read -r rate; do
    [[ -n "$rate" ]] || continue
    run_trial "$case_json" "$rate" "$num_prompts" descending
    met="$("$PYTHON_BIN" "$metrics_py" targets-met "$trials_csv" "$case_json")"
    if [[ "$met" == 1 ]]; then
      echo "[INFO] 所有 TTFT SLA 已找到满足点，停止继续降速"
      break
    fi
  done < <(printf "%s\n" "$rates_json" | "$PYTHON_BIN" -c 'import json,sys; print("\n".join(format(float(x), ".2f") for x in json.load(sys.stdin)))')
}

dry_run_case() {
  local case_json="$1"
  local input_tokens
  input_tokens="$(json_get "$case_json" input_tokens)"
  build_bench_args "$input_tokens" inf "${PREFILL_NUM_PROMPTS:-$PREFILL_INITIAL_NUM_PROMPTS}"
  print_bench_command
}

refresh_and_plot() {
  refresh_summary
  [[ "$PLOT_REQUESTED" == 1 && "$DRY_RUN" == 0 ]] || return 0
  if [[ ! -f "$plot_py" ]]; then
    echo "[WARN] 缺少作图模块，跳过图片" >&2
    return 0
  fi
  "$PYTHON_BIN" "$plot_py" --summary "$summary_tsv" --output "$out_dir/vllm_prefill_matrix.png" --cell "$PREFILL_PLOT_CELL" || echo "[WARN] 作图失败，不影响 CSV/TSV" >&2
}

on_exit() {
  local code=$?
  trap - EXIT
  refresh_summary
  echo "============================================================"
  echo "vLLM Prefill benchmark finished"
  echo "Result directory : $out_dir"
  echo "Trials CSV       : $trials_csv"
  echo "Summary TSV      : $summary_tsv"
  [[ -f "$out_dir/vllm_prefill_matrix.png" ]] && echo "Plot             : $out_dir/vllm_prefill_matrix.png"
  echo "============================================================"
  exit "$code"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case)
      [[ $# -ge 2 ]] || die "--case 缺少参数"
      CASE_SPECS+=("$2")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --plot)
      PLOT_REQUESTED=1
      shift
      ;;
    --no-plot)
      PLOT_REQUESTED=0
      shift
      ;;
    --cell)
      [[ $# -ge 2 ]] || die "--cell 缺少参数"
      PREFILL_PLOT_CELL="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "未知参数: $1"
      ;;
  esac
done

if [[ ${#CASE_SPECS[@]} -eq 0 && -n "${VLLM_PREFILL_CASES:-}" ]]; then
  IFS=";" read -ra CASE_SPECS <<< "$VLLM_PREFILL_CASES"
fi
validate_config

for spec in "${CASE_SPECS[@]}"; do
  case_json="$("$PYTHON_BIN" "$metrics_py" parse-case "$spec" "$TOKEN_UNIT")" || die "非法 case: $spec"
  CASE_JSONS+=("$case_json")
done
write_cases_file
load_parallel_config
write_run_metadata

trap "exit 130" INT
trap "exit 143" TERM
trap on_exit EXIT

echo "BASE_URL=$BASE_URL MODEL=$MODEL TOKEN_UNIT=$TOKEN_UNIT PARALLEL_CONFIG=$PARALLEL_CONFIG"
echo "Output length: 1; max-concurrency: ${FIXED_CONCURRENCY:-not set}"
echo "Result directory: $out_dir"

if [[ "$DRY_RUN" == 1 ]]; then
  for case_json in "${CASE_JSONS[@]}"; do
    dry_run_case "$case_json"
  done
  exit 0
fi

command -v vllm >/dev/null 2>&1 || die "找不到 vllm 命令"
for case_json in "${CASE_JSONS[@]}"; do
  run_case "$case_json" || exit 1
done
refresh_and_plot
exit 0
