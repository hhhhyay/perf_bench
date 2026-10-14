#!/usr/bin/env bash
# ============================================================
# SGLang 单侧 Prefill TTFT 矩阵压测
#
# 使用与 sglc.sh 相同的服务配置和 bench_serving 参数，
# 只改变输入长度、per-DP BS、输出 1 token 和 request-rate 搜索。
# ============================================================

set -uo pipefail

run_id="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
out_dir="./all_result/bench_prefill_ttft_${run_id}"
log_dir="${out_dir}/logs"
trials_csv="${out_dir}/prefill_ttft_trials.csv"
matrix_csv="${out_dir}/prefill_ttft_matrix.csv"

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
PREFILL_TTFT_TARGETS_MS="${PREFILL_TTFT_TARGETS_MS:-3000 5000 10000}"
PREFILL_TTFT_SEARCH_ITERS="${PREFILL_TTFT_SEARCH_ITERS:-7}"
PREFILL_TTFT_BS_RULES="${PREFILL_TTFT_BS_RULES:-8:256,16:128,32:64,64:32,128:16}"
PREFILL_TTFT_TEST_MODE="${PREFILL_TTFT_TEST_MODE:-}"

if [[ ! "${TOKEN_UNIT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] TOKEN_UNIT 必须是正整数，当前为: ${TOKEN_UNIT}" >&2
  exit 1
fi

if [[ ! "${PREFILL_TTFT_SEARCH_ITERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] PREFILL_TTFT_SEARCH_ITERS 必须是正整数" >&2
  exit 1
fi

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
    --max-concurrency "${global_concurrency}"
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
}
ttft = {
    "Mean TTFT (ms)": "mean_ttft_ms",
    "Median TTFT (ms)": "median_ttft_ms",
    "P50 TTFT (ms)": "p50_ttft_ms",
    "P75 TTFT (ms)": "p75_ttft_ms",
    "P90 TTFT (ms)": "p90_ttft_ms",
    "P95 TTFT (ms)": "p95_ttft_ms",
    "P99 TTFT (ms)": "p99_ttft_ms",
    "P999 TTFT (ms)": "p999_ttft_ms",
    "Max TTFT (ms)": "max_ttft_ms",
}
labels = {**prefix, **ttft}
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
  local target_ms="$5"
  local trial="$6"
  local phase="$7"
  local input_rate="$8"
  local case_status="$9"
  local log_file="${10}"
  local metrics_json
  metrics_json="$(parse_benchmark_log "${log_file}")"

  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" - "${trials_csv}" "${input_k}" "${input_len}" "${per_dp_bs}" "${global_concurrency}" "${target_ms}" "${trial}" "${phase}" "${input_rate}" "${case_status}" "${log_file}" "${metrics_json}" <<'PY_CSV'
import csv
import json
import os
import sys

path = sys.argv[1]
input_k, input_len, per_dp_bs, global_concurrency, target_ms, trial, phase, input_rate, case_status, log_file = sys.argv[2:12]
metrics = json.loads(sys.argv[12])
fieldnames = [
    "input_k", "input_len", "output_len", "per_dp_bs", "global_concurrency",
    "target_ttft_ms", "trial", "search_phase", "input_request_rate",
    "actual_rps", "backend", "traffic_request_rate", "max_request_concurrency",
    "successful_requests", "benchmark_duration_s", "total_input_tokens",
    "total_input_text_tokens", "total_generated_tokens",
    "total_generated_tokens_retokenized", "request_throughput",
    "input_throughput", "output_throughput", "peak_output_throughput",
    "peak_concurrency", "total_throughput", "concurrency", "accept_length",
    "mean_ttft_ms", "median_ttft_ms", "p50_ttft_ms", "p75_ttft_ms",
    "p90_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "p999_ttft_ms",
    "max_ttft_ms", "case_status", "target_status", "log_file",
]
row = {
    "input_k": input_k, "input_len": input_len, "output_len": "1",
    "per_dp_bs": per_dp_bs, "global_concurrency": global_concurrency,
    "target_ttft_ms": target_ms, "trial": trial, "search_phase": phase,
    "input_request_rate": input_rate, "case_status": case_status,
    "log_file": log_file,
}
row.update(metrics)
row["actual_rps"] = metrics.get("request_throughput", "")
mean = metrics.get("mean_ttft_ms")
if not target_ms:
    row["target_status"] = "BASELINE"
elif isinstance(mean, (int, float)):
    row["target_status"] = "MEET" if mean <= float(target_ms) else "EXCEED"
else:
    row["target_status"] = "NO_METRIC"

rows = []
if os.path.exists(path):
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        rows = []
key = (input_k, target_ms, trial, input_rate, phase)
rows = [
    old for old in rows
    if (old.get("input_k", ""), old.get("target_ttft_ms", ""),
        old.get("trial", ""), old.get("input_request_rate", ""),
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
  local target_ms="$4"
  local trial="$5"
  local phase="$6"
  local request_rate="$7"
  local global_concurrency=$((DP_SIZE * per_dp_bs))
  local num_prompts=$((global_concurrency * NUM_PROMPTS_MULTIPLIER))
  local warmup_requests=0
  local target_tag rate_tag base_name log_file exit_file exit_code

  if (( per_dp_bs == 1 )); then
    warmup_requests=$((DP_SIZE * 16))
  fi
  target_tag="${target_ms:-baseline}"
  rate_tag="${request_rate//./p}"
  rate_tag="${rate_tag//-/m}"
  base_name="input${input_k}k_perdp${per_dp_bs}_target${target_tag}_trial${trial}_rate${rate_tag}"
  log_file="${log_dir}/${base_name}.log"
  exit_file="${log_file}.exitcode"

  if [[ -s "${log_file}" && -f "${exit_file}" ]] && [[ "$(cat "${exit_file}" 2>/dev/null)" == "0" ]]; then
    echo "[SKIP] input=${input_k}K target=${target_ms:-baseline} rate=${request_rate}"
  else
    build_bench_command "${input_len}" "${global_concurrency}" "${num_prompts}" "${warmup_requests}" "${request_rate}"
    echo "[RUN] input=${input_k}K per_dp_bs=${per_dp_bs} target=${target_ms:-baseline} phase=${phase} rate=${request_rate}"
    if [[ "${PREFILL_TTFT_TEST_MODE}" == "command" ]]; then
      printf '%q ' "${cmd[@]}"
      echo
      return 0
    fi
    mkdir -p "${log_dir}"
    "${cmd[@]}" 2>&1 | tee "${log_file}"
    exit_code="${PIPESTATUS[0]}"
    printf "%s\n" "${exit_code}" > "${exit_file}"
  fi

  exit_code="$(cat "${exit_file}" 2>/dev/null || echo 1)"
  local metrics_json actual_rps mean_ttft
  metrics_json="$(parse_benchmark_log "${log_file}")"
  actual_rps="$(printf "%s" "${metrics_json}" | "${PYTHON_BIN}" -c 'import json,sys
d=json.load(sys.stdin); v=d.get("request_throughput")
print(v if v is not None else "")')"
  mean_ttft="$(printf "%s" "${metrics_json}" | "${PYTHON_BIN}" -c 'import json,sys
d=json.load(sys.stdin); v=d.get("mean_ttft_ms")
print(v if v is not None else "")')"

  local status="FAIL"
  [[ "${exit_code}" == "0" ]] && status="PASS"
  csv_upsert_trial "${input_k}" "${input_len}" "${per_dp_bs}" "${global_concurrency}" "${target_ms}" "${trial}" "${phase}" "${request_rate}" "${status}" "${log_file}"

  LAST_ACTUAL_RPS="${actual_rps}"
  LAST_MEAN_TTFT="${mean_ttft}"
  LAST_CASE_STATUS="${status}"
  LAST_LOG_FILE="${log_file}"
}

search_target() {
  local input_k="$1"
  local input_len="$2"
  local per_dp_bs="$3"
  local target_ms="$4"
  local baseline_rps="$5"
  local baseline_ttft="$6"
  local baseline_log="$7"
  local high_rate="${baseline_rps}"
  local low_rate=""
  local low_rps=""
  local low_ttft=""
  local trial=0
  local probe_rate
  local target_met

  if awk -v mean="${baseline_ttft}" -v target="${target_ms}" 'BEGIN { exit !(mean <= target) }' 2>/dev/null; then
    csv_upsert_trial "${input_k}" "${input_len}" "${per_dp_bs}" "$((DP_SIZE * per_dp_bs))" \
      "${target_ms}" "0" "baseline-target" "inf" "PASS" "${baseline_log}"
    echo "[TARGET-PASS] input=${input_k}K target=${target_ms}ms uses inf baseline actual_rps=${baseline_rps}"
    return 0
  fi

  while (( trial < PREFILL_TTFT_SEARCH_ITERS )); do
    trial=$((trial + 1))
    if [[ -z "${low_rate}" ]]; then
      probe_rate="$(awk -v high="${high_rate}" 'BEGIN { printf "%.6f", high / 2 }')"
      phase="descending"
    else
      probe_rate="$(awk -v low="${low_rate}" -v high="${high_rate}" 'BEGIN { printf "%.6f", (low + high) / 2 }')"
      phase="binary"
    fi
    if awk -v rate="${probe_rate}" 'BEGIN { exit !(rate > 0) }' 2>/dev/null; then :; else break; fi

    run_probe "${input_k}" "${input_len}" "${per_dp_bs}" "${target_ms}" "${trial}" "${phase}" "${probe_rate}"
    target_met=1
    if [[ "${LAST_CASE_STATUS}" != "PASS" || -z "${LAST_MEAN_TTFT}" ]]; then
      target_met=0
    elif ! awk -v mean="${LAST_MEAN_TTFT}" -v target="${target_ms}" 'BEGIN { exit !(mean <= target) }' 2>/dev/null; then
      target_met=0
    fi

    if (( target_met == 1 )); then
      low_rate="${probe_rate}"
      low_rps="${LAST_ACTUAL_RPS}"
      low_ttft="${LAST_MEAN_TTFT}"
      if [[ "${phase}" == "descending" ]]; then
        echo "[BRACKET] input=${input_k}K target=${target_ms}ms fail_rate=${high_rate} pass_rate=${low_rate}"
      fi
    else
      high_rate="${probe_rate}"
      if [[ -n "${low_rate}" ]] && awk -v low="${low_rate}" -v high="${high_rate}" 'BEGIN { exit !(high > low) }' 2>/dev/null; then :; fi
    fi

    [[ -n "${low_rate}" ]] || continue
    if [[ "${phase}" == "binary" ]] && [[ "${low_rate}" == "${high_rate}" ]]; then
      break
    fi
  done

  if [[ -n "${low_rate}" ]]; then
    echo "[TARGET-PASS] input=${input_k}K target=${target_ms}ms candidate_rps=${low_rps} rate=${low_rate} mean_ttft=${low_ttft}"
  else
    echo "[TARGET-FAIL] input=${input_k}K target=${target_ms}ms no passing request rate"
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
    "target_ttft_ms", "trial", "search_phase", "input_request_rate",
    "actual_rps", "backend", "traffic_request_rate", "max_request_concurrency",
    "successful_requests", "benchmark_duration_s", "total_input_tokens",
    "total_input_text_tokens", "total_generated_tokens",
    "total_generated_tokens_retokenized", "request_throughput",
    "input_throughput", "output_throughput", "peak_output_throughput",
    "peak_concurrency", "total_throughput", "concurrency", "accept_length",
    "mean_ttft_ms", "median_ttft_ms", "p50_ttft_ms", "p75_ttft_ms",
    "p90_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "p999_ttft_ms",
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
    row.get("target_ttft_ms", "") for row in rows
    if row.get("target_ttft_ms", "")
}, key=lambda x: float(x))
out = []
for input_k in inputs:
    baseline = next(
        (row for row in rows
         if row.get("input_k") == input_k
         and row.get("target_ttft_ms", "") == ""
         and row.get("case_status") == "PASS"),
        None,
    )
    target_values = [t for t in targets if any(
        row.get("input_k") == input_k and row.get("target_ttft_ms") == t
        for row in rows
    )]
    for target in target_values:
        candidates = [
            row for row in rows
            if row.get("input_k") == input_k
            and row.get("target_ttft_ms") == target
            and row.get("case_status") == "PASS"
            and row.get("target_status") == "MEET"
            and num(row, "actual_rps") is not None
        ]
        if baseline and num(baseline, "mean_ttft_ms") is not None and num(baseline, "mean_ttft_ms") <= float(target):
            base = dict(baseline)
            base["target_ttft_ms"] = target
            base["target_status"] = "MEET"
            candidates.append(base)
        if candidates:
            best = max(candidates, key=lambda row: (num(row, "actual_rps"), -num(row, "mean_ttft_ms")))
            best["selection_status"] = "BEST"
            out.append({field: best.get(field, "") for field in fields})
        else:
            out.append({
                "input_k": input_k, "target_ttft_ms": target,
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
  local input_len per_dp_bs baseline_rps baseline_ttft baseline_log target

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

  run_probe "${input_k}" "${input_len}" "${per_dp_bs}" "" "0" "baseline" "inf"
  baseline_rps="${LAST_ACTUAL_RPS}"
  baseline_ttft="${LAST_MEAN_TTFT}"
  baseline_log="${LAST_LOG_FILE}"
  if [[ "${LAST_CASE_STATUS}" != "PASS" || -z "${baseline_rps}" || -z "${baseline_ttft}" ]]; then
    echo "[ERROR] input=${input_k}K 的 inf 基准缺少实际 RPS 或 Mean TTFT，跳过矩阵" >&2
    return 1
  fi

  local target_list="${PREFILL_TTFT_TARGETS_MS//,/ }"
  for target in ${target_list}; do
    if [[ "${target}" =~ ^[1-9][0-9]*$ ]]; then
      search_target "${input_k}" "${input_len}" "${per_dp_bs}" "${target}" "${baseline_rps}" "${baseline_ttft}" "${baseline_log}"
    else
      echo "[ERROR] 非法 TTFT 目标: ${target}" >&2
      return 1
    fi
  done
}

main() {
  mkdir -p "${log_dir}"
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

  echo "[INFO] Prefill TTFT matrix"
  echo "[INFO] BASE_URL=${BASE_URL}"
  echo "[INFO] MODEL=${MODEL}"
  echo "[INFO] DP_SIZE=${DP_SIZE}"
  echo "[INFO] TOKEN_UNIT=${TOKEN_UNIT}"
  echo "[INFO] INPUTS=${PREFILL_TTFT_INPUTS}"
  echo "[INFO] TARGETS_MS=${PREFILL_TTFT_TARGETS_MS}"
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
  echo "[DONE] trials=${trials_csv}"
  echo "[DONE] matrix=${matrix_csv}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
