#!/usr/bin/env bash
# ============================================================
# SGLang 客户端容量压测
#
# 1) 优先从 /v1/loads?include=all 一次性读取 dp_size / max_total_num_tokens / max_running_requests
# 2) /v1/loads 不可用时回退 /metrics + 手动 max_running_requests
# 3) 理论 per-DP BS 同时受两项约束：
#      token_theoretical_per_dp_bs = MAX_TOTAL_TOKENS / tokens_per_request
#      request_limit               = MAX_RUNNING_REQUESTS
#      theoretical_per_dp_bs       = min(token_theoretical_per_dp_bs, request_limit)
#    若 MAX_RUNNING_REQUESTS 未获取到/<=0，则仅使用 token capacity。
#    自动 BS 扫描围绕最终 effective theoretical BS，而不是只围绕 KV token 理论值。
# 4) BS 扫描：优先使用 /server_info decode CUDA Graph BS；拿不到时回退 DP_SIZE 布局；再叠加 effective-theory(±1/±2) 与 SLA inline 连续加密
# 5) 汇总 jsonl + log -> sum_all.csv
# ============================================================

set -uo pipefail

# ============================================================
# 1. 基础配置
# ============================================================

run_id="$(date +%Y%m%d_%H%M%S)"

out_dir="./all_result/bench_theory_bs_${run_id}"
jsonl_dir="${out_dir}/jsonl"
log_dir="${out_dir}/logs"

mkdir -p "${jsonl_dir}"
mkdir -p "${log_dir}"

# 是否生成 Excel 版汇总。


# ============================================================
# 2. 服务配置（可用环境变量覆盖）
# ============================================================

BASE_URL="${BASE_URL:-http://12.12.12.37:10015}"

# 空则尝试从 /metrics 的 model_name 标签自动解析
MODEL="${MODEL:-/public4/opendas/DL_DATA/llm-models/DeepSeek-V4-Flash-0731-FP8-Channel}"


# DP 数：默认从 /metrics 自动识别 dp_rank 数量。
# 如 metrics 无 dp_rank，可通过环境变量 DP_SIZE 手动回退。
DP_SIZE="${DP_SIZE:-}"

# 一般留空：由 curl /metrics 自动识别 sglang:max_total_num_tokens
# 仅在 metrics 不可用时才手动覆盖
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"

# 示例：
#
# 支持日志格式包括：
#   max_running_requests=20
#   max_running_requests: 20
#   "max_running_requests": 20
#   --max-running-requests 20

# 日志不可用时可手动覆盖：
#   MAX_RUNNING_REQUESTS=20 bash this_script.sh
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"

# /server_info 自动提取的并行与 CUDA Graph 配置（TP/DP/PP/EP/ACP/KV cache dtype）。
# 若接口不可用，则回退原 DP_SIZE 布局策略。
TP_SIZE="${TP_SIZE:-}"
PP_SIZE="${PP_SIZE:-}"
EP_SIZE="${EP_SIZE:-}"
ATTN_CP_SIZE="${ATTN_CP_SIZE:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
DP_ATTENTION="${DP_ATTENTION:-}"
CUDA_GRAPH_DECODE_BS="${CUDA_GRAPH_DECODE_BS:-}"
PARALLEL_CONFIG="${PARALLEL_CONFIG:-}"

# Mean TPOT 接近该值时，在相邻 BS 区间自动补点。
SLA_DENSE_ENABLE="${SLA_DENSE_ENABLE:-1}"
# 通用 SLA 规则：metric:target:near，多个规则用逗号分隔
# 例如：mean_tpot_ms:30:2,p99_tpot_ms:75:5,mean_ttft_ms:1000:100
SLA_RULES="${SLA_RULES:-mean_tpot_ms:50:5,p99_tpot_ms:75:5}"
# 默认 1=开启 SLA inline 自动加密；设 0 可关闭
# SLA_DENSE_ENABLE=0 bash this_script.sh
# TPOT 附近连续加密：
# - 默认关闭；SLA_DENSE_ENABLE=1 时开启
# - 同时检测 Mean TPOT / P99 TPOT 是否接近目标值
# - 一旦当前 BS 命中 target±near，就立刻把“当前 BS 到下一个基础/CUDA-Graph BS 之间”按连续整数 BS 跑完
# - 不再有第二阶段补测，不回头、不重复；随后继续原 CUDA-Graph/theory BS 顺序。

# 真实服务默认关闭；FakePD / 仅测 decode 时设 1

USE_FAKE_PREFILL="${USE_FAKE_PREFILL:-0}"

# ============================================================
# 固定并发接口（可选）
#
# 默认都为空：继续使用“理论 BS 自动扫描”模式。
#
# 方式 1：固定 per-DP BS
#   FIXED_PER_DP_BS=12 bash bench_theory_bs_with_fixed_concurrency.sh
#
# 方式 2：固定全局并发
#   FIXED_CONCURRENCY=192 bash bench_theory_bs_with_fixed_concurrency.sh
#
# 注意：
# - FIXED_PER_DP_BS 与 FIXED_CONCURRENCY 不能同时设置。
# - FIXED_CONCURRENCY 必须能被 DP_SIZE 整除，以保持
#   per_dp_bs = global_concurrency / DP_SIZE 的语义准确。
# - 也支持空格/逗号分隔的多个固定点，例如：
#   FIXED_PER_DP_BS="10,12,14,16"
#   FIXED_CONCURRENCY="160 192 224 256"
# ============================================================
FIXED_PER_DP_BS="${FIXED_PER_DP_BS:-}"
FIXED_CONCURRENCY="${FIXED_CONCURRENCY:-}"

# 请求数量倍率：
#   num_prompts = max_concurrency * NUM_PROMPTS_MULTIPLIER
# 默认 1，保持旧行为；例如每个并发位跑 4 个请求：
#   NUM_PROMPTS_MULTIPLIER=4 bash this_script.sh
NUM_PROMPTS_MULTIPLIER="${NUM_PROMPTS_MULTIPLIER:-1}"



# ============================================================
# 3. Input / Output（单位：K）
#
# 4 1    => 4K in / 1K out   → theory = 258304 / 5 / 1024 ≈ 50.45
# 64 16  => 64K / 16K
# 1024 1 => 1M / 1K
# ============================================================

CASES=(
#  "4 1"
  "8 1"
#  "16 1"
#  "36 1"
#  "64 16"
#  "128 1"
#  "1024 1"
)


# ============================================================
# 4. BS 扫描配置
# ============================================================


# DP<=3 时使用较稀疏但更实用的基础点。
# 基础序列：
#   1/2/4/8/16/32/48/64/96/128
# 当理论 BS > 128 时，继续按 32 的步长扩展：
#   160/192/224/256/...
LOW_DP_BS=(
  1
  2
  4
  8
  16
  32
  48
  64
  96
  128
)

# DP>3 时使用关键 anchor 加密的 per-DP BS。
# 重点观察 16 / 32 / 64 附近的 scheduler / kernel / cudagraph 性能拐点。
# >72 后继续按 16 粗扫，再叠加 effective theory 附近 ±1/±2。
DENSE_DP_BS=(
  1
  2
  4
  8

  12
  14
  15
  16
  17
  18
  20

  24
  28
  30
  31
  32
  33
  34
  36
  40

  48
  56
  60
  62
  63
  64
  65
  66
  68
  72
)

# effective theory 附近加密点
THEORY_DELTAS=(
  -8
  -4
  -2
  -1
  0
  1
  2
  4
  8
)



# ============================================================
# 5. 优先从 /v1/loads?include=all 一次性提取：
#    DP_SIZE / MAX_TOTAL_TOKENS / MAX_RUNNING_REQUESTS
# ============================================================
load_capacity_from_loads_api() {
  local loads_url="${BASE_URL%/}/v1/loads?include=all"
  local values=""

  values="$(
    curl -fsS "${loads_url}" 2>/dev/null \
    | python3 -c 'import sys,json
d=json.load(sys.stdin)
l=d.get("loads") or []
assert l
print(
    d.get("dp_rank_count") or len(l),
    min(int(x["max_total_num_tokens"]) for x in l if x.get("max_total_num_tokens") is not None),
    min(int(x["max_running_requests"]) for x in l if x.get("max_running_requests") is not None),
)' 2>/dev/null \
    || true
  )"

  if [[ -n "${values}" ]]; then
    read -r DP_SIZE MAX_TOTAL_TOKENS MAX_RUNNING_REQUESTS <<< "${values}"
    echo "[INFO] capacity: DP_SIZE=${DP_SIZE} MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS} MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}"
    return 0
  fi
  return 1
}

# ============================================================
# 5. 从 /metrics 自动识别 max_total_num_tokens / DP_SIZE
#
# 正式 benchmark 运行期间不再轮询 /metrics。
# 启动前只抓取一次，用于：
#   - max_total_num_tokens
#   - dp_size
#   - model_name
# ============================================================

parse_metric_value() {
  local metric_name="$1"
  local metrics_text="$2"

  # 取该 metric 行最后一个数字字段，例如 258304.0 -> 258304
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



# 从 /metrics 自动识别 DP Size：
# 1) 优先统计所有唯一 dp_rank="N"
# 2) 如果没有 dp_rank，再尝试 sglang:dp_size
# 3) 都没有时返回空，由调用方回退到手动 DP_SIZE 或 1
parse_dp_size_from_metrics() {
  local metrics_text="$1"
  local rank_count
  local explicit_dp

  rank_count="$(
    printf "%s\n" "${metrics_text}" \
      | grep -oE 'dp_rank="[0-9]+"' \
      | sed -E 's/.*"([0-9]+)"/\1/' \
      | sort -nu \
      | wc -l \
      | awk '{print $1}'
  )"

  if [[ "${rank_count:-0}" =~ ^[0-9]+$ ]] && (( rank_count > 0 )); then
    printf "%s\n" "${rank_count}"
    return
  fi

  explicit_dp="$(parse_metric_value "sglang:dp_size" "${metrics_text}")"
  if [[ -n "${explicit_dp}" ]] && [[ "${explicit_dp}" =~ ^[0-9]+$ ]] && (( explicit_dp > 0 )); then
    printf "%s\n" "${explicit_dp}"
    return
  fi

  printf "\n"
}


fetch_model_from_metrics() {
  local metrics_url="$1"
  local raw

  raw="$(curl -s --connect-timeout 5 --max-time 30 "${metrics_url}" || true)"
  printf "%s\n" "${raw}" \
    | sed -n 's/.*model_name="\([^"]*\)".*/\1/p' \
    | head -n 1
}

load_server_info() {
  local info_url="${BASE_URL%/}/server_info"
  local values=""

  values="$(
    curl -fsS "${info_url}" 2>/dev/null \
    | python3 -c 'import sys,json
d=json.load(sys.stdin)
tp=d.get("tp_size"); dp=d.get("dp_size"); pp=d.get("pp_size"); ep=d.get("ep_size")
acp=d.get("attn_cp_size"); dpa=d.get("enable_dp_attention")
kv=d.get("kv_cache_dtype")
cfg=d.get("cuda_graph_config") or {}; dec=cfg.get("decode") or {}; bs=dec.get("bs") or []
def sv(v):
    if isinstance(v,bool): return "true" if v else "false"
    return "" if v is None else str(v)
print(sv(tp),sv(dp),sv(pp),sv(ep),sv(acp),sv(dpa),sv(kv),",".join(str(int(x)) for x in bs if isinstance(x,(int,float))))' 2>/dev/null \
    || true
  )"

  if [[ -n "${values}" ]]; then
    read -r _tp _dp _pp _ep _acp _dpa _kv _cudabs <<< "${values}"
    [[ -n "${_tp}" ]] && TP_SIZE="${_tp}"
    [[ -n "${_dp}" ]] && DP_SIZE="${_dp}"
    [[ -n "${_pp}" ]] && PP_SIZE="${_pp}"
    [[ -n "${_ep}" ]] && EP_SIZE="${_ep}"
    [[ -n "${_acp}" ]] && ATTN_CP_SIZE="${_acp}"
    [[ -n "${_dpa}" ]] && DP_ATTENTION="${_dpa}"
    [[ -n "${_kv}" ]] && KV_CACHE_DTYPE="${_kv}"
    [[ -n "${_cudabs}" ]] && CUDA_GRAPH_DECODE_BS="${_cudabs}"

    local attn_tag
    if [[ "${DP_ATTENTION}" == "true" ]]; then attn_tag="DPA"; else attn_tag="TPA"; fi
    PARALLEL_CONFIG="TP${TP_SIZE:-NA}-DP${DP_SIZE:-NA}(${attn_tag})-PP${PP_SIZE:-NA}-EP${EP_SIZE:-NA}-ACP${ATTN_CP_SIZE:-NA}-KV${KV_CACHE_DTYPE:-NA}"

    echo "[INFO] server_info: ${PARALLEL_CONFIG}"
    if [[ -n "${CUDA_GRAPH_DECODE_BS}" ]]; then
      echo "[INFO] server_info decode cuda graph bs: ${CUDA_GRAPH_DECODE_BS}"
    else
      echo "[WARN] server_info 未返回 decode cuda graph bs，后续回退 DP_SIZE 布局"
    fi
    return 0
  fi

  echo "[WARN] /server_info 不可用，后续回退 DP_SIZE 布局"
  return 1
}

load_server_capacity() {
  if load_capacity_from_loads_api; then
    return 0
  fi

  echo "[WARN] /v1/loads 不可用，回退 /metrics；MAX_RUNNING_REQUESTS 需手动设置"

  local metrics_url="${BASE_URL%/}/metrics"
  local raw_metrics=""
  local max_tokens=""
  local detected_dp=""

  raw_metrics="$(curl -fsS "${metrics_url}" 2>/dev/null || true)"

  if [[ -n "${raw_metrics}" ]]; then
    max_tokens="$(
      printf "%s\n" "${raw_metrics}" \
      | awk '/^sglang:max_total_num_tokens([{ ]|$)/ {print $NF}' \
      | sort -n \
      | head -n 1
    )"

    detected_dp="$(parse_dp_size_from_metrics "${raw_metrics}")"
    [[ -n "${detected_dp}" ]] && DP_SIZE="${detected_dp}"
    [[ -n "${max_tokens}" ]] && MAX_TOTAL_TOKENS="${max_tokens%.*}"
  fi

  DP_SIZE="${DP_SIZE:-1}"
  MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-0}"

  if [[ -z "${MAX_TOTAL_TOKENS}" ]]; then
    echo "[ERROR] 无法自动获取 MAX_TOTAL_TOKENS，请手动设置"
    exit 1
  fi

  echo "[INFO] fallback capacity: DP_SIZE=${DP_SIZE} MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS} MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}"
}


# ============================================================
# 6. 名称格式
# ============================================================

format_k_name() {
  local value="$1"

  if (( value == 1024 )); then
    echo "1m"
  elif (( value > 1024 && value % 1024 == 0 )); then
    echo "$((value / 1024))m"
  else
    echo "${value}k"
  fi
}



# ============================================================
# 固定并发参数解析
# ============================================================

normalize_int_list() {
  # 接受：
  #   "12"
  #   "10 12 14"
  #   "10,12,14"
  #   "10, 12, 14"
  #
  # 输出为空格分隔的正整数列表，并去重排序。
  local raw="$1"

  printf "%s\n" "${raw}" \
    | tr ',' ' ' \
    | tr -s '[:space:]' '\n' \
    | awk '
        /^[0-9]+$/ && $1 > 0 { seen[$1] = 1 }
        END {
          for (v in seen) print v
        }
      ' \
    | sort -n \
    | tr '\n' ' '
}

validate_fixed_mode() {
  if [[ -n "${FIXED_PER_DP_BS}" && -n "${FIXED_CONCURRENCY}" ]]; then
    echo "[ERROR] FIXED_PER_DP_BS 和 FIXED_CONCURRENCY 不能同时设置"
    exit 1
  fi

  if [[ -n "${FIXED_PER_DP_BS}" ]]; then
    local normalized
    normalized="$(normalize_int_list "${FIXED_PER_DP_BS}")"
    if [[ -z "${normalized// }" ]]; then
      echo "[ERROR] FIXED_PER_DP_BS 没有解析到有效正整数: ${FIXED_PER_DP_BS}"
      exit 1
    fi
    FIXED_PER_DP_BS="${normalized% }"
  fi

  if [[ -n "${FIXED_CONCURRENCY}" ]]; then
    local normalized
    local c

    normalized="$(normalize_int_list "${FIXED_CONCURRENCY}")"
    if [[ -z "${normalized// }" ]]; then
      echo "[ERROR] FIXED_CONCURRENCY 没有解析到有效正整数: ${FIXED_CONCURRENCY}"
      exit 1
    fi

    for c in ${normalized}; do
      if (( c % DP_SIZE != 0 )); then
        echo "[ERROR] FIXED_CONCURRENCY=${c} 不能被 DP_SIZE=${DP_SIZE} 整除"
        echo "[ERROR] 为保证 per_dp_bs 语义准确，请改用可整除的全局并发，"
        echo "        或直接使用 FIXED_PER_DP_BS。"
        exit 1
      fi
    done

    FIXED_CONCURRENCY="${normalized% }"
  fi
}

get_test_bs_list() {
  local auto_bs_list="$1"

  if [[ -n "${FIXED_PER_DP_BS}" ]]; then
    printf "%s\n" "${FIXED_PER_DP_BS}"
    return
  fi

  if [[ -n "${FIXED_CONCURRENCY}" ]]; then
    local c
    for c in ${FIXED_CONCURRENCY}; do
      printf "%s " "$((c / DP_SIZE))"
    done
    echo
    return
  fi

  printf "%s\n" "${auto_bs_list}"
}

get_test_mode_name() {
  if [[ -n "${FIXED_PER_DP_BS}" ]]; then
    echo "fixed_per_dp_bs"
  elif [[ -n "${FIXED_CONCURRENCY}" ]]; then
    echo "fixed_concurrency"
  else
    echo "auto_theory_bs"
  fi
}

# ============================================================
# 7. 理论 BS
#
# token_theoretical_per_dp_bs =
#   MAX_TOTAL_TOKENS / ((input_k + output_k) * 1024)
#
# request_limit_per_dp_bs =
#   MAX_RUNNING_REQUESTS
#
# effective theoretical_per_dp_bs =
#   min(token_theoretical_per_dp_bs, MAX_RUNNING_REQUESTS)
#
# 若 MAX_RUNNING_REQUESTS <= 0，则退化为 token_theoretical_per_dp_bs。
# ============================================================

get_token_theoretical_bs() {
  local input_k="$1"
  local output_k="$2"

  awk \
    -v max_tokens="${MAX_TOTAL_TOKENS}" \
    -v input_k="${input_k}" \
    -v output_k="${output_k}" \
    'BEGIN {
      printf "%.3f",
      max_tokens / ((input_k + output_k) * 1024)
    }'
}

get_theoretical_bs() {
  local input_k="$1"
  local output_k="$2"
  local token_bs

  token_bs="$(get_token_theoretical_bs "${input_k}" "${output_k}")"

  if [[ "${MAX_RUNNING_REQUESTS:-0}" =~ ^[0-9]+$ ]] && (( MAX_RUNNING_REQUESTS > 0 )); then
    awk       -v token_bs="${token_bs}"       -v max_req="${MAX_RUNNING_REQUESTS}"       'BEGIN {
        if (token_bs < max_req) printf "%.3f", token_bs;
        else                    printf "%.3f", max_req;
      }'
  else
    printf "%s\n" "${token_bs}"
  fi
}

get_theory_limit_factor() {
  local input_k="$1"
  local output_k="$2"
  local token_bs

  token_bs="$(get_token_theoretical_bs "${input_k}" "${output_k}")"

  if [[ "${MAX_RUNNING_REQUESTS:-0}" =~ ^[0-9]+$ ]] && (( MAX_RUNNING_REQUESTS > 0 )); then
    awk       -v token_bs="${token_bs}"       -v max_req="${MAX_RUNNING_REQUESTS}"       'BEGIN {
        if (max_req <= token_bs) print "max_running_requests";
        else                     print "max_total_num_tokens";
      }'
  else
    echo "max_total_num_tokens"
  fi
}


# ============================================================
# 8. 自动生成 BS 列表
#
# - DP<=3：1/2/4/8/16/32/48/64/96/128... + 理论附近加密
# - DP>3 ：per-DP BS 使用更密扫描
# - 理论附近：floor(effective_theory)-2/-1/0/+1/+2
# - scan_max = effective_theory_floor + 8
# - 小范围 (<=8)：连续扫描
# ============================================================

get_bs_list() {
  local input_k="$1"
  local output_k="$2"

  local tokens_per_request theoretical_floor scan_max bs delta
  tokens_per_request=$(((input_k + output_k) * 1024))
  theoretical_floor=$((MAX_TOTAL_TOKENS / tokens_per_request))

  if [[ "${MAX_RUNNING_REQUESTS:-0}" =~ ^[0-9]+$ ]] && (( MAX_RUNNING_REQUESTS > 0 )); then
    if (( MAX_RUNNING_REQUESTS < theoretical_floor )); then
      theoretical_floor="${MAX_RUNNING_REQUESTS}"
    fi
  fi
  if (( theoretical_floor < 1 )); then theoretical_floor=1; fi
  scan_max=$((theoretical_floor + 2))

  declare -A selected=()

  # 第一优先级：服务真实 decode CUDA Graph bucket
  if [[ -n "${CUDA_GRAPH_DECODE_BS}" ]]; then
    while IFS= read -r bs; do
      [[ -n "${bs}" ]] || continue
      if [[ "${bs}" =~ ^[0-9]+$ ]] && (( bs >= 1 && bs <= scan_max )); then
        selected["${bs}"]=1
      fi
    done < <(
      printf "%s\n" "${CUDA_GRAPH_DECODE_BS}" | tr ',' '\n' | awk '/^[0-9]+$/ {print $1}' | sort -nu
    )
  else
    # Fallback：拿不到 server_info CUDA Graph BS 时才按 DP_SIZE 布局
    if (( DP_SIZE > 3 )); then
      for bs in "${DENSE_DP_BS[@]}"; do
        (( bs <= scan_max )) && selected["${bs}"]=1
      done
      bs=80
      while (( bs <= scan_max )); do selected["${bs}"]=1; bs=$((bs + 16)); done
    else
      for bs in "${LOW_DP_BS[@]}"; do
        (( bs <= scan_max )) && selected["${bs}"]=1
      done
      bs=160
      while (( bs <= scan_max )); do selected["${bs}"]=1; bs=$((bs + 32)); done
    fi
  fi

  # 第二层：effective theory 附近 ±1/±2
  for delta in -2 -1 0 1 2; do
    bs=$((theoretical_floor + delta))
    if (( bs >= 1 && bs <= scan_max )); then selected["${bs}"]=1; fi
  done

  selected["1"]=1
  selected["${scan_max}"]=1

  for ((bs=1; bs<=scan_max; bs++)); do
    [[ -n "${selected[$bs]:-}" ]] && printf "%s " "${bs}"
  done
  echo
}

# ============================================================
# 9. 汇总 LOG + JSONL -> sum_all.csv
# CSV 理论 BS 仅保留：token_theoretical_per_dp_bs / effective_theoretical_per_dp_bs
# ============================================================

merge_results() {

python3 - \
  "${jsonl_dir}" \
  "${log_dir}" \
  "${out_dir}/sum_all.csv" \
  "${MAX_TOTAL_TOKENS}" \
  "${DP_SIZE}" \
  "${MAX_RUNNING_REQUESTS}" \
  "${NUM_PROMPTS_MULTIPLIER}" \
  "${PARALLEL_CONFIG:-UNKNOWN}" <<'PY'

import csv
import json
import math
import re
import sys
from pathlib import Path

jsonl_dir = Path(sys.argv[1])
log_dir = Path(sys.argv[2])
output_csv = Path(sys.argv[3])
max_total_tokens = int(sys.argv[4])
dp_size = int(sys.argv[5])
max_running_requests = int(sys.argv[6])
num_prompts_multiplier = int(sys.argv[7])
parallel_config = sys.argv[8]

pattern = re.compile(
    r"(?P<case>.+?)"
    r"_in(?P<input>\d+)"
    r"_out(?P<output>\d+)"
    r"_perdp(?P<perdp>\d+)"
    r"_global(?P<global>\d+)"
    r"\.(?:log|jsonl)$"
)

SPECIAL_KEYS = {
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
    "Median E2E Latency (ms)": "median_e2e_latency_ms",
    "P50 E2E Latency (ms)": "p50_e2e_latency_ms",
    "P75 E2E Latency (ms)": "p75_e2e_latency_ms",
    "P90 E2E Latency (ms)": "p90_e2e_latency_ms",
    "P95 E2E Latency (ms)": "p95_e2e_latency_ms",
    "P99 E2E Latency (ms)": "p99_e2e_latency_ms",
    "P999 E2E Latency (ms)": "p999_e2e_latency_ms",
    "Max E2E Latency (ms)": "max_e2e_latency_ms",
    "Mean TTFT (ms)": "mean_ttft_ms",
    "Median TTFT (ms)": "median_ttft_ms",
    "P50 TTFT (ms)": "p50_ttft_ms",
    "P75 TTFT (ms)": "p75_ttft_ms",
    "P90 TTFT (ms)": "p90_ttft_ms",
    "P95 TTFT (ms)": "p95_ttft_ms",
    "P99 TTFT (ms)": "p99_ttft_ms",
    "P999 TTFT (ms)": "p999_ttft_ms",
    "Max TTFT (ms)": "max_ttft_ms",
    "Mean TPOT (ms)": "mean_tpot_ms",
    "Median TPOT (ms)": "median_tpot_ms",
    "P50 TPOT (ms)": "p50_tpot_ms",
    "P75 TPOT (ms)": "p75_tpot_ms",
    "P90 TPOT (ms)": "p90_tpot_ms",
    "P95 TPOT (ms)": "p95_tpot_ms",
    "P99 TPOT (ms)": "p99_tpot_ms",
    "P999 TPOT (ms)": "p999_tpot_ms",
    "Max TPOT (ms)": "max_tpot_ms",
    "Mean ITL (ms)": "mean_itl_ms",
    "Median ITL (ms)": "median_itl_ms",
    "P50 ITL (ms)": "p50_itl_ms",
    "P75 ITL (ms)": "p75_itl_ms",
    "P90 ITL (ms)": "p90_itl_ms",
    "P95 ITL (ms)": "p95_itl_ms",
    "P99 ITL (ms)": "p99_itl_ms",
    "P999 ITL (ms)": "p999_itl_ms",
    "Max ITL (ms)": "max_itl_ms",
    "Benchmark duration (s)": "duration_s",
    "Input token throughput (tok/s)": "input_throughput_tok_s",
    "Max concurrent requests": "max_concurrency",
    "Output token throughput (tok/s)": "generate_throughput_tok_s",
    "Peak concurrent requests": "peak_concurrency",
    "Request throughput (req/s)": "rps",
    "Successful requests": "num_prompts",
    "Total token throughput (tok/s)": "total_throughput_tok_s",

}


def normalize_key(label):
    if label in SPECIAL_KEYS:
        return SPECIAL_KEYS[label]
    key = label.strip().lower()
    key = key.replace("%", "percent")
    key = re.sub(r"\(req/s\)", "", key)
    key = re.sub(r"\(tok/s\)", "", key)
    key = re.sub(r"\(ms\)", "_ms", key)
    key = re.sub(r"\(s\)", "_s", key)
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return re.sub(r"_+", "_", key).strip("_")


def parse_value(value):
    value = value.strip()
    if not value:
        return ""
    if value.lower() in {"inf", "+inf", "-inf", "nan", "none", "null", "n/a"}:
        return value
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", value):
            return float(value)
    except ValueError:
        pass
    return value


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q / 100.0
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return values[low]
    return values[low] * (high - pos) + values[high] * (pos - low)


def calc_stats(values, scale=1.0):
    clean = []

    def flatten(v):
        if isinstance(v, list):
            for x in v:
                flatten(x)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            if math.isfinite(float(v)):
                clean.append(float(v) * scale)

    flatten(values)
    if not clean:
        return {}
    mean = sum(clean) / len(clean)
    variance = sum((x - mean) ** 2 for x in clean) / len(clean)
    return {
        "count": len(clean),
        "min": min(clean),
        "mean": mean,
        "std": math.sqrt(variance),
        "p50": percentile(clean, 50),
        "p75": percentile(clean, 75),
        "p90": percentile(clean, 90),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
        "p999": percentile(clean, 99.9),
        "max": max(clean),
    }


def add_stats(row, metric, values, scale=1.0):
    for stat, value in calc_stats(values, scale).items():
        if stat == "count":
            row[f"count_{metric}"] = value
        else:
            row[f"{stat}_{metric}_ms"] = value


def parse_log(path):
    result = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return result, False

    starts = [i for i, line in enumerate(lines) if "Serving Benchmark Result" in line]
    if not starts:
        return result, False

    start = starts[-1]
    kv_pattern = re.compile(r"^\s*([^:]+?)\s*:\s*(.*?)\s*$")
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("====") or stripped.startswith("----"):
            continue
        match = kv_pattern.match(line)
        if not match:
            continue
        label = match.group(1).strip()
        value = match.group(2).strip()
        if not label:
            continue
        result[normalize_key(label)] = parse_value(value)
    return result, True


def parse_jsonl(path):
    result = {}
    if not path.exists():
        return result
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return result

    records = []
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            records.append(obj)
        elif isinstance(obj, list):
            records.extend(x for x in obj if isinstance(x, dict))
    except json.JSONDecodeError:
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)

    if not records:
        return result

    obj = records[-1]
    for key, value in obj.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            result[normalize_key(key)] = value

    latencies = obj.get("latencies") or obj.get("e2e_latencies") or obj.get("request_latencies")
    if latencies:
        add_stats(result, "e2e_latency", latencies, 1000.0)

    ttfts = obj.get("ttfts") or obj.get("raw_ttfts")
    if ttfts:
        add_stats(result, "ttft", ttfts, 1000.0)

    itls = obj.get("itls") or obj.get("raw_itls")
    if itls:
        add_stats(result, "itl", itls, 1000.0)

    tpots = obj.get("tpots") or obj.get("raw_tpots")
    if tpots:
        add_stats(result, "tpot", tpots, 1000.0)
    elif latencies and ttfts and obj.get("output_lens"):
        flat_latency, flat_ttft, flat_out = [], [], []

        def flatten(v, target):
            if isinstance(v, list):
                for x in v:
                    flatten(x, target)
            elif isinstance(v, (int, float)):
                target.append(float(v))

        flatten(latencies, flat_latency)
        flatten(ttfts, flat_ttft)
        flatten(obj["output_lens"], flat_out)
        if len(flat_latency) == len(flat_ttft) == len(flat_out):
            calculated = []
            for latency, ttft, outlen in zip(flat_latency, flat_ttft, flat_out):
                if outlen <= 1:
                    continue
                value = (latency - ttft) / (outlen - 1)
                if value >= 0:
                    calculated.append(value)
            if calculated:
                add_stats(result, "tpot", calculated, 1000.0)

    return result


def read_exit_code(log_path):
    path = Path(str(log_path) + ".exitcode")
    if not path.exists():
        return ""
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return ""


def read_time_info(log_path):
    path = Path(str(log_path) + ".time")
    info = {
        "start_time": "",
        "end_time": "",
        "elapsed_s": "",
    }
    if not path.exists():
        return info
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key not in info:
                continue
            if key == "elapsed_s":
                try:
                    info[key] = int(float(value)) if value != "" else ""
                except ValueError:
                    info[key] = value
            else:
                info[key] = value
    except Exception:
        pass
    return info

log_files = []
for path in log_dir.glob("*.log"):
    match = pattern.fullmatch(path.name)
    if not match:
        continue
    log_files.append(
        (
            int(match.group("input")),
            int(match.group("output")),
            int(match.group("perdp")),
            path,
            match,
        )
    )

log_files.sort(key=lambda x: (x[0], x[1], x[2]))
rows = []

for input_len, output_len, perdp, log_path, match in log_files:
    case_name = match.group("case")
    global_concurrency = int(match.group("global"))
    jsonl_path = jsonl_dir / (log_path.stem + ".jsonl")

    json_result = parse_jsonl(jsonl_path)
    log_result, found_result = parse_log(log_path)

    row = {}
    row.update(json_result)
    row.update(log_result)

    tokens_per_request = input_len + output_len
    token_theoretical_bs = max_total_tokens / tokens_per_request

    if max_running_requests > 0:
        theoretical_bs = min(token_theoretical_bs, max_running_requests)
        limiting_factor = (
            "max_running_requests"
            if max_running_requests <= token_theoretical_bs
            else "max_total_num_tokens"
        )
    else:
        theoretical_bs = token_theoretical_bs
        limiting_factor = "max_total_num_tokens"

    theoretical_global_bs = theoretical_bs * dp_size

    row["case_name"] = case_name
    row["input_len"] = input_len
    row["output_len"] = output_len
    row["input_k"] = input_len / 1024
    row["output_k"] = output_len / 1024
    row["parallel_config"] = parallel_config
    row["max_total_tokens"] = max_total_tokens
    row["max_running_requests"] = max_running_requests if max_running_requests > 0 else ""
    row["tokens_per_request"] = tokens_per_request
    row["token_theoretical_per_dp_bs"] = round(token_theoretical_bs, 4)
    row["effective_theoretical_per_dp_bs"] = round(theoretical_bs, 4)
    row["per_dp_bs"] = perdp

    # ========================================================
    # 常用指标统一别名：用于让 sum_all.csv 第一屏直接看到核心性能数据
    # ========================================================
    # max_concurrency 就是本 case 的 global concurrency；
    # num_prompts 独立表示总请求数，不再与 max_concurrency 重复。
    row["max_concurrency"] = global_concurrency
    row["num_prompts"] = global_concurrency * num_prompts_multiplier
    row["request_rate"] = row.get("request_rate", row.get("traffic_request_rate", ""))

    # 优先使用 benchmark 自己输出的 Benchmark duration (s)。
    # elapsed_s 包含 warmup / 启动收尾等 shell 外围耗时，只作为 fallback。
    row["duration_s"] = row.get("duration_s", row.get("benchmark_duration_s", read_time_info(log_path).get("elapsed_s", "")))

    # canonical 字段若已由日志 parser 提取，不要再被旧 alias 的空值覆盖。
    row["rps"] = row.get("rps", row.get("request_throughput", ""))
    row["input_throughput_tok_s"] = row.get("input_throughput_tok_s", row.get("input_throughput", ""))
    row["generate_throughput_tok_s"] = row.get("generate_throughput_tok_s", row.get("output_throughput", ""))
    row["total_throughput_tok_s"] = row.get("total_throughput_tok_s", row.get("total_throughput", ""))

    for metric in ["e2e_latency", "ttft", "tpot", "itl"]:
        median_key = f"median_{metric}_ms"
        p50_key = f"p50_{metric}_ms"
        if median_key in row and p50_key not in row:
            row[p50_key] = row[median_key]

    exit_code = read_exit_code(log_path)
    row["benchmark_result_found"] = 1 if found_result else 0
    row["case_exit_code"] = exit_code
    if exit_code == 0:
        row["case_status"] = "PASS"
    elif exit_code == "":
        row["case_status"] = "RESULT_FOUND" if found_result else "UNKNOWN"
    else:
        row["case_status"] = "FAIL"

    rows.append(row)

preferred_columns = [
    # 固定核心列：前面统一展示，后面不再追加语义重复字段
    "case_name",
    "input_len",
    "output_len",
    "input_k",
    "output_k",
    "parallel_config",
    "max_running_requests",
    "max_total_tokens",
    "tokens_per_request",
    "token_theoretical_per_dp_bs",      # 纯 KV/token 容量理论 BS
    "effective_theoretical_per_dp_bs",  # 实际理论 BS=min(token theory, max_running_requests)
    "per_dp_bs",
    "num_prompts",
    "max_concurrency",
    "concurrency",
    "peak_output_throughput",
    "peak_concurrency",
    "request_rate",
    "duration_s",
    "rps",
    "input_throughput_tok_s",
    "generate_throughput_tok_s",
    "total_throughput_tok_s",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "accept_length",
    "mean_e2e_latency_ms",
    "case_status",
    "case_exit_code",
    "benchmark_result_found",

    # 独立的 token 统计（不是前面字段的别名）
    "total_input_tokens",
    "total_input_text_tokens",
    "total_generated_tokens",
    "total_generated_tokens_retokenized",

    # 额外统计：只保留与前面 mean/p95/p99 不重复的维度
    "std_e2e_latency_ms",
    "p75_e2e_latency_ms",
    "p90_e2e_latency_ms",
    "p999_e2e_latency_ms",
    "max_e2e_latency_ms",
    "std_ttft_ms",
    "p75_ttft_ms",
    "p90_ttft_ms",
    "p999_ttft_ms",
    "max_ttft_ms",
    "std_tpot_ms",
    "p75_tpot_ms",
    "p90_tpot_ms",
    "p999_tpot_ms",
    "max_tpot_ms",
    "std_itl_ms",
    "p75_itl_ms",
    "p90_itl_ms",
    "p999_itl_ms",
    "max_itl_ms",
]

all_keys = set()
for row in rows:
    all_keys.update(row.keys())
# 语义重复字段不要再次追加到 CSV。
# 左侧为后端/原始字段，右侧表示其语义已经由前面的标准列覆盖。
duplicate_aliases = {
    # workload
    "random_input_len": "input_len",
    "random_output_len": "output_len",
    "input_length": "input_len",
    "output_length": "output_len",

    # requests / concurrency
    "successful_requests": "num_prompts",  # 实际成功数不再重复出列；num_prompts 为配置请求数
    "completed": "num_prompts",
    "max_concurrent_requests": "max_concurrency",
    "max_request_concurrency": "max_concurrency",
    "global_concurrency": "max_concurrency",

    # duration / request rate
    "duration": "duration_s",
    "benchmark_duration": "duration_s",
    "benchmark_duration_s": "duration_s",
    "elapsed_s": "duration_s",
    "request_throughput": "rps",
    "request_throughput_req_s": "rps",
    "traffic_request_rate": "request_rate",

    # throughput
    "input_throughput": "input_throughput_tok_s",
    "input_token_throughput": "input_throughput_tok_s",
    "output_throughput": "generate_throughput_tok_s",
    "output_throughput_tok_s": "generate_throughput_tok_s",
    "generation_throughput": "generate_throughput_tok_s",
    "generation_throughput_tok_s": "generate_throughput_tok_s",
    "total_token_throughput": "total_throughput_tok_s",
    "total_token_throughput_tok_s": "total_throughput_tok_s",
    "total_throughput": "total_throughput_tok_s",

    # median 与 p50 重复，只保留一套；最终表两者都不追加
    "median_e2e_latency_ms": "p50_e2e_latency_ms",
    "p50_e2e_latency_ms": "median_e2e_latency_ms",
    "median_ttft_ms": "p50_ttft_ms",
    "p50_ttft_ms": "median_ttft_ms",
    "median_tpot_ms": "p50_tpot_ms",
    "p50_tpot_ms": "median_tpot_ms",
    "median_itl_ms": "p50_itl_ms",
    "p50_itl_ms": "median_itl_ms",

    # theory / trace / file
    "request_theoretical_per_dp_bs": "hidden_theory",
    "theoretical_global_bs": "hidden_theory",
    "theory_limiting_factor": "hidden_theory",
    "start_time": "trace_only",
    "end_time": "trace_only",
    "log_file": "trace_only",
    "json_file": "trace_only",
    "jsonl_file": "trace_only",
    "source_file": "trace_only",
    "theoretical_per_dp_bs": "effective_theoretical_per_dp_bs",
    "bs_vs_theory_ratio": "hidden_theory",
}

# 这些属于纯辅助/内部字段，即使没有 alias，也不需要出现在最终 CSV。
hidden_columns = {
    "dp_size",
    "theoretical_per_dp_bs",
    "request_theoretical_per_dp_bs",
    "theoretical_global_bs",
    "theory_limiting_factor",
    "bs_vs_theory_ratio",
    "start_time",
    "end_time",
    "elapsed_s",
    "log_file",
    "json_file",
    "jsonl_file",
    "source_file",
    "backend",
    "traffic_request_rate",
    "max_request_concurrency",
    "successful_requests",
    "benchmark_duration_s",
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "median_e2e_latency_ms",
    "p50_e2e_latency_ms",
    "median_ttft_ms",
    "p50_ttft_ms",
    "median_tpot_ms",
    "p50_tpot_ms",
    "median_itl_ms",
    "p50_itl_ms",
}

extra_columns = sorted(
    key for key in all_keys
    if key not in preferred_columns
    and key not in hidden_columns
    and key not in duplicate_aliases
)

columns = preferred_columns + extra_columns

tmp_csv = Path(str(output_csv) + ".tmp")
with tmp_csv.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
tmp_csv.replace(output_csv)

print()
print("============================================================")
print(f"[MERGE] 已汇总 BS 数量 : {len(rows)}")
print(f"[MERGE] CSV 指标列数   : {len(columns)}")
print(f"[MERGE] 输出           : {output_csv}")
print("============================================================")
PY

}



# ============================================================
#
# 目的：
# - 避免 Excel 把 p90_e2e_latency_ms 等纯数字误显示成日期
# - *_ms / duration_s / rps / throughput 等显式设置为数值格式
# - CSV 仍然作为原始结果保留
# ============================================================


# ============================================================
# 10. Ctrl+C / TERM / 正常退出
# ============================================================

on_exit() {
  local exit_code=$?
  trap - EXIT
  echo
  echo "============================================================"
  echo "脚本退出，正在汇总已有结果..."
  echo "============================================================"
  merge_results || true
  echo
  echo "结果目录：${out_dir}"
  echo "JSONL   ：${jsonl_dir}"
  echo "LOG     ：${log_dir}"
  echo "CSV     ：${out_dir}/sum_all.csv"
  exit "${exit_code}"
}

trap 'exit 130' INT
trap 'exit 143' TERM
trap on_exit EXIT


# ============================================================
# 11. 拉 metrics + 打印配置
# ============================================================

load_server_info || true
load_server_capacity
validate_fixed_mode
python3 - "${SLA_RULES}" <<'PY2' || exit 1
import sys
for raw in sys.argv[1].split(','):
    raw=raw.strip()
    if not raw: continue
    p=raw.split(':')
    if len(p)!=3 or not p[0]:
        print(f'[ERROR] 非法 SLA_RULES 项: {raw}',file=sys.stderr); raise SystemExit(1)
    try: float(p[1]); float(p[2])
    except: print(f'[ERROR] SLA target/near 非数字: {raw}',file=sys.stderr); raise SystemExit(1)
PY2

if [[ ! "${NUM_PROMPTS_MULTIPLIER}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] NUM_PROMPTS_MULTIPLIER 必须是正整数，当前=${NUM_PROMPTS_MULTIPLIER}"
  exit 1
fi

echo
echo "============================================================"
echo "Benchmark Configuration"
echo "============================================================"
echo
echo "BASE_URL         : ${BASE_URL}"
echo "MODEL            : ${MODEL}"
echo "DP_SIZE          : ${DP_SIZE}"
echo "MAX_TOTAL_TOKENS : ${MAX_TOTAL_TOKENS}"
echo "MAX_RUNNING_REQS  : ${MAX_RUNNING_REQUESTS:-0}"
echo "PARALLEL_CONFIG   : ${PARALLEL_CONFIG:-UNKNOWN}"
echo "CUDA_GRAPH_BS     : ${CUDA_GRAPH_DECODE_BS:-FALLBACK_DP_LAYOUT}"
echo "SLA DENSE          : enable=${SLA_DENSE_ENABLE} rules=${SLA_RULES} mode=inline"
echo "USE_FAKE_PREFILL : ${USE_FAKE_PREFILL}"
echo "TEST_MODE        : $(get_test_mode_name)"
echo "FIXED_PER_DP_BS  : ${FIXED_PER_DP_BS:-AUTO}"
echo "FIXED_CONCURRENCY: ${FIXED_CONCURRENCY:-AUTO}"
echo "NUM_PROMPTS_MULT : ${NUM_PROMPTS_MULTIPLIER}"
echo
echo "Output Directory : ${out_dir}"
echo


# ============================================================
# TPOT=50ms 附近自适应加密
# ============================================================

run_one_bs() {
  local per_dp_bs="$1"
  local global_concurrency=$((DP_SIZE * per_dp_bs))
  local num_prompts=$((global_concurrency * NUM_PROMPTS_MULTIPLIER))
  local warmup_requests=0

  if (( per_dp_bs == 1 )); then
    warmup_requests=$((DP_SIZE * 16))
  fi

  local base_name="${case_name}_in${input_len}_out${output_len}_perdp${per_dp_bs}_global${global_concurrency}"
  local output_file="${jsonl_dir}/${base_name}.jsonl"
  local log_file="${log_dir}/${base_name}.log"
  local exit_file="${log_file}.exitcode"

  # 已有成功结果则跳过，方便续跑/补点。
  if [[ -s "${output_file}" && -f "${exit_file}" ]] && [[ "$(cat "${exit_file}" 2>/dev/null)" == "0" ]]; then
    echo "[SKIP] 已完成 ${case_name} per_dp_bs=${per_dp_bs}"
    return 0
  fi

  local bs_start_time bs_start_epoch bs_end_time bs_end_epoch bs_elapsed_s
  local case_exit_code
  bs_start_time="$(date '+%F %T')"
  bs_start_epoch="$(date +%s)"

  echo
  echo "============================================================"
  echo "CASE               : ${case_name}"
  echo "Input Length       : ${input_len}"
  echo "Output Length      : ${output_len}"
  echo "Theoretical BS/DP  : ${theoretical_bs}"
  echo "Max Running Reqs   : ${MAX_RUNNING_REQUESTS:-0}"
  echo "Per-DP BS          : ${per_dp_bs}"
  echo "Global Concurrency : ${global_concurrency}"
  echo "Num Prompts        : ${num_prompts} (${global_concurrency} x ${NUM_PROMPTS_MULTIPLIER})"
  echo "Warmup Requests    : ${warmup_requests}"
  echo "Start Time         : ${bs_start_time}"
  echo "JSONL              : ${output_file}"
  echo "LOG                : ${log_file}"
  echo "============================================================"
  echo

  cmd=(
    python3 -m sglang.bench_serving
    --backend sglang
    --base-url "${BASE_URL}"
    --model "${MODEL}"
    --dataset-name random-ids
    --tokenize-prompt
    --num-prompts "${num_prompts}"
    --random-input-len "${input_len}"
    --random-output-len "${output_len}"
    --random-range-ratio 1.0
    --request-rate inf
    --max-concurrency "${global_concurrency}"
    --warmup-requests "${warmup_requests}"
    --extra-request-body
    "{\"sampling_params\":{\"temperature\":0.6,\"top_p\":0.95,\"max_new_tokens\":${output_len},\"ignore_eos\":true}}"
    --output-details
    --output-file "${output_file}"
    --disable-tqdm
  )

  if (( USE_FAKE_PREFILL == 1 )); then
    cmd+=(--fake-prefill)
  fi

  "${cmd[@]}" 2>&1 | tee "${log_file}"
  case_exit_code="${PIPESTATUS[0]}"
  printf "%s\n" "${case_exit_code}" > "${exit_file}"

  bs_end_time="$(date '+%F %T')"
  bs_end_epoch="$(date +%s)"
  bs_elapsed_s=$((bs_end_epoch - bs_start_epoch))

  {
    echo "start_time=${bs_start_time}"
    echo "end_time=${bs_end_time}"
    echo "elapsed_s=${bs_elapsed_s}"
  } > "${log_file}.time"

  if (( case_exit_code == 0 )); then
    echo "[PASS] ${case_name} per_dp_bs=${per_dp_bs} global=${global_concurrency} elapsed=${bs_elapsed_s}s"
  else
    echo "[FAIL] ${case_name} per_dp_bs=${per_dp_bs} global=${global_concurrency} exit=${case_exit_code}"
  fi

  merge_results
}



# ============================================================
# Inline TPOT dense helper
# 当前 BS 测完后立即检查 SLA。
# 任一指标进入 target±near，就把当前 BS 到下一个基础 BS 之间的整数 BS 连续跑完。
# ============================================================
is_sla_near_target() {
  local current_case="$1"
  local current_bs="$2"
  local csv_file="${out_dir}/sum_all.csv"
  [[ -s "${csv_file}" ]] || return 1
  python3 - "${csv_file}" "${current_case}" "${current_bs}" "${SLA_RULES}" <<'PY2'
import csv,sys,math
csv_file,case_name,bs_s,rules_text=sys.argv[1:5]
bs=int(bs_s)
row=None
with open(csv_file,newline='',encoding='utf-8-sig') as f:
    for r in csv.DictReader(f):
        if r.get('case_name')!=case_name: continue
        try: rbs=int(float(r.get('per_dp_bs','')))
        except: continue
        if rbs==bs: row=r
if not row: raise SystemExit(1)
hits=[]; states=[]
for raw in rules_text.split(','):
    raw=raw.strip()
    if not raw: continue
    p=raw.split(':')
    if len(p)!=3:
        states.append(raw+'=INVALID'); continue
    metric,target_s,near_s=p
    try:
        target=float(target_s); near=float(near_s); value=float(row.get(metric,''))
        if not math.isfinite(value): raise ValueError
    except:
        states.append(metric+'=NA'); continue
    hit=abs(value-target)<=near
    states.append(f'{metric}={value:.4f} target={target:g}±{near:g} hit={hit}')
    if hit: hits.append(metric)
print(f'[SLA-CHECK] bs={bs} '+' | '.join(states),file=sys.stderr)
raise SystemExit(0 if hits else 1)
PY2
}

# ============================================================
# 12. 主循环
# ============================================================

for case_config in "${CASES[@]}"; do
  read -r input_k output_k <<< "${case_config}"

  input_len=$((input_k * 1024))
  output_len=$((output_k * 1024))
  input_name="$(format_k_name "${input_k}")"
  output_name="$(format_k_name "${output_k}")"
  case_name="${input_name}_${output_name}"

  tokens_per_request=$((input_len + output_len))
  token_theoretical_bs="$(get_token_theoretical_bs "${input_k}" "${output_k}")"
  theoretical_bs="$(get_theoretical_bs "${input_k}" "${output_k}")"
  theory_limit_factor="$(get_theory_limit_factor "${input_k}" "${output_k}")"
  theoretical_global_bs="$(awk -v bs="${theoretical_bs}" -v dp="${DP_SIZE}" 'BEGIN { printf "%.3f", bs * dp }')"
  auto_bs_list="$(get_bs_list "${input_k}" "${output_k}")"
  bs_list="$(get_test_bs_list "${auto_bs_list}")"

  echo
  echo
  echo "############################################################"
  echo "# CASE              : ${case_name}"
  echo "# Input             : ${input_k}K = ${input_len}"
  echo "# Output            : ${output_k}K = ${output_len}"
  echo "# Tokens / Request  : ${tokens_per_request}"
  echo "# DP Size           : ${DP_SIZE}"
  echo "# Max Total Tokens  : ${MAX_TOTAL_TOKENS}"
  echo "# Max Running Reqs  : ${MAX_RUNNING_REQUESTS:-0}"
  echo "# Token Theory/DP   : ${token_theoretical_bs}"
  echo "# Effective Theory  : ${theoretical_bs}"
  echo "# Theory Global BS  : ${theoretical_global_bs}"
  echo "# Limiting Factor   : ${theory_limit_factor}"
  echo "# Test Mode         : $(get_test_mode_name)"
  echo "# Auto BS           : ${auto_bs_list}"
  if [[ -n "${CUDA_GRAPH_DECODE_BS}" ]]; then
    echo "# BS Source         : server_info.cuda_graph_config.decode.bs + theory ±1/±2"
  else
    echo "# BS Source         : DP_SIZE fallback layout + theory ±1/±2"
  fi
  echo "# Test BS           : ${bs_list}"
  echo "############################################################"

  # 单阶段顺序执行：
  # 1) 按 CUDA Graph / fallback / theory 生成的基础 BS 从小到大跑
  # 2) 每个 BS 跑完立即检查 SLA
  # 3) 若当前点命中 target±near，则直接连续跑到下一个基础 BS 前一位
  #    例如基础点 ...18,22,24...，BS18 命中后立即跑 19/20/21，然后继续 22、24...
  mapfile -t base_bs_array < <(printf "%s\n" ${bs_list} | awk 'NF' | sort -n)

  for ((i=0; i<${#base_bs_array[@]}; i++)); do
    per_dp_bs="${base_bs_array[$i]}"
    run_one_bs "${per_dp_bs}"

    # 最后一个基础点没有“下一个基础点”，无需区间加密。
    if (( i + 1 >= ${#base_bs_array[@]} )); then
      continue
    fi

    if [[ "${SLA_DENSE_ENABLE}" != "1" ]]; then
      continue
    fi

    next_base_bs="${base_bs_array[$((i + 1))]}"

    # 当前点 Mean/P99 任一接近目标，立即连续测试到下一个基础点之前。
    if is_sla_near_target "${case_name}" "${per_dp_bs}"; then
      if (( next_base_bs > per_dp_bs + 1 )); then
        echo
        echo "############################################################"
        echo "# Inline SLA Dense Scan"
        echo "# Current BS        : ${per_dp_bs}"
        echo "# Next Base BS      : ${next_base_bs}"
        echo "# SLA Rules         : ${SLA_RULES}"
        echo "# Dense BS          : $((per_dp_bs + 1)) .. $((next_base_bs - 1))"
        echo "############################################################"

        for ((dense_bs=per_dp_bs + 1; dense_bs<next_base_bs; dense_bs++)); do
          run_one_bs "${dense_bs}"
        done
      fi
    fi
  done

done


# ============================================================
# 13. 完成
# ============================================================

echo
echo
echo "============================================================"
echo "全部 Benchmark 执行结束"
echo "============================================================"
echo
echo "结果目录："
echo "${out_dir}"
echo
echo "JSONL："
echo "${jsonl_dir}"
echo
echo "Client LOG："
echo "${log_dir}"
echo
echo "汇总 CSV："
echo "${out_dir}/sum_all.csv"
echo
echo
