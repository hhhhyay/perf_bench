#!/usr/bin/env bash
# 配置来源优先级：
#   1) /server_info
#   2) SERVER_LOG（尤其用于 max_num_seqs）
#   3) /metrics
#   4) 手动环境变量覆盖/兜底
# ============================================================
# vLLM Prefix-Salt 标准容量压测
#
# 规则：
# 1) 不清 Prefix Cache。
# 2) 每个 Case / BS 使用唯一 cache_salt：
#      ${RUN_ID}_${case_name}_perdp${per_dp_bs}
#    从而隔离不同 BS、不同 run 的 Prefix Cache。
# 3) /metrics 只在启动阶段读取容量信息；正式 benchmark 不轮询。
# 4) CSV 前置常用指标顺序固定，并与 SGLang 客户端口径对齐。
# 5) 同义字段统一到标准列；后面只追加真正非重复的新指标。
# 6) 理论 BS = min(token theory, max_num_seqs)，若 max_num_seqs 不可得则退回 token theory。
# 7) 通用 SLA inline 加密：当前 BS 命中任一 SLA 规则后，立即连续跑到下一个基础 BS。
# ============================================================

set -uo pipefail

run_id="$(date +%Y%m%d_%H%M%S)"
out_dir="./all_result/bench_vllm_prefix_salt_${run_id}"
jsonl_dir="${out_dir}/jsonl"
log_dir="${out_dir}/logs"
mkdir -p "${jsonl_dir}" "${log_dir}"
STARTUP_READY=0


# ============================================================
# 服务配置
# ============================================================

BASE_URL="${BASE_URL:-http://12.12.12.108:8332}"
SERVER_INFO_URL="${SERVER_INFO_URL:-${BASE_URL%/}/server_info}"
SERVER_LOG="${SERVER_LOG:-}"   # 可选：vLLM 服务端启动日志路径，用于提取 max_num_seqs 等未暴露配置
MODEL="${MODEL:-/model/Qwen3-8B}"
CACHE_MODE="prefix_salt_no_flush"

SLA_DENSE_ENABLE="${SLA_DENSE_ENABLE:-1}"
# 通用 SLA 规则：metric:target:near，多个规则用逗号分隔
# 示例：
#   mean_tpot_ms:30:2
#   p99_tpot_ms:40:3
#   mean_ttft_ms:1000:100
#   p99_ttft_ms:1500:100
# 默认保持 TPOT 规则：
SLA_RULES="${SLA_RULES:-mean_tpot_ms:50:5,p99_tpot_ms:75:5}"

# 请求数量倍率：num_prompts = max_concurrency * NUM_PROMPTS_MULTIPLIER
NUM_PROMPTS_MULTIPLIER="${NUM_PROMPTS_MULTIPLIER:-1}"

# 一般留空，由 metrics 自动识别
DP_SIZE="${DP_SIZE:-}"
KV_TOKENS="${KV_TOKENS:-}"

# 对标 SGLang max_running_requests。
# 优先尝试从 /metrics 中的 max_num_seqs label/metric 自动识别；
# 当前 vLLM 若未暴露该配置，可手动传：
#   MAX_NUM_SEQS=128 bash this_script.sh
# 注意：
#   VLLM_MAX_N_SEQUENCES != max_num_seqs
#   前者是 OpenAI 参数 n 的最大值，不可用于 effective theory BS。
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"

# vLLM 不同版本暴露的配置字段不同；以下字段优先从 /metrics label 尝试读取，
# 获取不到时可手动覆盖。最终只合并成 parallel_config 一列。
TP_SIZE="${TP_SIZE:-}"
PP_SIZE="${PP_SIZE:-}"
EP_SIZE="${EP_SIZE:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-}"
PARALLEL_CONFIG="${PARALLEL_CONFIG:-}"

# ============================================================
# 固定并发接口（可选）
#
# 默认留空：按 theoretical BS 自动扫描。
#
# 固定 per-DP BS：
#   FIXED_PER_DP_BS=12 bash bench_vllm_prefix_salt_standard_v3.sh
#
# 固定全局并发：
#   FIXED_CONCURRENCY=192 bash bench_vllm_prefix_salt_standard_v3.sh
#
# 也支持多个点：
#   FIXED_PER_DP_BS="10,12,14,16"
#   FIXED_CONCURRENCY="160 192 224 256"
#
# 注意：
# - 两者不能同时设置。
# - FIXED_CONCURRENCY 必须能被 DP_SIZE 整除。
# ============================================================
FIXED_PER_DP_BS="${FIXED_PER_DP_BS:-}"
FIXED_CONCURRENCY="${FIXED_CONCURRENCY:-}"

# 压测结果百分位（可覆盖）
METRIC_PERCENTILES="${METRIC_PERCENTILES:-95,99}"
PERCENTILE_METRICS="${PERCENTILE_METRICS:-ttft,tpot,itl,e2el}"



# ============================================================
# Input / Output（单位 K）
# ============================================================

CASES=(
  "1 1"
  "8 1"
  "16 1"
  "32 1"
  "64 16"
  "128 1"
  "1024 1"
)


# DP<=3 时使用较稀疏但更实用的基础点：
#   1/2/4/8/16/32/48/64/96/128
# 理论 BS > 128 后继续按 32 的步长扩展：
#   160/192/224/256/...
LOW_DP_BS=(1 2 4 8 16 32 48 64 96 128)

# DP>3 时使用更密的 per-DP BS 基础点：
#   1/2/4/6/8/10/12/14/16/20/24/28/32/40/48/56/64
# 理论 BS > 64 后继续按 16 的步长扩展：
#   80/96/112/128/...
DENSE_DP_BS=(1 2 4 6 8 10 12 14 16 20 24 28 32 40 48 56 64)
THEORY_DELTAS=(-2 -1 0 1 2)


# ============================================================
# metrics 解析
# ============================================================

fetch_vllm_metrics() {
  curl -s --connect-timeout 5 --max-time 30 "${BASE_URL%/}/metrics" || true
}

# 等价于：
# m=$(curl -s .../metrics)
# DP_SIZE=$(echo "$m" | grep '^vllm:cache_config_info{' | grep -oE 'engine="[0-9]+"' | sort -u | wc -l)
# KV_TOKENS=$(echo "$m" | grep '^vllm:cache_config_info{' | sed ... | awk '{print $1*$2}')

load_vllm_server_info() {
  local info
  info="$(curl -fsS --max-time 5 "${SERVER_INFO_URL}" 2>/dev/null || true)"
  if [[ -z "${info}" ]]; then
    echo "[WARN] /server_info 不可用: ${SERVER_INFO_URL}；TP/DP/PP/KV/CUDA Graph 后续使用 metrics/环境变量/fallback"
    return 0
  fi

  local parsed
  parsed="$(
    python3 - "${info}" <<'PY'
import sys, json, re
try:
    d = json.loads(sys.argv[1])
except Exception:
    raise SystemExit(1)

cfg = d.get("vllm_config", "")
if not isinstance(cfg, str):
    cfg = str(cfg)

def pick_int(name):
    m = re.search(rf'\b{name}=([0-9]+)\b', cfg)
    return m.group(1) if m else ""

def pick_str(name):
    m = re.search(rf'\b{name}=([^,\s]+)', cfg)
    return m.group(1) if m else ""

tp = pick_int("tensor_parallel_size")
dp = pick_int("data_parallel_size")
pp = pick_int("pipeline_parallel_size")
ep = pick_int("expert_parallel_size")
kv = pick_str("kv_cache_dtype")

# cudagraph_capture_sizes=[...]
cg = ""
m = re.search(r"'cudagraph_capture_sizes':\s*\[([^\]]*)\]", cfg)
if m:
    vals = []
    for x in m.group(1).split(","):
        x = x.strip()
        if x.isdigit():
            vals.append(x)
    cg = ",".join(vals)

# Explicitly DO NOT treat VLLM_MAX_N_SEQUENCES as max_num_seqs.
# Try real max_num_seqs only if it appears in the resolved config string.
mns = ""
m = re.search(r'\bmax_num_seqs=([0-9]+)\b', cfg)
if m:
    mns = m.group(1)

print("|".join([tp, dp, pp, ep, kv, mns, cg]))
PY
  )"

  if [[ -z "${parsed}" ]]; then
    echo "[WARN] /server_info 返回了数据，但解析失败"
    return 0
  fi

  local _tp _dp _pp _ep _kv _mns _cg
  IFS='|' read -r _tp _dp _pp _ep _kv _mns _cg <<< "${parsed}"

  [[ -z "${TP_SIZE}" && -n "${_tp}" ]] && TP_SIZE="${_tp}"
  [[ -z "${DP_SIZE}" && -n "${_dp}" ]] && DP_SIZE="${_dp}"
  [[ -z "${PP_SIZE}" && -n "${_pp}" ]] && PP_SIZE="${_pp}"
  [[ -z "${EP_SIZE}" && -n "${_ep}" ]] && EP_SIZE="${_ep}"
  [[ -z "${KV_CACHE_DTYPE}" && -n "${_kv}" ]] && KV_CACHE_DTYPE="${_kv}"
  [[ -z "${MAX_NUM_SEQS}" && -n "${_mns}" ]] && MAX_NUM_SEQS="${_mns}"
  [[ -z "${CUDA_GRAPH_BS}" && -n "${_cg}" ]] && CUDA_GRAPH_BS="${_cg}"

  # Sanity check: KV dtype should not accidentally become a CUDA graph list.
  if [[ "${KV_CACHE_DTYPE:-}" == *,* ]]; then
    echo "[WARN] KV_CACHE_DTYPE 解析异常: ${KV_CACHE_DTYPE}; 清空并回退"
    KV_CACHE_DTYPE=""
  fi

  echo "[INFO] server_info:"
  echo "       TP_SIZE=${TP_SIZE:-N/A}"
  echo "       DP_SIZE=${DP_SIZE:-N/A}"
  echo "       PP_SIZE=${PP_SIZE:-N/A}"
  echo "       EP_SIZE=${EP_SIZE:-N/A}"
  echo "       KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-N/A}"
  echo "       MAX_NUM_SEQS=${MAX_NUM_SEQS:-N/A}"
  echo "       CUDA_GRAPH_BS=${CUDA_GRAPH_BS:-N/A}"
}


load_vllm_server_log_config() {
  [[ -n "${SERVER_LOG}" ]] || return 0
  if [[ ! -r "${SERVER_LOG}" ]]; then
    echo "[WARN] SERVER_LOG 不可读: ${SERVER_LOG}"
    return 0
  fi

  local mns mbt tp pp dp kv
  mns="$(
    grep -Eio 'max[_-]num[_-]seqs[=: ]+[0-9]+' "${SERVER_LOG}" \
      | tail -1 | grep -Eo '[0-9]+' || true
  )"
  mbt="$(
    grep -Eio 'max[_-]num[_-]batched[_-]tokens[=: ]+[0-9]+' "${SERVER_LOG}" \
      | tail -1 | grep -Eo '[0-9]+' || true
  )"
  tp="$(
    grep -Eio '(tensor_parallel_size|tensor-parallel-size|tp_size)[=: ]+[0-9]+' "${SERVER_LOG}" \
      | tail -1 | grep -Eo '[0-9]+' || true
  )"
  pp="$(
    grep -Eio '(pipeline_parallel_size|pipeline-parallel-size|pp_size)[=: ]+[0-9]+' "${SERVER_LOG}" \
      | tail -1 | grep -Eo '[0-9]+' || true
  )"
  dp="$(
    grep -Eio '(data_parallel_size|data-parallel-size|dp_size)[=: ]+[0-9]+' "${SERVER_LOG}" \
      | tail -1 | grep -Eo '[0-9]+' || true
  )"
  kv="$(
    grep -Eio 'kv_cache_dtype[=: ]+[^, ]+' "${SERVER_LOG}" \
      | tail -1 | sed -E 's/.*kv_cache_dtype[=: ]+//' || true
  )"

  [[ -z "${MAX_NUM_SEQS}" && -n "${mns}" ]] && MAX_NUM_SEQS="${mns}"
  [[ -z "${TP_SIZE}" && -n "${tp}" ]] && TP_SIZE="${tp}"
  [[ -z "${PP_SIZE}" && -n "${pp}" ]] && PP_SIZE="${pp}"
  [[ -z "${DP_SIZE}" && -n "${dp}" ]] && DP_SIZE="${dp}"
  [[ -z "${KV_CACHE_DTYPE}" && -n "${kv}" ]] && KV_CACHE_DTYPE="${kv}"

  if [[ -n "${mns}${mbt}${tp}${pp}${dp}${kv}" ]]; then
    echo "[INFO] server log config:"
    echo "       MAX_NUM_SEQS=${MAX_NUM_SEQS:-N/A}"
    echo "       MAX_NUM_BATCHED_TOKENS=${mbt:-N/A}"
    echo "       TP_SIZE=${TP_SIZE:-N/A}"
    echo "       DP_SIZE=${DP_SIZE:-N/A}"
    echo "       PP_SIZE=${PP_SIZE:-N/A}"
    echo "       KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-N/A}"
  fi
}

load_vllm_capacity() {
  local m
  local dp
  local kv
  local max_num_seqs
  local tp_size pp_size ep_size kv_cache_dtype

  echo
  echo "============================================================"
  echo "自动识别 vLLM DP_SIZE / KV_TOKENS / MAX_NUM_SEQS"
  echo "============================================================"
  echo "URL: ${BASE_URL%/}/metrics"
  echo

  m="$(fetch_vllm_metrics)"
  if [[ -z "${m}" ]]; then
    echo "[ERROR] 无法访问 metrics: ${BASE_URL%/}/metrics"
    if [[ -z "${KV_TOKENS}" ]]; then
      echo "[ERROR] KV_TOKENS 无法自动获取；请先确认 BASE_URL，或手动设置 KV_TOKENS=..."
      echo "[HINT] 当前 BASE_URL=${BASE_URL}"
      exit 1
    fi
    [[ -z "${DP_SIZE}" ]] && DP_SIZE=1
    echo "[WARN] 使用已有/手动容量：DP_SIZE=${DP_SIZE} KV_TOKENS=${KV_TOKENS} MAX_NUM_SEQS=${MAX_NUM_SEQS:-N/A}"
    return 0
  fi

  printf "%s\n" "${m}" > "${out_dir}/metrics_snapshot.prom"

  # 从 cache_config_info 的 engine 标签统计 DP 数。
  dp="$(
    printf "%s
" "${m}" \
      | grep '^vllm:cache_config_info{' \
      | grep -oE 'engine="[0-9]+"' \
      | sort -u \
      | wc -l \
      | tr -d ' '
  )"

  kv="$(
    printf "%s\n" "${m}" \
      | grep '^vllm:cache_config_info{' \
      | sed -n 's/.*block_size="\([0-9]*\)".*num_gpu_blocks="\([0-9]*\)".*/\1 \2/p' \
      | head -1 \
      | awk '{print $1*$2}'
  )"

  # 兼容 label 顺序：num_gpu_blocks 在前、block_size 在后
  if [[ -z "${kv}" ]]; then
    kv="$(
      printf "%s\n" "${m}" \
        | grep '^vllm:cache_config_info{' \
        | sed -n 's/.*num_gpu_blocks="\([0-9]*\)".*block_size="\([0-9]*\)".*/\1 \2/p' \
        | head -1 \
        | awk '{print $1*$2}'
    )"
  fi

  # 尝试从任何带 max_num_seqs label 的 info metric 中提取。
  # 兼容例如：
  #   ...{...,max_num_seqs="128",...} 1
  max_num_seqs="$(
    printf "%s\n" "${m}" \
      | grep -E 'max_num_seqs="[0-9]+"' \
      | grep -oE 'max_num_seqs="[0-9]+"' \
      | head -1 \
      | grep -oE '[0-9]+' \
      || true
  )"

  # 兼容未来/自定义版本直接暴露数值 metric 的情况：
  #   vllm:max_num_seqs 128
  if [[ -z "${max_num_seqs}" ]]; then
    max_num_seqs="$(
      printf "%s\n" "${m}" \
        | awk '/^vllm:max_num_seqs([ {]|$)/ {print $NF; exit}' \
        | sed 's/\..*$//' \
        || true
    )"
  fi


  # best-effort：不同 vLLM 版本 label 名称可能不同。
  tp_size="$(
    printf "%s\n" "${m}" \
      | grep -oE '(tensor_parallel_size|tp_size)="[0-9]+"' \
      | head -1 | grep -oE '[0-9]+' || true
  )"
  pp_size="$(
    printf "%s\n" "${m}" \
      | grep -oE '(pipeline_parallel_size|pp_size)="[0-9]+"' \
      | head -1 | grep -oE '[0-9]+' || true
  )"
  ep_size="$(
    printf "%s\n" "${m}" \
      | grep -oE '(expert_parallel_size|ep_size)="[0-9]+"' \
      | head -1 | grep -oE '[0-9]+' || true
  )"
  kv_cache_dtype="$(
    printf "%s\n" "${m}" \
      | grep -oE '(kv_cache_dtype|cache_dtype)="[^"]+"' \
      | head -1 | sed -E 's/^[^=]+="([^"]+)"/\1/' || true
  )"

  echo "detected DP_SIZE     = ${dp:-N/A}"
  echo "detected KV_TOKENS   = ${kv:-N/A}"
  echo "detected MAX_NUM_SEQS= ${max_num_seqs:-N/A}"

  if [[ -n "${dp}" && "${dp}" -gt 0 ]]; then
    if [[ -n "${DP_SIZE}" && "${DP_SIZE}" != "${dp}" ]]; then
      echo "[WARN] 手动 DP_SIZE=${DP_SIZE} 与 metrics 提取值 ${dp} 不一致"
      echo "[WARN] 优先使用 metrics 提取值，避免理论 BS / global concurrency 算错"
    fi
    DP_SIZE="${dp}"
    echo "[INFO] 自动识别 DP_SIZE=${DP_SIZE} (source=metrics)"
  elif [[ -n "${DP_SIZE}" ]]; then
    echo "[WARN] metrics 未识别到 DP_SIZE，回退使用手动 DP_SIZE=${DP_SIZE}"
  else
    DP_SIZE=1
    echo "[WARN] metrics 未识别到 DP_SIZE，且未手动设置；按 DP_SIZE=1 回退"
  fi

  if [[ -z "${KV_TOKENS}" ]]; then
    if [[ -z "${kv}" || "${kv}" -le 0 ]]; then
      echo "[ERROR] 未识别到 KV_TOKENS，请手动设置 KV_TOKENS=..."
      exit 1
    fi
    KV_TOKENS="${kv}"
  fi

  if [[ -n "${MAX_NUM_SEQS}" ]]; then
    if [[ -n "${max_num_seqs}" && "${max_num_seqs}" =~ ^[0-9]+$ && "${max_num_seqs}" -gt 0 && "${MAX_NUM_SEQS}" != "${max_num_seqs}" ]]; then
      echo "[WARN] 已有 MAX_NUM_SEQS=${MAX_NUM_SEQS} 与 metrics=${max_num_seqs} 不一致；保留已有值"
    fi
  elif [[ -n "${max_num_seqs}" && "${max_num_seqs}" =~ ^[0-9]+$ && "${max_num_seqs}" -gt 0 ]]; then
    MAX_NUM_SEQS="${max_num_seqs}"
  else
    echo "[WARN] 未获得 max_num_seqs；MAX_NUM_SEQS=N/A，effective theory 暂按 token theory"
  fi


  [[ -z "${TP_SIZE}" && -n "${tp_size}" ]] && TP_SIZE="${tp_size}"
  [[ -z "${PP_SIZE}" && -n "${pp_size}" ]] && PP_SIZE="${pp_size}"
  [[ -z "${EP_SIZE}" && -n "${ep_size}" ]] && EP_SIZE="${ep_size}"
  [[ -z "${KV_CACHE_DTYPE}" && -n "${kv_cache_dtype}" ]] && KV_CACHE_DTYPE="${kv_cache_dtype}"

  if [[ -z "${PARALLEL_CONFIG}" ]]; then
    parts=()
    [[ -n "${TP_SIZE}" ]] && parts+=("TP${TP_SIZE}")
    parts+=("DP${DP_SIZE}")
    [[ -n "${PP_SIZE}" ]] && parts+=("PP${PP_SIZE}")
    [[ -n "${EP_SIZE}" ]] && parts+=("EP${EP_SIZE}")
    [[ -n "${KV_CACHE_DTYPE}" ]] && parts+=("KV${KV_CACHE_DTYPE}")
    PARALLEL_CONFIG="$(IFS=-; echo "${parts[*]}")"
  fi

  echo
  echo "[INFO] 最终 DP_SIZE=${DP_SIZE}  KV_TOKENS=${KV_TOKENS}  MAX_NUM_SEQS=${MAX_NUM_SEQS:-N/A}"
  echo "[INFO] PARALLEL_CONFIG=${PARALLEL_CONFIG:-UNKNOWN}"
}

# ============================================================
# 固定并发参数解析
# ============================================================

normalize_int_list() {
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
        echo "[ERROR] 请改用可整除的全局并发，或直接指定 FIXED_PER_DP_BS"
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

# ============================================================
# BS 列表
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

get_token_theoretical_bs() {
  local input_k="$1"
  local output_k="$2"
  awk -v max_tokens="${KV_TOKENS}" -v input_k="${input_k}" -v output_k="${output_k}" \
    'BEGIN { printf "%.3f", max_tokens / ((input_k + output_k) * 1024) }'
}

get_theoretical_bs() {
  local input_k="$1"
  local output_k="$2"
  local token_bs
  token_bs="$(get_token_theoretical_bs "${input_k}" "${output_k}")"

  if [[ "${MAX_NUM_SEQS:-}" =~ ^[0-9]+$ ]] && (( MAX_NUM_SEQS > 0 )); then
    awk -v t="${token_bs}" -v m="${MAX_NUM_SEQS}" \
      'BEGIN { if (t < m) printf "%.3f", t; else printf "%.3f", m }'
  else
    printf "%s
" "${token_bs}"
  fi
}

get_bs_list() {
  local input_k="$1"
  local output_k="$2"
  local tokens_per_request
  local token_floor
  local theoretical_floor
  local scan_max
  local bs
  local delta

  tokens_per_request=$(((input_k + output_k) * 1024))
  token_floor=$((KV_TOKENS / tokens_per_request))
  theoretical_floor="${token_floor}"

  # vLLM 中 max_num_seqs 相当于 scheduler 同时运行请求数上限。
  # 若可得，则 effective theory 取 token capacity 与 max_num_seqs 的较小值。
  if [[ "${MAX_NUM_SEQS:-}" =~ ^[0-9]+$ ]] && (( MAX_NUM_SEQS > 0 )); then
    if (( MAX_NUM_SEQS < theoretical_floor )); then
      theoretical_floor="${MAX_NUM_SEQS}"
    fi
  fi

  (( theoretical_floor < 1 )) && theoretical_floor=1
  scan_max=$((theoretical_floor + 2))

  declare -A selected=()

  # 若用户/环境提供了真实 CUDA Graph BS，优先作为基础扫描点。
  if [[ -n "${CUDA_GRAPH_BS}" ]]; then
    while IFS= read -r bs; do
      [[ "${bs}" =~ ^[0-9]+$ ]] || continue
      (( bs >= 1 && bs <= scan_max )) && selected["${bs}"]=1
    done < <(
      printf "%s\n" "${CUDA_GRAPH_BS}" \
      | tr ',' ' ' | tr -s '[:space:]' '\n' \
      | awk '/^[0-9]+$/ {print $1}' | sort -nu
    )
  else
    # fallback：沿用 DP_SIZE 布局
    if (( DP_SIZE > 3 )); then
      for bs in "${DENSE_DP_BS[@]}"; do
        (( bs <= scan_max )) && selected["${bs}"]=1
      done
      bs=80
      while (( bs <= scan_max )); do
        selected["${bs}"]=1
        bs=$((bs + 16))
      done
    else
      for bs in "${LOW_DP_BS[@]}"; do
        (( bs <= scan_max )) && selected["${bs}"]=1
      done
      bs=160
      while (( bs <= scan_max )); do
        selected["${bs}"]=1
        bs=$((bs + 32))
      done
    fi
  fi

  # effective theory 附近只加 ±1/±2
  for delta in "${THEORY_DELTAS[@]}"; do
    bs=$((theoretical_floor + delta))
    (( bs >= 1 && bs <= scan_max )) && selected["${bs}"]=1
  done

  selected["1"]=1
  selected["${scan_max}"]=1

  for ((bs = 1; bs <= scan_max; bs++)); do
    [[ -n "${selected[$bs]:-}" ]] && printf "%s " "${bs}"
  done
  echo
}

# ============================================================
# 汇总 CSV（轻量）
# ============================================================

merge_results() {
python3 - \
  "${jsonl_dir}" \
  "${log_dir}" \
  "${out_dir}/sum_all.csv" \
  "${KV_TOKENS}" \
  "${DP_SIZE}" \
  "${MAX_NUM_SEQS:-}" \
  "${run_id}" \
  "${CACHE_MODE}" \
  "${PARALLEL_CONFIG:-UNKNOWN}" \
  "${NUM_PROMPTS_MULTIPLIER}" <<'PY'
import csv, json, re, sys
from pathlib import Path

if len(sys.argv) < 11:
    raise SystemExit(f"merge_results argv mismatch: expected 10 args, got {len(sys.argv)-1}")

jsonl_dir, log_dir, output_csv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
def to_int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None

kv_tokens, dp_size = to_int_or_none(sys.argv[4]), to_int_or_none(sys.argv[5])
max_num_seqs_raw = sys.argv[6].strip()
max_num_seqs = int(max_num_seqs_raw) if max_num_seqs_raw.isdigit() else ""
run_id, cache_mode = sys.argv[7], sys.argv[8]
parallel_config = sys.argv[9]
num_prompts_multiplier = int(sys.argv[10])

pattern = re.compile(
    r"(?P<case>.+?)_in(?P<input>\d+)_out(?P<output>\d+)_perdp(?P<perdp>\d+)_global(?P<global>\d+)\.(?:log|json)$"
)


def first_present(row, *keys):
    for key in keys:
        if key in row and row[key] not in ("", None):
            return row[key]
    return ""


def percentile_from_list(values, q):
    """支持 [(50,v)...] / [[50,v]...] / {50:v,'95':v}"""
    if not values:
        return ""
    if isinstance(values, dict):
        for key, val in values.items():
            try:
                if abs(float(key) - q) < 1e-6:
                    return float(val)
            except Exception:
                continue
        return ""
    if not isinstance(values, list) or not values:
        return ""
    first = values[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        for item in values:
            try:
                pf, vf = float(item[0]), float(item[1])
            except Exception:
                continue
            if abs(pf - q) < 1e-6:
                return vf
    return ""


def has_value(row, key):
    return key in row and row[key] not in ("", None)


def parse_percentiles_from_log(log_path):
    """从客户端 log 文本补 P50/P95/P99。"""
    out = {}
    if not log_path.exists():
        return out
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out

    # 兼容多种打印格式
    patterns = [
        # P99 TTFT (ms): 12.3
        re.compile(
            r"P\s*(?P<pct>50|95|99)\s+(?P<metric>TTFT|TPOT|ITL|E2E(?:L| Latency)?)"
            r"[^\d\n]*?(?P<val>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            re.IGNORECASE,
        ),
        # TTFT P99 (ms): 12.3
        re.compile(
            r"(?P<metric>TTFT|TPOT|ITL|E2E(?:L| Latency)?)\s+P\s*(?P<pct>50|95|99)"
            r"[^\d\n]*?(?P<val>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            re.IGNORECASE,
        ),
    ]
    metric_map = {
        "ttft": "ttft",
        "tpot": "tpot",
        "itl": "itl",
        "e2e": "e2e_latency",
        "e2el": "e2e_latency",
        "e2e latency": "e2e_latency",
    }
    for pat in patterns:
        for m in pat.finditer(text):
            metric = re.sub(r"\s+", " ", m.group("metric").strip().lower())
            metric = metric_map.get(metric, metric)
            if metric not in {"ttft", "tpot", "itl", "e2e_latency"}:
                continue
            key = f"p{int(m.group('pct'))}_{metric}_ms"
            try:
                out[key] = float(m.group("val"))
            except Exception:
                continue
    return out


def normalize_row(row, log_path=None):
    # ========================================================
    # 标准字段 alias：与 SGLang CSV 口径对齐
    # ========================================================
    row["num_prompts"] = first_present(
        row, "num_prompts", "successful_requests", "completed", "global_concurrency"
    )
    row["max_concurrency"] = first_present(
        row, "max_concurrency", "max_concurrent_requests",
        "max_request_concurrency", "global_concurrency"
    )
    row["concurrency"] = first_present(
        row, "concurrency", "max_concurrent_requests", "global_concurrency"
    )
    row["peak_output_throughput"] = first_present(
        row, "peak_output_throughput",
        "peak_output_token_throughput",
        "max_output_tokens_per_s"
    )
    row["peak_concurrency"] = first_present(
        row, "peak_concurrency",
        "peak_concurrent_requests",
        "max_concurrent_requests"
    )
    row["request_rate"] = first_present(
        row, "request_rate", "traffic_request_rate"
    )
    if row["request_rate"] in ("", None):
        row["request_rate"] = "inf"

    # benchmark 自身 duration 优先；shell elapsed 只 fallback。
    row["duration_s"] = first_present(
        row, "duration_s", "benchmark_duration_s", "duration", "elapsed_s"
    )

    row["rps"] = first_present(
        row, "rps", "request_throughput", "request_throughput_req_s"
    )
    row["generate_throughput_tok_s"] = first_present(
        row, "generate_throughput_tok_s",
        "output_throughput", "output_throughput_tok_s"
    )
    row["total_throughput_tok_s"] = first_present(
        row, "total_throughput_tok_s",
        "total_token_throughput", "total_throughput"
    )
    row["mean_e2e_latency_ms"] = first_present(
        row, "mean_e2e_latency_ms", "mean_e2el_ms"
    )
    row["accept_length"] = first_present(
        row, "accept_length", "mean_acceptance_length", "accept_len"
    )

    # log 里的 P50/P95/P99 先补进 row
    if log_path is not None:
        for k, v in parse_percentiles_from_log(log_path).items():
            if not has_value(row, k):
                row[k] = v

    metric_map = [
        ("ttft", "ttft"),
        ("tpot", "tpot"),
        ("itl", "itl"),
        ("e2el", "e2e_latency"),
    ]
    completed = row.get("completed", "")
    try:
        completed_n = int(completed)
    except Exception:
        completed_n = -1

    for src, dst in metric_map:
        plist = row.get(f"percentiles_{src}_ms")
        for q in (50, 95, 99):
            out_key = f"p{q}_{dst}_ms"
            if not has_value(row, out_key) and plist not in ("", None):
                val = percentile_from_list(plist, q)
                if val != "":
                    row[out_key] = val

        # p50 <- median
        p50_key = f"p50_{dst}_ms"
        if not has_value(row, p50_key):
            row[p50_key] = first_present(
                row, p50_key, f"median_{dst}_ms", f"median_{src}_ms"
            )

        mean_key = f"mean_{dst}_ms" if dst != "e2e_latency" else "mean_e2e_latency_ms"
        mean_alt = f"mean_{src}_ms"
        base = first_present(row, p50_key, mean_key, mean_alt, f"median_{src}_ms")

        # 样本很少时（尤其 BS=1），P95/P99 应等于该次样本，不能留空
        if base != "":
            for q in (50, 95, 99):
                out_key = f"p{q}_{dst}_ms"
                if not has_value(row, out_key) and completed_n in (0, 1):
                    row[out_key] = base
            # 即使 completed 字段缺失，mean==median 也按单样本补齐
            mean_v = first_present(row, mean_key, mean_alt)
            med_v = first_present(row, f"median_{dst}_ms", f"median_{src}_ms", p50_key)
            if (
                mean_v != ""
                and med_v != ""
                and abs(float(mean_v) - float(med_v)) < 1e-9
            ):
                for q in (95, 99):
                    out_key = f"p{q}_{dst}_ms"
                    if not has_value(row, out_key):
                        row[out_key] = mean_v

    return row




rows = []
for log_path in sorted(log_dir.glob("*.log")):
    m = pattern.fullmatch(log_path.name)
    if not m:
        continue
    input_len = int(m.group("input"))
    output_len = int(m.group("output"))
    perdp = int(m.group("perdp"))
    global_c = int(m.group("global"))
    case_name = m.group("case")

    json_path = jsonl_dir / (log_path.stem + ".json")
    time_path = Path(str(log_path) + ".time")
    exit_path = Path(str(log_path) + ".exitcode")

    row = {
        "case_name": case_name,
        "input_len": input_len,
        "output_len": output_len,
        "input_k": input_len / 1024,
        "output_k": output_len / 1024,
        "parallel_config": parallel_config,
        "kv_tokens": kv_tokens,
        "max_num_seqs": max_num_seqs,
        "max_total_tokens": kv_tokens,
        "tokens_per_request": input_len + output_len,
        "token_theoretical_per_dp_bs": round(kv_tokens / (input_len + output_len), 4) if kv_tokens else "",
        "effective_theoretical_per_dp_bs": (
            round(
                min(kv_tokens / (input_len + output_len), max_num_seqs)
                if isinstance(max_num_seqs, int) and max_num_seqs > 0
                else kv_tokens / (input_len + output_len),
                4
            ) if kv_tokens else ""
        ),
        "per_dp_bs": perdp,
        "num_prompts": global_c * num_prompts_multiplier,
        "max-concurrency": global_c,
        "max_concurrency": global_c,
        "concurrency": global_c,
        "global_concurrency": global_c,
        "cache_mode": cache_mode,
        "cache_salt": f"{run_id}_{case_name}_perdp{perdp}",
    }

    if time_path.exists():
        for line in time_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                row[k.strip()] = v.strip()

    if exit_path.exists():
        try:
            row["case_exit_code"] = int(exit_path.read_text().strip())
        except Exception:
            row["case_exit_code"] = ""
    else:
        row["case_exit_code"] = ""

    row["case_status"] = (
        "PASS" if row.get("case_exit_code") == 0
        else ("FAIL" if row.get("case_exit_code") not in ("", None) else "UNKNOWN")
    )

    if json_path.exists():
        try:
            text = json_path.read_text(encoding="utf-8", errors="replace").strip()
            obj = None
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                for line in text.splitlines()[::-1]:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                        # keep percentile lists for later extraction; skip huge arrays
                        if isinstance(v, list) and k in {
                            "ttfts", "itls", "tpots", "latencies",
                            "input_lens", "output_lens", "errors",
                            "generated_texts", "start_times",
                        }:
                            continue
                        row[k] = v
        except Exception:
            pass

    row = normalize_row(row, log_path)
    row["log_file"] = log_path.name
    row["json_file"] = json_path.name
    rows.append(row)

# ============================================================
# 固定前置字段：与 SGLang 常用指标口径保持一致。
# 后面只追加非重复的新指标。
# ============================================================
preferred = [
    "case_name",
    "input_len",
    "output_len",
    "input_k",
    "output_k",
    "parallel_config",
    "max_num_seqs",
    "max_total_tokens",
    "tokens_per_request",
    "token_theoretical_per_dp_bs",
    "effective_theoretical_per_dp_bs",
    "per_dp_bs",
    "num_prompts",
    "max_concurrency",
    "concurrency",
    "peak_output_throughput",
    "peak_concurrency",
    "request_rate",
    "duration_s",
    "rps",
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
    "cache_mode",
    "cache_salt",

    # Case / debug
    "case_status",
    "case_exit_code",
]

duplicate_aliases = {
    "random_input_len",
    "random_output_len",
    "input_length",
    "output_length",
    "successful_requests",
    "completed",
    "max_concurrent_requests",
    "max_request_concurrency",
    "duration",
    "benchmark_duration_s",
    "elapsed_s",
    "request_throughput",
    "request_throughput_req_s",
    "traffic_request_rate",
    "output_throughput",
    "output_throughput_tok_s",
    "total_token_throughput",
    "total_throughput",
    "theoretical_per_dp_bs",
    "kv_tokens",
}

hidden_columns = {
    "backend", "model", "random_input_len", "random_output_len",
    "request_throughput", "output_throughput", "total_token_throughput",
    "mean_ttft", "median_ttft", "p95_ttft", "p99_ttft",
    "mean_tpot", "median_tpot", "p95_tpot", "p99_tpot",
    "mean_itl", "median_itl", "p95_itl", "p99_itl",
    "mean_e2el", "median_e2el", "p95_e2el", "p99_e2el",
    "dp_size", "global_concurrency", "bs_vs_theory_ratio",
    "start_time", "end_time", "elapsed_s", "log_file", "json_file",
    "ttfts", "itls", "tpots", "latencies",
    "input_lens", "output_lens", "errors",
    "generated_texts", "start_times",
}

keys = set()
for r in rows:
    keys.update(r.keys())

extra = sorted(
    k for k in keys
    if k not in preferred
    and k not in duplicate_aliases
    and k not in hidden_columns
)
cols = preferred + extra

tmp = Path(str(output_csv) + ".tmp")
with tmp.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in cols})
tmp.replace(output_csv)
print(f"[MERGE] rows={len(rows)} -> {output_csv}")
PY
}

on_exit() {
  local exit_code=$?
  trap - EXIT
  merge_results || true
  echo
  echo "结果目录：${out_dir}"
  echo "CSV     ：${out_dir}/sum_all.csv"
  exit "${exit_code}"
}
trap 'exit 130' INT
trap 'exit 143' TERM
trap on_exit EXIT



# ============================================================
# TPOT≈50ms 自适应加密
# ============================================================


is_sla_near_target() {
  local current_case="$1"
  local current_bs="$2"
  local csv_file="${out_dir}/sum_all.csv"

  [[ -s "${csv_file}" ]] || return 1

  python3 - "${csv_file}" "${current_case}" "${current_bs}" "${SLA_RULES}" <<'PY'
import csv, sys, math

csv_file, case_name, bs_s, rules_text = sys.argv[1:5]
bs = int(bs_s)

row_match = None
with open(csv_file, newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r.get("case_name") != case_name:
            continue
        try:
            r_bs = int(float(r.get("per_dp_bs", "")))
        except Exception:
            continue
        if r_bs == bs:
            row_match = r

if not row_match:
    sys.exit(1)

hits = []
states = []
for raw in rules_text.split(","):
    raw = raw.strip()
    if not raw:
        continue
    p = raw.split(":")
    if len(p) != 3:
        states.append(f"{raw}=INVALID")
        continue

    metric, target_s, near_s = p
    try:
        target = float(target_s)
        near = float(near_s)
        value = float(row_match.get(metric, ""))
        if not math.isfinite(value):
            raise ValueError
    except Exception:
        states.append(f"{metric}=NA")
        continue

    hit = abs(value - target) <= near
    states.append(f"{metric}={value:.4f} target={target:g}±{near:g} hit={hit}")
    if hit:
        hits.append(metric)

print(f"[SLA-CHECK] bs={bs} " + " | ".join(states), file=sys.stderr)
sys.exit(0 if hits else 1)
PY
}

case_result_exists() {
  local current_case="$1"
  local per_dp_bs="$2"
  local global_concurrency=$((DP_SIZE * per_dp_bs))
  local base_name="${current_case}_in${input_len}_out${output_len}_perdp${per_dp_bs}_global${global_concurrency}"
  local result_file="${jsonl_dir}/${base_name}.json"
  local exit_file="${log_dir}/${base_name}.log.exitcode"

  [[ -s "${result_file}" && -f "${exit_file}" ]] || return 1
  [[ "$(tr -d '[:space:]' < "${exit_file}" 2>/dev/null)" == "0" ]]
}

run_one_bs() {
  local per_dp_bs="$1"
  local global_concurrency=$((DP_SIZE * per_dp_bs))
  local num_prompts=$((global_concurrency * NUM_PROMPTS_MULTIPLIER))
  local warmup_requests="${WARMUP_REQUESTS:-0}"

  local base_name="${case_name}_in${input_len}_out${output_len}_perdp${per_dp_bs}_global${global_concurrency}"
  local result_file="${jsonl_dir}/${base_name}.json"
  local log_file="${log_dir}/${base_name}.log"
  local exit_file="${log_file}.exitcode"

  # 已成功跑过的 BS 自动跳过，适合续跑/自动补点。
  if case_result_exists "${case_name}" "${per_dp_bs}"; then
    echo "[SKIP] ${case_name} per_dp_bs=${per_dp_bs} 已完成"
    return 0
  fi

  local bs_start_time bs_start_epoch bs_end_time bs_end_epoch bs_elapsed_s
  local case_exit_code bench_pid cache_salt

  bs_start_time="$(date '+%F %T')"
  bs_start_epoch="$(date +%s)"
  cache_salt="${run_id}_${case_name}_perdp${per_dp_bs}"

  echo
  echo "============================================================"
  echo "CASE               : ${case_name}"
  echo "Per-DP BS          : ${per_dp_bs}"
  echo "Global Concurrency : ${global_concurrency}"
  echo "Num Prompts        : ${num_prompts} (${global_concurrency} x ${NUM_PROMPTS_MULTIPLIER})"
  echo "Theoretical BS     : ${theoretical_bs}"
  echo "Max Num Seqs       : ${MAX_NUM_SEQS:-N/A}"
  echo "Start Time         : ${bs_start_time}"
  echo "LOG                : ${log_file}"
  echo "JSON               : ${result_file}"
  echo "Cache Mode         : ${CACHE_MODE}"
  echo "Cache Salt         : ${cache_salt}"
  echo "============================================================"
  echo

  (
    set -o pipefail
    vllm bench serve \
      --backend openai \
      --base-url "${BASE_URL}" \
      --model "${MODEL}" \
      --dataset-name random \
      --random-input-len "${input_len}" \
      --random-output-len "${output_len}" \
      --random-range-ratio 0.0 \
      --max-concurrency "${global_concurrency}" \
      --num-prompts "${num_prompts}" \
      --request-rate inf \
      --num-warmups "${warmup_requests}" \
      --ignore-eos \
      --percentile-metrics "${PERCENTILE_METRICS}" \
      --metric-percentiles "${METRIC_PERCENTILES}" \
      --save-result \
      --save-detailed \
      --result-dir "${jsonl_dir}" \
      --result-filename "${base_name}.json" \
      --disable-tqdm \
      --extra-body "{\"cache_salt\":\"${cache_salt}\"}" \
      2>&1 | tee "${log_file}"
    echo $? > "${exit_file}"
  ) &
  bench_pid=$!

  wait "${bench_pid}"

  if [[ -f "${exit_file}" ]]; then
    case_exit_code="$(tr -d '[:space:]' < "${exit_file}")"
  else
    case_exit_code=1
    printf "%s\n" "${case_exit_code}" > "${exit_file}"
  fi

  bs_end_time="$(date '+%F %T')"
  bs_end_epoch="$(date +%s)"
  bs_elapsed_s=$((bs_end_epoch - bs_start_epoch))

  {
    echo "start_time=${bs_start_time}"
    echo "end_time=${bs_end_time}"
    echo "elapsed_s=${bs_elapsed_s}"
    echo "kv_tokens=${KV_TOKENS}"
    echo "dp_size=${DP_SIZE}"
  } > "${log_file}.time"

  if (( case_exit_code == 0 )); then
    echo "[PASS] ${case_name} per_dp_bs=${per_dp_bs} global=${global_concurrency} elapsed=${bs_elapsed_s}s"
  else
    echo "[FAIL] ${case_name} per_dp_bs=${per_dp_bs} global=${global_concurrency} exit=${case_exit_code} elapsed=${bs_elapsed_s}s"
  fi

  merge_results
}

# ============================================================
# 启动
# ============================================================

load_vllm_server_info
load_vllm_server_log_config
load_vllm_capacity
validate_fixed_mode

if [[ ! "${NUM_PROMPTS_MULTIPLIER}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] NUM_PROMPTS_MULTIPLIER 必须是正整数"
  exit 1
fi

python3 - "${SLA_RULES}" <<'PY' || exit 1
import sys
for raw in sys.argv[1].split(","):
    raw = raw.strip()
    if not raw:
        continue
    p = raw.split(":")
    if len(p) != 3 or not p[0]:
        print(f"[ERROR] 非法 SLA_RULES 项: {raw}", file=sys.stderr)
        raise SystemExit(1)
    try:
        float(p[1]); float(p[2])
    except ValueError:
        print(f"[ERROR] SLA target/near 非数字: {raw}", file=sys.stderr)
        raise SystemExit(1)
PY

STARTUP_READY=1

echo
echo "============================================================"
echo "Benchmark Configuration"
echo "============================================================"
echo "BASE_URL   : ${BASE_URL}"
echo "MODEL      : ${MODEL}"
echo "DP_SIZE      : ${DP_SIZE}"
echo "MAX_NUM_SEQS : ${MAX_NUM_SEQS:-N/A}"
echo "KV_TOKENS    : ${KV_TOKENS}"
echo "CACHE_MODE : ${CACHE_MODE}"
echo "PARALLEL   : ${PARALLEL_CONFIG:-UNKNOWN}"
echo "CUDA_GRAPH_BS: ${CUDA_GRAPH_BS:-FALLBACK_DP_LAYOUT}"
echo "SERVER_INFO  : ${SERVER_INFO_URL}"
echo "SERVER_LOG   : ${SERVER_LOG:-N/A}"
echo "FLUSH      : NEVER"
echo "SLA DENSE  : enable=${SLA_DENSE_ENABLE} rules=${SLA_RULES} mode=inline"
echo "FIXED_PER_DP_BS   : ${FIXED_PER_DP_BS:-AUTO}"
echo "FIXED_CONCURRENCY : ${FIXED_CONCURRENCY:-AUTO}"
echo "NUM_PROMPTS_MULT  : ${NUM_PROMPTS_MULTIPLIER}"
echo "PERCENTILES: metrics=${PERCENTILE_METRICS}  pct=${METRIC_PERCENTILES}"
echo "OUT        : ${out_dir}"
echo


for case_config in "${CASES[@]}"; do
  read -r input_k output_k <<< "${case_config}"

  input_len=$((input_k * 1024))
  output_len=$((output_k * 1024))
  case_name="$(format_k_name "${input_k}")_$(format_k_name "${output_k}")"
  tokens_per_request=$((input_len + output_len))
  token_theoretical_bs="$(get_token_theoretical_bs "${input_k}" "${output_k}")"
  theoretical_bs="$(get_theoretical_bs "${input_k}" "${output_k}")"
  auto_bs_list="$(get_bs_list "${input_k}" "${output_k}")"
  bs_list="$(get_test_bs_list "${auto_bs_list}")"

  echo
  echo "############################################################"
  echo "# CASE           : ${case_name}"
  echo "# Input/Output   : ${input_len} / ${output_len}"
  echo "# Tokens/Request : ${tokens_per_request}"
  echo "# KV_TOKENS      : ${KV_TOKENS}"
  echo "# Token Theory/DP: ${token_theoretical_bs}"
  echo "# Effective Theory/DP: ${theoretical_bs}"
  if [[ -n "${CUDA_GRAPH_BS}" ]]; then
    echo "# BS Source      : /server_info cudagraph_capture_sizes + theory ±1/±2"
  else
    echo "# BS Source      : DP_SIZE fallback + theory ±1/±2"
  fi
  echo "# Auto BS        : ${auto_bs_list}"
  echo "# Test BS        : ${bs_list}"
  echo "############################################################"

  # 单阶段顺序执行：
  # 基础/CUDA Graph/theory BS 按升序跑；每个 BS 结束后立即检查 SLA。
  # 任一规则命中，就连续跑到下一个基础 BS 之前，不做第二阶段回头补测。
  mapfile -t base_bs_array < <(printf "%s\n" ${bs_list} | awk 'NF' | sort -n)

  for ((i=0; i<${#base_bs_array[@]}; i++)); do
    per_dp_bs="${base_bs_array[$i]}"
    run_one_bs "${per_dp_bs}"

    if (( i + 1 >= ${#base_bs_array[@]} )); then
      continue
    fi

    if [[ "${SLA_DENSE_ENABLE}" != "1" || -n "${FIXED_PER_DP_BS}" || -n "${FIXED_CONCURRENCY}" ]]; then
      continue
    fi

    next_base_bs="${base_bs_array[$((i + 1))]}"

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

echo
echo "全部 Benchmark 执行结束"
echo "CSV: ${out_dir}/sum_all.csv"
