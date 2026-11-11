#!/usr/bin/env bash
# ============================================================
# prefill_sla_bench.sh v6.2 — 可变命中率 prefix 数据集 + np 饱和扫描 + 0.1 回测 (+decode 模式)
# 每条 trial 统一落 27 列标准指标 (Group/InOut/parallel_config/命中率/np/并发/
# server 峰值/rate/duration/rps/吞吐×3/TTFT·TPOT·ITL mean|p95|p99/E2E),
# sla_backtest.tsv 追加 sla_metric/target_s/lock_type 3 列锁点元数据。
# v6.2: decode 限并发阶梯每档以 --max-concurrency C 逐档下压 (修复全局固定值);
# ============================================================
# 数据集: sla_bench/prefix_hit.py (PREFIX_HIT 参数化, input/output len 任意)
# 模式分流 (按 OUTPUT 自动判定):
#   OUTPUT=1  → prefill 模式: np 饱和扫描 + 0.1 降率 SLA 回测 (下述规则)
#   OUTPUT>1  → decode 模式: --max-concurrency 限并发 (默认 inf=不传该参),
#               只跑 np 扫描/标定, 不做请求率回测
#   decode 限并发时: --np-list 每个元素 = 该档并发数 C, 以 --max-concurrency C
#   逐档下压 (阶梯开关=--max-concurrency 传有限值; 全局值兼作档位上限告警),
#   实际请求数 = C × --np-mult (默认 3), 请求池持续补位形成稳态并发,
#   避免池子=并发跑完即止。
#   decode SLA 边界: 升压扫描中某档 SLA 指标 (默认 mean TTFT, --sla-stop-ms
#     阈值, 缺省取 --sla-targets 首档) 超标即停止升压, 锁定最后一个满足档
#     → decode_sla_lock (无档满足 → decode_sla_not_found)。
# 客户端 prefill 模式默认不加 --max-concurrency, 只用 --request-rate 限流。
# 回测规则 (参考 gsp_prefix_bench 口径):
#   阶段0 np 饱和扫描: rate=inf 下请求池从 --np-start 起按 --np-step 倍增,
#     实测 rps 增益 >= --sat-gain% 才继续上打, 增益不足(只高一点点)即止,
#     best_np = 扫描中 rps 最大档; 给 --np-list 则跑固定列表(无自动止)。
#   阶段1 SLA 回测: 用 best_np 的 inf 标定为基准, 对每档 SLA (默认
#     mean TTFT 3|4|5|6 s): inf 满足 → inf_lock; 不满足则以 --rate-step
#     (默认 0.1) 固定步长降率, 首个满足点 → descend_lock。
#     全程 num_prompts=best_np (与最高吞吐同一请求池, 口径一致)。
# 用法 (perf_bench 仓库根目录, 客户端容器内):
#   BASE_URL=http://192.168.0.65:30000 MODEL=/mnt/gbj/models/xxx \
#   bash sla_bench/prefill_sla_bench.sh \
#        --case '64k,1,80' --case '128k,1,80' --case '256k,1,90' \
#        [--np-start 64 --np-step 2 --sat-gain 3]
# case 语法: INPUT,OUTPUT,HIT   INPUT 任意(k/K 后缀), OUTPUT 任意, HIT 0-100(0=冷)
# 参数: --np-start N      inf 饱和扫描起始请求数 (默认 64)
#       --np-step F       扫描倍增因子 (默认 2, 即 64→128→256...)
#       --sat-gain P      继续上打的 rps 增益门槛% (默认 3, 不足即停)
#       --np-max N        扫描请求数上限 (默认 1024)
#       --np-list L,N,... 固定扫描列表, 给定则忽略 start/step 自动止
#                          (decode 限并发时列表元素=并发数)
#       --np-mult M       decode 限并发时 请求数=并发×M (默认 3; prefill 固定 1)
#       --num-prompts N   固定单点 (等价 --np-list N, 跳过扫描)
#       --sla-metric M     SLA 指标, ttft/tpot/itl × mean/p95/p99 共 9 种
#                          (默认 mean_ttft_ms)
#       --sla-targets S|.. 秒阈值列表 (默认 3|4|5|6)
#       --sla-stop-ms N    decode 升压扫描 SLA 停止阈值 ms (默认=首档×1000)
#       --rate-step F      固定降率步长 (默认 0.1, 仅 prefill 模式)
#       --max-concurrency N|inf  最大并发 (默认 inf=不加该参数, prefill
#                          同样生效; decode 模式传有限值=启用并发阶梯)
#       --max-trials N     单 SLA 档 trial 上限 (默认 24, 仅 prefill)
#       --run-id ID --dry-run
# 环境变量: BASE_URL MODEL PYTHON_BIN SEED
# 输出: ./all_result/sla_${RUN_ID}/{np_scan.tsv, bs_scan.tsv,
#       sla_backtest.tsv, logs/}
# ============================================================

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
MODEL="${MODEL:-}"
SEED="${SEED:-42}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN=0
# 兼容容器内 sglang 源码未安装到 site-packages 的场景:
# 显式设置 PYTHONPATH 后, 所有 "$PYTHON_BIN" 调用自动继承
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}${SGLANG_PYTHONPATH:-/sgl-workspace/sglang/python}"

SLA_METRIC="mean_ttft_ms"
SLA_TARGETS="3|4|5|6"
SLA_STOP_MS=""          # decode 升压 SLA 停止阈值 ms; 空=首档×1000
NUM_PROMPTS="${NUM_PROMPTS:-64}"
NP_START="${NP_START:-64}"
NP_STEP="2"
SAT_GAIN="3"
NP_MAX="1024"
NP_LIST=""
RATE_STEP="0.1"
MAX_TRIALS=24
MAX_CONC="inf"          # inf = 不传 --max-concurrency
NP_MULT=""              # 空=自动 (decode 限并发→3, 其余→1); 传值强制覆盖
mc_args=()
[[ "$MAX_CONC" != "inf" ]] && mc_args=(--max-concurrency "$MAX_CONC")

# ---- 27 列元数据来源 ----
GROUP="${GROUP:-}"                       # Group 列: 战役/分组标签
PARALLEL_CONFIG="${PARALLEL_CONFIG:-}"   # 空=启动后自动抓 /get_server_info
PCFG="$PARALLEL_CONFIG"
MC_COL=""                                # 并发两列默认值 (解析后按 MAX_CONC 重建)
MC_ROW=""                                # 并发阶梯逐行覆盖: 该档并发数
PEAK_SAMPLING="${PEAK_SAMPLING:-1}"      # trial 期间采样 server /metrics 峰值
PEAK_INT="${PEAK_INT:-2}"                # 采样间隔 s
PCT_METRICS="${PERCENTILE_METRICS:-ttft,tpot,itl,e2el}"
PCT_VALUES="${METRIC_PERCENTILES:-95,99}"

declare -a CASES=()
declare -a PASSTHROUGH=()

usage() { awk '/^# ={10,}/{c++; next} c==1{sub(/^# ?/,""); print} c>1{exit}' "$0"; exit 0; }
log()  { echo "[INFO]  $(date '+%F %T') $*"; }
warn() { echo "[WARN]  $(date '+%F %T') $*"; }
die()  { echo "[ERROR] $(date '+%F %T') $*" >&2; exit "${2:-1}"; }

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
    --case)         CASES+=("$2"); shift 2 ;;
    --np-start)     NP_START="$2"; shift 2 ;;
    --np-step)      NP_STEP="$2"; shift 2 ;;
    --sat-gain)     SAT_GAIN="$2"; shift 2 ;;
    --np-max)       NP_MAX="$2"; shift 2 ;;
    --np-list)      NP_LIST="$2"; shift 2 ;;
    --num-prompts)  NP_LIST="$2"; shift 2 ;;
    --sla-metric)   SLA_METRIC="$2"; shift 2 ;;
    --sla-targets)  SLA_TARGETS="$2"; shift 2 ;;
    --sla-stop-ms)  SLA_STOP_MS="$2"; shift 2 ;;
    --rate-step)    RATE_STEP="$2"; shift 2 ;;
    --max-concurrency) MAX_CONC="$2"; mc_args=(); [[ "$MAX_CONC" != "inf" ]] && mc_args=(--max-concurrency "$MAX_CONC"); shift 2 ;;
    --np-mult)      NP_MULT="$2"; shift 2 ;;
    --max-trials)   MAX_TRIALS="$2"; shift 2 ;;
    --run-id)       RUN_ID="$2"; shift 2 ;;
    --dry-run)      DRY_RUN=1; shift ;;
    -h|--help|help) usage ;;
    *)              PASSTHROUGH+=("$1"); shift ;;  # 其余参数透传给 bench CLI
  esac
done

# 并发两列默认值: 依赖 CLI --max-concurrency, 解析后重建
MC_COL=""; [[ "$MAX_CONC" != "inf" ]] && MC_COL="$MAX_CONC"

# OUT_DIR 必须在参数解析后赋值, 否则 --run-id 不生效
# 用绝对路径, 避免 run_trial() 里 subshell cd 后父 shell 找不到 jsonl
OUT_DIR="$(pwd)/all_result/sla_${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

case "$SLA_METRIC" in
  mean_ttft_ms|p95_ttft_ms|p99_ttft_ms|mean_tpot_ms|p95_tpot_ms|p99_tpot_ms|mean_itl_ms|p95_itl_ms|p99_itl_ms) ;;
  *) die "sla-metric 仅支持 ttft/tpot/itl × mean/p95/p99 (9 种)" ;;
esac

if (( DRY_RUN == 0 )); then
  curl -s --max-time 5 -o /dev/null "$BASE_URL/health" \
    || die "server 不可达: $BASE_URL/health"
  "$PYTHON_BIN" -c 'from sglang.benchmark.datasets.common import DatasetRow, get_available_tokens' \
    || die "sglang.benchmark.datasets 不可用"
  log "server OK: $BASE_URL"
fi

# parallel_config 27 列自动识别: /get_server_info 任意层级找 *_size (sglang, 非 1 才拼)
if [[ -z "$PCFG" && "$DRY_RUN" == 0 ]]; then
  info=$(curl -s --max-time 5 "${BASE_URL%/}/get_server_info" 2>/dev/null || true)
  if [[ -n "$info" ]]; then
    PCFG=$("$PYTHON_BIN" - "$info" <<'PYPCFG'
import json, sys
def dig(d, key):
    if isinstance(d, dict):
        v = d.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
        for x in d.values():
            r = dig(x, key)
            if r is not None:
                return r
    return None
def gint(d, key):
    v = dig(d, key)
    return v if isinstance(v, int) and not isinstance(v, bool) else None
def gfloat(d, key):
    v = dig(d, key)
    return v if isinstance(v, float) else None
def gstr(d, key):
    v = dig(d, key)
    return v if isinstance(v, str) else None
try:
    d = json.loads(sys.argv[1])
except Exception:
    raise SystemExit
parts = []
# TP
tp = gint(d, "tp_size") or 1
parts.append(f"TP{tp}")
# DP (+ DPA 标记)
dp = gint(d, "dp_size") or 1
dp_attn = dig(d, "enable_dp_attention") is True
parts.append(f"DP{dp}" + ("(DPA)" if dp_attn else ""))
# PP (1 省略)
pp = gint(d, "pp_size") or 1
if pp > 1:
    parts.append(f"PP{pp}")
# EP (1 省略)
ep = gint(d, "ep_size") or 1
if ep > 1:
    parts.append(f"EP{ep}")
# ACP (attention cp, 1 省略)
acp = gint(d, "attn_cp_size") or 1
if acp > 1:
    parts.append(f"ACP{acp}")
# KV cache dtype
kv = gstr(d, "kv_cache_dtype") or "auto"
qnt = gstr(d, "quantization")
if qnt:
    parts.append(f"KV{qnt}")
elif kv and kv != "auto":
    parts.append(f"KV{kv}")
# MEM fraction
mem = gfloat(d, "mem_fraction_static")
if mem is not None:
    parts.append(f"MEM{mem:.1f}")
print("-".join(parts))
PYPCFG
)
  fi
  [[ -n "$PCFG" ]] || PCFG="$PARALLEL_CONFIG"
  [[ -n "$PCFG" ]] && log "[server] parallel_config=$PCFG"
fi

# 摘指标: rps metric dur completed actual_hit; completed 不齐视为失败(全 0)
pick_metrics() { # json
  "$PYTHON_BIN" - "$1" "$SLA_METRIC" "$NUM_PROMPTS" <<'PYEOF'
import json, sys
try:
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
except Exception:
    print("0 0 0 0 0"); sys.exit(0)
j = rows[-1] if rows else {}
def g(k):
    try: return float(j.get(k))
    except (TypeError, ValueError): return 0.0
hit = 0.0
try: hit = float((j.get("cache_report") or {}).get("cache_hit_rate_pct") or 0)
except Exception: pass
ok = g("completed") == float(sys.argv[3])
if not ok:
    print("0 0 0 0 0"); sys.exit(0)
metric = sys.argv[2] if len(sys.argv) > 2 else "mean_ttft_ms"
print(" ".join(str(round(x, 4)) for x in
    (g("request_throughput"), g(metric), g("duration"), g("completed"), hit)))
PYEOF
}

meet() { "$PYTHON_BIN" -c "import sys; sys.exit(0 if ${1} > 0 and ${1} < ${2} else 1)"; }
fstep() { "$PYTHON_BIN" -c "print(max(round(float('${1}') - float('${2}'), 2), 0.1))"; }
fmul() { "$PYTHON_BIN" -c "import math; print(int(math.floor(float('${1}') * float('${2}'))))"; }

# ---- trial 峰值采样: server /metrics (num_running_reqs / gen_throughput) ----
peak_start() { # $1=输出文件, 返回后台 pid (未启用返回空)
  (( PEAK_SAMPLING == 1 )) || return 0
  ( while :; do
      s=$(curl -s --max-time 2 "${BASE_URL%/}/metrics" 2>/dev/null || true)
      [[ -n "$s" ]] && \
        echo "$(awk '/^sglang:num_running_reqs/{t+=$NF} END{if(t>0)printf "%.0f",t}' <<<"$s") $(awk '/^sglang:gen_throughput/{t+=$NF} END{if(t>0)printf "%.1f",t}' <<<"$s")"
      sleep "$PEAK_INT"
    done ) >> "$1" 2>/dev/null &
  echo $!
}
peak_stop() { [[ -n "${1:-}" ]] && { kill "$1" 2>/dev/null; wait "$1" 2>/dev/null; }; return 0; }

peaks_of() { # $1=json → "peak_gen\tpeak_run", 无采样 → ""
  local f="$1.peaks" r=""
  [[ -s "$f" ]] \
    && r=$(awk '{if($1+0>R)R=$1+0;if($2+0>G)G=$2+0} END{printf "%s\t%s",(G>0?G:""),(R>0?R:"")}' "$f")
  printf '%s' "$r"
}

# ---- 27 列动态字段: bench jsonl 末行 summary (缺失回退计算, 无则留空) ----
pick_extra() { # $1=json → 13 个 tab 分隔值: 吞吐×3 + ttft/tpot/itl×3 + e2e
  "$PYTHON_BIN" - "$1" <<'PYXTRA'
import json, sys
rows = []
try:
    rows = [json.loads(s) for s in open(sys.argv[1], encoding="utf-8") if s.strip()]
except Exception:
    pass
j = rows[-1] if rows else {}
d = j.get("duration")
def num(v): return isinstance(v, (int, float)) and not isinstance(v, bool)
def g(*keys):
    for k in keys:
        v = j.get(k)
        if num(v):
            return v
    return ""
ti, to = j.get("total_input_tokens"), j.get("total_output_tokens")
in_tp = g("input_token_throughput") or (round(ti / d, 4) if num(ti) and d else "")
gen_tp = g("output_throughput") or (round(to / d, 4) if num(to) and d else "")
tot_tp = g("total_token_throughput") or \
    (round((ti + to) / d, 4) if num(ti) and num(to) and d else "")
def met(m, p="mean"):
    v = j.get(f"{p}_{m}_ms")
    return round(v, 4) if num(v) else ""
vals = [in_tp, gen_tp, tot_tp]
for m in ("ttft", "tpot", "itl"):
    vals += [met(m, "mean"), met(m, "p95"), met(m, "p99")]
vals.append(g("mean_e2el_ms", "mean_e2e_latency_ms"))
print("\t".join("" if x == "" else str(x) for x in vals))
PYXTRA
}

# ---- 27 列行组装 (列序见文件头), 静态列取自脚本侧变量 ----
row_core() { # rate np json rps mm dur hit → 打印一行 27 列 (tab)
  local ex pk core
  ex=$(pick_extra "$3")
  pk=$(peaks_of "$3")
  core=$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' \
    "${GROUP}" "${IN_LEN}" "${OUT_LEN}" "${PCFG}" "${hit_s}" "$7" "$2" \
    "${MC_ROW:-$MC_COL}" "${MC_ROW:-$MC_COL}" \
    "${pk%%$'\t'*}" "${pk#*$'\t'}" "$1" "$6" "$4")
  printf '%s\t%s\n' "${core}" "${ex}"
}

NP_TSV="${OUT_DIR}/np_scan.tsv"
SCAN_TSV="${OUT_DIR}/bs_scan.tsv"
BACK_TSV="${OUT_DIR}/sla_backtest.tsv"
COLS="Group,Input tokens,Generated tokens,parallel_config,Target cache hit(%),Actual cache hit(%),num_prompts,max_concurrency,concurrency,peak_output_throughput,peak_concurrency,request_rate,duration_s,rps,input_throughput_tok_s,generate_throughput_tok_s,total_throughput_tok_s,mean_ttft_ms,p95_ttft_ms,p99_ttft_ms,mean_tpot_ms,p95_tpot_ms,p99_tpot_ms,mean_itl_ms,p95_itl_ms,p99_itl_ms,mean_e2e_latency_ms"
COLS_TSV="${COLS//,/$'\t'}"
printf '%s\t%s\n' "$COLS_TSV" "gain_vs_prev_pct" > "$NP_TSV"
printf '%s\n' "$COLS_TSV" > "$SCAN_TSV"
printf '%s\t%s\t%s\t%s\n' "$COLS_TSV" "sla_metric" "target_s" "lock_type" > "$BACK_TSV"

for raw_case in "${CASES[@]}"; do
  IFS=',' read -r in_s out_s hit_s <<< "$raw_case"
  [[ "$hit_s" =~ ^[0-9]+$ ]] || die "case 的 HIT 必须是百分数: $raw_case"
  (( hit_s >= 0 && hit_s <= 100 )) || die "hit 须为 0..100: $raw_case"
  IN_LEN=$(parse_len "$in_s")
  OUT_LEN=$(parse_len "$out_s")
  CACHED_LEN=$(( IN_LEN * hit_s / 100 / 64 * 64 ))

  tag="${in_s}o${out_s}h${hit_s}"
  decode_mode=0; (( OUT_LEN > 1 )) && decode_mode=1
  (( decode_mode )) \
    && log "== case=$tag L=$IN_LEN O=$OUT_LEN hit=${hit_s}% (cached=$CACHED_LEN) DECODE max_concurrency=$MAX_CONC" \
    || log "== case=$tag L=$IN_LEN O=$OUT_LEN hit=${hit_s}% (cached=$CACHED_LEN) PREFILL rate-only"

  run_trial() { # rate json clog
    local rate=$1 json=$2 clog=$3 rc pid
    pid=$(peak_start "$json.peaks")
    # [flush-patch] 每 trial 前清 radix cache, 保证首触碰命中率口径 (防跨 trial 污染)
    curl -s --max-time 10 -X POST "${BASE_URL}/flush_cache" >/dev/null 2>&1 || true
    sleep 2
    local pct_args=()
    # 兼容旧版 prefix_hit.py: 不支持 --percentile-metrics 时不传
    if "$PYTHON_BIN" "$script_dir/prefix_hit.py" --help 2>&1 | grep -q "percentile-metrics"; then
      pct_args=(--percentile-metrics "$PCT_METRICS" --metric-percentiles "$PCT_VALUES")
    fi
    ( cd "$script_dir/.." && PREFIX_HIT="$hit_s" PREFIX_SEED="$SEED" \
        "$PYTHON_BIN" "$script_dir/prefix_hit.py" \
        --backend sglang --base-url "$BASE_URL" --model "$MODEL" \
        --dataset-name random-ids --tokenize-prompt \
        --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
        --random-range-ratio 1.0 --request-rate "$rate" \
        --num-prompts "$NUM_PROMPTS" ${mc_args[@]+"${mc_args[@]}"} \
        --warmup-requests 0 --cache-report --output-details \
        ${pct_args[@]+"${pct_args[@]}"} \
        --output-file "$json" --disable-tqdm \
        --extra-request-body '{"sampling_params":{"temperature":0.6,"top_p":0.95,"ignore_eos":true}}' \
        ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"} \
    ) > "$clog" 2>&1
    rc=$?
    peak_stop "$pid"
    return "$rc"
  }
  record() { # rate rps metric dur completed hit
    printf '%s\n' "$(row_core "$1" "$NUM_PROMPTS" "$json" "$2" "$3" "$4" "$6")" >> "$SCAN_TSV"
  }

  # ---- 阶段0: np 饱和扫描 (rate=inf, 决定请求池 best_np) ----
  conc_ladder=0; eff_mult=1
  if (( decode_mode )) && [[ "$MAX_CONC" != "inf" ]]; then
    conc_ladder=1; eff_mult="${NP_MULT:-3}"   # 档位=并发数C, 请求数=C×M (持续补位)
  elif [[ -n "$NP_MULT" ]] && (( decode_mode )); then
    eff_mult="$NP_MULT"                        # decode 无限并发: 仅记录用
  fi
  # decode SLA 停止阈值: 未显式给 --sla-stop-ms 则取 --sla-targets 首档
  dec_stop_ms="$SLA_STOP_MS"
  if (( decode_mode )) && [[ -z "$dec_stop_ms" ]]; then
    first_t="${SLA_TARGETS%%|*}"
    dec_stop_ms=$("$PYTHON_BIN" -c "print(int(float(${first_t})*1000))")
  fi
  last_ok_np=""; last_ok_rps=""; last_ok_m=""; last_ok_dur=""; last_ok_done=""; last_ok_hit=""
  if [[ -n "$NP_LIST" ]]; then
    IFS=',' read -ra NP_SEQ <<< "$NP_LIST"
    auto_mode=0
  else
    NP_SEQ=(); cur_np="$NP_START"
    while (( cur_np <= NP_MAX )); do NP_SEQ+=("$cur_np"); cur_np=$(fmul "$cur_np" "$NP_STEP"); done
    auto_mode=1
  fi
  log "[np-scan] 档位: ${NP_SEQ[*]} (auto=$auto_mode sat_gain=${SAT_GAIN}% conc_ladder=$conc_ladder mult=$eff_mult)"
  best_np=""; best_rps="-1"; best_m=""; best_dur=""; best_done=""; best_hit=""; prev_rps=""
  np_i=0
  for np in "${NP_SEQ[@]}"; do
    if (( conc_ladder )); then
      np_eff=$(fmul "$np" "$eff_mult"); MC_ROW="$np"
      (( np > MAX_CONC )) && warn "[np-scan] 档=$np 超过 --max-concurrency=$MAX_CONC 上限 (仍按 $np 出压)"
      mc_args=(--max-concurrency "$np")   # v6.2: 每档以其并发数下压
    else
      np_eff="$np"; MC_ROW=""
      mc_args=(); [[ "$MAX_CONC" != "inf" ]] && mc_args=(--max-concurrency "$MAX_CONC")
    fi
    NUM_PROMPTS="$np_eff"
    json="${LOG_DIR}/${tag}_inf_c${np}.jsonl"; clog="${LOG_DIR}/${tag}_inf_c${np}.log"
    log "[np-scan] 档=$np np=$np_eff rate=inf -> $clog"
    if (( DRY_RUN )); then m="0 0 0 0 0"
    else
      if ! run_trial inf "$json" "$clog"; then warn "[np-scan] 档=$np trial 失败, 见 $clog (跳过)"; continue; fi
      m=$(pick_metrics "$json")
    fi
    read -r rps mm dur done_ hitp <<< "$m"
    gain="-"
    [[ -n "$prev_rps" && "$prev_rps" != "0" ]] \
      && gain=$("$PYTHON_BIN" -c "print(round((float('$rps')-float('$prev_rps'))/float('$prev_rps')*100,2))")
    printf '%s\t%s\n' "$(row_core inf "$np_eff" "$json" "$rps" "$mm" "$dur" "$hitp")" "$gain" >> "$NP_TSV"
    log "[np-scan] 档=$np np=$np_eff rps=$rps ${SLA_METRIC}=$mm gain=${gain}%"
    if (( DRY_RUN == 0 )) && "$PYTHON_BIN" -c "import sys; sys.exit(0 if float('$rps') > float('$best_rps') else 1)"; then
      best_np=$np_eff; best_rps=$rps; best_m=$mm; best_dur=$dur; best_done=$done_; best_hit=$hitp
    fi
    # decode SLA 硬边界: 超标即停升压, 锁最后一个满足档
    if (( decode_mode && DRY_RUN == 0 )) && [[ -n "$dec_stop_ms" ]]; then
      if "$PYTHON_BIN" -c "import sys; sys.exit(0 if float('$mm') > float('$dec_stop_ms') else 1)"; then
        log "[np-scan] 档=$np ${SLA_METRIC}=$mm 超过 SLA ${dec_stop_ms}ms, 停止升压"
        break
      fi
      last_ok_np=$np_eff; last_ok_rps=$rps; last_ok_m=$mm; last_ok_dur=$dur; last_ok_done=$done_; last_ok_hit=$hitp
    fi
    if (( auto_mode && DRY_RUN == 0 && np_i > 0 )); then
      if ! "$PYTHON_BIN" -c "import sys; g='${gain}'; sys.exit(0 if g != '-' and float(g) >= float('${SAT_GAIN}') else 1)"; then
        log "[np-scan] rps 增益 ${gain}% < ${SAT_GAIN}% (边际提升), 停止上打"
        break
      fi
    fi
    prev_rps=$rps; np_i=$((np_i + 1))
  done

  # ---- calib: 复用 best_np 档的 inf 结果 ----
  if [[ -z "$best_np" ]]; then   # dry-run/全失败 → 回退首档
    if (( conc_ladder )); then best_np=$(fmul "${NP_SEQ[0]}" "$eff_mult"); else best_np="${NP_SEQ[0]}"; fi
  fi
  NUM_PROMPTS="$best_np"
  lad="$best_np"; (( conc_ladder )) && lad=$(( best_np / eff_mult ))
  MC_ROW=""; (( conc_ladder )) && MC_ROW="$lad"
  # 恢复全局 mc_args (阶梯循环曾按档覆盖); calib 回退重跑时再按需改回首档
  mc_args=(); [[ "$MAX_CONC" != "inf" ]] && mc_args=(--max-concurrency "$MAX_CONC")
  json="${LOG_DIR}/${tag}_inf.jsonl"; clog="${LOG_DIR}/${tag}_inf.log"
  if [[ -n "$best_rps" && "$best_rps" != "-1" ]]; then
    inf_rps=$best_rps; inf_m=$best_m; inf_dur=$best_dur; inf_done=$best_done; inf_hit=$best_hit
    [[ -f "${LOG_DIR}/${tag}_inf_c${lad}.jsonl" ]] \
      && cp "${LOG_DIR}/${tag}_inf_c${lad}.jsonl" "$json" 2>/dev/null || true
  elif (( DRY_RUN )); then
    read -r inf_rps inf_m inf_dur inf_done inf_hit <<< "0 0 0 0 0"
  else
    log "[calib] rate=inf np=${NP_SEQ[0]} (扫描全失败, 回退) -> $clog"
    (( conc_ladder )) && mc_args=(--max-concurrency "${NP_SEQ[0]}")
    if ! run_trial inf "$json" "$clog"; then
      warn "case $tag inf 标定失败, 见 $clog (跳过该 case)"; continue
    fi
    read -r inf_rps inf_m inf_dur inf_done inf_hit <<< "$(pick_metrics "$json")"
  fi
  # bs_scan rate 列: prefill 模式显式写 inf_rps (= 回测基准请求率),
  # decode 模式保持 inf (decode 不做请求率回测)
  if (( decode_mode )); then record inf "$inf_rps" "$inf_m" "$inf_dur" "$inf_done" "$inf_hit"
  else                           record "$inf_rps" "$inf_rps" "$inf_m" "$inf_dur" "$inf_done" "$inf_hit"; fi
  log "[calib] best_np=$NUM_PROMPTS rps=$inf_rps ${SLA_METRIC}=$inf_m cache_hit=${inf_hit}%"

  # ---- decode 模式: SLA 边界锁定, 不做请求率回测 ----
  if (( decode_mode )); then
    if [[ -n "$last_ok_np" ]]; then
      lock_type="decode_sla_lock"
      l_np=$last_ok_np; l_rps=$last_ok_rps; l_m=$last_ok_m; l_hit=$last_ok_hit; l_dur=$last_ok_dur
      l_tgt=$("$PYTHON_BIN" -c "print(float('${dec_stop_ms}')/1000)")
      log "[decode] SLA 锁定: np=$l_np rps=$l_rps ${SLA_METRIC}=$l_m <= ${dec_stop_ms}ms"
    else
      lock_type="decode_sla_not_found"
      l_np=$NUM_PROMPTS; l_rps=$inf_rps; l_m=$inf_m; l_hit=$inf_hit; l_dur=$inf_dur
      l_tgt=$("$PYTHON_BIN" -c "print(float('${dec_stop_ms}')/1000)" 2>/dev/null || echo -)
      warn "[decode] 无档满足 SLA<=${dec_stop_ms:-any}ms (最大 rps=$inf_rps)"
    fi
    l_lad="$l_np"; (( conc_ladder )) && l_lad=$(( l_np / eff_mult ))
    MC_ROW=""; (( conc_ladder )) && MC_ROW="$l_lad"
    lf_json="$json"
    if [[ -n "$last_ok_np" && "$last_ok_np" != "$NUM_PROMPTS" && "$eff_mult" -gt 1 ]]; then
      lad=$(( last_ok_np / eff_mult ))
      [[ -f "${LOG_DIR}/${tag}_inf_c${lad}.jsonl" ]] && lf_json="${LOG_DIR}/${tag}_inf_c${lad}.jsonl" && MC_ROW="$lad"
    fi
    printf '%s\t%s\t%s\t%s\n' "$(row_core inf "$l_np" "$lf_json" "$l_rps" "$l_m" "$l_dur" "$l_hit")" "$SLA_METRIC" "$l_tgt" "$lock_type" >> "$BACK_TSV"
    continue
  fi

  # ---- SLA 逐档回测 ----
  IFS='|' read -ra TARGETS <<< "$SLA_TARGETS"
  for tgt_s in "${TARGETS[@]}"; do
    [[ "$tgt_s" =~ ^[0-9.]+$ ]] || die "非法 SLA 阈值: $tgt_s"
    tgt_ms=$("$PYTHON_BIN" -c "print(int(float(${tgt_s})*1000))")
    if meet "$inf_m" "$tgt_ms"; then
      printf '%s\t%s\t%s\t%s\n' "$(row_core inf "$NUM_PROMPTS" "$json" "$inf_rps" "$inf_m" "$inf_dur" "$inf_hit")" "$SLA_METRIC" "$tgt_s" "inf_lock" >> "$BACK_TSV"
      log "[backtest] <${tgt_s}s: rate=inf rps=$inf_rps (inf 已满足, 锁定)"
      continue
    fi

    searches=0
    lo=""; lo_m=""; lo_rps=""; lo_hit="-"; lo_dur="-"; cur=$inf_rps; prev=""
    # 固定步长降率找首个满足点 (rate-step 粒度)
    while (( searches < MAX_TRIALS )); do
      [[ "$cur" == "$prev" ]] && break  # 已到 0.1 下限
      prev=$cur
      cur=$(fstep "$cur" "$RATE_STEP")
      json="${LOG_DIR}/${tag}_t${tgt_s}_r${cur}.jsonl"; clog="${LOG_DIR}/${tag}_t${tgt_s}_r${cur}.log"
      log "[descend] <${tgt_s}s rate=$cur (prev ${prev})"
      searches=$(( searches + 1 ))
      if (( DRY_RUN )); then m="0 0 0 0 0"; else
        run_trial "$cur" "$json" "$clog" || warn "trial rc=$? 继续"
        m=$(pick_metrics "$json")
      fi
      read -r rps mm dur done_ hitp <<< "$m"
      record "$cur" "$rps" "$mm" "$dur" "$done_" "$hitp"
      if meet "$mm" "$tgt_ms"; then lo=$cur; lo_m=$mm; lo_rps=$rps; lo_hit=$hitp; lo_dur=$dur; break; fi
    done

    if [[ -z "$lo" ]]; then
      printf '%s\t%s\t%s\t%s\n' "$(row_core "$cur" "$NUM_PROMPTS" "$json" - - - -)" "$SLA_METRIC" "$tgt_s" "not_found" >> "$BACK_TSV"
      warn "[backtest] <${tgt_s}s 未找到满足点 (rate 到 0.1 仍不满足)"
      continue
    fi

    printf '%s\t%s\t%s\t%s\n' "$(row_core "$lo" "$NUM_PROMPTS" "$json" "$lo_rps" "$lo_m" "$lo_dur" "$lo_hit")" "$SLA_METRIC" "$tgt_s" "descend_lock" >> "$BACK_TSV"
    log "[backtest] <${tgt_s}s: rate=$lo 锁定 (${SLA_METRIC}=$lo_m, cache_hit=$lo_hit%)"
  done
done

log "完成。扫描: $SCAN_TSV  回测: $BACK_TSV"
(( DRY_RUN == 1 )) || column -t -s $'\t' "$BACK_TSV"
