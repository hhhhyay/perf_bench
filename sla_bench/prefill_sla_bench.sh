#!/usr/bin/env bash
# ============================================================
# prefill_sla_bench.sh v6 — 可变命中率 prefix 数据集 + np 饱和扫描 + 0.1 回测 (+decode 模式)
# ============================================================
# 数据集: sla_bench/prefix_hit.py (PREFIX_HIT 参数化, input/output len 任意)
# 模式分流 (按 OUTPUT 自动判定):
#   OUTPUT=1  → prefill 模式: np 饱和扫描 + 0.1 降率 SLA 回测 (下述规则)
#   OUTPUT>1  → decode 模式: --max-concurrency 限并发 (默认 inf=不传该参),
#               只跑 np 扫描/标定, 不做请求率回测
#   decode 限并发时: 扫描/标定的档位是并发数 C, 实际请求数 = C × --np-mult
#   (默认 3), 请求池持续补位形成稳态并发, 避免池子=并发跑完即止。
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
#       --max-concurrency N|inf  最大并发 (默认 inf=不加该参数;
#                          decode 模式即按此并发出压, prefill 同样生效)
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

declare -a CASES=()
declare -a PASSTHROUGH=()

usage() { sed -n '2,44p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
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

# OUT_DIR 必须在参数解析后赋值, 否则 --run-id 不生效
OUT_DIR="./all_result/sla_${RUN_ID}"
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

NP_TSV="${OUT_DIR}/np_scan.tsv"
SCAN_TSV="${OUT_DIR}/bs_scan.tsv"
BACK_TSV="${OUT_DIR}/sla_backtest.tsv"
echo -e "case\tinput_len\toutput_len\thit_pct\tnum_prompts\trate\tactual_rps\tduration_s\t${SLA_METRIC}\tgain_vs_prev_pct" > "$NP_TSV"
echo -e "case\tinput_len\toutput_len\thit_pct\tcached_len\tnum_prompts\trate\tactual_rps\t${SLA_METRIC}\tduration_s\tcompleted\tactual_cache_hit_pct" > "$SCAN_TSV"
echo -e "case\tinput_len\toutput_len\thit_pct\tsla_metric\ttarget_s\tnum_prompts\tlocked_rate\tlocked_rps\t${SLA_METRIC}\tduration_s\tactual_cache_hit_pct\tlock_type" > "$BACK_TSV"

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
    local rate=$1 json=$2 clog=$3
    # [flush-patch] 每 trial 前清 radix cache, 保证首触碰命中率口径 (防跨 trial 污染)
    curl -s --max-time 10 -X POST "${BASE_URL}/flush_cache" >/dev/null 2>&1 || true
    sleep 2
    ( cd "$script_dir/.." && PREFIX_HIT="$hit_s" PREFIX_SEED="$SEED" \
        "$PYTHON_BIN" "$script_dir/prefix_hit.py" \
        --backend sglang --base-url "$BASE_URL" --model "$MODEL" \
        --dataset-name random-ids --tokenize-prompt \
        --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
        --random-range-ratio 1.0 --request-rate "$rate" \
        --num-prompts "$NUM_PROMPTS" ${mc_args[@]+"${mc_args[@]}"} \
        --warmup-requests 0 --cache-report --output-details \
        --output-file "$json" --disable-tqdm \
        --extra-request-body '{"sampling_params":{"temperature":0.6,"top_p":0.95,"ignore_eos":true}}' \
        ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"} \
    ) > "$clog" 2>&1
  }
  record() { # rate rps metric dur completed hit
    echo -e "$tag\t$IN_LEN\t$OUT_LEN\t${hit_s}\t$CACHED_LEN\t$NUM_PROMPTS\t$1\t$2\t$3\t$4\t$5\t$6" >> "$SCAN_TSV"
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
    if (( conc_ladder )); then np_eff=$(fmul "$np" "$eff_mult"); else np_eff="$np"; fi
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
    echo -e "$tag\t$IN_LEN\t$OUT_LEN\t${hit_s}\t$np_eff\tinf\t$rps\t$dur\t$mm\t$gain" >> "$NP_TSV"
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
  json="${LOG_DIR}/${tag}_inf.jsonl"; clog="${LOG_DIR}/${tag}_inf.log"
  if [[ -n "$best_rps" && "$best_rps" != "-1" ]]; then
    inf_rps=$best_rps; inf_m=$best_m; inf_dur=$best_dur; inf_done=$best_done; inf_hit=$best_hit
    [[ -f "${LOG_DIR}/${tag}_inf_np${best_np}.jsonl" || -f "${LOG_DIR}/${tag}_inf_c${best_np}.jsonl" ]] \
      && cp "${LOG_DIR}/${tag}_inf_np${best_np}.jsonl" "$json" 2>/dev/null \
      || cp "${LOG_DIR}/${tag}_inf_c${best_np}.jsonl" "$json" 2>/dev/null || true
  elif (( DRY_RUN )); then
    read -r inf_rps inf_m inf_dur inf_done inf_hit <<< "0 0 0 0 0"
  else
    log "[calib] rate=inf np=${NP_SEQ[0]} (扫描全失败, 回退) -> $clog"
    if ! run_trial inf "$json" "$clog"; then
      warn "case $tag inf 标定失败, 见 $clog (跳过该 case)"; continue
    fi
    read -r inf_rps inf_m inf_dur inf_done inf_hit <<< "$(pick_metrics "$json")"
  fi
  record inf "$inf_rps" "$inf_m" "$inf_dur" "$inf_done" "$inf_hit"
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
    echo -e "$tag\t$IN_LEN\t$OUT_LEN\t${hit_s}\t$SLA_METRIC\t${l_tgt}\t$l_np\tinf\t$l_rps\t$l_m\t${l_dur:--}\t$l_hit\t$lock_type" >> "$BACK_TSV"
    continue
  fi

  # ---- SLA 逐档回测 ----
  IFS='|' read -ra TARGETS <<< "$SLA_TARGETS"
  for tgt_s in "${TARGETS[@]}"; do
    [[ "$tgt_s" =~ ^[0-9.]+$ ]] || die "非法 SLA 阈值: $tgt_s"
    tgt_ms=$("$PYTHON_BIN" -c "print(int(float(${tgt_s})*1000))")
    if meet "$inf_m" "$tgt_ms"; then
      echo -e "$tag\t$IN_LEN\t$OUT_LEN\t${hit_s}\t$SLA_METRIC\t${tgt_s}\t$NUM_PROMPTS\tinf\t$inf_rps\t$inf_m\t$inf_dur\t$inf_hit\tinf_lock" >> "$BACK_TSV"
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
      echo -e "$tag\t$IN_LEN\t$OUT_LEN\t${hit_s}\t$SLA_METRIC\t${tgt_s}\t$NUM_PROMPTS\t-\t-\t-\t-\t-\tnot_found" >> "$BACK_TSV"
      warn "[backtest] <${tgt_s}s 未找到满足点 (rate 到 0.1 仍不满足)"
      continue
    fi

    echo -e "$tag\t$IN_LEN\t$OUT_LEN\t${hit_s}\t$SLA_METRIC\t${tgt_s}\t$NUM_PROMPTS\t$lo\t$lo_rps\t$lo_m\t$lo_dur\t$lo_hit\tdescend_lock" >> "$BACK_TSV"
    log "[backtest] <${tgt_s}s: rate=$lo 锁定 (${SLA_METRIC}=$lo_m, cache_hit=$lo_hit%)"
  done
done

log "完成。扫描: $SCAN_TSV  回测: $BACK_TSV"
(( DRY_RUN == 1 )) || column -t -s $'\t' "$BACK_TSV"
