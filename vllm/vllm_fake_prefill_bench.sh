#!/usr/bin/env bash
# ==============================================================================
#  vllm_fake_prefill_bench.sh -- vLLM fake-prefill 客户端测试驱动
#
#  fake-prefill 原理 (纯客户端方式, 无需 PD 分离/fake 后端):
#    1) 客户端先发 1 条 warmup 请求, 把共享随机前缀烘进服务端 APC KV cache
#       (真实 prefill 只算一次)
#    2) 主请求 = 共享前缀 + 唯一 start token + 随机尾部
#       -> 除 1 个 unique token 外的 prefill 全部由 cache 命中 "伪造"
#       -> 整场压测等价于纯 decode 压力 (TTFT≈0, 与 SGLang PD fake 语义一致)
#
#  双重验证:
#    - 服务端 usage.prompt_tokens_details.cached_tokens 实测 token 命中率
#    - warm TTFT / cold TTFT 比值 (ratio <= FAKE_MAX_TTFT_RATIO 视为生效)
#
#  用法:
#    bash vllm_fake_prefill_bench.sh --help
#    bash vllm_fake_prefill_bench.sh --run --clear                 # 干净重跑默认 case
#    bash vllm_fake_prefill_bench.sh --run --case '4k,1,0|100'     # 指定 case
#    bash vllm_fake_prefill_bench.sh --last-run                    # 上一轮汇总
#    bash vllm_fake_prefill_bench.sh --check                       # 离线自检
#
#  case 格式: INPUT,OUTPUT,PHASES
#    PHASES: '|' 分隔的 prefix_hit 配比, 常见 "0|100"(冷基线 -> fake-prefill)
#    只给一个 >0 的配比(如 '100')时不产出配对判定, 只记录明细
#
#  主要环境变量 (默认值):
#    BASE_URL            服务地址            http://127.0.0.1:18901
#    MODEL               模型路径            /model/Qwen3-8B
#    TOKENIZER           tokenizer 路径      $MODEL
#    NUM_PROMPTS         每轮请求数          64
#    REQUEST_RATE        请求速率, inf/N     inf
#    MAX_CONCURRENCY     并发上限, 0 不限    0
#    PAGE_SIZE           前缀页大小 token    64
#    FAKE_MAX_TTFT_RATIO 判定: warm/cold 比  0.30
#    FAKE_MIN_HIT_PCT    判定: 最小实测命中  90
#    PREFIX_SEED         数据种子            1234
#    OUTPUT_ROOT         结果根目录          all_result
#    PYTHON_BIN          python 解释器       python3
# ==============================================================================

dry_run=1
new_run=0
clear=0
last_run=0
self_check=0
run_cases=()
run_num_prompts=-1
run_request_rate=

OUTPUT_ROOT=${OUTPUT_ROOT:-"all_result"}
PYTHON_BIN=${PYTHON_BIN:-"$(command -v python3 || echo python3)"}
if ! "$PYTHON_BIN" -c 'import argparse' >/dev/null 2>&1; then
  echo "FATAL: python3 not workable, set a valid PYTHON_BIN"
  exit 1
fi
PYTHON=${PYTHON_BIN}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CLIENT="${SCRIPT_DIR}/vllm_fake_prefill_client.py"

BASE_URL=${BASE_URL:-"http://127.0.0.1:18901"}
MODEL=${MODEL:-"/model/Qwen3-8B"}
TOKENIZER=${TOKENIZER:-"$MODEL"}
PAGE_SIZE=${PAGE_SIZE:-64}
PREFIX_SEED=${PREFIX_SEED:-1234}
FAKE_MAX_TTFT_RATIO=${FAKE_MAX_TTFT_RATIO:-0.30}
FAKE_MIN_HIT_PCT=${FAKE_MIN_HIT_PCT:-90}
REQUEST_RATE=${REQUEST_RATE:-inf}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-0}
NUM_PROMPTS=${NUM_PROMPTS:-64}

# --------------------------------------------------------------------------- #
usage() {
  sed -n '3,39p' "$0" | sed 's/^# \{0,1\}//'
  echo "  options:"
  echo "    --run            正式执行压测 (默认 dry-run)"
  echo "    --dry-run        只打印配置不压测"
  echo "    --new-run        生成新 RUN_ID 开新一轮"
  echo "    --last-run       打印上一轮的结果汇总"
  echo "    --clear          清空该 RUN_ID 下已测记录后重测"
  echo "    --check          离线自检: tokenizer + 数据集构造 (不发请求)"
  echo "    --case SPEC      形如 input,output,phases 例: 4k,1,0|100"
  echo "    --num-prompts N  覆盖 NUM_PROMPTS"
  echo "    --request-rate R 覆盖 REQUEST_RATE (inf 或数字)"
}

runid_last() { ls -dr "$OUTPUT_ROOT"/bench_vllm_fake_prefill_* 2>/dev/null | head -n1 | xargs -n1 basename 2>/dev/null; }
runid_make() {
  local ts; ts="$(date +%Y%m%d_%H%M%S)"
  echo "bench_vllm_fake_prefill_${ts}_$(printf '%04x' $((RANDOM % 65536)))"
}

len_tokens() {  # 1k/2k/4k/1m/纯数字 -> token 数
  local input="$1"
  if [[ "${input,,}" =~ ^([0-9]+)k$ ]]; then
    echo $(( ${input:0:${#input}-1} * 1024 ))
  elif [[ "${input,,}" =~ ^([0-9]+)m$ ]]; then
    echo $(( ${input:0:${#input}-1} * 1024 * 1024 ))
  else
    echo "$input"
  fi
}

TRIAL_COLS=(run_id ts case_id input_tokens output_tokens prefix_hit_pct phase
  num_prompts request_rate successful_requests failed_requests duration_s
  request_throughput input_tput output_tput total_tput
  mean_ttft_ms p50_ttft_ms p95_ttft_ms p99_ttft_ms
  mean_tpot_ms mean_itl_ms mean_e2el_ms
  measured_hit_pct hit_source ttft_ratio_vs_cold verdict log)

ensure_dirs() {
  [[ -d "$LOGS_DIR" ]] || mkdir -p "$LOGS_DIR"
  [[ -d "$WORK_DIR" ]] || mkdir -p "$WORK_DIR"
}

ensure_csv() {
  if [[ ! -f "$RUN_DIR/vllm_fake_prefill_trials.csv" ]]; then
    printf '%s\n' "$(IFS=,; echo "${TRIAL_COLS[*]}")" > "$RUN_DIR/vllm_fake_prefill_trials.csv"
  fi
}

clear_run() {
  rm -rf "$LOGS_DIR" "$WORK_DIR" 2>/dev/null
  rm -f "$RUN_DIR/vllm_fake_prefill_trials.csv" \
        "$RUN_DIR/vllm_fake_prefill_summary.tsv" 2>/dev/null
  ensure_dirs
  ensure_csv
}

json_num() { # file key -> raw json number (stdout)
  "$PYTHON" - "$1" "$2" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    data = json.load(fh)
value = data.get(sys.argv[2], "")
print("" if value is None else value)
PY
}

append_trial_row() { # result_json logname
  "$PYTHON" - "$RUN_DIR/vllm_fake_prefill_trials.csv" \
    "$1" "$2" "$3" "$(date +%F_%T)" "$REQUEST_RATE" <<'PY'
import csv, json, os, sys
trials_csv, result_json, logname, run_id, ts, request_rate = sys.argv[1:7]
with open(result_json, encoding="utf-8") as fh:
    d = json.load(fh)
exported = {
    "run_id": run_id, "ts": ts,
    "case_id": os.environ.get("CASE_ID", ""),
    "input_tokens": d["input_len"], "output_tokens": d["output_len"],
    "prefix_hit_pct": d["prefix_hit_pct"],
    "phase": "fake_prefill" if d["prefix_hit_pct"] > 0 else "cold_baseline",
    "num_prompts": d["num_prompts"], "request_rate": request_rate,
    "successful_requests": d["successful_requests"],
    "failed_requests": d["failed_requests"],
    "duration_s": d["duration_s"],
    "request_throughput": d["request_throughput"],
    "input_tput": d["input_throughput"],
    "output_tput": d["output_throughput"],
    "total_tput": d["total_throughput"],
    "mean_ttft_ms": d["mean_ttft_ms"], "p50_ttft_ms": d["p50_ttft_ms"],
    "p95_ttft_ms": d["p95_ttft_ms"], "p99_ttft_ms": d["p99_ttft_ms"],
    "mean_tpot_ms": d["mean_tpot_ms"] if d["mean_tpot_ms"] is not None else "",
    "mean_itl_ms": d["mean_itl_ms"] if d["mean_itl_ms"] is not None else "",
    "mean_e2el_ms": d["mean_e2el_ms"],
    "measured_hit_pct": d["measured_cache_hit_pct"],
    "hit_source": d.get("measured_hit_source", ""),
    "ttft_ratio_vs_cold": "", "verdict": "",
    "log": logname,
}
cols = open(trials_csv, encoding="utf-8").readline().strip().split(",")
with open(trials_csv, "a", encoding="utf-8", newline="") as fh:
    csv.writer(fh).writerow([exported.get(c, "") for c in cols])
PY
}

write_meta_header() { # logfile
  {
    printf '%-26s %s\n' run_id "$RUN_ID"
    printf '%-26s %s\n' ts "$(date +%F' '%T)"
    printf '%-26s %s\n' base_url "$BASE_URL"
    printf '%-26s %s\n' model "$MODEL"
    printf '%-26s %s\n' tokenizer "$TOKENIZER"
    printf '%-26s %s\n' num_prompts "$NUM_PROMPTS"
    printf '%-26s %s\n' request_rate "$REQUEST_RATE"
    printf '%-26s %s\n' max_concurrency "$MAX_CONCURRENCY"
    printf '%-26s %s\n' page_size "$PAGE_SIZE"
    printf '%-26s %s\n' prefix_seed "$PREFIX_SEED"
    printf '%-26s %s\n' fake_max_ttft_ratio "$FAKE_MAX_TTFT_RATIO"
    printf '%-26s %s\n' fake_min_hit_pct "$FAKE_MIN_HIT_PCT"
    printf '%-26s %s\n' client "$CLIENT"
  } > "$1"
}

run_one() { # input_len output_len hit
  local in="$1" out="$2" hit="$3"
  local dir; dir="${in}_${out}"
  local tag; tag="hit${hit%%.*}"
  local phase="fake_prefill"; [[ "$hit" == "0" ]] && phase="cold_baseline"
  local logname="vllm_fake_prefill_${dir}_${tag}.log"
  local logpath="$LOGS_DIR/$logname"
  local result_json="$WORK_DIR/fake_prefill_${dir}_${tag}.json"
  local salt="fakep_${RUN_ID}_${dir}_${tag}_$(date +%s)"  # 时间戳隔离服务端 APC, 防止重跑冷基线被旧缓存污染

  write_meta_header "$logpath"
  if [[ "$dry_run" == "1" ]]; then
    echo "    [dry-run] python $CLIENT --base-url $BASE_URL --model $MODEL \\"
    echo "        --tokenizer $TOKENIZER --input-len $in --output-len $out \\"
    echo "        --num-prompts $NUM_PROMPTS --prefix-hit $hit --request-rate $REQUEST_RATE \\"
    echo "        --max-concurrency $MAX_CONCURRENCY --page-size $PAGE_SIZE \\"
    echo "        --seed $PREFIX_SEED --cache-salt $salt"
    return 0
  fi

  echo "    run: input=$in output=$out hit=${hit} prompts=$NUM_PROMPTS rate=$REQUEST_RATE"
  if ! CASE_ID="$case_id" "$PYTHON" "$CLIENT" \
      --base-url "$BASE_URL" \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --input-len "$in" \
      --output-len "$out" \
      --num-prompts "$NUM_PROMPTS" \
      --prefix-hit "$hit" \
      --request-rate "$REQUEST_RATE" \
      --max-concurrency "$MAX_CONCURRENCY" \
      --page-size "$PAGE_SIZE" \
      --seed "$PREFIX_SEED" \
      --cache-salt "$salt" \
      --result-json "$result_json" \
      >> "$logpath" 2>&1; then
    echo "    [ERROR] client failed, see $logpath"
    return 1
  fi
  local nok; nok="$(json_num "$result_json" successful_requests)"
  if [[ -z "$nok" || "$nok" == "0" ]]; then
    echo "    [ERROR] empty/failed result: $result_json"
    return 1
  fi
  append_trial_row "$result_json" "$logname" "$RUN_ID"
  return 0
}

run_case() { # INPUT,OUTPUT,PHASES
  local spec="$1"
  local in_name out_name phases
  IFS=',' read -r in_name out_name phases <<<"$spec"
  phases=${phases:-"0|100"}
  case_id="${in_name}_${out_name}"
  export CASE_ID="$case_id"
  local in; in="$(len_tokens "$in_name")"
  local out; out="$out_name"
  mkdir -p "$RUN_DIR/logs" "$RUN_DIR/work"
  echo "  case: input=$in output=$out phases=$phases"
  local meet=0 total=0
  local phases_arr
  IFS='|' read -ra phases_arr <<<"$phases"
  for hit in "${phases_arr[@]}"; do
    if run_one "$in" "$out" "$hit"; then
      meet=$((meet + 1))
    fi
    total=$((total + 1))
  done
  "$PYTHON" - "$RUN_DIR/vllm_fake_prefill_trials.csv" "$RUN_ID" "$case_id" \
    "$FAKE_MAX_TTFT_RATIO" "$FAKE_MIN_HIT_PCT" \
    "$RUN_DIR/vllm_fake_prefill_summary.tsv" <<'PY'
import csv, os, sys
trials_csv, run_id, case_id, max_ratio, min_hit_pct, summary_tsv = sys.argv[1:7]
max_ratio, min_hit_pct = float(max_ratio), float(min_hit_pct)
rows = []
with open(trials_csv, encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["run_id"] == run_id and r["case_id"] == case_id:
            rows.append(r)
cold = next((r for r in rows if r["phase"] == "cold_baseline"), None)
warm = next((r for r in rows if r["phase"] == "fake_prefill"), None)
if cold and warm:
    c = float(cold["mean_ttft_ms"])
    w = float(warm["mean_ttft_ms"])
    ratio = (w / c) if c > 0 else float("inf")
    hitp = float(warm["measured_hit_pct"])
    verdict = "PASS" if (ratio <= max_ratio and hitp >= min_hit_pct) else "FAIL"
    for r in rows:
        r["verdict"] = verdict
        r["ttft_ratio_vs_cold"] = round(ratio, 4)
    with open(trials_csv, "w", encoding="utf-8", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    header = not os.path.isfile(summary_tsv)
    with open(summary_tsv, "a", encoding="utf-8", newline="") as fh:
        hdr = ["run_id", "case_id", "input_tokens", "output_tokens",
               "cold_mean_ttft_ms", "fake_mean_ttft_ms", "ttft_ratio",
               "measured_hit_pct", "verdict"]
        wr = csv.DictWriter(fh, fieldnames=hdr, delimiter="\t")
        if header:
            wr.writeheader()
        wr.writerow(dict(run_id=run_id, case_id=case_id,
                         input_tokens=cold["input_tokens"],
                         output_tokens=cold["output_tokens"],
                         cold_mean_ttft_ms=c, fake_mean_ttft_ms=w,
                         ttft_ratio=round(ratio, 4),
                         measured_hit_pct=hitp, verdict=verdict))
    print(f"    [verdict] ratio={ratio:.4f} (<= {max_ratio}), "
          f"measured_hit={hitp:.1f}% (>= {min_hit_pct}) -> {verdict}")
elif warm is None:
    print("    [verdict] no fake_prefill row (run failed or phases=0 only)")
elif cold is None:
    print("    [verdict] no cold_baseline row, pair verdict skipped")
PY
  echo "  case [$spec] done: $meet/$total phase runs ok"
}

# --------------------------------------------------------------------------- #
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) dry_run=0 ;;
    --dry-run) dry_run=1 ;;
    --new-run) new_run=1 ;;
    --clear) clear=1 ;;
    --last-run) last_run=1 ;;
    --check) self_check=1 ;;
    --case) run_cases+=("$2"); shift ;;
    --num-prompts) run_num_prompts="$2"; shift ;;
    --request-rate) run_request_rate="$2"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1"; usage; exit 1 ;;
  esac
  shift
done

if [[ "$last_run" == "1" ]]; then
  last="$(runid_last)"
  if [[ -z "$last" ]]; then
    echo "no previous run"
    exit 1
  fi
  echo "== last run: $last =="
  if [[ -f "$last/vllm_fake_prefill_summary.tsv" ]]; then
    cat "$last/vllm_fake_prefill_summary.tsv"
  else
    echo "(empty summary)"
  fi
  exit 0
fi

RUN_ID="$(runid_last)"
if [[ -z "$RUN_ID" || "$new_run" == "1" ]]; then
  RUN_ID="$(runid_make)"
  echo "new run id: $RUN_ID"
fi
export RUN_ID
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
LOGS_DIR="$RUN_DIR/logs"
WORK_DIR="$RUN_DIR/work"
ensure_dirs
ensure_csv

if [[ "$self_check" == "1" ]]; then
  echo "offline self check:"
  "$PYTHON" "$CLIENT" --self-check --tokenizer "$TOKENIZER" --input-len 4096 \
    --prefix-hit 100 --num-prompts 4 --page-size "$PAGE_SIZE"
  exit $?
fi

[[ "$run_num_prompts" -ge 0 ]] && NUM_PROMPTS=$run_num_prompts
[[ -n "$run_request_rate" ]] && REQUEST_RATE=$run_request_rate
[[ "$clear" == "1" ]] && clear_run

if [[ ${#run_cases[@]} -eq 0 ]]; then
  run_cases=("1k,1,0|100" "4k,1,0|100")
fi

echo "== vllm fake-prefill bench =="
echo "   run_id : $RUN_ID"
echo "   server : $BASE_URL"
echo "   model  : $MODEL (tokenizer: $TOKENIZER)"
echo "   prompts: $NUM_PROMPTS rate=$REQUEST_RATE maxc=$MAX_CONCURRENCY page=$PAGE_SIZE seed=$PREFIX_SEED"
echo "   verdict: ratio<=$FAKE_MAX_TTFT_RATIO and measured_hit>=$FAKE_MIN_HIT_PCT%"

for spec in "${run_cases[@]}"; do
  run_case "$spec"
done

echo "== summary =="
if [[ -f "$RUN_DIR/vllm_fake_prefill_summary.tsv" ]]; then
  if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$RUN_DIR/vllm_fake_prefill_summary.tsv"
  else
    cat "$RUN_DIR/vllm_fake_prefill_summary.tsv"
  fi
else
  echo "(no summary yet)"
fi
