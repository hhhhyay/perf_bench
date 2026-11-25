#!/usr/bin/env bash
# ==============================================================================
#  vllm_prefix_cache_bench.sh -- vLLM prefix caching(APC) 命中率扫描测试
#
#  原理:
#    用 vllm_fake_prefill_client.py 对同一 (input,output) 配置扫描一组
#    PREFIX_HIT (理论命中配比), 每档:
#      hit=0          纯随机 prompt, 无共享前缀  -> 冷基线
#      hit>0          先 warmup 把共享前缀烘进 APC, 主请求按配比命中
#    采集 TTFT/TPOT/吞吐 与服务端 reported cached_tokens (实测命中率),
#    产出 "理论命中 vs 实测命中 vs TTFT 加速比" 矩阵, 观察 prefix caching
#    收益边界 (命中率越高, TTFT 越低; 100%≈decode-only)。
#
#  用法:
#    bash vllm_prefix_cache_bench.sh --help
#    bash vllm_prefix_cache_bench.sh --run --clear
#    bash vllm_prefix_cache_bench.sh --run --case '4k,1,0,25,50,75,90,100'
#    bash vllm_prefix_cache_bench.sh --last-run
#
#  case 格式: INPUT,OUTPUT,HITS
#    HITS: 逗号分隔的命中配比列表, 至少包含 0 (作为基线), 例 0,50,100
#
#  环境变量 (默认值): 同 vllm_fake_prefill_bench.sh, 另有
#    NUM_PROMPTS=32  REQUEST_RATE=inf  PAGE_SIZE=64  PREFIX_SEED=1234
#    OUTPUT_ROOT=all_result  PYTHON_BIN=python3
# ==============================================================================

dry_run=1
new_run=0
clear=0
last_run=0
run_cases=()

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
REQUEST_RATE=${REQUEST_RATE:-inf}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-0}
NUM_PROMPTS=${NUM_PROMPTS:-32}

usage() {
  sed -n '3,36p' "$0" | sed 's/^# \{0,1\}//'
  echo "  options:"
  echo "    --run            正式执行 (默认 dry-run)"
  echo "    --dry-run        只打印配置"
  echo "    --new-run        新 RUN_ID 开新一轮"
  echo "    --last-run       打印上一轮汇总"
  echo "    --clear          清空该 RUN_ID 重测"
  echo "    --check          离线自检 (tokenizer + 数据集)"
  echo "    --case SPEC      input,output,hits 例: 2k,1,0,25,50,75,90,100"
  echo "    --num-prompts N  覆盖 NUM_PROMPTS"
}

runid_last() { ls -dr "$OUTPUT_ROOT"/bench_vllm_prefix_cache_* 2>/dev/null | head -n1 | xargs -n1 basename 2>/dev/null; }
runid_make() {
  local ts; ts="$(date +%Y%m%d_%H%M%S)"
  echo "bench_vllm_prefix_cache_${ts}_$(printf '%04x' $((RANDOM % 65536)))"
}

len_tokens() {
  local input="$1"
  if [[ "${input,,}" =~ ^([0-9]+)k$ ]]; then
    echo $(( ${input:0:${#input}-1} * 1024 ))
  elif [[ "${input,,}" =~ ^([0-9]+)m$ ]]; then
    echo $(( ${input:0:${#input}-1} * 1024 * 1024 ))
  else
    echo "$input"
  fi
}

TRIAL_COLS=(run_id ts case_id input_tokens output_tokens prefix_hit_pct
  num_prompts request_rate successful_requests failed_requests duration_s
  request_throughput input_tput output_tput total_tput
  mean_ttft_ms p50_ttft_ms p95_ttft_ms p99_ttft_ms
  mean_tpot_ms mean_itl_ms mean_e2el_ms measured_hit_pct ttft_speedup_vs_cold log)

ensure_dirs() { [[ -d "$LOGS_DIR" ]] || mkdir -p "$LOGS_DIR"; [[ -d "$WORK_DIR" ]] || mkdir -p "$WORK_DIR"; }
ensure_csv() {
  if [[ ! -f "$RUN_DIR/vllm_prefix_cache_trials.csv" ]]; then
    printf '%s\n' "$(IFS=,; echo "${TRIAL_COLS[*]}")" > "$RUN_DIR/vllm_prefix_cache_trials.csv"
  fi
}
clear_run() {
  rm -rf "$LOGS_DIR" "$WORK_DIR" 2>/dev/null
  rm -f "$RUN_DIR/vllm_prefix_cache_trials.csv" "$RUN_DIR/vllm_prefix_cache_matrix.tsv" 2>/dev/null
  ensure_dirs; ensure_csv
}

write_meta_header() {
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
  } > "$1"
}

run_one() { # input output hit case_id
  local in="$1" out="$2" hit="$3" case_id="$4"
  local logname="prefix_cache_${case_id}_hit${hit%%.*}.log"
  local logpath="$LOGS_DIR/$logname"
  local result_json="$WORK_DIR/prefix_cache_${case_id}_hit${hit%%.*}.json"
  local salt="pc_${RUN_ID}_${case_id}_hit${hit%%.*}_$(date +%s)"  # 时间戳隔离, 防重跑冷基线被旧缓存污染

  write_meta_header "$logpath"
  if [[ "$dry_run" == "1" ]]; then
    echo "    [dry-run] python $CLIENT --input-len $in --output-len $out --prefix-hit $hit ..."
    return 0
  fi
  echo "    run: ${in}x${out} hit=${hit}"
  if ! CASE_ID="$case_id" "$PYTHON" "$CLIENT" \
      --base-url "$BASE_URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
      --input-len "$in" --output-len "$out" \
      --num-prompts "$NUM_PROMPTS" --prefix-hit "$hit" \
      --request-rate "$REQUEST_RATE" --max-concurrency "$MAX_CONCURRENCY" \
      --page-size "$PAGE_SIZE" --seed "$PREFIX_SEED" \
      --cache-salt "$salt" --result-json "$result_json" \
      >> "$logpath" 2>&1; then
    echo "    [ERROR] client failed, see $logpath"
    return 1
  fi
  local nok; nok="$("$PYTHON" - "$result_json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    d = json.load(fh)
print(d.get("successful_requests", 0))
PY
)"
  if [[ -z "$nok" || "$nok" == "0" ]]; then
    echo "    [ERROR] empty/failed result: $result_json"
    return 1
  fi
  "$PYTHON" - "$RUN_DIR/vllm_prefix_cache_trials.csv" "$result_json" \
    "$logname" "$RUN_ID" "$case_id" "$(date +%F_%T)" "$REQUEST_RATE" <<'PY'
import csv, json, sys
trials_csv, result_json, logname, run_id, case_id, ts, request_rate = sys.argv[1:8]
with open(result_json, encoding="utf-8") as fh:
    d = json.load(fh)
exported = {
    "run_id": run_id, "ts": ts, "case_id": case_id,
    "input_tokens": d["input_len"], "output_tokens": d["output_len"],
    "prefix_hit_pct": d["prefix_hit_pct"],
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
    "ttft_speedup_vs_cold": "",
    "log": logname,
}
cols = open(trials_csv, encoding="utf-8").readline().strip().split(",")
with open(trials_csv, "a", encoding="utf-8", newline="") as fh:
    csv.writer(fh).writerow([exported.get(c, "") for c in cols])
PY
  return 0
}

run_case() { # INPUT,OUTPUT,HITS
  local spec="$1"
  local in_name out_name hits
  IFS=',' read -r in_name out_name hits <<<"$spec"
  hits=${hits:-"0,50,100"}
  if [[ ",${hits}," != *",0,"* ]]; then
    hits="0,${hits}"          # 基线必须存在
  fi
  hits=${hits//,/ }           # "0,50,100" -> "0 50 100"
  case_id="${in_name}_${out_name}"
  local in; in="$(len_tokens "$in_name")"
  local out; out="$out_name"
  mkdir -p "$RUN_DIR/logs" "$RUN_DIR/work"
  echo "  case: input=$in output=$out hits=[$hits]"
  local hit
  for hit in $hits; do
    run_one "$in" "$out" "$hit" "$case_id" || true
  done
  # 汇总矩阵: 命中率->TTFT/加速比/实测命中
  "$PYTHON" - "$RUN_DIR/vllm_prefix_cache_trials.csv" "$RUN_ID" "$case_id" \
    "$RUN_DIR/vllm_prefix_cache_matrix.tsv" <<'PY'
import csv, os, sys
trials_csv, run_id, case_id, matrix_tsv = sys.argv[1:5]
rows = []
with open(trials_csv, encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["run_id"] == run_id and r["case_id"] == case_id:
            rows.append(r)
by_hit = {float(r["prefix_hit_pct"]): r for r in rows if r["successful_requests"] not in ("", "0")}
if not by_hit:
    print("    [matrix] no successful runs yet")
    sys.exit(0)
cold = by_hit.get(0.0)
cold_ttft = float(cold["mean_ttft_ms"]) if cold else None
header = not os.path.isfile(matrix_tsv)
with open(matrix_tsv, "a", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh, delimiter="\t")
    if header:
        w.writerow(["run_id", "case_id", "input", "output", "theoretical_hit_pct",
                    "measured_hit_pct", "mean_ttft_ms", "p50_ttft_ms", "p99_ttft_ms",
                    "ttft_speedup_vs_cold", "input_tput", "output_tput"])
    for hit in sorted(by_hit):
        r = by_hit[hit]
        ttft = float(r["mean_ttft_ms"])
        speedup = (cold_ttft / ttft) if (cold_ttft and ttft > 0) else ""
        w.writerow([run_id, case_id, r["input_tokens"], r["output_tokens"],
                    hit, r["measured_hit_pct"], ttft, r["p50_ttft_ms"],
                    r["p99_ttft_ms"],
                    round(speedup, 3) if speedup != "" else "",
                    r["input_tput"], r["output_tput"]])
print(f"    [matrix] {case_id}: {len(by_hit)} hit levels -> {matrix_tsv}")
if cold_ttft:
    for hit in sorted(by_hit):
        r = by_hit[hit]
        ttft = float(r["mean_ttft_ms"])
        print(f"      hit={hit:>5.1f}% measured={float(r['measured_hit_pct']):>6.2f}% "
              f"ttft={ttft:>9.2f}ms speedup={cold_ttft / ttft if ttft > 0 else float('inf'):>6.2f}x")
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) dry_run=0 ;;
    --dry-run) dry_run=1 ;;
    --new-run) new_run=1 ;;
    --clear) clear=1 ;;
    --last-run) last_run=1 ;;
    --check) "$PYTHON" "$CLIENT" --self-check --tokenizer "$TOKENIZER" \
              --input-len 2048 --prefix-hit 100 --num-prompts 4 --page-size "$PAGE_SIZE"; exit $? ;;
    --case) run_cases+=("$2"); shift ;;
    --num-prompts) NUM_PROMPTS="$2"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1"; usage; exit 1 ;;
  esac
  shift
done

if [[ "$last_run" == "1" ]]; then
  last="$(runid_last)"
  if [[ -z "$last" ]]; then
    echo "no previous run"; exit 1
  fi
  echo "== last run: $last =="
  [[ -f "$last/vllm_prefix_cache_matrix.tsv" ]] && cat "$last/vllm_prefix_cache_matrix.tsv" || echo "(empty)"
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
ensure_dirs; ensure_csv
[[ "$clear" == "1" ]] && clear_run

if [[ ${#run_cases[@]} -eq 0 ]]; then
  run_cases=("2k,1,0,25,50,75,90,100")
fi

echo "== vllm prefix cache bench =="
echo "   run_id : $RUN_ID"
echo "   server : $BASE_URL"
echo "   model  : $MODEL"
echo "   prompts: $NUM_PROMPTS rate=$REQUEST_RATE maxc=$MAX_CONCURRENCY page=$PAGE_SIZE seed=$PREFIX_SEED"

for spec in "${run_cases[@]}"; do
  run_case "$spec"
done

echo "== matrix =="
if [[ -f "$RUN_DIR/vllm_prefix_cache_matrix.tsv" ]]; then
  if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$RUN_DIR/vllm_prefix_cache_matrix.tsv"
  else
    cat "$RUN_DIR/vllm_prefix_cache_matrix.tsv"
  fi
else
  echo "(no matrix yet)"
fi
