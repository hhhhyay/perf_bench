#!/bin/bash
# v5 repeat-inf: 每场景 3 次重复 calib(inf), 唯一 run-id 避开 SKIP-RESUME, 参数与首轮 calib 一致
cd /tmp || exit 9
B=http://192.168.0.35:30000
M=/model/DeepSeek-V4-Flash-0731-FP8-DSpark
F=/tmp/gsp_v2.sh
echo "=== REP-START $(date '+%F %T') ==="
run_case() {  # in out hit tag
  local IN="$1" OUT="$2" HIT="$3" TAG="$4"
  for ci in c1 c2 c3; do
    local LOG=/tmp/drvrep_${TAG}_${ci}.log
    echo "--- [${TAG}:${ci}] $(date '+%F %T') ---"
    bash "$F" calib --base-url "$B" --model "$M" \
      --case "${IN},${OUT},${HIT},mean_ttft_ms,3,mean_tpot_ms,999999" \
      --groups 4 --qpp 16 --run-id "v5rep_${TAG}_${ci}" > "$LOG" 2>&1
    grep CALIB_RESULT "$LOG" | tail -1 || echo "[${TAG}:${ci}] CALIB_FAIL"
  done
}
run_case 4096 1 0 4k
run_case 65536 1 90 64k
run_case 131072 1 90 128k
run_case 131072 1024 90 128k1k
echo "=== REPEAT-SUMMARY ==="
for TAG in 4k 64k 128k 128k1k; do
  vals=""
  for ci in c1 c2 c3; do
    v=$(grep -o 'inf_rps=[0-9.]*' /tmp/drvrep_${TAG}_${ci}.log 2>/dev/null | tail -1 | cut -d= -f2)
    vals="$vals $v"
  done
  echo "REPEAT_SUMMARY ${TAG}:${vals}"
done
echo "=== REP-DONE $(date '+%F %T') ==="
echo "REPEAT_ALL_DONE $(date '+%F %T')"
