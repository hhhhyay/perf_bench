#!/bin/bash
# v5 新口径驱动: output=1 表示 1 token (非 1k); usage:
#   bash _drv_v5.sh <in> <out> <hit> <ttft_targets> <tag>
# calib 一次 inf 标定 -> grid 从 inf 实际 rps 降序 step 0.1 网格, 档位锁定即停
cd /tmp || exit 9
IN="${1:?in}"; OUT="${2:?out}"; HIT="${3:?hit}"; TT="${4:?ttft}"; TAG="${5:?tag}"
B=http://192.168.0.35:30000
M=/model/DeepSeek-V4-Flash-0731-FP8-DSpark
F=/tmp/gsp_v2.sh
LOG=/tmp/drv5_${TAG}.log
COMMON="--groups 4 --qpp 16 --skip-confirm --rate-step 0.1 --mode grid"
HIF=/tmp/.hi_v5_${TAG}
echo "=== [v5:${TAG}] start $(date '+%F %T') case=${IN}_${OUT}_${HIT} ttft=${TT} ==="
if [[ ! -s "$HIF" ]]; then
  bash "$F" calib --base-url "$B" --model "$M" \
    --case "${IN},${OUT},${HIT},mean_ttft_ms,${TT%%|*},mean_tpot_ms,999999" \
    --groups 4 --qpp 16 --run-id "v5_${TAG}_calib" >> "$LOG" 2>&1
  HI="$(awk '/CALIB_RESULT/{for(i=1;i<=NF;i++) if($i ~ /^inf_rps=/){sub(/^inf_rps=/,"",$i); print $i}}' "$LOG" | tail -1)"
  [[ -n "$HI" ]] || { echo "[v5:${TAG}] CALIB_FAIL"; exit 8; }
  echo "$HI" > "$HIF"
  echo "=== [v5:${TAG}] CALIB ok HI=${HI} $(date '+%F %T') ==="
else
  HI="$(cat "$HIF")"
  echo "=== [v5:${TAG}] CALIB reuse HI=${HI} ==="
fi
bash "$F" run --base-url "$B" --model "$M" \
  --case "${IN},${OUT},${HIT},mean_ttft_ms,${TT},mean_tpot_ms,999999" \
  --rate-hi "$HI" $COMMON --run-id "v5_${TAG}_grid" >> "$LOG" 2>&1
echo "=== [v5:${TAG}] GRID done $(date '+%F %T') ==="
echo "DRIVER_V5_${TAG}_DONE $(date '+%F %T')"
