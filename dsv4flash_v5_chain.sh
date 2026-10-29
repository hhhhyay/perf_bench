#!/bin/bash
setsid bash -c "bash /tmp/_drv_v5.sh 4096 1 0 '3|4|5' 4k; bash /tmp/_drv_v5.sh 65536 1 90 '3|4|5|6' 64k; bash /tmp/_drv_v5.sh 131072 1 90 '3|4|5|6' 128k; bash /tmp/_drv_v5.sh 131072 1024 90 '3|4|5|6' 128k1k" > /tmp/drv_chain_v5.log 2>&1 < /dev/null &
sleep 1
exit 0
