# perf_bench

SGLang / vLLM 推理服务容量压测脚本集合，重点用于按照 per-DP batch size 扫描服务吞吐、TTFT、TPOT、ITL、E2EL 等指标，并把单次压测输出汇总为 CSV。

## 目录结构

```text
perf_bench/
├── sglang/
│   ├── sglc.sh      # SGLang 客户端容量压测
│   ├── prefill_ttft.sh # 单侧 Prefill TTFT 矩阵压测
│   └── gsp_prefix_bench.sh # GSP prefix 命中率下的 TTFT/TPOT 容量压测
└── vllm/
    └── vllmc.sh     # vLLM Prefix-Salt 标准容量压测
```

## 脚本说明

### SGLang: `sglang/sglc.sh`

- 优先通过 `/v1/loads?include=all` 读取 `dp_size`、`max_total_num_tokens`、`max_running_requests`。
- `/v1/loads` 不可用时回退 `/metrics` 和环境变量。
- 自动读取 `/server_info` 中的 TP / DP / PP / EP / ACP / KV cache dtype / CUDA Graph decode BS。
- 理论 per-DP BS 同时受 KV token capacity 和 `max_running_requests` 约束。
- 每个 benchmark case 使用 `--flush-cache`；warmup 完成后清理服务端 Prefix Cache，避免预热请求影响正式测量。
- 支持 `SLA_RULES` inline 加密扫描：当当前 BS 命中任一 SLA 规则后，连续补测到下一个基础 BS 前。
- 支持 `HARD_STOP_RULES` 硬停止：指标命中阈值后停止当前 case 的后续 BS 测试。
- 直接从 benchmark 日志提取 RPS、Token 吞吐、请求数、耗时和时延分位数，不保存逐请求 JSONL。

### SGLang Prefill: `sglang/prefill_ttft.sh`

- 使用与 `sglc.sh` 相同的服务配置和 `bench_serving` 参数。
- 输出长度固定为 1 token，不再配置输出长度。
- 默认测试输入 `32K/64K/128K`，按 `PREFILL_TTFT_BS_RULES` 自动选择一个 per-DP BS，不扫描全部 BS。
- 对每个输入先跑 `request_rate=inf`，再从高请求率向下寻找满足每条 SLA 规则的临界值。
- 试探请求率统一保留两位小数；二分搜索区间默认缩小到 `0.1 req/s` 后停止，避免用过小步长制造没有实际意义的伪精度。
- SLA 规则按输入长度绑定，格式为 `input_k:metric:target_seconds:near_seconds`；target 和 near 使用秒，可填写小数。
- 保留平台固定的 `Mean/P95/P99 TTFT`、`Mean/P95/P99 TPOT`、`Mean/P95/P99 ITL` 和 `Mean E2E Latency`，并在其后放置 `P50/Std/P75/P90/P999/Max TTFT`。
- 不提取与 `P50 TTFT` 重复的 `Median TTFT`、其他 E2E 分位数及额外 TPOT/ITL 分位数，不保存逐请求 JSONL。

### SGLang GSP Prefix: `sglang/gsp_prefix_bench.sh`

只要测试需求包含 prefix/cache 命中率，就使用这个脚本。它调用官方
`generated-shared-prefix` 数据集，同时支持 Prefill、输出大于 1 的普通推理和后续 PD
分离场景，不把请求数命中比例当成 token 级 cache 命中率。

命中率按官方 `--cache-report` 的结果计算：

```text
实际命中率 = total_cached_tokens / total_prompt_tokens
```

脚本使用 `--gsp-ordered`，并把每个 prefix group 的首个请求视为冷 miss。给定总输入
长度 `L`、目标命中率 `H` 和每组请求数 `P`，自动计算：

```text
system_prompt_len = round((H / 100) * P * L / (P - 1))
question_len = L - system_prompt_len
理论命中率 = (P - 1) * system_prompt_len / (P * L)
```

默认使用 `32` 个 group、每组 `16` 个 prompt，即 `512` 个请求；默认 `TOKEN_UNIT=1024`，
需要十进制 k 时显式设置 `TOKEN_UNIT=1000`。输入和输出支持小数，例如 `0.5k`。

一个命令可以连续测试多个输入/输出、命中率和 SLA：

```bash
BASE_URL=http://127.0.0.1:30000 \
MODEL=/path/to/model \
bash sglang/gsp_prefix_bench.sh \
  --case '16k,1k,80,mean_ttft_ms,3|5,mean_tpot_ms,30' \
  --case '32k,0.5k,80,mean_ttft_ms,5|7,mean_tpot_ms,50'
```

case 格式为：

```text
INPUT,OUTPUT,HIT_RATE,TTFT_METRIC,TTFT_TARGETS_S,TPOT_METRIC,TPOT_TARGET_MS[,MAX_CONCURRENCY]
```

例如 `3|5` 表示分别查找 TTFT 小于 3 秒和 5 秒时的最大实际客户端 RPS；TPOT
限制为 30ms。有限请求率从高到低回测，记录实际 `request_throughput`，保留两位小数，
最小步长不低于 `0.1 req/s`。只有实测命中率在目标容差内、TTFT 和 TPOT 同时满足的点
才会进入容量结果。

默认只生成 CSV 和日志；需要矩阵图时指定格子中央显示的指标：

```bash
# 格子显示最大实际客户端 RPS
bash sglang/gsp_prefix_bench.sh --plot --cell rps \
  --case '16k,1k,80,mean_ttft_ms,3|5,mean_tpot_ms,30' \
  --case '32k,0.5k,80,mean_ttft_ms,5|7,mean_tpot_ms,50'

# 格子显示客户端输入 token throughput
bash sglang/gsp_prefix_bench.sh --plot --cell input_throughput \
  --case '16k,1k,80,mean_ttft_ms,3|5,mean_tpot_ms,30'
```

结果位于 `all_result/bench_gsp_prefix_<RUN_ID>/`：

- `logs/`：每个 case/请求率的官方 benchmark 日志和 `.exitcode`。
- `gsp_prefix_trials.csv`：所有请求率 trial、指定/理论/实际 cache 命中率、总 prompt/cache token、通用客户端吞吐、请求数、TTFT、TPOT、ITL、Mean E2E 和日志路径；Mean E2E 及之前的固定客户端指标顺序与现有脚本保持一致。
- `gsp_prefix_summary.csv`：每个 case/TTFT SLA 下满足 cache hit、TTFT、TPOT 的最大实际 RPS。
- `gsp_prefix_matrix.png`：指定 `--plot` 后生成的容量矩阵，列为输入/输出/命中率，行为 TTFT SLA，格子中央为 RPS 或输入吞吐。

脚本支持 `--dry-run` 核对官方命令而不连接服务；同一 `RUN_ID` 重新执行时会跳过已有成功
日志。缺少 `--cache-report` 或 benchmark 失败时保留日志并标记为失败，不用理论值伪造实测命中率。

### vLLM: `vllm/vllmc.sh`

- 使用 Prefix-Salt no-flush 模式；每个 case / BS 使用唯一 `cache_salt`，隔离不同 BS 与不同 run 的 Prefix Cache。
- 通过 `/server_info`、`SERVER_LOG`、`/metrics` 和环境变量提取服务配置与容量信息。
- CSV 前置常用指标与 SGLang 脚本口径对齐。
- 理论 BS = `min(token theory, max_num_seqs)`；如果无法获取 `max_num_seqs`，退回 token theory。
- 支持与 SGLang 相同的 `SLA_RULES` inline 加密扫描。
- 直接从 vLLM benchmark 日志提取 RPS、Token 吞吐、请求数、耗时、平均时延和分位数，不保存结果 JSON。

## 长度口径

脚本中的 `CASES` 默认使用二进制 K，即 `TOKEN_UNIT=1024`。例如 `"4 1"` 表示输入
`4096 tokens`、输出 `1024 tokens`；`"0.5 1"` 表示输入 `512 tokens`、输出 `1024 tokens`。
独立 Prefill 脚本的输出长度固定为 `1 token`；输入长度仍使用 `TOKEN_UNIT` 换算。
需要十进制口径时设置 `TOKEN_UNIT=1000`，此时 `"1 1"` 表示输入和输出各 `1000 tokens`。
小数换算后会四舍五入为整数 token；结果可根据 `input_len/output_len` 与 `input_k/output_k` 判断实际口径。

## 快速运行

### SGLang

```bash
cd perf_bench
BASE_URL=http://127.0.0.1:30000 \
MODEL=/path/to/model \
bash sglang/sglc.sh
```

### SGLang 单侧 Prefill TTFT 矩阵

```bash
cd perf_bench
BASE_URL=http://127.0.0.1:30000 \
MODEL=/path/to/model \
PREFILL_TTFT_INPUTS="32 64 128" \
PREFILL_SLA_RULES="32:mean_ttft_ms:3:0,32:mean_ttft_ms:5:0,32:mean_ttft_ms:10:0,64:mean_ttft_ms:3:0,64:mean_ttft_ms:5:0,64:mean_ttft_ms:10:0,128:mean_ttft_ms:3:0,128:mean_ttft_ms:5:0,128:mean_ttft_ms:10:0" \
PREFILL_TTFT_BS_RULES="8:256,16:128,32:64,64:32,128:16" \
bash sglang/prefill_ttft.sh
```

例如 8K/BS64 要求 P99 TTFT 小于 5 秒和 6 秒，32K/BS16 要求 Mean TTFT 小于 4 秒和 5 秒：

```bash
PREFILL_TTFT_INPUTS="8 32" \
PREFILL_TTFT_BS_RULES="8:64,32:16" \
PREFILL_SLA_RULES="8:p99_ttft_ms:5:0.5,8:p99_ttft_ms:6:0.5,32:mean_ttft_ms:4:0.5,32:mean_ttft_ms:5:0.5" \
PREFILL_TTFT_RATE_STEP=0.1 \
bash sglang/prefill_ttft.sh --plot sla
```

请求率搜索示例输出如下，`inf` 仅用于基线，有限请求率均保留两位小数；当当前通过点和失败点的区间不超过 `0.1 req/s` 时停止继续细分：

```text
[RUN] input=8K per_dp_bs=64 target=p99_ttft_ms<=5000 phase=binary rate=2.70
[RUN] input=8K per_dp_bs=64 target=p99_ttft_ms<=5000 phase=binary rate=2.60
[TARGET-PASS] input=8K p99_ttft_ms<=5000ms candidate_rps=2.70 rate=2.60 value=2990.44
```

Prefill 默认只输出 CSV。需要画图时，在同一次测试命令后追加 `--plot`：

```bash
# 参考容量矩阵：行是 TTFT SLA，列是输入长度/BS，格子是最大实际客户端 RPS
bash sglang/prefill_ttft.sh --plot mean
bash sglang/prefill_ttft.sh --plot p99

# 每个输入 case 使用自己的 SLA 指标，例如 8K=P99、32K=Mean
bash sglang/prefill_ttft.sh --plot sla

# 切换格子中央的数值：显示客户端输入 token 吞吐（tok/s）
bash sglang/prefill_ttft.sh --plot mean --cell input_throughput

# 也可以一次生成多种图
bash sglang/prefill_ttft.sh --plot mean --plot p99 --plot line
```

其中 `mean` 和 `p99` 分别按统一的 Mean TTFT、P99 TTFT 生成容量矩阵；`sla` 按每个输入 case 的 `target_metric` 和 `target_value_ms` 使用各自 SLA。列始终显示输入 token/BS；默认 `--cell rps`，格子显示日志提取的最大实际客户端 RPS；使用 `--cell input_throughput` 时，格子显示客户端输入 token 吞吐（tok/s），也可简写为 `--cell input_tp`。`line` 生成 TTFT-请求率曲线，`heatmap` 生成原始 TTFT 热力图。图片与 CSV 位于同一个结果目录；不指定 `--plot` 时不会生成图片。

脚本内部会把秒换算成毫秒参与判断，例如 `5.5` 秒变成 `5500ms`，`0.25` 秒变成 `250ms`。CSV 仍保留 `target_value_ms` 和 `target_near_ms` 字段，以兼容现有数据平台。

### vLLM

```bash
cd perf_bench
BASE_URL=http://127.0.0.1:8000 \
MODEL=/path/to/model \
bash vllm/vllmc.sh
```

## 常用环境变量

| 变量 | 说明 |
| --- | --- |
| `BASE_URL` | OpenAI 兼容服务地址。 |
| `MODEL` | benchmark 请求中使用的模型名或模型路径。 |
| `FIXED_PER_DP_BS` | 固定 per-DP BS 测试点，支持逗号或空格分隔多个值。 |
| `FIXED_CONCURRENCY` | 固定全局并发测试点，必须能被 `DP_SIZE` 整除。 |
| `NUM_PROMPTS_MULTIPLIER` | 请求数量倍率，`num_prompts = max_concurrency * NUM_PROMPTS_MULTIPLIER`。 |
| `SLA_DENSE_ENABLE` | 是否开启 SLA inline 加密扫描，默认 `1`。 |
| `SLA_RULES` | SLA 规则，格式为 `metric:target:near`，多个规则用逗号分隔。 |
| `HARD_STOP_RULES` | SGLang 硬停止规则，格式为 `metric:op:threshold`，多个规则用逗号分隔；默认关闭。 |
| `RUN_ID` | 固定结果目录后缀，用于中断后续测；不设置时自动使用当前时间戳。 |
| `DP_SIZE` | 手动指定 DP 数；通常由脚本自动识别。 |
| `KV_TOKENS` / `MAX_TOTAL_TOKENS` | 容量识别失败时的手动兜底值。 |
| `MAX_NUM_SEQS` / `MAX_RUNNING_REQUESTS` | vLLM / SGLang 的请求数上限兜底值。 |
| `TOKEN_UNIT` | K 的 token 定义，默认 `1024`；设为 `1000` 使用十进制 K。 |
| `PREFILL_TTFT_INPUTS` | Prefill 输入长度，默认 `32 64 128`，单位为 K。 |
| `PREFILL_SLA_RULES` | Prefill SLA 规则，格式为 `input_k:metric:target_seconds:near_seconds`，例如 `8:p99_ttft_ms:5:0.5`；默认对每个输入使用 Mean TTFT 的 3s/5s/10s。 |
| `PREFILL_TTFT_TARGETS_S` | 未显式设置 `PREFILL_SLA_RULES` 时使用的默认目标秒数，默认 `3 5 10`。 |
| `PREFILL_TTFT_SEARCH_ITERS` | 每个 TTFT 目标最多搜索次数，默认 `7`。 |
| `PREFILL_TTFT_RATE_STEP` | 请求率搜索的最小区间，默认 `0.1 req/s`；不允许设置小于 `0.1`。有限请求率传给 benchmark 并写入 CSV 时保留两位小数。 |
| `PREFILL_TTFT_BS_RULES` | 输入上限 K 到 per-DP BS 的规则，默认 `8:256,16:128,32:64,64:32,128:16`。 |
| `--plot mean` / `--plot p99` / `--plot sla` | 可选作图入口，生成统一指标或按 case SLA 的容量矩阵；也支持 `--plot line` 和 `--plot heatmap`。可重复指定多个模式。 |
| `--cell` | 容量矩阵格子中央的数值，支持 `rps`、`input_throughput`（简写 `input_tp`）；默认 `rps`。列固定为输入 token/BS。 |

`SLA_RULES` 示例：

```bash
SLA_RULES="mean_tpot_ms:50:5,p99_tpot_ms:75:5,mean_ttft_ms:1000:100"
```

含义是当某个 BS 的指标落入 `target ± near` 区间时，立即连续补测当前基础 BS 到下一个基础 BS 之间的整数 BS。

`HARD_STOP_RULES` 示例：

```bash
HARD_STOP_RULES="mean_tpot_ms:ge:100,p99_tpot_ms:ge:150,mean_ttft_ms:ge:5000" \
bash sglang/sglc.sh
```

规则格式为 `metric:op:threshold`：

- `metric` 使用 `sum_all.csv` 中的指标名，例如 `mean_tpot_ms`、`p99_tpot_ms`、`mean_ttft_ms`、`p99_ttft_ms`。
- `op` 支持 `ge`（greater than or equal，大于等于，`>=`）、`gt`（greater than，大于，`>`）、`le`（less than or equal，小于等于，`<=`）和 `lt`（less than，小于，`<`）。
- 多条规则用逗号分隔，任意一条命中就会触发硬停止。
- 每个 BS 完成并汇总到 `sum_all.csv` 后检查；命中后停止当前 case 的后续 BS，不影响已经完成的结果。
- 默认值为空，表示关闭硬停止；如果同时命中 `SLA_RULES`，会先完成 SLA inline dense 补点，再执行硬停止。

## 续测

脚本默认使用时间戳创建新的结果目录。需要续测时，使用原结果目录的时间戳作为
`RUN_ID`，并保持服务地址、模型、`CASES`、DP 配置、`TOKEN_UNIT` 和 BS 配置一致：

```bash
# 例如原目录为 ./all_result/bench_theory_bs_20261012_143000
RUN_ID=20261012_143000 \
BASE_URL=http://127.0.0.1:30000 \
MODEL=/path/to/model \
bash sglang/sglc.sh
```

vLLM 使用相同方式，目录前缀为 `bench_vllm_prefix_salt_`：

```bash
RUN_ID=20261012_143000 \
BASE_URL=http://127.0.0.1:8000 \
MODEL=/path/to/model \
bash vllm/vllmc.sh
```

续测规则：

- 已有非空日志且对应 `.log.exitcode` 为 `0` 的 case / BS 会输出 `[SKIP]` 并跳过。
- 失败或未完成的 case / BS 会重新执行；`sum_all.csv` 会根据当前日志重新汇总。
- 如果上次因 `HARD_STOP_RULES` 命中而停止，需要清空或调整该规则后再续测，否则会在相同阈值处再次停止。
- 不设置 `RUN_ID` 会创建全新的结果目录，不会读取旧目录中的结果。

Prefill TTFT 矩阵也通过 `RUN_ID` 续测：

```bash
RUN_ID=20261015_143000 \
BASE_URL=http://127.0.0.1:30000 \
MODEL=/path/to/model \
bash sglang/prefill_ttft.sh
```

已有成功日志和 `.log.exitcode=0` 的 Prefill trial 会跳过，结果保存在
`all_result/bench_prefill_ttft_<RUN_ID>/`。

## 固定并发示例

固定 per-DP BS：

```bash
FIXED_PER_DP_BS="8,12,16" bash sglang/sglc.sh
FIXED_PER_DP_BS="8,12,16" bash vllm/vllmc.sh
```

固定全局并发：

```bash
FIXED_CONCURRENCY="64 96 128" bash sglang/sglc.sh
FIXED_CONCURRENCY="64 96 128" bash vllm/vllmc.sh
```

## 输出

脚本会在当前目录生成 `all_result/` 子目录，每次运行一个独立结果目录，包含：

- `logs/`：每个 case / BS 的执行日志。
- `sum_all.csv`：汇总后的核心指标表。
- `prefill_ttft_trials.csv`：Prefill 基准和搜索 trial 明细。
- `prefill_ttft_matrix.csv`：每个输入长度和 TTFT 目标下的最优达标结果。
- `prefill_ttft_*_capacity_matrix_*.png`：指定 `--plot mean` 或 `--plot p99` 时生成的容量矩阵，文件名末尾标记中央指标。
- `prefill_ttft_line.png` / `prefill_ttft_heatmap.png`：指定对应绘图模式时生成的辅助图。

Prefill CSV 的客户端字段布局与 `sglc.sh` 对齐：保留前置客户端指标、`Mean/P95/P99 TTFT`、`Mean/P95/P99 TPOT`、`Mean/P95/P99 ITL`、`Mean E2E Latency`，并在其后放置 `P50/Std/P75/P90/P999/Max TTFT`。`Median TTFT`、其他 E2E 分位数及额外 TPOT/ITL 分位数不提取。矩阵目标列同时记录 `target_metric`、`target_value_ms` 和 `target_near_ms`。

`all_result/` 是运行产物目录，建议不要提交到仓库。
