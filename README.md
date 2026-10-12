# perf_bench

SGLang / vLLM 推理服务容量压测脚本集合，重点用于按照 per-DP batch size 扫描服务吞吐、TTFT、TPOT、ITL、E2EL 等指标，并把单次压测输出汇总为 CSV。

## 目录结构

```text
perf_bench/
├── sglang/
│   └── sglc.sh      # SGLang 客户端容量压测
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

`all_result/` 是运行产物目录，建议不要提交到仓库。
