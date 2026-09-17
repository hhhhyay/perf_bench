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
- 支持 `SLA_RULES` inline 加密扫描：当当前 BS 命中任一 SLA 规则后，连续补测到下一个基础 BS 前。

### vLLM: `vllm/vllmc.sh`

- 使用 Prefix-Salt no-flush 模式；每个 case / BS 使用唯一 `cache_salt`，隔离不同 BS 与不同 run 的 Prefix Cache。
- 通过 `/server_info`、`SERVER_LOG`、`/metrics` 和环境变量提取服务配置与容量信息。
- CSV 前置常用指标与 SGLang 脚本口径对齐。
- 理论 BS = `min(token theory, max_num_seqs)`；如果无法获取 `max_num_seqs`，退回 token theory。
- 支持与 SGLang 相同的 `SLA_RULES` inline 加密扫描。

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
| `DP_SIZE` | 手动指定 DP 数；通常由脚本自动识别。 |
| `KV_TOKENS` / `MAX_TOTAL_TOKENS` | 容量识别失败时的手动兜底值。 |
| `MAX_NUM_SEQS` / `MAX_RUNNING_REQUESTS` | vLLM / SGLang 的请求数上限兜底值。 |

`SLA_RULES` 示例：

```bash
SLA_RULES="mean_tpot_ms:50:5,p99_tpot_ms:75:5,mean_ttft_ms:1000:100"
```

含义是当某个 BS 的指标落入 `target ± near` 区间时，立即连续补测当前基础 BS 到下一个基础 BS 之间的整数 BS。

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

脚本会在当前目录生成 `all_result/` 子目录，每次运行一个独立结果目录，通常包含：

- `jsonl/`：原始 benchmark 输出。
- `logs/`：每个 case / BS 的执行日志。
- `sum_all.csv`：汇总后的核心指标表。

`all_result/` 是运行产物目录，建议不要提交到仓库。
