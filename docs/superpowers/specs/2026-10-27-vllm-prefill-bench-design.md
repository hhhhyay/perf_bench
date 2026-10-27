# vLLM Prefill Benchmark 设计

## 1. 背景与目标

当前仓库的 `vllm/vllmc.sh` 面向 Decode/容量扫描。新增独立脚本
`vllm/vllm_prefill_bench.sh`，用于固定输出长度为 1 的 Prefill 测试，按 TTFT SLA 查找
最大实际客户端请求率，不改变现有 Decode 脚本行为。

目标：

1. 使用官方 `vllm bench serve`，输入长度支持小数 K 和 1000/1024 两种口径。
2. 先以 `request_rate=inf` 找到输入吞吐不再提升的基线，再从高到低回测。
3. 回测步长默认 0.1 req/s，所有请求率保留两位小数。
4. 支持一次传入多个输入长度和多个 TTFT 目标。
5. 公共性能字段严格复用 `vllmc.sh` 的解析、别名和命名，只在 Prefill 侧增加 `SLO` 与请求率回测相关信息。
6. 输出与当前 GSP/SGLC 尽量对齐的日志、CSV/TSV 和容量矩阵图。

不在范围内：

- 修改 `vllm/vllmc.sh` 的 Decode/BS 扫描逻辑。
- vLLM Prefix Cache 命中率测试。
- 启动或重启 vLLM 服务。
- 用 TPOT 作为 Prefill 容量门槛；输出为 1 时 TPOT/ITL 可能没有有效含义，仍可保留原始字段。

## 2. 使用接口

推荐命令：

```bash
BASE_URL=http://127.0.0.1:8000 MODEL=/data/models/Qwen3.8-27B-Channel-INT8-w8a8 PREFILL_TARGET_DURATION_S=60 bash vllm/vllm_prefill_bench.sh --case '4k,mean_ttft_ms,3|4' --case '64k,p99_ttft_ms,5|6'
```

`--case` 格式：

```text
INPUT,TTFT_METRIC,TTFT_TARGETS_S
```

约定：

- `INPUT` 支持 `4k`、`0.5k`；默认 `k=1024`，设置 `TOKEN_UNIT=1000` 时使用十进制 K。
- `TTFT_METRIC` 支持 `mean_ttft_ms`、`p50_ttft_ms`、`p95_ttft_ms`、`p99_ttft_ms`。
- `TTFT_TARGETS_S` 使用 `|` 分隔多个秒级目标，例如 `3|4|5`。
- 输出长度固定为 `1`，不作为 case 参数。

主要环境变量：

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `BASE_URL` | `http://127.0.0.1:8000` | vLLM OpenAI 兼容服务地址 |
| `MODEL` | 无 | served model 或模型路径 |
| `TOKENIZER` | 空 | 客户端本地 tokenizer |
| `TOKEN_UNIT` | `1024` | K 的 token 定义 |
| `PREFILL_TARGET_DURATION_S` | `60` | 每个 case 目标时长 |
| `PREFILL_RATE_STEP` | `0.1` | 有限请求率下降步长，不能小于 0.1 |
| `PREFILL_INF_MAX_TRIALS` | `5` | inf 稳定性试跑上限 |
| `PREFILL_INF_NO_IMPROVEMENT_LIMIT` | `2` | 输入吞吐连续不提升次数 |
| `FIXED_CONCURRENCY` | 空 | 设置后才传递 `--max-concurrency` |
| `PREFILL_NUM_PROMPTS` | 自动 | 显式固定请求数 |
| `RUN_ID` | 当前时间 | 结果目录后缀，支持续测 |
| `PREFILL_PLOT` | `1` | 默认生成容量矩阵 |
| `PREFILL_PLOT_CELL` | `rps` | 矩阵格子显示 `rps` 或 `input_throughput` |

默认不传客户端 `--max-concurrency`。需要固定客户端并发时可设置 `FIXED_CONCURRENCY=64`。

## 3. 测试流程

每个 case 独立执行：

```text
解析 case -> 计算输入 token -> inf 试跑 -> 选择最高输入吞吐基线
-> 从基线实际 RPS 由高到低按 0.1 req/s 回测 -> 选择各 TTFT SLA 的最大 RPS
-> 写入 trials、summary 和矩阵图
```

### 3.1 inf 基线

- 固定 `--input-len` 和 `--output-len 1`，请求率使用 `inf`。
- 默认不设置 `--max-concurrency`；若设置 `FIXED_CONCURRENCY`，inf 和回测必须一致。
- 未显式固定请求数时，按实际 duration 动态调整请求数，默认目标为 60 秒。
- 重复 inf，直到 `input_throughput_tok_s` 连续不提升达到限制或达到最大次数。
- 只有成功请求数大于 0、核心指标可解析且 benchmark 成功的 trial 才能作为基线。
- 以有效 inf trial 中输入吞吐最高者作为有限请求率回测起点。

### 3.2 请求率回测

- 起点使用 inf 的实际客户端 RPS，不用字符串 `inf` 替代。
- 请求率从高到低递减，保留两位小数，步长不小于 0.1 req/s。
- 每个点保持输入长度、输出长度、请求数策略、tokenizer 和并发配置一致。
- 每个 TTFT 目标判断 `selected_metric <= target_seconds * 1000`，选择满足条件的最大实际 RPS。
- 所有目标找到结果或请求率降到最低后停止该 case；无满足点时保留 `NO_MEET`。

### 3.3 vLLM 指标统一解析

Prefill 不复制一套新的 vLLM 日志解析规则，而是复用 `vllmc.sh` 当前使用的标准化映射：

- `Request throughput` 映射为 `rps`，配置值 `Traffic request rate` 映射为 `request_rate`。
- `Input/Output/Total token throughput` 分别映射为 `input_throughput_tok_s`、
  `generate_throughput_tok_s`、`total_throughput_tok_s`。
- TTFT、TPOT、ITL 和 E2E 的 `mean/p95/p99` 使用相同字段名和单位（毫秒）。
- `Benchmark duration`、`Successful requests`、`Failed requests`、`Concurrency`、
  `Peak concurrent requests` 和 `Peak output token throughput` 使用相同字段名。
- P50、`Mean E2EL` 等 vLLM 日志别名沿用现有补齐逻辑，不由 Prefill 脚本另行解释。

实现时将公共解析/标准化逻辑提取为可被 `vllmc.sh` 和 Prefill 共同调用的 Python 模块，
保持现有 `vllmc.sh` 的 `sum_all.csv` 公共字段兼容；Prefill 只过滤不适用的 Decode/cache
字段，不改变公共字段的名字和数值口径。

## 4. 输出设计

结果目录：

```text
all_result/bench_vllm_prefill_<RUN_ID>/
├── run_metadata.json
├── logs/
├── vllm_prefill_trials.csv
├── vllm_prefill_summary.tsv
└── vllm_prefill_matrix.png
```

`vllm_prefill_trials.csv` 保存所有 inf、有限请求率、失败和未完成 trial，至少包含输入/输出长度、
请求数、并发、配置请求率、实际 RPS、duration、输入/生成/总吞吐、TTFT 分位数、TPOT/ITL/E2E
原始指标、退出码、成功请求数和日志路径。

`vllm_prefill_summary.tsv` 是最终复制文件。除 Prefill 专用的 `SLO` 外，公共字段严格使用
`vllmc.sh` 的标准顺序和命名；`input_k`、`output_k`、cache 字段、per-DP BS 字段和过程状态
字段不进入 Prefill summary：

```text
case_name, input_len, output_len, parallel_config, SLO, successful_requests, failed_requests,
num_prompts, max_concurrency, concurrency, peak_output_throughput, peak_concurrency, request_rate,
duration_s, rps, input_throughput_tok_s, generate_throughput_tok_s, total_throughput_tok_s,
mean_ttft_ms, p95_ttft_ms, p99_ttft_ms, mean_tpot_ms, p95_tpot_ms, p99_tpot_ms,
mean_itl_ms, p95_itl_ms, p99_itl_ms, accept_length, mean_e2e_latency_ms, p95_e2e_latency_ms,
p99_e2e_latency_ms
```

`SLO` 使用统一 JSON，例如 `{"ttft":{"mean":[3,4]}}`。summary 同时保留 inf 基线和有限
请求率结果；同一 case 内按实际请求率从高到低排列，inf 放在首行。总 token 统计列、过程字段
和重复别名不进入 summary。

矩阵图每列代表一个输入 case，每行代表一个 TTFT target；格子选择满足 SLA 的最大实际客户端
RPS，默认显示 `rps`，也可显示 `input_throughput`，没有满足点显示 `NO MEET`。绘图失败
不影响 CSV/TSV 和日志生成。

## 5. 错误处理与续测

- 服务不可达、参数不支持、模型不匹配或 benchmark 失败时保留完整命令和日志，不伪造指标。
- 成功请求数为 0、存在失败请求或核心指标缺失时，该 trial 不得作为有效容量点。
- 收到中断信号时刷新已完成的 trials 和 summary，不删除日志。
- 同一 `RUN_ID` 续测时跳过完整成功 trial，失败或未完成 trial 重新执行。
- 日志和 `run_metadata.json` 写入模型、tokenizer、节点 hostname、服务地址、运行时间和脚本 commit。

## 6. 验证计划

1. 静态检查：Shell/Python 语法、参数校验和 `git diff --check`。
2. dry-run/fixture：验证多 case、多 TTFT target、1024/1000 token 单位、0.5k、请求率序列、
   summary 排序和矩阵筛选。
3. vLLM 日志 fixture 对比：同一份日志分别经过 `vllmc.sh` 和 Prefill 的公共解析模块，
   核心字段值和字段名必须一致。
4. 真实 smoke：用单个小输入 case 验证官方命令、日志解析、CSV/TSV 和图片生成；确认未设置
   `FIXED_CONCURRENCY` 时命令不含 `--max-concurrency`，设置后 inf/回测命令一致。

真实性能结论只使用通过服务健康、成功请求和核心指标校验的 trial；dry-run 和 smoke 只标记
为流程验证，不标记为正式性能数据。
