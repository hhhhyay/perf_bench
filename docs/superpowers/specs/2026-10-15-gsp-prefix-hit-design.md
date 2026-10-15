# GSP Prefix 命中率测试设计

## 背景

为 `sglang.bench_serving --dataset-name generated-shared-prefix` 增加一个独立的命中率容量测试入口，覆盖 Prefill、输出大于 1 的普通推理，以及后续 PD 分离场景。测试必须以 SGLang 官方 GSP 数据集和 `--cache-report` 为依据，不能把请求数命中比例冒充 token 级 prefix 命中率。

## 目标

- 新增独立脚本 `sglang/gsp_prefix_bench.sh`，不改变 `sglc.sh` 和 `prefill_ttft.sh` 的既有行为。
- 支持按目标总输入长度、输出长度和目标 prefix token 命中率自动生成 GSP 的 system/question 长度。
- 理论命中率考虑每个 prefix group 的首个请求冷 miss，尽量与指定目标一致。
- 支持多个 workload case 连续测试，每个 case 可以配置 TTFT 指标、多个 TTFT SLA 和 TPOT SLA。
- 从高请求率向低请求率搜索，记录实际客户端 RPS；有限请求率保留两位小数，搜索步长不小于 0.1。
- 保存官方原始日志、完整 trial CSV、每个 case 的最优结果和可选容量矩阵图。
- 默认不保存逐请求 JSONL，避免大文件；依赖官方 `--cache-report` 输出汇总缓存指标。

## 口径与公式

### Token 级 prefix 命中率

目标命中率定义为：

```text
实际命中率 = total_cached_tokens / total_prompt_tokens
```

对均匀分组且 `--gsp-ordered` 的请求序列，理论值按每组第一个请求不命中计算：

```text
theoretical_hit_rate =
  (gsp_prompts_per_group - 1) * gsp_system_prompt_len
  / (gsp_prompts_per_group * (gsp_system_prompt_len + gsp_question_len))
```

给定总输入长度 `L`、目标命中率 `H` 和每组请求数 `P`，脚本反推：

```text
system_prompt_len = round(H * P * L / (P - 1))
question_len = L - system_prompt_len
```

脚本会检查长度为正数且 system prompt 不超过总输入长度。由于 tokenization、cache block 对齐和服务实现可能带来误差，结果中同时保存理论命中率和官方 `--cache-report` 的实测命中率；不得用理论值替代实测值。

`TOKEN_UNIT` 默认是 1024，设置 `TOKEN_UNIT=1000` 时才使用十进制 k。输入和输出支持小数，例如 `0.5k`。

### 请求与 GSP 参数

默认参数沿用用户给出的官方调用：

```text
gsp-num-groups=32
gsp-prompts-per-group=16
gsp-ordered
num-prompts=32*16=512
seed=42
flush-cache
cache-report
```

`num-prompts` 必须等于 `gsp-num-groups * gsp-prompts-per-group`，避免官方数据集顺序和理论公式不一致。用户可以通过环境变量覆盖 groups、prompts-per-group、seed、host、port、model 和请求率搜索参数。

## 用例配置

通过重复的 `--case` 指定多个 case，格式为：

```text
INPUT,OUTPUT,HIT_RATE,TTFT_METRIC,TTFT_TARGETS_S,TPOT_METRIC,TPOT_TARGET_MS[,MAX_CONCURRENCY]
```

`TTFT_TARGETS_S` 使用 `|` 分隔多个目标，目标支持小数秒。示例：

```bash
bash sglang/gsp_prefix_bench.sh \
  --case '16k,1k,80,mean_ttft_ms,3|5,mean_tpot_ms,30' \
  --case '32k,0.5k,80,mean_ttft_ms,5|7,mean_tpot_ms,50'
```

每个 case 使用同一组 GSP workload 参数，输出长度不再写死为 1，因此覆盖 Decode 输出大于 1 的场景。可选的 `MAX_CONCURRENCY` 固定该 case 的客户端最大并发，未填写时使用全局 `GSP_MAX_CONCURRENCY`；这相当于现有 Prefill 测试中的固定 BS/全局并发约束。支持的 TTFT 指标包括 `mean_ttft_ms`、`p50_ttft_ms`、`p90_ttft_ms`、`p95_ttft_ms`、`p99_ttft_ms` 和 `max_ttft_ms`；TPOT 使用官方输出的 `mean_tpot_ms`、`p50_tpot_ms`、`p90_tpot_ms`、`p95_tpot_ms` 或 `p99_tpot_ms`。

## 请求率搜索

每个 case 先运行 `--request-rate inf`，作为不限制请求率的基线，并读取实际客户端 RPS。随后从该基线实际 RPS 向下搜索，每次请求率保留两位小数，最小步长固定不低于 0.1。默认使用高到低的保守回测序列，并支持通过环境变量提供显式请求率列表。

对每个 TTFT SLA，结果选择同时满足 TTFT 和 TPOT 条件的最大实际客户端 RPS。未达到 SLA 的点仍保留在 trial CSV；如果所有点均不满足，结果状态为 `NO_MEET`，不伪造容量结果。命中率偏离目标时，保留该结果但标记 `HIT_RATE_MISMATCH`，并在 summary 中显示理论/实测差值。

## 输出与作图

默认输出到：

```text
all_result/bench_gsp_prefix_<RUN_ID>/
  logs/
  gsp_prefix_trials.csv
  gsp_prefix_summary.csv
```

每个 log 同时保存 `.exitcode`，成功结果可以续跑跳过。CSV 至少包含：case 配置、GSP system/question 长度、理论/实测命中率、请求率、实际 RPS、输入/输出 throughput、TTFT 全部主要分位数、TPOT 全部主要分位数、目标状态和 log 路径。

指定 `--plot` 才生成 PNG，默认与 CSV 同目录。支持：

```bash
bash sglang/gsp_prefix_bench.sh --plot --cell rps
bash sglang/gsp_prefix_bench.sh --plot --cell input_throughput
```

热力图列为输入/输出/目标命中率 case，行为 TTFT SLA，格子中央显示实际客户端 RPS 或输入 token throughput；颜色表示对应容量。绘图使用脚本内联 Python 和现有运行环境中的 matplotlib，不新增独立 Python 文件。未指定 `--plot` 时只生成 CSV。

## CLI 与官方参数兼容性检查

脚本提供 `--help`、重复 `--case`、`--plot`、`--cell` 和 `--dry-run`。`--dry-run` 只完成参数解析、命中率长度计算和官方命令打印，不连接服务、不写 benchmark log，便于在 811 节点启动真实测试前核对 workload。

脚本只调用官方 `python3 -m sglang.bench_serving`，使用：

- `--dataset-name generated-shared-prefix`
- `--gsp-num-groups`
- `--gsp-prompts-per-group`
- `--gsp-system-prompt-len`
- `--gsp-question-len`
- `--gsp-output-len`
- `--gsp-ordered`
- `--num-prompts`
- `--flush-cache`
- `--seed`
- `--cache-report`

正式测试前检查本地 `bench_serving --help` 是否包含所需参数；缺少参数时立即失败并提示安装的 SGLang 版本，而不是静默退化为不同 workload。真实执行时按 case 传入 `--max-concurrency`（如果设置），并且 `--num-prompts` 仍由 GSP groups 和 prompts-per-group 相乘得到。脚本不实现自己的请求客户端、不生成自己的 prompt 数据集，也不解析逐请求 JSONL。

## 错误处理与续测

- case 格式、token 单位、命中率、SLA、输出长度和搜索步长非法时在运行前失败。
- server 不可达、bench 命令失败或缺少 cache-report 指标时，保留日志和 exitcode，并将 case 标记为 `FAILED` 或 `NO_CACHE_REPORT`。
- 成功的 log 和 exitcode 为 0 时跳过同一 trial，支持中断后续测。
- 不停止、不重启、不清理用户已有服务；`--flush-cache` 只调用官方缓存刷新接口。

## 验证计划

1. `bash -n sglang/gsp_prefix_bench.sh`。
2. `--help`、非法 case、非法命中率和非法步长校验。
3. dry-run 检查两组示例会生成正确的 system/question/output 和官方参数。
4. 在 811 节点使用两组示例真实运行，核对 server log、官方 cache-report、TTFT/TPOT、RPS、CSV 和 PNG。
5. 检查重复执行能续测，且 `all_result/`、JSONL 和缓存数据不会被提交。
6. README 补充命中率公式、示例命令、SLA 语法、CSV 字段和作图方式。

