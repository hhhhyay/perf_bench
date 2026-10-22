# GSP 压测脚本保守精简设计

## 背景

`sglang/gsp_prefix_bench.sh` 当前约 1867 行，同时负责 GSP/Prefill workload 选择、命中率校准、inf 基线、动态请求数、请求率回测、CSV/TSV 汇总和容量矩阵作图。脚本中仍保留了一段未被主流程调用的旧版汇总实现，增加了维护成本和阅读负担。

## 目标

- 减少明确无效的代码，降低脚本阅读和维护成本。
- 保持现有命令行参数、环境变量、测试顺序和结果文件格式不变。
- 保留 0% cache 的 `random-ids` Prefill 流程和大于 0% 的官方 GSP 流程。
- 保留命中率校准、inf 吞吐选择、动态时长、请求率回测、summary/trials CSV、TSV 和作图能力。

## 范围

### 删除

- 删除未被调用的 `write_summary_legacy()` 旧版汇总函数，约 113 行。

### 保留

- `write_summary()` 当前汇总实现。
- `prune_confirmed_trials()` 对 trials/summary 的整理逻辑。
- `parse_benchmark_log()` 及其指标读取函数。
- GSP layout 候选、命中率校准和自动时长调整。
- `HIT_RATE=0` 的 `random-ids --tokenize-prompt` 路径。
- `--plot`、`gsp_prefix_matrix.png`、CSV 和 TSV 输出。

### 不做

- 不重写 shell/Python 混合架构。
- 不删除仍被调用的兼容参数或环境变量。
- 不修改 `prefill_ttft.sh`。
- 不修改 CSV 列名、列顺序、日志命名和测试算法。

## 验证方案

1. `bash -n sglang/gsp_prefix_bench.sh`。
2. `bash sglang/gsp_prefix_bench.sh --help`。
3. 使用 dry-run 验证 `HIT_RATE=0` 仍生成 `random-ids` 命令。
4. 使用 dry-run 验证 `HIT_RATE>0` 仍生成 `generated-shared-prefix` 命令。
5. `git diff --check`，并确认只改动脚本及本设计文档。

## 验收标准

- 脚本行数减少约 113 行。
- 删除函数没有任何残留调用或语法错误。
- 0% 和非 0% 两条 workload 分支的命令生成保持不变。
- 现有工作区中的未跟踪结果文件不被修改或删除。
