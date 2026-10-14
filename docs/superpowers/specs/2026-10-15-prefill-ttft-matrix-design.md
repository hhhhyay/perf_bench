# 单侧 Prefill TTFT 矩阵回测设计

## 背景

现有单侧 Prefill（输出 1 token）回测使用固定的绝对 RPS 步长。该方式不能直接回答指定 Mean TTFT 约束下的最大 Prefill 实际 RPS，也无法覆盖多输入长度和多 TTFT 目标的组合。

## 目标

- 删除旧的 `-0.1,-0.1,-0.1,-0.2,-0.2,-0.3,-0.5` 固定回测。
- 针对输入 `32K/64K/128K` 和 TTFT 目标 `3s/5s/10s` 建立 3 x 3 矩阵。
- 每个矩阵点从高请求率开始向下寻找 TTFT 达标区间，再在区间内细化临界请求率。
- 报告满足目标的最大实际客户端 RPS，同时保留当时的输入请求率和 Mean TTFT。
- 保留日志、`--flush-cache` 和 RUN_ID 续测能力，不保存逐请求 JSONL。

## 非目标

- 不改变普通容量扫描、BS 扫描、SLA inline 扫描或 HARD_STOP_RULES 的判定逻辑。
- 不把 TTFT 矩阵结果混入容量汇总 `sum_all.csv`。
- 不把配置的 `request_rate` 当作最终吞吐；最终吞吐必须来自日志中的实际 `Request throughput (req/s)`。

## 配置接口

SGLang 脚本新增以下环境变量，默认值如下：

```bash
PREFILL_TTFT_INPUTS="32 64 128"
PREFILL_TTFT_TARGETS_MS="3000 5000 10000"
PREFILL_TTFT_SEARCH_ITERS=7
```

输入值表示 K，实际 token 数继续由 `TOKEN_UNIT` 换算。输出长度固定为 1 token，默认通过 `0.000976562K` 表示；`TOKEN_UNIT=1000` 时使用等价的十进制换算。

TTFT 目标使用毫秒，`3000/5000/10000` 分别对应 3 秒、5 秒、10 秒。

## 搜索算法

对每个输入长度先执行一次 `request_rate=inf` 基准测试，提取成功日志中的实际 RPS，作为高请求率上界。随后对每个 TTFT 目标执行临界值搜索：

1. 先测试高请求率。若 `mean_ttft_ms <= target_ttft_ms`，直接将该结果作为当前目标的最大候选。
2. 若高请求率超标，则继续向低请求率测试，直到找到一个达标点；维护“高点超标、低点达标”的区间。
3. 在该区间内二分请求率，最多执行 `PREFILL_TTFT_SEARCH_ITERS` 次搜索测试。达标时提高低点，超标时降低高点。
4. 在所有成功且达标的结果中，按 `actual_rps` 取最大值；若实际 RPS 相同，则优先选择 `mean_ttft_ms` 更低的结果。
5. 若降到最小可执行请求率仍不达标，则该矩阵点标记为 `FAIL`，不伪造性能结果。

搜索变量是命令行传入的 `request_rate`，最终性能指标是日志提取的 `actual_rps`。每次测试都使用 `--flush-cache`，并保留独立日志和 `.exitcode`。

## 输出与续测

结果写入：

```text
<result_dir>/prefill_backtest/prefill_ttft_matrix.csv
```

至少包含以下字段：

```text
case_name,input_len,output_len,target_ttft_ms,trial,
input_request_rate,actual_rps,mean_ttft_ms,case_status,log_file
```

每个输入长度和目标 TTFT 使用稳定的 case/target 标识。已有成功日志且 `.exitcode=0` 时跳过并复用；失败或未完成测试重新执行。CSV 按矩阵点和 trial 去重，支持使用同一个 `RUN_ID` 续测。

## 失败处理

- 基准 `inf` 测试失败或无法提取实际 RPS：跳过该输入长度的矩阵搜索并保留失败日志。
- 回测日志缺少 `Request throughput` 或 `Mean TTFT`：该 trial 标记为 `FAIL`。
- benchmark 非 0 退出：保留日志和退出码，不作为候选结果。
- 搜索过程中服务异常：停止当前矩阵点，保留已完成结果，不将部分结果标记为完整最优结果。

## 验证

- 使用 `bash -n` 检查脚本语法。
- 使用 `git diff --check` 检查补丁格式。
- 用模拟的基准 RPS 和 TTFT 序列验证：搜索顺序先从高到低，最终只选择达标且实际 RPS 最大的结果。
- 验证 9 个矩阵点的输入长度、目标 TTFT、日志命名和 CSV 字段互不串行。
- 验证相同 `RUN_ID` 下成功 trial 可跳过，失败 trial 可续测。

