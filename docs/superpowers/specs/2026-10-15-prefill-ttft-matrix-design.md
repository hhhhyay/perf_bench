# 单侧 Prefill TTFT 矩阵回测设计

## 背景

此前单侧 Prefill（输出 1 token）回测挂在 `sglc.sh` 的普通 BS 扫描中，并使用固定的绝对 RPS 步长。该方式不能直接回答指定 Mean TTFT 约束下的最大 Prefill 实际 RPS，也会让普通容量脚本混入专用回测逻辑。

## 目标

- 删除旧的 `-0.1,-0.1,-0.1,-0.2,-0.2,-0.3,-0.5` 固定回测。
- 将 `sglc.sh` 恢复为加入 Prefill 回测前的普通容量扫描脚本。
- 新增独立的 `sglang/prefill_ttft.sh`，只负责单侧 Prefill TTFT 矩阵。
- 针对输入 `32K/64K/128K` 和 TTFT 目标 `3s/5s/10s` 建立 3 x 3 矩阵。
- 不扫描所有 BS；根据输入长度规则自动选择一个 `per_dp_bs`。
- 每个矩阵点从高请求率开始向下寻找 TTFT 达标区间，再在区间内细化临界请求率。
- 报告满足目标的最大实际客户端 RPS，同时保留当时的输入请求率和 Mean TTFT。
- 保留日志、`--flush-cache` 和 RUN_ID 续测能力，不保存逐请求 JSONL。

## 非目标

- 不改变 `sglc.sh` 的普通容量扫描、BS 扫描、SLA inline 扫描、HARD_STOP_RULES 或 `--flush-cache` 行为。
- 不把 TTFT 矩阵结果混入容量汇总 `sum_all.csv`。
- 不把配置的 `request_rate` 当作最终吞吐；最终吞吐必须来自日志中的实际 `Request throughput (req/s)`。

## 脚本边界

`sglang/sglc.sh` 只保留普通容量扫描；`sglang/prefill_ttft.sh` 是独立入口，不调用普通脚本的主循环，也不修改普通脚本的结果目录。Prefill 脚本沿用 `sglc.sh` 的服务配置、模型、DP/容量兜底、并行配置、`TOKEN_UNIT`、`RUN_ID`、请求数量和 fake-prefill 等同名参数，以及同一组 `bench_serving` 参数；只改变输入长度、选定 BS、输出 1 token 和 `request_rate` 搜索方式。

## 配置接口

Prefill 脚本沿用 `sglc.sh` 的已有环境变量和默认值，并新增以下矩阵参数：

```bash
PREFILL_TTFT_INPUTS="32 64 128"
PREFILL_TTFT_TARGETS_MS="3000 5000 10000"
PREFILL_TTFT_SEARCH_ITERS=7
PREFILL_TTFT_BS_RULES="8:256,16:128,32:64,64:32,128:16"
```

输入值表示 K，实际 token 数继续由 `TOKEN_UNIT` 换算。输出长度固定写死为 1 token，不作为矩阵参数，也不通过 `TOKEN_UNIT` 换算。

TTFT 目标使用毫秒，`3000/5000/10000` 分别对应 3 秒、5 秒、10 秒。

`PREFILL_TTFT_BS_RULES` 使用 `输入上限K:per_dp_bs` 逗号分隔，按第一个满足 `input_k <= 上限K` 的规则选择 BS。例如默认规则表示：`8K` 以内使用 256，`8K~16K` 使用 128，`16K~32K` 使用 64，`32K~64K` 使用 32，`64K~128K` 使用 16。规则由环境变量覆盖，不在代码中固定单个 BS。

除矩阵参数外，服务地址、模型、DP、容量、并行、请求数量、`USE_FAKE_PREFILL`、`TOKEN_UNIT` 和续测参数均使用与 `sglc.sh` 相同的名字和语义。benchmark 命令参数保持一致，仅由矩阵逻辑填充输入长度、输出长度、选定并发和限流请求率。

## 搜索算法

对每个输入长度先根据 `PREFILL_TTFT_BS_RULES` 选择一次 `per_dp_bs`，然后执行一次 `request_rate=inf` 基准测试，提取成功日志中的实际 RPS，作为高请求率上界。随后对每个 TTFT 目标执行临界值搜索：

1. 先测试高请求率。若 `mean_ttft_ms <= target_ttft_ms`，直接将该结果作为当前目标的最大候选。
2. 若高请求率超标，则继续向低请求率测试，直到找到一个达标点；维护“高点超标、低点达标”的区间。
3. 在该区间内二分请求率，最多执行 `PREFILL_TTFT_SEARCH_ITERS` 次搜索测试。达标时提高低点，超标时降低高点。
4. 在所有成功且达标的结果中，按 `actual_rps` 取最大值；若实际 RPS 相同，则优先选择 `mean_ttft_ms` 更低的结果。
5. 若降到最小可执行请求率仍不达标，则该矩阵点标记为 `FAIL`，不伪造性能结果。

搜索变量是命令行传入的 `request_rate`，最终性能指标是日志提取的 `actual_rps`。每个输入长度不遍历其他 BS。每次测试都使用 `--flush-cache`，并保留独立日志和 `.exitcode`。

## 输出与续测

结果写入独立脚本自己的结果目录：

```text
<result_dir>/logs/
<result_dir>/prefill_ttft_trials.csv
<result_dir>/prefill_ttft_matrix.csv
```

`prefill_ttft_trials.csv` 保存所有基准和搜索 trial。客户端字段布局与 `sglc.sh` 对齐：保留前置客户端指标、`Mean/P95/P99 TTFT`、`Mean/P95/P99 TPOT`、`Mean/P95/P99 ITL` 和 `Mean E2E Latency`；在 `Mean E2E Latency` 后放置 `P50/Std/P75/P90/P999/Max TTFT`，替代 E2E 后续分位数位置。`Median TTFT`、其他 E2E 分位数及额外 TPOT/ITL 分位数不提取。不同日志 label 的字段按各自语义分别保存，不合并成一个指标。至少包含以下字段：

```text
input_k,input_len,output_len,per_dp_bs,global_concurrency,
target_ttft_ms,trial,search_phase,input_request_rate,actual_rps,
backend,traffic_request_rate,max_request_concurrency,
successful_requests,benchmark_duration_s,total_input_tokens,
total_input_text_tokens,total_generated_tokens,
total_generated_tokens_retokenized,request_throughput,
input_throughput,output_throughput,peak_output_throughput,
peak_concurrency,total_throughput,concurrency,accept_length,
mean_ttft_ms,p95_ttft_ms,p99_ttft_ms,
mean_tpot_ms,p95_tpot_ms,p99_tpot_ms,
mean_itl_ms,p95_itl_ms,p99_itl_ms,mean_e2e_latency_ms,
p50_ttft_ms,std_ttft_ms,p75_ttft_ms,p90_ttft_ms,p999_ttft_ms,max_ttft_ms,
case_status,log_file
```

`prefill_ttft_matrix.csv` 每个输入长度和目标 TTFT 只保留最终最优达标结果，按 `actual_rps` 最大选择，并保留该结果的全部 TTFT 指标；已有成功日志且 `.exitcode=0` 时跳过并复用，失败或未完成测试重新执行。CSV 按输入、目标和 trial 去重，支持使用同一个 `RUN_ID` 续测。

## 失败处理

- BS 规则不覆盖输入长度：跳过该输入长度并报告配置错误。
- 基准 `inf` 测试失败或无法提取实际 RPS/Mean TTFT：跳过该输入长度的矩阵搜索并保留失败日志。
- 回测日志缺少 `Request throughput` 或 `Mean TTFT`：该 trial 标记为 `FAIL`。
- benchmark 非 0 退出：保留日志和退出码，不作为候选结果。
- 搜索过程中服务异常：停止当前矩阵点，保留已完成结果，不将部分结果标记为完整最优结果。

## 验证

- 使用 `bash -n` 检查脚本语法。
- 使用 `git diff --check` 检查补丁格式。
- 用模拟的基准 RPS 和 TTFT 序列验证：搜索顺序先从高到低，最终只选择达标且实际 RPS 最大的结果。
- 验证 9 个矩阵点的输入长度、目标 TTFT、日志命名和 CSV 字段互不串行。
- 验证 `sglc.sh` 与恢复前提交 `007f969` 一致，且不再创建 Prefill 回测目录。
- 验证相同 `RUN_ID` 下成功 trial 可跳过，失败 trial 可续测。
