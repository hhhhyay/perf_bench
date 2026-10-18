# GSP inf 命中率硬门槛设计

## 背景

GSP 测试使用 `request_rate=inf` 获取无客户端限流下的原始最高吞吐，同时还需要保证指定的 token-level cache 命中率。现有汇总逻辑只从有限请求率 trial 中选择满足条件的结果，因此可能出现 `inf` 命中率不达标、低速 trial 达标却被报告为 `MEET` 的情况。

## 目标

- 每个输入/输出/目标命中率 case 的 `inf` trial 必须成功并包含官方 `cache-report`。
- `inf` 的实际命中率必须落在指定命中率 ± `GSP_HIT_RATE_TOLERANCE_PCT` 内。
- 只有在 `inf` 通过后，有限请求率 trial 才能参与 TTFT、TPOT 和容量筛选。
- 任何有限请求率的候选也必须同时满足实际命中率、TTFT 和 TPOT 条件。
- 保留 `inf` 原始 RPS、耗时和命中率，便于区分原始上限与合规容量。

## 行为设计

1. 每个 case 先执行 `inf` baseline。
2. 如果 baseline 失败、缺少 cache-report、缺少实际命中率或命中率超出容差：
   - 不把该 case 的任何有限请求率结果写成 `MEET`；
   - summary 状态写为 `BASELINE_NO_CACHE_REPORT`、`BASELINE_FAILED` 或 `BASELINE_HIT_RATE_MISMATCH`；
   - 仍保留所有 trial 日志和 CSV 行。
3. baseline 通过后，按现有策略从高到低执行有限请求率，并选择满足所有 SLA 的最高实际 RPS。
4. summary 增加 baseline 命中率、baseline 命中率偏差和 baseline 状态字段。
5. 绘图只使用 baseline 通过且有限请求率满足 SLA 的 summary 行。

## 不改变的内容

- `GSP_NUM_GROUPS`、`GSP_PROMPTS_PER_GROUP`、`--gsp-ordered`、`--gsp-input-ids`、`--flush-cache`、`--cache-report` 的语义不变。
- 不自动添加 `--max-concurrency`。
- 请求率仍保留两位小数，最小步长不低于 `0.1 req/s`。
- `GSP_SYSTEM_LEN_EXTRA_TOKENS` 仍只是基于官方 cache-report 的 page/token 校准，不作为命中率容差放宽手段。

## 验证标准

- 使用一个 baseline 命中率不达标、但低速 trial 达标的 fixture，summary 必须不是 `MEET`。
- 使用 baseline 和有限 trial 都达标的 fixture，summary 才能选择最高合规 RPS。
- 真实 811 节点回测时，必须分别报告 `inf` 原始 RPS/命中率和最终合规 RPS/命中率；不能将此前 `0.20 req/s` 的结果继续标记为 inf 合规结果。
