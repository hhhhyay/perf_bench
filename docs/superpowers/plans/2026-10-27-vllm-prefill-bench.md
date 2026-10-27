# vLLM Prefill Benchmark Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 新增独立的 vLLM Prefill 压测脚本，固定输出长度为 1，先用 inf 找到输入吞吐基线，再按 TTFT SLA 从高到低回测最大实际 RPS，并与现有 vllmc.sh 公共指标严格对齐。

**Architecture:** 将 vllmc.sh 中的 vLLM 日志解析和字段标准化提取为共享 Python 模块；vllmc.sh 和新的 Prefill 脚本共同调用该模块。Prefill shell 只负责编排 case、调用官方 vLLM benchmark、持久化日志和触发 Python 汇总；Prefill 指标/请求率选择和矩阵作图分别由小型 Python 模块负责。

**Tech Stack:** Bash、Python 3 标准库、vllm bench serve、unittest、可选 matplotlib/numpy。

## Global Constraints

- Prefill 的 --output-len 固定为 1，默认不传客户端 --max-concurrency。
- TOKEN_UNIT 默认 1024，设置为 1000 时使用十进制 K；小数 K 换算后四舍五入为整数 token。
- inf 基线和有限请求率回测必须使用同一输入长度、输出长度、请求数策略、tokenizer 和并发配置。
- 有限请求率从 inf 实际客户端 RPS 高到低回测，默认步长为 0.1 req/s，保留两位小数，不能使用小于 0.1 的步长。
- vLLM 公共字段的日志解析、别名和单位必须与现有 vllm/vllmc.sh 相同；Prefill 只能过滤不适用的 Decode/cache 字段。
- 失败请求、成功请求数为 0 或核心指标缺失的 trial 不得作为有效容量结果；原始日志和退出码必须保留。
- 不提交 all_result/、临时目录、Python 缓存和其他未跟踪测试产物。

---

### Task 1: 提取共享 vLLM 日志解析并建立回归 fixture

**Files:**
- Create: vllm/vllm_bench_metrics.py
- Create: vllm/tests/fixtures/bench_serving.log
- Create: vllm/tests/test_vllm_bench_metrics.py
- Modify: vllm/vllmc.sh:800-1124，将内嵌的百分位解析、汇总解析和 normalize_row 改为调用共享模块。

**Interfaces:**
- parse_percentiles_from_text(text: str) -> dict[str, float]
- parse_summary_from_log(log_path: pathlib.Path) -> dict[str, object]
- normalize_row(row: dict[str, object], log_path: pathlib.Path | None = None) -> dict[str, object]
- vllmc.sh 的 sum_all.csv 前置字段和现有隐藏/重复字段行为保持不变。

- [ ] Step 1: 写入 vLLM benchmark 日志 fixture

在 vllm/tests/fixtures/bench_serving.log 中写入以下最小完整输出：

~~~text
Traffic request rate: inf
Successful requests: 2
Failed requests: 0
Benchmark duration (s): 10.00
Request throughput (req/s): 0.20
Input token throughput (tok/s): 819.20
Output token throughput (tok/s): 0.20
Total token throughput (tok/s): 819.40
Peak output token throughput (tok/s): 0.20
Peak concurrent requests: 2
Concurrency: 1.00
Mean TTFT (ms): 150.00
Median TTFT (ms): 140.00
P95 TTFT (ms): 180.00
P99 TTFT (ms): 200.00
Mean TPOT (ms): 30.00
P95 TPOT (ms): 35.00
P99 TPOT (ms): 40.00
Mean ITL (ms): 30.00
P95 ITL (ms): 35.00
P99 ITL (ms): 40.00
Mean E2EL (ms): 160.00
P95 E2EL (ms): 190.00
P99 E2EL (ms): 210.00
~~~

- [ ] Step 2: 先写共享解析的失败测试

在 vllm/tests/test_vllm_bench_metrics.py 中使用 unittest，验证：

~~~python
row = normalize_row(parse_summary_from_log(FIXTURE), FIXTURE)
assert row["rps"] == 0.20
assert row["request_rate"] == "inf"
assert row["input_throughput_tok_s"] == 819.20
assert row["generate_throughput_tok_s"] == 0.20
assert row["total_throughput_tok_s"] == 819.40
assert row["mean_ttft_ms"] == 150.0
assert row["p95_ttft_ms"] == 180.0
assert row["p99_ttft_ms"] == 200.0
assert row["mean_e2e_latency_ms"] == 160.0
~~~

运行：
~~~bash
python -m unittest discover -s vllm/tests -p 'test_*.py' -v
~~~

预期：在共享模块尚未存在时失败，并指出导入或函数缺失。

- [ ] Step 3: 实现共享解析模块

将现有 vllmc 内嵌 Python 逻辑迁移为上述函数，保留 Request throughput、Input/Output/Total token throughput 的 alias，Mean E2EL 到 mean_e2e_latency_ms 的 alias，P50/P95/P99 两种打印顺序和 median/p95/p99 补齐逻辑，以及 parse_scalar 对 inf、nan、数字和普通字符串的处理。共享模块只负责解析和标准化，不写 CSV，不读取服务端，不改变 vllmc 的 BS 计算。

- [ ] Step 4: 让 vllmc.sh 调用共享模块并运行回归测试

在 vllmc 的汇总 heredoc 中通过仓库根目录导入 vllm.vllm_bench_metrics，删除重复的三组函数，保持现有 normalize_row 调用点和字段输出不变。

~~~bash
python -m unittest discover -s vllm/tests -p 'test_*.py' -v
git diff --check
~~~

预期：fixture 解析通过，且 vllmc 脚本的标准化字段名称不发生变化。

- [ ] Step 5: 提交本任务

~~~bash
git add vllm/vllm_bench_metrics.py vllm/vllmc.sh vllm/tests
git commit -m "refactor: 共享 vLLM benchmark 指标解析"
~~~

### Task 2: 实现 Prefill case、token 单位和请求率回测核心

**Files:**
- Modify: vllm/vllm_bench_metrics.py，增加 Prefill case、请求率和 SLA 选择函数。
- Modify: vllm/tests/test_vllm_bench_metrics.py，增加 Prefill 核心逻辑测试。

**Interfaces:**
- PrefillCase 数据类字段：case_id、input_tokens、output_tokens、metric、targets_s。
- parse_token_value(raw: str, token_unit: int) -> int
- parse_prefill_case(spec: str, token_unit: int = 1024) -> PrefillCase
- build_rate_sequence(start_rps: float, step: float = 0.1) -> list[float]
- select_sla_rows(rows: list[dict[str, object]], case: PrefillCase) -> dict[float, dict[str, object] | None]

- [ ] Step 1: 写失败测试覆盖 case 和请求率规则

至少加入：

~~~python
case = parse_prefill_case("0.5k,p99_ttft_ms,5|6", 1024)
assert case.input_tokens == 512
assert case.output_tokens == 1
assert case.targets_s == (5.0, 6.0)
assert parse_prefill_case("0.5k,mean_ttft_ms,3", 1000).input_tokens == 500
assert build_rate_sequence(1.03, 0.1) == [1.03, 0.93, 0.83, 0.73, 0.63, 0.53, 0.43, 0.33, 0.23, 0.13]
~~~

非法 metric、空 target、非正 token、步长小于 0.1 必须抛出 ValueError。

- [ ] Step 2: 实现 case 和回测核心

使用 Decimal 或整数百分之一避免浮点长小数；序列从 floor(start_rps * 100) / 100 开始，每次减去 round(step * 100)，在小于 0.10 前停止。SLA 选择必须使用实际 rps，不能使用配置的 request_rate；相同实际 RPS 时保留 duration 更接近目标的 row。

- [ ] Step 3: 运行单元测试并提交

~~~bash
python -m unittest discover -s vllm/tests -p 'test_*.py' -v
git diff --check
git add vllm/vllm_bench_metrics.py vllm/tests/test_vllm_bench_metrics.py
git commit -m "feat: 增加 vLLM Prefill 回测核心"
~~~

### Task 3: 新增 vLLM Prefill 压测编排脚本

**Files:**
- Create: vllm/vllm_prefill_bench.sh
- Modify: vllm/vllm_bench_metrics.py，增加 trial CSV、summary TSV 的命令行入口。
- Create: vllm/tests/test_vllm_prefill_cli.py，验证 dry-run 和文件生成。

**Interfaces:**
- CLI：重复 --case SPEC、--help、--dry-run、--plot、--no-plot、--cell rps|input_throughput。
- 环境变量：BASE_URL、MODEL、TOKENIZER、TOKEN_UNIT、PREFILL_TARGET_DURATION_S、PREFILL_RATE_STEP、PREFILL_INF_MAX_TRIALS、PREFILL_INF_NO_IMPROVEMENT_LIMIT、PREFILL_NUM_PROMPTS、FIXED_CONCURRENCY、RUN_ID。
- Python CLI：parse-case SPEC TOKEN_UNIT、rate-list START STEP、append-trial TRIALS_CSV ROW_JSON、summary TRIALS_CSV SUMMARY_TSV CASES_JSON。

- [ ] Step 1: 先写 dry-run 失败测试

~~~bash
BASE_URL=http://127.0.0.1:8000 MODEL=/model/demo bash vllm/vllm_prefill_bench.sh --dry-run --case '4k,mean_ttft_ms,3|4'
~~~

测试需要验证 dry-run 输出包含 --backend openai、--base-url http://127.0.0.1:8000、--random-input-len 4096、--random-output-len 1、--request-rate inf、--num-prompts，且未设置 FIXED_CONCURRENCY 时不包含 --max-concurrency。

- [ ] Step 2: 实现服务元数据和命令构造

Prefill 使用与 vllmc 相同的关键参数：

~~~bash
vllm bench serve --backend openai --base-url "$BASE_URL" --model "$MODEL" --dataset-name random --random-input-len "$input_tokens" --random-output-len 1 --random-range-ratio 0.0 --num-prompts "$num_prompts" --request-rate "$rate" --num-warmups 0 --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 95,99 --disable-tqdm
~~~

有 TOKENIZER 时追加 --tokenizer；有 FIXED_CONCURRENCY 时追加 --max-concurrency，否则不追加。每个日志开头写入模型、tokenizer、hostname、BASE_URL、RUN_ID 和脚本 commit。

- [ ] Step 3: 实现 inf 稳定性和动态请求数

每个 case 先运行短 pilot；没有 PREFILL_NUM_PROMPTS 时按实际 duration / PREFILL_TARGET_DURATION_S 估算下一轮请求数，至少保证请求数为 1。inf 最多运行 PREFILL_INF_MAX_TRIALS 次；当输入吞吐连续 PREFILL_INF_NO_IMPROVEMENT_LIMIT 次不再提升时选择最高有效输入吞吐作为基线。显式 PREFILL_NUM_PROMPTS 时不自动修改请求数，但 inf 和有限请求率仍必须使用同一请求数。

- [ ] Step 4: 实现有限请求率回测和中断续测

对确认后的基线调用 build_rate_sequence；每个 rate 写入独立 .log、.log.exitcode 和 .time，完成后立即调用 append-trial 和 summary，确保中断时已完成数据可用。收到 INT/TERM 时刷新 summary 后退出，不删除日志。成功请求数为 0、日志缺少 Request throughput 或核心 TTFT 字段时，将退出码改为失败，并继续保留该 trial，不把它送入容量选择。

- [ ] Step 5: 实现命令行校验和续测

校验 MODEL 非空、TOKEN_UNIT 为正整数、PREFILL_RATE_STEP >= 0.1、case metric 属于允许集合、目标秒数为正数。相同 RUN_ID 下，已有完整成功日志跳过，失败或缺失 .exitcode 的 trial 重新执行。

- [ ] Step 6: 运行 dry-run/fixture 测试并提交

~~~bash
python -m unittest discover -s vllm/tests -p 'test_*.py' -v
bash -n vllm/vllm_prefill_bench.sh
BASE_URL=http://127.0.0.1:8000 MODEL=/model/demo bash vllm/vllm_prefill_bench.sh --dry-run --case '4k,mean_ttft_ms,3|4'
git diff --check
git add vllm/vllm_prefill_bench.sh vllm/vllm_bench_metrics.py vllm/tests
git commit -m "feat: 增加 vLLM Prefill 压测与请求率回测"
~~~

### Task 4: 实现 Prefill summary 和容量矩阵图

**Files:**
- Create: vllm/vllm_prefill_plot.py
- Modify: vllm/vllm_bench_metrics.py，实现最终 summary 字段顺序和每个目标的容量筛选。
- Modify: vllm/tests/test_vllm_prefill_cli.py，增加 summary 排序和矩阵选择测试。

**Interfaces:**
- write_prefill_summary(trials_csv: Path, summary_tsv: Path, cases: list[PrefillCase]) -> None
- plot_prefill_matrix(summary_tsv: Path, output_png: Path, cell: str = "rps") -> None
- cell 只允许 rps 或 input_throughput；绘图依赖缺失时抛出可捕获异常，由 shell 只打印 warning。

- [ ] Step 1: 写 summary 字段和排序测试

构造同一个 case 的 inf、0.93、0.83 三条 row，验证：

~~~python
assert header[:5] == ["case_name", "input_len", "output_len", "parallel_config", "SLO"]
assert header[header.index("rps") + 1] == "input_throughput_tok_s"
assert rows[0]["request_rate"] == "inf"
assert [row["request_rate"] for row in rows[1:]] == ["0.93", "0.83"]
assert rows[0]["SLO"] == '{"ttft":{"mean":[3,4]}}'
~~~

- [ ] Step 2: 实现 summary 生成

summary 仅输出公共性能字段、SLO 和 Prefill 必需的 workload 字段；不输出 input_k、output_k、cache、per-DP BS、过程状态、重复别名和 token 总量统计。rps 使用实际请求吞吐，request_rate 使用配置值；inf 放在同一 case 的有限请求率之前。

- [ ] Step 3: 实现矩阵图

按输入 case 形成列，按 TTFT target 形成行；只选择指标满足 target 的有效有限请求率 row，取实际 RPS 最大者。cell=rps 显示 RPS，cell=input_throughput 显示输入 token throughput，无满足结果显示 NO MEET。

- [ ] Step 4: 运行 summary/plot 测试并提交

~~~bash
python -m unittest discover -s vllm/tests -p 'test_*.py' -v
git diff --check
git add vllm/vllm_bench_metrics.py vllm/vllm_prefill_plot.py vllm/tests
git commit -m "feat: 增加 vLLM Prefill 汇总与容量矩阵"
~~~

### Task 5: 更新 README 并完成端到端验证

**Files:**
- Modify: README.md，增加 vLLM Prefill 用法、case 格式、回测规则、输出文件和环境变量。
- Modify: vllm/tests/test_vllm_prefill_cli.py，补齐文档示例对应的 dry-run 断言。

- [ ] Step 1: 更新 README

新增 vllm/vllm_prefill_bench.sh 小节，包含以下可直接复制的命令和说明：

~~~bash
BASE_URL=http://127.0.0.1:8000 MODEL=/data/models/Qwen3.8-27B-Channel-INT8-w8a8 PREFILL_TARGET_DURATION_S=60 bash vllm/vllm_prefill_bench.sh --case '4k,mean_ttft_ms,3|4' --case '64k,p99_ttft_ms,5|6'
~~~

明确说明输出固定为 1、默认不设置 max-concurrency、inf 后按实际 RPS 由高到低以 0.1 回测、rps 与 request_rate 的区别，以及 CSV/TSV/PNG 路径。

- [ ] Step 2: 执行完整本地验证

~~~bash
python -m unittest discover -s vllm/tests -p 'test_*.py' -v
bash -n vllm/vllm_prefill_bench.sh
BASE_URL=http://127.0.0.1:8000 MODEL=/model/demo bash vllm/vllm_prefill_bench.sh --dry-run --case '4k,mean_ttft_ms,3|4' --case '0.5k,p99_ttft_ms,5|6'
git diff --check
~~~

预期：所有单元测试通过；dry-run 展示两个 case、输出长度 1、inf 命令和从实际 RPS 生成的有限请求率计划；未设置 FIXED_CONCURRENCY 时命令不含 --max-concurrency。

- [ ] Step 3: 真实 smoke 验证

在用户提供的 vLLM 服务上使用一个小输入 case，确认 /v1/models 可用、至少一个 inf 和一个有限请求率 benchmark 成功、日志可解析、summary 有数据、矩阵图生成失败不会影响 TSV。真实 smoke 只作为流程验证，不作为正式性能结论。

- [ ] Step 4: 最终检查并提交文档

~~~bash
git status --short
git diff --check
git add README.md vllm/tests
git commit -m "docs: 补充 vLLM Prefill 使用说明"
~~~

完成后再根据用户明确要求决定是否 push；不提交现有 all_result/、临时目录和其他未跟踪文件。
