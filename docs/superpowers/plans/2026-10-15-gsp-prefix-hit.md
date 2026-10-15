# GSP Prefix 命中率测试 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个独立的 SGLang GSP prefix 命中率容量测试脚本，支持目标命中率、输出长度、TTFT/TPOT SLA、多 case 连测、RPS 搜索、CSV 汇总和可选热力图。

**Architecture:** `sglang/gsp_prefix_bench.sh` 作为唯一新增运行脚本，调用官方 `python3 -m sglang.bench_serving` 生成请求和采集指标。Shell 负责参数、流程、续测和进程退出码；少量内联 Python 只负责结构化解析 CSV/日志和生成 PNG，不创建独立 Python 文件。README 只增加使用说明，不改变现有 benchmark 脚本。

**Tech Stack:** Bash 4+、Python 3 标准库、官方 `sglang.bench_serving`、可选 matplotlib、curl、awk、sed、git。

## Global Constraints

- 命中率使用 `total_cached_tokens / total_prompt_tokens`，不是请求数命中比例。
- 理论值按 `(prompts_per_group - 1) * system_prompt_len / (prompts_per_group * total_input_len)` 计算。
- 官方 workload 必须使用 `generated-shared-prefix`、`--gsp-ordered`、`--flush-cache`、`--cache-report`。
- 默认 `gsp-num-groups=32`、`gsp-prompts-per-group=16`、`seed=42`，请求数等于两者乘积。
- `TOKEN_UNIT` 默认 1024；只有明确设置 `TOKEN_UNIT=1000` 时使用十进制 k；输入和输出支持小数。
- 有限请求率保留两位小数，搜索最小步长不小于 0.1。
- 默认只保存 log、CSV 和成功时的 PNG，不保存逐请求 JSONL。
- 不停止、不重启、不删除现有服务或用户文件；`--flush-cache` 只调用官方接口。
- 不修改 `sglc.sh` 和 `prefill_ttft.sh` 的行为。

---

### Task 1: 建立 GSP case 配置和理论命中率计算

**Files:**
- Create: `sglang/gsp_prefix_bench.sh`
- Test: shell syntax check、`--help`、`--dry-run`

**Interfaces:**
- Consumes: environment variables `BASE_URL`, `MODEL`, `TOKEN_UNIT`, `GSP_NUM_GROUPS`, `GSP_PROMPTS_PER_GROUP`, `GSP_MAX_CONCURRENCY`, `RUN_ID` and repeated `--case` arguments.
- Produces: `parse_case`, `tokens_from_k`, `derive_gsp_lengths`, `theoretical_hit_rate`, `format_rate` and validated case records consumed by later tasks.

- [ ] **Step 1: Add the executable shell skeleton and usage text**

  Use `set -uo pipefail`, resolve `script_dir`, initialize `RUN_ID`, `out_dir`, `log_dir`, `trials_csv` and `summary_csv`. Add `--help`, repeated `--case`, `--plot`, `--cell`, and `--dry-run` parsing. Do not start a server or call the benchmark during `--help` or `--dry-run`.

- [ ] **Step 2: Implement token-unit parsing and validation**

  Accept `16k`, `0.5k`, `16K`, and plain positive numbers. Convert k values with `TOKEN_UNIT` using an inline Python calculation or `awk`, round to the nearest integer token, and reject zero/negative values. Validate `TOKEN_UNIT` as a positive integer and `GSP_NUM_GROUPS`/`GSP_PROMPTS_PER_GROUP` as positive integers.

- [ ] **Step 3: Implement the cold-miss-adjusted GSP length calculation**

  For case input `L`, hit rate `H` in percent, and prompts per group `P`, calculate:

  ```text
  system_len = round((H / 100) * P * L / (P - 1))
  question_len = L - system_len
  theory_hit_pct = 100 * (P - 1) * system_len / (P * L)
  ```

  Reject `P <= 1`, `system_len <= 0`, or `question_len <= 0`; print both lengths and the resulting theoretical percentage in dry-run output.

- [ ] **Step 4: Parse and validate the case grammar**

  Parse `INPUT,OUTPUT,HIT_RATE,TTFT_METRIC,TTFT_TARGETS_S,TPOT_METRIC,TPOT_TARGET_MS[,MAX_CONCURRENCY]`. Split TTFT targets on `|`, convert seconds to milliseconds, and allow only the official metric names `mean_ttft_ms`, `p50_ttft_ms`, `p90_ttft_ms`, `p95_ttft_ms`, `p99_ttft_ms`, `max_ttft_ms`, `mean_tpot_ms`, `p50_tpot_ms`, `p90_tpot_ms`, `p95_tpot_ms`, and `p99_tpot_ms` in their corresponding positions. Reject malformed or non-positive values before the first benchmark call.

- [ ] **Step 5: Verify the first task without a server**

  Run:

  ```bash
  bash -n sglang/gsp_prefix_bench.sh
  bash sglang/gsp_prefix_bench.sh --help
  TOKEN_UNIT=1024 bash sglang/gsp_prefix_bench.sh --dry-run \
    --case '16k,1k,80,mean_ttft_ms,3|5,mean_tpot_ms,30' \
    --case '32k,0.5k,80,mean_ttft_ms,5|7,mean_tpot_ms,50'
  ```

  Expected: syntax passes, help lists all flags, and dry-run prints positive system/question lengths whose theoretical hit rate is within token rounding error of 80%.

- [ ] **Step 6: Commit the configuration layer**

  ```bash
  git add sglang/gsp_prefix_bench.sh
  git commit -m "增加GSP命中率测试参数解析"
  ```

### Task 2: 调用官方 bench_serving 并解析完整指标

**Files:**
- Modify: `sglang/gsp_prefix_bench.sh`
- Test: `--dry-run` command inspection and a controlled failing endpoint

**Interfaces:**
- Consumes: validated case records from Task 1.
- Produces: `build_bench_command`, `check_bench_flags`, `run_trial`, `parse_benchmark_log` and normalized metric JSON for CSV and SLA evaluation.

- [ ] **Step 1: Add the official flag compatibility gate**

  Run `${PYTHON_BIN} -m sglang.bench_serving --help`, capture output, and require `generated-shared-prefix`, all five GSP length/group flags, `--gsp-ordered`, `--flush-cache`, `--seed`, and `--cache-report`. On failure, print the missing flag names and exit before touching the server.

- [ ] **Step 2: Build only the official command**

  Construct an array, never an interpolated command string:

  ```bash
  python3 -m sglang.bench_serving \
    --backend sglang --base-url "$BASE_URL" --model "$MODEL" \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups "$GSP_NUM_GROUPS" \
    --gsp-prompts-per-group "$GSP_PROMPTS_PER_GROUP" \
    --gsp-system-prompt-len "$system_len" \
    --gsp-question-len "$question_len" \
    --gsp-output-len "$output_len" \
    --gsp-ordered --num-prompts "$num_prompts" \
    --flush-cache --seed "$SEED" --cache-report \
    --request-rate "$request_rate" --disable-tqdm
  ```

  Append `--max-concurrency` only when the case or global setting is positive. Do not add `--output-details`, so official JSONL is not created.

- [ ] **Step 3: Persist one log and exitcode per trial**

  Name logs with a stable case slug and request-rate slug. If a non-empty log has `.exitcode` equal to `0`, skip it. Otherwise run the array through `tee`, capture `${PIPESTATUS[0]}`, and write the exit code even on failure.

- [ ] **Step 4: Parse official text metrics and cache-report**

  Use inline Python to map official labels to normalized fields: `request_throughput`, `input_throughput`, `output_throughput`, `mean/p50/p90/p95/p99/max_ttft_ms`, `mean/p50/p90/p95/p99_tpot_ms`, `total_prompt_tokens`, `total_cached_tokens`, and `cache_hit_rate_pct`. If cache-report fields are missing, return `NO_CACHE_REPORT` rather than calculating a fake value from theory.

- [ ] **Step 5: Verify command construction and failure retention**

  Run dry-run and set `BASE_URL=http://127.0.0.1:1` with a single small case. Expected: dry-run prints the exact official GSP flags; the unreachable run exits nonzero, creates a log and `.exitcode`, and does not create JSONL.

- [ ] **Step 6: Commit the official runner layer**

  ```bash
  git add sglang/gsp_prefix_bench.sh
  git commit -m "增加官方GSP命中率测试执行器"
  ```

### Task 3: Implement descending request-rate search and SLA summary

**Files:**
- Modify: `sglang/gsp_prefix_bench.sh`
- Test: deterministic synthetic log parser/SLA checks through `--dry-run` and local fixture logs created under a temporary directory outside the repository

**Interfaces:**
- Consumes: normalized trial metrics from Task 2 and case SLA definitions from Task 1.
- Produces: `evaluate_trial`, `append_trial_csv`, `write_summary_csv`, `is_sla_meet` and the final per-case/per-SLA rows.

- [ ] **Step 1: Add request-rate candidate generation**

  Run each case first at `inf`. Read its actual RPS from the parsed `request_throughput`. Generate finite candidates in descending order using `RATE_OFFSETS` defaulting to `0,-0.1,-0.1,-0.1,-0.2,-0.2,-0.3,-0.5`, applying cumulative subtraction from the baseline and clamping to a positive two-decimal value. Enforce `RATE_STEP >= 0.1`; support `GSP_REQUEST_RATES` as an explicit comma-separated override.

- [ ] **Step 2: Evaluate both TTFT and TPOT constraints**

  For every trial and every requested TTFT target, require:

  ```text
  selected_ttft_ms <= target_ttft_ms
  selected_tpot_ms <= target_tpot_ms
  actual cache hit rate is present
  ```

  Save `MEET`, `EXCEED`, `NO_CACHE_REPORT`, or `FAILED`. Select the maximum actual RPS among `MEET` rows; do not use configured request rate as the capacity value.

- [ ] **Step 3: Write stable CSV schemas**

  `gsp_prefix_trials.csv` must include one row per case/rate with case input/output/target, GSP lengths, theory hit percentage, measured cache fields, configured rate, actual RPS, throughput, all supported TTFT/TPOT metrics, status, and log path. `gsp_prefix_summary.csv` must include one row per case/TTFT target with selected actual RPS, selected metric values, TPOT, measured/theory hit rates, and `NO_MEET` when no valid row satisfies the rules.

- [ ] **Step 4: Add resume behavior**

  On rerun, reuse only logs whose exitcode is `0` and whose normalized metrics can be parsed. Rebuild CSVs from all available logs so an interrupted run can continue without duplicate rows.

- [ ] **Step 5: Verify SLA selection with a fixture**

  Feed the parser fixture rows where configured rates are `10.00`, `9.90`, and `9.80`, actual RPS are `9.8`, `9.7`, and `9.6`, and only the last row satisfies TTFT/TPOT. Expected summary selects actual RPS `9.6`, not configured rate `9.80`, and preserves the other two rows in trials.

- [ ] **Step 6: Commit the search and summary layer**

  ```bash
  git add sglang/gsp_prefix_bench.sh
  git commit -m "增加GSP命中率SLA和请求率搜索"
  ```

### Task 4: Add optional matrix plotting and README documentation

**Files:**
- Modify: `sglang/gsp_prefix_bench.sh`
- Modify: `README.md`
- Test: local CSV fixture and image existence/size checks

**Interfaces:**
- Consumes: `gsp_prefix_summary.csv` and `--cell rps|input_throughput`.
- Produces: `gsp_prefix_matrix.png` next to the CSV when `--plot` is set, plus documented commands and field meanings.

- [ ] **Step 1: Implement inline heatmap generation**

  Read summary CSV with inline Python, use columns as case labels (`16K/1K · hit 80%`, etc.), rows as TTFT SLA labels, and choose `selected_actual_rps` for `rps` or `selected_input_throughput` for `input_throughput`. Render each cell’s numeric value in the center, leave `NO_MEET` as an empty/annotated cell, and save beside the summary CSV. Fail clearly if matplotlib is unavailable.

- [ ] **Step 2: Add GSP README section**

  Document token-level hit-rate semantics, cold-miss formula, `TOKEN_UNIT`, the two sample cases, `--plot --cell`, output files, cache-report fields, resume behavior, and the fact that measured cache hit rate is authoritative.

- [ ] **Step 3: Verify the plot and documentation**

  Create a temporary summary fixture with two cases and two TTFT rows, run the plot path, check the PNG exists and is non-empty, and inspect README links/commands with `rg`.

- [ ] **Step 4: Commit script and documentation**

  ```bash
  git add sglang/gsp_prefix_bench.sh README.md
  git commit -m "补充GSP命中率矩阵作图和使用说明"
  ```

### Task 5: Run remote verification, review changes, and push

**Files:**
- Modify: only files already listed above if verification exposes a real defect
- Test: 811 node live benchmark and local static checks

**Interfaces:**
- Consumes: completed script, README, and two-case command from Tasks 1-4.
- Produces: verified output directory, reviewable git diff, and pushed GitHub commit.

- [ ] **Step 1: Run local static checks**

  ```bash
  bash -n sglang/gsp_prefix_bench.sh
  git diff --check HEAD~4..HEAD
  git status --short
  ```

  Confirm only intended tracked files are staged; leave existing untracked `all_result/` and sheet artifacts untouched.

- [ ] **Step 2: Confirm the 811 endpoint and installed SGLang flags**

  Before running, verify the exact node, endpoint, served model, and the installed `bench_serving --help`. Do not restart or alter the service. Abort the live run if the official flags or cache-report are unavailable.

- [ ] **Step 3: Run the two requested cases**

  ```bash
  bash sglang/gsp_prefix_bench.sh \
    --case '16k,1k,80,mean_ttft_ms,3|5,mean_tpot_ms,30' \
    --case '32k,0.5k,80,mean_ttft_ms,5|7,mean_tpot_ms,50' \
    --plot --cell rps
  ```

  Verify each successful trial contains official total prompt/cached tokens, actual cache hit rate, output length greater than 1, TTFT and TPOT. Verify the PNG uses actual RPS values from summary rows.

- [ ] **Step 4: Review evidence and fix only proven defects**

  Compare theory versus measured hit rate, inspect failed logs, confirm resume skips only successful trials, and rerun only affected cases if necessary. Do not convert missing cache-report or failed benchmark output into a passing result.

- [ ] **Step 5: Push the completed work**

  ```bash
  git status --short
  git log --oneline -5
  git push origin main
  ```

Report the pushed commit, GitHub branch, verification output directory, measured hit rates, and any blocked live cases separately.
