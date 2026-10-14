# Prefill TTFT Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent SGLang Prefill TTFT matrix runner for configurable inputs and SLA metric/threshold rules, with default 32K/64K/128K inputs and 3s/5s/10s Mean TTFT targets, while keeping `sglang/sglc.sh` unchanged from its pre-Prefill state.

**Architecture:** `sglang/prefill_ttft.sh` will reuse the existing `sglc.sh` environment names, service discovery, `bench_serving` command arguments, token-unit conversion, flush-cache behavior, and RUN_ID conventions, but will select one per-DP BS per input and search request rate from high to low. It will keep the fixed client columns, retain the standard TTFT/TPOT/ITL fields and Mean E2E, then use P50 plus distinct TTFT fields in the post-E2E slots, and write detailed trials plus one best row per matrix point.

**Tech Stack:** Bash, `python3`, `curl`, `awk`, SGLang `bench_serving`, CSV files, Git Bash syntax checks.

## Global Constraints

- `sglang/sglc.sh` must remain identical to commit `007f969` and must not create Prefill result directories.
- The Prefill runner must use the same benchmark command parameters as `sglc.sh`; only input length, output length, selected BS, and request-rate search change.
- Default `TOKEN_UNIT=1024`; `TOKEN_UNIT=1000` remains an explicit decimal-K option.
- Prefill output length is exactly 1 token.
- Default matrix inputs are `32 64 128`; default SLA rules are Mean TTFT targets `3000/5000/10000` milliseconds.
- `PREFILL_SLA_RULES` accepts `metric:target:near` rules and drives both target judgment and matrix dimensions; supported target metrics include extracted TTFT fields such as `mean_ttft_ms`, `p50_ttft_ms`, `p95_ttft_ms`, and `p99_ttft_ms`.
- Default BS rules are `8:256,16:128,32:64,64:32,128:16`, matched by the first input upper bound that contains the input.
- Every Prefill probe uses `--flush-cache` and does not save per-request JSONL.
- Final performance selection uses parsed `Request throughput (req/s)` as `actual_rps`, never the configured request rate.
- Keep the fixed client fields aligned with `sglc.sh`, retain `Mean/P95/P99 TTFT`, `Mean/P95/P99 TPOT`, `Mean/P95/P99 ITL`, and `Mean E2E Latency`, then use `P50` plus distinct TTFT fields (`Std`/`P75`/`P90`/`P999`/`Max`) in the post-E2E slots; skip duplicate Median TTFT, other E2E percentiles, and extra TPOT/ITL percentiles.
- Do not add or commit the unrelated untracked `sheet_*.json` files.

---

### Task 1: Lock the restored baseline and create the independent runner skeleton

**Files:**
- Create: `sglang/prefill_ttft.sh`
- Verify: `sglang/sglc.sh`

**Interfaces:**
- Consumes the same common environment variables as `sglc.sh`, including `BASE_URL`, `MODEL`, `DP_SIZE`, `MAX_TOTAL_TOKENS`, `MAX_RUNNING_REQUESTS`, `TP_SIZE`, `PP_SIZE`, `EP_SIZE`, `ATTN_CP_SIZE`, `KV_CACHE_DTYPE`, `MEM_FRACTION_STATIC`, `DP_ATTENTION`, `CUDA_GRAPH_DECODE_BS`, `PARALLEL_CONFIG`, `USE_FAKE_PREFILL`, `NUM_PROMPTS_MULTIPLIER`, `TOKEN_UNIT`, and `RUN_ID`.
- Produces `all_result/bench_prefill_ttft_<RUN_ID>/` with `logs/`, `prefill_ttft_trials.csv`, and `prefill_ttft_matrix.csv`.

- [ ] **Step 1: Verify the restored baseline before editing the new runner**

Run:

```powershell
git diff 007f969 -- sglang/sglc.sh
& 'D:\\Git\\bin\\bash.exe' -n 'sglang/sglc.sh'
```

Expected: no diff output and a zero shell syntax exit code.

- [ ] **Step 2: Add the standalone script configuration and guarded entry point**

Add the following defaults near the top of `sglang/prefill_ttft.sh`:

```bash
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PREFILL_TTFT_INPUTS="${PREFILL_TTFT_INPUTS:-32 64 128}"
PREFILL_SLA_RULES="${PREFILL_SLA_RULES:-mean_ttft_ms:3000:0,mean_ttft_ms:5000:0,mean_ttft_ms:10000:0}"
PREFILL_TTFT_SEARCH_ITERS="${PREFILL_TTFT_SEARCH_ITERS:-7}"
PREFILL_TTFT_BS_RULES="${PREFILL_TTFT_BS_RULES:-8:256,16:128,32:64,64:32,128:16}"
TOKEN_UNIT="${TOKEN_UNIT:-1024}"
```

Use the same default service variables and validation rules as `sglc.sh`. Keep the executable main flow in a `main()` function and invoke it only when the file is executed directly, so helper functions can be sourced by shell-level tests.

- [ ] **Step 3: Run the skeleton syntax check**

Run:

```powershell
& 'D:\\Git\\bin\\bash.exe' -n 'sglang/prefill_ttft.sh'
```

Expected: zero exit code.

- [ ] **Step 4: Commit the standalone runner skeleton**

```powershell
git add -- sglang/prefill_ttft.sh
git commit -m "新增Prefill TTFT矩阵脚本骨架"
```

### Task 2: Reuse SGLang service and benchmark command semantics

**Files:**
- Modify: `sglang/prefill_ttft.sh`

**Interfaces:**
- Produces `resolve_service_config`, `tokens_from_k`, `select_prefill_bs`, and `build_bench_command` shell functions.
- `build_bench_command` returns the same command array as `sglc.sh`, with `--request-rate` supplied by the caller.

- [ ] **Step 1: Port service configuration probes without changing `sglc.sh`**

Implement the same read-only `/v1/loads`, `/metrics`, and `/server_info` fallback behavior needed to resolve `DP_SIZE`, `MAX_TOTAL_TOKENS`, `MAX_RUNNING_REQUESTS`, and parallel metadata. Manual environment overrides must win over probes, matching `sglc.sh`.

- [ ] **Step 2: Implement token conversion and BS rule selection**

For each `input_k`, convert with the same rounding behavior as `sglc.sh`. Parse `PREFILL_TTFT_BS_RULES` as `upper_k:per_dp_bs` pairs and select the first rule satisfying `input_k <= upper_k`; reject an uncovered input with an explicit error.

For the default matrix this must resolve to:

```text
32K  -> per_dp_bs=64
64K  -> per_dp_bs=32
128K -> per_dp_bs=16
```

- [ ] **Step 3: Build the exact common benchmark command**

Use this command shape for every baseline and finite-rate probe:

```bash
cmd=(
  python3 -m sglang.bench_serving
  --backend sglang
  --base-url "${BASE_URL}"
  --model "${MODEL}"
  --dataset-name random-ids
  --tokenize-prompt
  --num-prompts "${num_prompts}"
  --random-input-len "${input_len}"
  --random-output-len 1
  --random-range-ratio 1.0
  --request-rate "${request_rate}"
  --max-concurrency "${global_concurrency}"
  --warmup-requests "${warmup_requests}"
  --flush-cache
  --disable-tqdm
)
```

Append `--fake-prefill` only when `USE_FAKE_PREFILL=1`. Use `global_concurrency=DP_SIZE*per_dp_bs`, `num_prompts=global_concurrency*NUM_PROMPTS_MULTIPLIER`, and the same warmup rule as `sglc.sh`.

- [ ] **Step 4: Test command construction without starting a service**

Run the script's helper test mode with `BASE_URL`, `MODEL`, `DP_SIZE=2`, `NUM_PROMPTS_MULTIPLIER=1`, and input values `32 64 128`. Assert the generated arrays contain `--random-output-len 1`, `--flush-cache`, `--disable-tqdm`, and the selected concurrency values `128`, `64`, and `32`.

### Task 3: Add log parsing and CSV persistence

**Files:**
- Modify: `sglang/prefill_ttft.sh`

**Interfaces:**
- Produces `parse_benchmark_log`, `append_trial_row`, and `write_matrix_summary`.
- `parse_benchmark_log <log>` returns the selected metrics as stable `key=value` records for the caller.

- [ ] **Step 1: Implement the metric allowlist**

Keep these fields from the log prefix before Mean E2E:

```text
Backend
Traffic request rate
Max request concurrency
Successful requests
Benchmark duration (s)
Total input tokens
Total input text tokens
Total generated tokens
Total generated tokens (retokenized)
Request throughput (req/s)
Input token throughput (tok/s)
Output token throughput (tok/s)
Peak output token throughput (tok/s)
Peak concurrency
Total token throughput (tok/s)
Concurrency
Accept length
```

Then keep these non-duplicated TTFT fields around the `Mean E2E` position:

```text
Mean TTFT (ms)
P95 TTFT (ms)
P99 TTFT (ms)
Mean TPOT (ms)
P95 TPOT (ms)
P99 TPOT (ms)
Mean ITL (ms)
P95 ITL (ms)
P99 ITL (ms)
Mean E2E Latency (ms)
P50 TTFT (ms)
Std TTFT (ms)
P75 TTFT (ms)
P90 TTFT (ms)
P999 TTFT (ms)
Max TTFT (ms)
```

Retain `Mean E2E Latency`, standard Mean/P95/P99 TPOT and ITL fields, skip other E2E percentiles and duplicate Median TTFT, and use P50 plus distinct TTFT fields in the post-E2E slots. Preserve distinct keys rather than merging labels. Set `actual_rps` from `Request throughput (req/s)` and use `mean_ttft_ms` for target decisions.

- [ ] **Step 2: Add atomic trial CSV upsert**

Write UTF-8 CSV through a temporary file and `os.replace`. Deduplicate by `(input_k, target_metric, target_value_ms, target_near_ms, trial, input_request_rate)` and include the metadata, all prefix metrics, all TTFT metrics, `case_status`, and `log_file`.

- [ ] **Step 3: Add best-row matrix summary**

For each `(input_k, target_metric, target_value_ms, target_near_ms)`, select rows with successful benchmark exit and `target_metric <= target_value_ms`; choose maximum `actual_rps`, then lower target metric value as the tie breaker. Write a `FAIL`/`NO_PASS` summary when no valid candidate exists.

- [ ] **Step 4: Test parsing with a synthetic log fixture**

Use a temporary log containing prefix fields, E2E fields, all selected TTFT fields including `P90 TTFT (ms): 1234`, and standard TPOT/ITL fields. Assert the output contains `mean_e2e_latency_ms`, `p90_ttft_ms=1234`, the standard TPOT/ITL fields, the prefix throughput fields, and no unselected E2E/TPOT/ITL keys.

### Task 4: Implement high-to-low TTFT search and resume

**Files:**
- Modify: `sglang/prefill_ttft.sh`

**Interfaces:**
- Produces `run_baseline`, `run_probe`, `search_target`, and `run_input_matrix`.
- `search_target` returns the best candidate row for one input and one TTFT target.

- [ ] **Step 1: Run one `inf` baseline per input length**

Create a stable baseline log name containing input length and selected BS. Reuse a successful `.log` plus `.exitcode=0` on the same `RUN_ID`. Extract baseline actual RPS and Mean TTFT before starting the target searches.

- [ ] **Step 2: Search from the high point down to a passing bracket**

Use baseline actual RPS as the high point. If baseline Mean TTFT meets the target, record it as a candidate. Otherwise probe lower request rates, starting at the midpoint between zero and the current high point, until a finite-rate result meets the target; keep the last failing rate and first passing rate as the bracket.

- [ ] **Step 3: Refine the bracket with bounded binary probes**

For at most `PREFILL_TTFT_SEARCH_ITERS` probes, test the midpoint of the failing/passing bracket. A passing result moves the lower bound upward; an exceeding result moves the upper bound downward. Format rates to six decimals, stop when the midpoint repeats, and never issue a zero request rate.

- [ ] **Step 4: Use stable logs and resume successful probes**

Name logs with input K, selected BS, target milliseconds, trial, and request-rate tag. If a matching log and `.exitcode=0` already exist, parse and reuse it; rerun only missing or failed probes. Every executed probe must pass `--flush-cache`.

- [ ] **Step 5: Test search behavior with mocked probe metrics**

Feed a deterministic sequence where high rates exceed the target and lower rates pass. Assert the first probe is lower than the baseline, the search narrows the failing/passing interval, and the selected row is the highest `actual_rps` whose Mean TTFT is within the target.

### Task 5: Add documentation and regression verification

**Files:**
- Modify: `README.md`
- Verify: `sglang/sglc.sh`, `sglang/prefill_ttft.sh`

**Interfaces:**
- Documents the independent entry point, inherited parameters, BS rules, matrix targets, output files, metric allowlist, and RUN_ID resume semantics.

- [ ] **Step 1: Remove obsolete single-side Prefill backtest documentation**

Delete the old seven-run delta description and `prefill_backtest/prefill_backtest.csv` references that describe behavior no longer present in `sglc.sh`.

- [ ] **Step 2: Document the independent command**

Add a runnable example:

```bash
PREFILL_TTFT_INPUTS="32 64 128" \
PREFILL_SLA_RULES="mean_ttft_ms:3000:0,mean_ttft_ms:5000:0,mean_ttft_ms:10000:0" \
PREFILL_TTFT_BS_RULES="8:256,16:128,32:64,64:32,128:16" \
bash sglang/prefill_ttft.sh
```

Document that common `sglc.sh` service and benchmark variables retain their names and semantics, while the new script uses output 1 token and searches request rate from high to low.

- [ ] **Step 3: Document the exact metrics**

State that the fixed client columns, Mean/P95/P99 TTFT, Mean/P95/P99 TPOT, Mean/P95/P99 ITL, and Mean E2E are retained; post-E2E slots use P50 plus distinct Std/P75/P90/P999/Max TTFT fields, while duplicate Median TTFT, other E2E percentiles, and extra TPOT/ITL percentiles are excluded.

- [ ] **Step 4: Run the full static verification**

```powershell
& 'D:\\Git\\bin\\bash.exe' -n 'sglang/sglc.sh'
& 'D:\\Git\\bin\\bash.exe' -n 'sglang/prefill_ttft.sh'
git diff --check
git diff 007f969 -- sglang/sglc.sh
```

Expected: both syntax checks and diff check pass, and the final `sglc.sh` diff is empty.

- [ ] **Step 5: Commit the implementation and documentation**

```powershell
git add -- README.md sglang/prefill_ttft.sh
git commit -m "新增Prefill TTFT矩阵回测"
```
