# Prefill SLA Seconds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow one Prefill run to configure input-specific TTFT SLA rules with target and near values expressed in seconds.

**Architecture:** Keep the benchmark and metric extraction paths unchanged. Add a rule parser that maps each input length to one or more internal millisecond rules, then feed only that input's rules into the existing search and summary functions. Preserve `target_value_ms` and `target_near_ms` in CSV output.

**Tech Stack:** Bash, awk, Python standard library snippets already embedded in `sglang/prefill_ttft.sh`, Markdown documentation.

## Global Constraints

- Rule syntax is `input_k:metric:target_seconds:near_seconds`.
- `target_seconds` and `near_seconds` accept positive decimal numbers and convert to milliseconds internally.
- Existing TTFT metric names remain unchanged.
- No benchmark service restart or remote test is part of this change.
- Do not modify `sglang/sglc.sh`.

---

### Task 1: Add input-scoped seconds rule parsing

**Files:**
- Modify: `sglang/prefill_ttft.sh` near `prefill_sla_rules_text`, `validate_prefill_sla_rules`, and `run_input_matrix`.
- Test: `sglang/prefill_ttft.sh` with `PREFILL_TTFT_TEST_MODE=command`.

**Interfaces:**
- Consumes `PREFILL_SLA_RULES` entries in `input_k:metric:target_seconds:near_seconds` format.
- Produces per-input rules in the existing internal `metric,target_value_ms,target_near_ms` shape.

- [ ] **Step 1: Add a parser that validates four fields and converts seconds to milliseconds**

Implement a Bash helper that splits each comma-separated rule, validates the input length, metric, positive decimal target and non-negative decimal near, then uses `awk` to multiply target and near by `1000` and normalize the values.

- [ ] **Step 2: Make `run_input_matrix` select only rules matching its input length**

Replace the current loop over all global rules with a loop over parsed rules whose first field equals `input_k`. Pass converted millisecond values to `search_target` and preserve them in CSV rows.

- [ ] **Step 3: Fail clearly when a rule has no input prefix or no input is configured**

Emit the supported syntax in the error message and stop before running a benchmark when a rule is malformed.

- [ ] **Step 4: Run command-mode validation**

Run:

```bash
PREFILL_TTFT_TEST_MODE=command \
PREFILL_TTFT_INPUTS="8 32" \
PREFILL_TTFT_BS_RULES="8:64,32:16" \
PREFILL_SLA_RULES="8:p99_ttft_ms:5:0.5,8:p99_ttft_ms:6:0.5,32:mean_ttft_ms:4:0.5,32:mean_ttft_ms:5:0.5" \
./sglang/prefill_ttft.sh
```

Expected: the command is accepted; no remote benchmark is started; malformed rules are rejected.

### Task 2: Update README and design references

**Files:**
- Modify: `README.md` sections describing Prefill SLA rules and examples.
- Modify: `docs/superpowers/specs/2026-10-15-prefill-ttft-matrix-design.md` and `docs/superpowers/plans/2026-10-15-prefill-ttft-matrix.md` references to the old syntax.

**Interfaces:**
- Documents the exact four-field syntax, decimal seconds, input-specific scoping, and CSV millisecond fields.

- [ ] **Step 1: Replace old examples with an input-specific seconds example**

Document:

```bash
PREFILL_SLA_RULES="8:p99_ttft_ms:5:0.5,8:p99_ttft_ms:6:0.5,32:mean_ttft_ms:4:0.5,32:mean_ttft_ms:5:0.5"
```

- [ ] **Step 2: Document the conversion and validation behavior**

State that `5` means 5 seconds, `0.5` means 500 milliseconds, and the CSV remains millisecond-based for compatibility.

- [ ] **Step 3: Search for stale three-field examples**

Run `rg -n "PREFILL_SLA_RULES=.*(mean|p99).*:[0-9]+" README.md docs` and update every Prefill-specific example that still implies the old global syntax.

### Task 3: Verify regression boundaries

**Files:**
- Test: `sglang/prefill_ttft.sh`, `README.md`.

- [ ] **Step 1: Check the existing command builder**

Run the command-mode invocation and confirm it still emits `--random-output-len 1`, `--flush-cache`, and the configured input/BS values.

- [ ] **Step 2: Check numeric conversions directly**

Use a shell assertion or command-mode output to verify `5.5` becomes `5500` and `0.25` becomes `250` in the internal target fields.

- [ ] **Step 3: Review the diff and status**

Run `git diff --check`, inspect only the intended script/docs changes, and leave unrelated untracked benchmark artifacts untouched.

