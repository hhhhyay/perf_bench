# GSP Script Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the unused legacy summary implementation from `sglang/gsp_prefix_bench.sh` while preserving all current benchmark behavior and output formats.

**Architecture:** Keep the existing shell/Python hybrid script unchanged except for deleting the unreferenced `write_summary_legacy()` function. The active `write_summary()` path remains the only summary implementation; workload selection, calibration, rate replay, plotting, and file outputs remain untouched.

**Tech Stack:** Bash, embedded Python, Git.

## Global Constraints

- Preserve all command-line arguments, environment variables, test ordering, CSV/TSV columns, log naming, and benchmark algorithms.
- Preserve `HIT_RATE=0` using `random-ids --tokenize-prompt` and `HIT_RATE>0` using `generated-shared-prefix`.
- Do not modify `sglang/prefill_ttft.sh`.
- Do not modify or remove existing untracked result files.
- Do not change the existing `docs/superpowers/specs/2026-10-22-gsp-script-cleanup-design.md`.

---

### Task 1: Remove Dead Legacy Summary Code

**Files:**
- Modify: `sglang/gsp_prefix_bench.sh:1299-1411`
- Test: shell syntax, help output, and dry-run command generation

**Interfaces:**
- Consumes: the current `write_summary()` function and the main execution flow at the end of `sglang/gsp_prefix_bench.sh`.
- Produces: the same `gsp_prefix_summary.csv`, `gsp_prefix_summary.tsv`, trial pruning, and plot outputs as before.

- [ ] **Step 1: Confirm the deletion target is unreferenced**

Run:

```powershell
rg -n "\bwrite_summary_legacy\b|\bwrite_summary\b" sglang/gsp_prefix_bench.sh
```

Expected: `write_summary_legacy` appears only at its function definition, while `write_summary` appears at its definition and active call site.

- [ ] **Step 2: Delete only `write_summary_legacy()`**

Remove the complete function body from its definition through the line immediately before `write_summary()`. Leave `write_summary()` and all surrounding function boundaries unchanged.

- [ ] **Step 3: Verify shell syntax and dead-code removal**

Run:

```powershell
git diff --check
rg -n "\bwrite_summary_legacy\b" sglang/gsp_prefix_bench.sh
```

Expected: no whitespace errors and no remaining `write_summary_legacy` match.

- [ ] **Step 4: Verify the CLI and both workload command paths**

Run:

```powershell
ssh -p 10035 public_user@8.130.89.93 'ssh h20-1 "docker exec gbj_811_q3.8_27b_fp8 bash -n /home/v4flash/perf_bench/sglang/gsp_prefix_bench.sh"'
```

Then run the script's help and dry-run paths in the test container. Confirm that the 0% case contains `--dataset-name random-ids --tokenize-prompt` and the non-zero case contains `--dataset-name generated-shared-prefix`.

- [ ] **Step 5: Review the final diff and commit**

Run:

```powershell
git diff --stat
git status --short
git add sglang/gsp_prefix_bench.sh
git commit -m "删除GSP脚本未使用汇总代码"
```

Expected: only the tracked benchmark script is staged; existing untracked result and sheet files remain untouched.
