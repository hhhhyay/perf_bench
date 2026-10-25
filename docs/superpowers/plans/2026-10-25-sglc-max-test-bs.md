# SGLC 自动 BS 测试上限 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `sglang/sglc.sh` 增加按 per-DP BS 限制自动扫描范围的配置，同时保留理论容量、固定测试模式和现有 CSV 结构。

**Architecture:** 将“自动扫描列表截断并补入用户指定边界”的纯 shell 逻辑放入可单测的 `sglc_bs_helpers.sh`，主脚本只负责配置解析、模式分流和日志展示。`MAX_TEST_PER_DP_BS` 仅在 `auto_theory_bs` 模式调用，固定 `FIXED_PER_DP_BS`/`FIXED_CONCURRENCY` 在更早分支返回，因此不会被静默截断。

**Tech Stack:** Bash、GNU `sort`/`awk`、Python 元数据写入、Markdown 文档。

## Global Constraints

- `MAX_TEST_PER_DP_BS` 为空时保持现有自动 BS 扫描行为。
- `MAX_TEST_PER_DP_BS` 必须是正整数，并按 per-DP BS 解释。
- 理论 BS 仍由现有 token/request 约束和 CUDA Graph/fallback 逻辑计算；新变量不扩大理论容量。
- 上限不在自动扫描点中且不超过理论 BS 时必须补入该边界；上限超过理论 BS 时只扫描到理论 BS。
- 自动列表被截到上限后仍保留 BS1；固定测试模式不受新变量影响。
- `run_metadata.json` 记录新环境变量；CSV 指标和列顺序不改变。
- 不停止服务、不修改服务端配置、不删除已有结果；保留工作区中与本功能无关的未跟踪文件。

---

## 文件结构

- Modify: `sglang/sglc.sh` — 读取、校验并记录变量，在自动模式接入上限函数，并打印理论/实际扫描边界。
- Create: `sglang/sglc_bs_helpers.sh` — 提供无副作用的 `cap_auto_bs_list <auto_bs_list> <max_test_per_dp_bs>`。
- Create: `sglang/tests/test_sglc_bs_helpers.sh` — 覆盖默认、截断、补边界、超过理论和 BS1 保留行为。
- Modify: `README.md` — 增加环境变量说明、自动模式示例和固定模式不受影响的说明。

### Task 1: Add and test the pure automatic-BS cap helper

**Files:**
- Create: `sglang/sglc_bs_helpers.sh`
- Create: `sglang/tests/test_sglc_bs_helpers.sh`

**Interfaces:**
- Consumes: 空格分隔的自动 BS 列表，例如 `1 2 4 8 16 32`；正整数上限或空字符串。
- Produces: 升序、去重、空格分隔的测试 BS 列表；空上限原样返回；上限大于理论列表最大值时不扩容。

- [ ] **Step 1: Write the failing shell assertions**

在测试文件中先写以下自包含断言；测试引用的 helper 此时尚不存在，因此应失败：

```bash
#!/usr/bin/env bash
set -euo pipefail
script_dir="$(pwd)/sglang"
source "$script_dir/sglc_bs_helpers.sh"

assert_eq() {
  local expected="$1" actual="$2" label="$3"
  [[ "$actual" == "$expected" ]] || {
    printf 'FAIL: %s expected=<%s> actual=<%s>\n' "$label" "$expected" "$actual" >&2
    exit 1
  }
}

assert_eq '1 2 4 8 16 32 ' "$(cap_auto_bs_list '1 2 4 8 16 32' '')" 'empty cap'
assert_eq '1 2 4 8 16 ' "$(cap_auto_bs_list '1 2 4 8 16 32' 16)" 'cap 16'
assert_eq '1 2 4 8 12 16 ' "$(cap_auto_bs_list '1 2 4 8 16' 12)" 'add missing boundary'
assert_eq '1 2 4 8 16 ' "$(cap_auto_bs_list '1 2 4 8 16' 64)" 'do not expand theory'
assert_eq '1 ' "$(cap_auto_bs_list '4 8 16' 2)" 'keep BS1'
printf 'PASS: sglc BS cap helper\n'
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
bash sglang/tests/test_sglc_bs_helpers.sh
```

Expected: `sglang/sglc_bs_helpers.sh` does not exist.

- [ ] **Step 3: Write the minimal helper**

Create `sglang/sglc_bs_helpers.sh` with `cap_auto_bs_list`. It must return the original list for an empty cap; find the largest existing BS as the theoretical ceiling; filter points above the effective ceiling; add the requested cap when it is at or below that ceiling; always add BS1; and finish with `sort -n -u` while preserving the current trailing-space output convention.

```bash
#!/usr/bin/env bash

cap_auto_bs_list() {
  local auto_bs_list="$1"
  local max_test_bs="$2"
  [[ -z "$max_test_bs" ]] && { printf '%s\n' "$auto_bs_list"; return 0; }

  local theory_max=0 bs effective_cap
  for bs in $auto_bs_list; do
    [[ "$bs" =~ ^[0-9]+$ ]] || continue
    (( bs > theory_max )) && theory_max="$bs"
  done
  (( theory_max >= 1 )) || { printf '\n'; return 0; }

  effective_cap="$max_test_bs"
  (( effective_cap > theory_max )) && effective_cap="$theory_max"
  {
    for bs in $auto_bs_list; do
      [[ "$bs" =~ ^[0-9]+$ ]] && (( bs <= effective_cap )) && printf '%s\n' "$bs"
    done
    (( max_test_bs <= theory_max )) && printf '%s\n' "$max_test_bs"
    printf '1\n'
  } | sort -n -u | tr '\n' ' '
}
```

- [ ] **Step 4: Run the focused test and verify it passes**

```bash
bash sglang/tests/test_sglc_bs_helpers.sh
```

Expected: `PASS: sglc BS cap helper`.

- [ ] **Step 5: Commit the focused helper and test**

```bash
git add sglang/sglc_bs_helpers.sh sglang/tests/test_sglc_bs_helpers.sh
git commit -m "test: 增加SGLC自动BS上限校验"
```

### Task 2: Integrate the cap into SGLC without changing fixed modes

**Files:**
- Modify: `sglang/sglc.sh` near the metadata prefix, fixed-mode variables, validation, `get_test_bs_list`, global configuration output, and per-case BS output.

**Interfaces:**
- Consumes: `MAX_TEST_PER_DP_BS` from the environment and `cap_auto_bs_list` from `sglang/sglc_bs_helpers.sh`.
- Produces: auto mode's `bs_list`, run metadata, configuration output, and per-case log lines that identify theoretical BS versus actual test BS upper bound.

- [ ] **Step 1: Add the integration check before editing**

```bash
bash -n sglang/sglc.sh
rg -n 'MAX_TEST_PER_DP_BS|sglc_bs_helpers|Auto Test Cap|Test BS Upper' sglang/sglc.sh
```

Expected: syntax passes, but the `rg` command has no integration matches.

- [ ] **Step 2: Add configuration, source, and validation**

Source the helper immediately after `script_dir` is set:

```bash
source "$script_dir/sglc_bs_helpers.sh"
```

Near `FIXED_PER_DP_BS` and `FIXED_CONCURRENCY`, add:

```bash
# 自动理论 BS 扫描的 per-DP 上限；为空时不限制。
MAX_TEST_PER_DP_BS="${MAX_TEST_PER_DP_BS:-}"
```

After the existing `NUM_PROMPTS_MULTIPLIER` validation, reject non-empty non-positive/non-integer values:

```bash
if [[ -n "$MAX_TEST_PER_DP_BS" && ! "$MAX_TEST_PER_DP_BS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] MAX_TEST_PER_DP_BS 必须是正整数，当前=$MAX_TEST_PER_DP_BS"
  exit 1
fi
```

- [ ] **Step 3: Wire only the auto branch**

Keep the existing `FIXED_PER_DP_BS` and `FIXED_CONCURRENCY` return branches in `get_test_bs_list` unchanged. Replace only the final automatic return with:

```bash
cap_auto_bs_list "$auto_bs_list" "$MAX_TEST_PER_DP_BS"
```

Thus fixed modes remain exact even if the new variable is present.

- [ ] **Step 4: Record metadata and the actual scan boundary**

Add `MAX_TEST_PER_DP_BS` to the Python metadata environment-prefix tuple. Print the setting in the global configuration block. In the per-case block, after `bs_list` is computed, print both the configured automatic cap and the last actual tested BS:

```bash
test_bs_upper="$(printf '%s\n' $bs_list | tail -n 1)"
echo "# Auto Test Cap     : ${MAX_TEST_PER_DP_BS:-NONE(theory)}"
echo "# Test BS Upper      : $test_bs_upper"
```

Retain `# Effective Theory` and `# Theory Global BS`; do not add or reorder CSV columns.

- [ ] **Step 5: Run integration checks**

```bash
bash -n sglang/sglc.sh
bash sglang/tests/test_sglc_bs_helpers.sh
rg -n 'MAX_TEST_PER_DP_BS|cap_auto_bs_list|Auto Test Cap|Test BS Upper' sglang/sglc.sh
```

Expected: syntax succeeds, helper tests pass, and all integration points are present.

- [ ] **Step 6: Commit the integration**

```bash
git add sglang/sglc.sh
git commit -m "feat: 增加SGLC自动BS测试上限"
```

### Task 3: Document usage and verify compatibility

**Files:**
- Modify: `README.md` in the common environment-variable table and fixed-concurrency examples.

- [ ] **Step 1: Add the README entry and example**

Add this row after `FIXED_CONCURRENCY`:

```markdown
| `MAX_TEST_PER_DP_BS` | 自动理论 BS 扫描的 per-DP 上限；为空时按理论容量扫描，只有自动模式生效，固定 BS/并发模式不截断。 |
```

Add this example after the fixed-concurrency examples:

```bash
# 理论 per-DP BS 为 32 时，本次自动扫描最多验证到 BS16；理论值更小时不扩容。
MAX_TEST_PER_DP_BS=16 bash sglang/sglc.sh
```

Explain that `Effective Theory`, `Auto Test Cap`, and `Test BS Upper` distinguish theoretical capacity from this run's test range.

- [ ] **Step 2: Verify documentation and metadata key**

```bash
rg -n 'MAX_TEST_PER_DP_BS|Auto Test Cap|Test BS Upper|理论 per-DP BS 为 32' README.md sglang/sglc.sh
bash -n sglang/sglc.sh
```

Expected: README contains the variable and example; the script remains syntactically valid.

- [ ] **Step 3: Review the final diff and working tree**

```bash
git diff --check
git status --short
git log -5 --oneline --decorate
```

Expected: unrelated pre-existing untracked files remain untouched.

- [ ] **Step 4: Commit the documentation**

```bash
git add README.md
git commit -m "docs: 说明SGLC自动BS测试上限"
```

## Verification Summary

After all tasks, run:

```bash
bash sglang/tests/test_sglc_bs_helpers.sh
bash -n sglang/sglc.sh
bash -n sglang/sglc_bs_helpers.sh
git diff --check
```

远程 benchmark 不属于本次代码改动；后续真实节点验证仍需先检查 GPU/DCU、容器、端口和剩余显存。

