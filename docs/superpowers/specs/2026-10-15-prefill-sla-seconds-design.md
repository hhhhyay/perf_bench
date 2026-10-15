# Prefill SLA 按输入长度和秒单位配置设计

## 目标

让 `prefill_ttft.sh` 在一次运行中接收不同输入长度的不同 TTFT SLA，并让用户直接用秒填写目标值和 `near` 窗口。

## 配置格式

`PREFILL_SLA_RULES` 使用逗号分隔的规则：

```text
input_k:metric:target_seconds:near_seconds
```

示例：

```bash
PREFILL_TTFT_INPUTS="8 32" \
PREFILL_TTFT_BS_RULES="8:64,32:16" \
PREFILL_SLA_RULES="8:p99_ttft_ms:5:0.5,8:p99_ttft_ms:6:0.5,32:mean_ttft_ms:4:0.5,32:mean_ttft_ms:5:0.5" \
./sglang/prefill_ttft.sh
```

其中 `target_seconds` 和 `near_seconds` 可以是小数，例如 `0.5`。脚本内部转换为毫秒后参与现有指标比较和回测，CSV 继续保存 `target_value_ms` 与 `target_near_ms`，避免破坏现有数据平台字段。

## 行为

- 每条规则只作用于对应的 `input_k`。
- 同一输入可以配置多个指标或多个 SLA 目标。
- 目标判断仍使用 `target_metric <= target_value_ms`。
- `near` 继续作为回测元数据，不改变目标达标判断。
- 缺少对应输入规则时，该输入不启动 SLA 回测并输出明确错误。
- 指标名继续使用现有提取字段，例如 `mean_ttft_ms`、`p99_ttft_ms`；只有目标值和 `near` 使用秒单位。

## 兼容性

旧的 `metric:target:near` 三段格式不再作为默认推荐格式，因为它无法表达不同输入的独立 SLA。脚本报错信息和 README 将只展示新的四段格式。

## 验证

- 命令模式检查 8K/32K 规则可以正确解析并生成命令。
- 小数秒转换检查：`5.5` 秒变成 `5500` 毫秒，`0.25` 秒变成 `250` 毫秒。
- 规则隔离检查：8K 规则不会出现在 32K 的目标矩阵中。
- 非法输入检查：缺少输入前缀、目标非正数、near 为负数或额外字段时失败。
