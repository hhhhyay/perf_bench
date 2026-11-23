# perf_bench

SGLang / vLLM 推理服务容量压测脚本集合，重点用于按照 per-DP batch size 扫描服务吞吐、TTFT、TPOT、ITL、E2EL 等指标，并把单次压测输出汇总为 CSV/TSV。

## 目录结构

```text
perf_bench/
├── sglang/
│   ├── sglc.sh      # SGLang 客户端容量压测
│   ├── sglc_plot.py # Decode 容量矩阵作图
│   ├── gsp_prefix_bench.sh # 统一的 Prefill/GSP TTFT/TPOT 容量压测入口
│   ├── gsp_prefix_metrics.py # benchmark 日志解析与 trials CSV
│   ├── gsp_prefix_summary.py # summary TSV 生成与正式 trial 清理
│   ├── gsp_prefix_plot.py # Prefill/GSP 容量矩阵作图
│   ├── prof_capture.sh # 在线抓 torch profiler (CPU/GPU) + 自动分析
│   ├── prof_analyze.py # trace 汇总: GPU busy / kernel TopN / CPU 顶层 op / 空档分桶 / 通信子类
│   └── prof_dashboard.py # 分析产物 → 自包含 HTML 可视化看板 (对账/类目/rank/空泡/调度通信)
└── vllm/
    ├── vllmc.sh     # vLLM Prefix-Salt 标准容量压测
    ├── vllm_bench_metrics.py # vLLM benchmark 公共指标解析与 Prefill 回测核心
    ├── vllm_prefill_bench.sh # vLLM 单侧 Prefill TTFT 回测入口
    └── vllm_prefill_plot.py # vLLM Prefill 容量矩阵作图
```

## 脚本说明

### SGLang: `sglang/sglc.sh`

- 优先通过 `/v1/loads?include=all` 读取 `dp_size`、`max_total_num_tokens`、`max_running_requests`。
- `/v1/loads` 不可用时回退 `/metrics` 和环境变量。
- 自动读取 `/server_info` 中的 TP / DP / PP / EP / ACP / KV cache dtype / CUDA Graph decode BS。
- 理论 per-DP BS 同时受 KV token capacity 和 `max_running_requests` 约束。
- 每个 benchmark case 使用 `--flush-cache`；warmup 完成后清理服务端 Prefix Cache，避免预热请求影响正式测量。
- 支持 `SLA_RULES` inline 加密扫描：当当前 BS 命中任一 SLA 规则后，连续补测到下一个基础 BS 前。
- 支持 `HARD_STOP_RULES` 硬停止：指标命中阈值后停止当前 case 的后续 BS 测试。
- 直接从 benchmark 日志提取 RPS、Token 吞吐、请求数、耗时和时延分位数，不保存逐请求 JSONL。
- 默认根据 `sum_all.csv` 和 `SLA_RULES` 生成 `decode_matrix.png`；行是 TPOT/TTFT SLA，列是输入/输出与 per-DP BS，格子显示满足 SLA 的最大实际客户端 RPS。
- `SGLC_PLOT_CELL=output_throughput` 可将格子改为输出 token throughput，`SGLC_PLOT=0` 可关闭默认作图。

### SGLang GSP Prefix: `sglang/gsp_prefix_bench.sh`

这是统一的 Prefill/GSP 压测入口。`HIT_RATE=0` 使用官方 `random-ids` 生成独立 token 输入，
等价于无 prefix-cache 的单侧 Prefill；`HIT_RATE>0` 调用官方
`generated-shared-prefix` 数据集。脚本同时支持 Prefill、输出大于 1 的普通推理和后续 PD
分离场景，不把请求数命中比例当成 token 级 cache 命中率。

命中率按官方 `--cache-report` 的结果计算：

```text
实际命中率 = total_cached_tokens / total_prompt_tokens
```

脚本使用 `--gsp-ordered`，并把每个 prefix group 的首个请求视为冷 miss。给定总输入
长度 `L`、目标命中率 `H` 和每组请求数 `P`，先自动计算目标共享前缀：

```text
raw_system_prompt_len = round((H / 100) * P * L / (P - 1))
system_prompt_len = page 对齐后的最接近长度
question_len = L - system_prompt_len
理论命中率 = (P - 1) * system_prompt_len / (P * L)
```

真实 KV cache 按 page 命中。脚本默认从 `/server_info` 读取 `page_size`，读取失败时回退到
`64`，并选择最接近目标的 page 对齐长度；CSV 同时记录未对齐理论值、page 对齐后的理论值
和官方 `--cache-report` 实测值。设置 `GSP_PAGE_ALIGN=0` 可关闭对齐，设置
`GSP_PAGE_SIZE` 可显式指定 page 大小。理论值只用于解释 workload，容量筛选始终使用实测命中率。
如果特定服务版本的 `cache-report` 相对理论值存在稳定的 page/token 偏差，可显式设置
`GSP_SYSTEM_LEN_EXTRA_TOKENS` 在基础 page 对齐长度上追加共享前缀 token，默认 `0`，不自动修改目标；
校准值应使用 page 大小的整数倍。

默认使用 `2` 个 group，并根据目标命中率在每组 `16/32/64/96/128/256` 个 prompt 中自动选择，
以减少请求数并覆盖高命中率场景；默认 `TOKEN_UNIT=1024`，
需要十进制 k 时显式设置 `TOKEN_UNIT=1000`。输入和输出支持小数，例如 `0.5k`。
默认使用官方 `--gsp-input-ids` 直接发送 token ID，避免随机 token 经文本解码和重新编码后产生输入长度漂移；
如需兼容不支持该参数的旧版 SGLang，可设置 `GSP_INPUT_IDS=0`，但实际输入长度和命中率可能发生偏差。

默认开启自动命中率校准：每个 `inf` 候选直接使用官方完整 workload 校验命中率；如果实测命中率不满足，
脚本按 `目标命中率 / 实测命中率` 计算方向，并默认只走 50% 的修正步长再按 page size 对齐，最多再执行 1 次完整校正，
避免一次校正跨过目标后又触发多轮反向完整压测。
默认 `inf` 容差为 0.5 个百分点；如需放宽 page 对齐和冷 miss 的正常波动，可显式设置
`GSP_BASELINE_HIT_RATE_TOLERANCE_PCT=1.0`。命中率确认后，
GSP 会重复执行 `inf`，以客户端 `input_throughput` 最高值作为基线；默认最多 5 次，连续 2 次不再提升就进入有限请求率回测。
仍不达标时自动切换 `prompts/group` 和 `groups` 候选布局。校准 trial 会保留在
`gsp_prefix_trials.csv`，用于保留命中率修正前后的完整试探过程；确认参数后，最终的 `inf` 和有限请求率结果会单独整理到
`gsp_prefix_summary.tsv`。布局切换只改变请求组织方式，
不改变单请求输入总 token 数和输出长度。

每个自动 case 都会优先从较小的 `prompts/group` 开始；首轮 `inf` probe 还会受
`GSP_PILOT_INPUT_TOKEN_BUDGET` 限制，默认总输入 token 不超过约 1M，避免任意输入长度直接
使用过大的请求数开始。首轮得到该 case 的实际 RPS 后，脚本先做一个短时长 inf，再根据
实测 duration/RPS 逐步调整到 `GSP_TARGET_DURATION_S` 的容差范围；显式固定
`GSP_NUM_GROUPS` 或关闭 `GSP_AUTO_DURATION` 时不做这层自动缩放。

`--case` 的基本格式为 `INPUT,OUTPUT,HIT_RATE,TTFT_METRIC,TTFT_TARGETS_S`；TPOT 条件可选，
需要约束 TPOT 时再追加 `TPOT_METRIC,TPOT_TARGET_MS`，例如：

```bash
--case '16k,1,80,mean_ttft_ms,5'
--case '32k,1,90,mean_ttft_ms,5,mean_tpot_ms,100'
```

`HIT_RATE=0` 表示使用官方 `random-ids` workload 生成独立 token 输入，不设置共享 system prefix，
用于真正的无 prefix-cache 对照；结果仍保留 `cache-report`。多个 case 可以在同一次运行中连续测试，
每个 case 独立执行 `flush-cache`、命中率校验和请求率回测，例如：

```bash
bash sglang/gsp_prefix_bench.sh \
  --case '8k,1,0,mean_ttft_ms,3|5' \
  --case '50k,1,90,mean_ttft_ms,3|5'
```

#### 811 节点官方 GSP 验证记录

`m09r3n06` 上使用 Qwen3-8B、TP4、官方 `generated-shared-prefix`，默认使用
`2` 个 group，并按目标命中率自动尝试每组 `16 -> 32 -> 64 -> 96 -> 128 -> 256` 个 prompt；不设置
`max-concurrency`，使用 `--gsp-input-ids --gsp-ordered --flush-cache --warmup-requests 0 --cache-report`。
每个 case 的 `inf` baseline 必须先满足目标命中率，之后才从该 baseline 的实际 RPS
向下回测；命中率始终按官方 `cache-report` 实测 token 比例判定。

旧版固定 `32x16` 的结果不能作为新版 `inf` 硬门槛结果：此前 `inf` 命中率约为
`84%`。在 H20 无限请求率下，每组实测约有四个并发请求未命中，因此脚本按
`GSP_CACHE_MISS_REQUESTS_PER_GROUP=4` 校准共享前缀长度，并将 `warmup-requests` 设为 `0`，
避免 warmup 提前写入 KV cache；例如 8K/90% 会优先尝试 `96` prompts/group。新版脚本会重新验证每个 case；如果所有
prompt/group 候选的 `inf` 命中率都不达标，summary 必须标记为 `BASELINE_HIT_RATE_MISMATCH`，不再
用低速结果标记 `MEET`。

本次 `m09r3n06` TP4 实测：8K/1、目标 90% 使用 `2x32`，实测命中率 `90.1%`、`inf`
实际 RPS `34.75`、Mean TTFT `1411 ms`；32K/1、目标 90% 使用 `2x32` 和
`31424/1344`，实测命中率 `89.9%`、`inf` 实际 RPS `6.26`、Mean TTFT `7437.55 ms`。
两组均未设置 `max-concurrency`，结果目录和完整日志保存在节点容器的
`/all_result/bench_gsp_prefix_20261019_inf_calibrated_8k90/` 和
`/all_result/bench_gsp_prefix_20261019_inf_calibrated_32k90_floor/`。

动态时长实测（`GSP_TARGET_DURATION_S=60`）：8K/1 自动扩展为 `66x32=2112` 请求，
`inf` 耗时 `61.78s`、实际 RPS `34.18`、命中率 `90.1%`；32K/1 自动扩展为
`12x32=384` 请求，`inf` 耗时 `61.28s`、实际 RPS `6.27`、命中率 `89.9%`。
同请求数回测耗时分别为 `63.78s` 和 `60.66s`，说明请求数会随 case 的 pilot RPS
自动变化，而不会固定为 64。

多请求率回测也已在同一轮验证：默认从动态 `inf` 实际 RPS 开始，按
`GSP_RATE_STEP` 均匀从高到低逐档测试，默认步长为 `0.1 req/s`。例如基线为
`1.55 req/s` 时，配置请求率依次为 `1.55、1.45、1.35、1.25、1.15、1.05、
0.95、0.85...`，不会再因累计偏移跳过中间档位。其中 `inf` 只作为命中率和性能
基线，不作为最终容量点。
在固定 GSP 分组、共享前缀长度和 `--flush-cache` 口径下，请求率通常不会大幅改变
token-level cache hit rate，但无 `max-concurrency` 时并发冷 miss 数可能随请求率变化。
因此脚本对 `inf` baseline 默认使用 `0.5` 个百分点容差，对有限请求率默认使用
`1.5` 个百分点容差，并且 CSV 仍完整记录每个点的实测命中率和偏差；超过有限速率容差
仍判定为 `NO_MEET`，不会用理论值替代实测值。对应结果目录为
`/all_result/bench_gsp_prefix_20261019_dynamic_duration_60/`。

`GSP_SYSTEM_LEN_EXTRA_TOKENS` 仍可用于已有服务的实测校准，但默认不追加额外 token；这不是
替代实测命中率。`inf` 默认使用 `0.5` 个百分点容差，有限请求率默认使用 `1.5` 个百分点
容差，并可分别用 `GSP_BASELINE_HIT_RATE_TOLERANCE_PCT` 和
`GSP_FINITE_HIT_RATE_TOLERANCE_PCT` 调整。

一个命令可以连续测试多个输入/输出、命中率和 SLA：

```bash
BASE_URL=http://127.0.0.1:30000 \
MODEL=/path/to/model \
bash sglang/gsp_prefix_bench.sh \
  --case '16k,1k,80,mean_ttft_ms,3|5,mean_tpot_ms,30' \
  --case '32k,0.5k,80,mean_ttft_ms,5|7,mean_tpot_ms,50'
```

case 格式为：

```text
INPUT,OUTPUT,HIT_RATE,TTFT_METRIC,TTFT_TARGETS_S,TPOT_METRIC,TPOT_TARGET_MS[,MAX_CONCURRENCY]
```

例如 `3|5` 表示分别查找 TTFT 小于 3 秒和 5 秒时的最大实际客户端 RPS；TPOT
限制为 30ms。每个 case 先按 `GSP_PROMPTS_PER_GROUP_CANDIDATES` 从小到大计算可行的共享前缀，
使用 `GSP_NUM_GROUPS_CANDIDATES` 组合测试 `inf`，选择首个实际命中率达标的组合；显式设置
`GSP_PROMPTS_PER_GROUP` 或 `GSP_NUM_GROUPS` 时只使用固定值。然后以该次 `inf` 的实际
`request_throughput` 为起点从高到低按 `0.1 req/s`（可由 `GSP_RATE_STEP` 调整）逐档回测，
记录实际请求率，保留两位小数。只有 baseline 和候选点都满足实测命中率、TTFT、TPOT，才会进入容量结果。
默认先用小请求数得到 pilot RPS，再按 `GSP_TARGET_DURATION_S` 动态扩展当前 case 的
`gsp_num_groups/num_prompts`；默认目标为 60 秒，最终每个 case 的 benchmark duration
通常落在 1～2 分钟。扩展时保持 prompts/group、system/question 长度和命中率口径不变。

只验证一组输入/输出的多请求率回测可以直接使用：

```bash
BASE_URL=http://127.0.0.1:30000 \
MODEL=/path/to/model \
GSP_TARGET_DURATION_S=60 \
RUN_ID=gsp_rate_sweep_demo \
bash sglang/gsp_prefix_bench.sh \
  --case '8k,1,90,mean_ttft_ms,3' \
  --case '32k,1,90,mean_ttft_ms,5'
```

不设置 `GSP_REQUEST_RATES` 时使用默认降速序列；如果只想指定有限请求率，设置
`GSP_REQUEST_RATES=34.18,33.98,33.48`，脚本仍会先执行 `inf` 命中率基线，再执行这些
有限请求率。每个有限点都会写入 `gsp_prefix_trials.csv`，其中
`configured_request_rate` 是客户端配置值，`request_throughput` 是实际客户端输出 RPS；
容量筛选和 summary 排序使用后者。

默认生成 CSV、日志和容量矩阵图；格子中央默认显示实际客户端 RPS。可以指定格子中央显示的指标：

```bash
# 格子显示最大实际客户端 RPS
bash sglang/gsp_prefix_bench.sh --plot --cell rps \
  --case '16k,1k,80,mean_ttft_ms,3|5,mean_tpot_ms,30' \
  --case '32k,0.5k,80,mean_ttft_ms,5|7,mean_tpot_ms,50'

# 格子显示客户端输入 token throughput
bash sglang/gsp_prefix_bench.sh --plot --cell input_throughput \
  --case '16k,1k,80,mean_ttft_ms,3|5,mean_tpot_ms,30'
```

如只需要 TSV/CSV 和日志，可关闭默认作图：

```bash
GSP_PLOT=0 bash sglang/gsp_prefix_bench.sh --no-plot \
  --case '16k,1k,80,mean_ttft_ms,3|5,mean_tpot_ms,30'
```

结果位于 `all_result/bench_gsp_prefix_<RUN_ID>/`：

- `run_metadata.json`：本次运行的模型、服务地址、节点 hostname、启动时间、脚本 commit 和关键环境参数。节点 hostname 默认由 `hostname` 获取，也可通过 `NODE_HOSTNAME` 显式覆盖。
- `logs/`：每个 case/请求率的官方 benchmark 日志和 `.exitcode`。
- 每个新生成的 `.log` 开头也会写入上述模型、节点 hostname 和服务地址，便于单独复制日志后追溯来源。
- `gsp_prefix_trials.csv`：只保留正式测试前的参数试探记录，包括未确认布局和命中率校准记录；保留指定/理论/page 对齐理论/实际 cache 命中率、候选长度、groups/prompts、请求数、duration、TTFT、TPOT、ITL、Mean E2E 和日志路径。最终确认参数的正式 `inf` 与有限请求率回测会从该文件移出并进入 summary。
- `gsp_prefix_summary.tsv`：只保留命中率修正后最终确认参数对应的完整明细；每个确认参数块包含 1 行 `inf` 和所有有限请求率回测行，使用 `request_rate=inf` 区分基线，并保留确认后的长度/分组/请求数、请求率、吞吐、实际 cache 命中率和完整客户端指标。`SLO` 统一记录 TTFT/TPOT 要求，TTFT 单位为秒、TPOT 单位为毫秒，例如 `{"ttft":{"mean":[5,6]},"tpot":{"mean":[20]}}`；`page_aligned`、`ttft_metric`、`ttft_targets_s`、`tpot_metric`、`tpot_target_ms` 等过程字段不放入 summary。summary 内部仍按确认后的 INF 实际 RPS 降序排列，同一参数块内再按请求率从高到低排列，`inf` 位于块首；总 token 统计列不再输出。
- 只有 `gsp_prefix_summary.tsv` 作为可复制汇总文件；它会在每个 trial 完成后刷新一次，收到中断信号时也会保存当前已完成结果。中断时不会执行 trial 清理，因此可以结合 `gsp_prefix_trials.csv` 和日志继续排查。
- GSP CSV/TSV 会记录 `parallel_config`，优先从服务 `/server_info` 生成；接口不可用时可通过 `PARALLEL_CONFIG` 手动指定。
- `gsp_prefix_summary.tsv`：唯一的 summary 汇总文件，列之间使用制表符，空值用 `-` 占位以避免终端复制时连续 Tab 导致列错位。执行 `cat gsp_prefix_summary.tsv` 后复制到 Excel/WPS，可直接按列粘贴。
- `gsp_prefix_matrix.png`：默认生成的容量矩阵，列为输入/输出/命中率，行为 TTFT SLA，格子中央为 RPS 或输入吞吐；使用 `GSP_PLOT=0` 或 `--no-plot` 关闭。
- 作图只调用现有 Python 环境中的 `matplotlib/numpy`，脚本不会自动下载依赖；绘图库缺失时只打印 warning，CSV/TSV、日志和 benchmark 退出状态不受影响。

Prefill/GSP 作图规则：每一列代表一个输入/输出/指定命中率 workload，每一行代表一个 TTFT SLA
目标。只使用命中率满足容差、TTFT/TPOT 满足 SLO 的有限请求率 trial；`inf` 只作为确认后的
基线，不直接作为容量格子。每个格子选择实际客户端 RPS 最大的 trial，格子中央默认显示 RPS，
也可以用 `--cell input_throughput` 显示客户端输入 token throughput。没有满足条件的格子显示
`NO MEET`。因此它是“约束下最大容量矩阵”，不是把所有原始 trial 直接画成热力图。

Decode 也可以采用同一套矩阵方式：把行从 TTFT SLA 换成 TPOT/ITL SLA，把列换成输入/输出、
BS 或并发 workload，格子选择满足 SLA 的最大实际 RPS 或输出 token throughput。当前
`sglc.sh` 仍以 `sum_all.csv` 为主，尚未默认接入 decode 矩阵图；GSP 拆分后的
`gsp_prefix_plot.py` 已将作图逻辑独立出来，后续可复用同样的筛选规则接入 SGLC，而不必再把
绘图代码塞回压测主流程。

脚本支持 `--dry-run` 核对官方命令而不连接服务；同一 `RUN_ID` 重新执行时会跳过已有成功
日志。缺少 `--cache-report` 或 benchmark 失败时保留日志并标记为失败，不用理论值伪造实测命中率。

### SGLang: `sglang/prof_capture.sh` + `prof_analyze.py` + `prof_dashboard.py`

在线抓取服务端 torch profiler（`/start_profile`），并在抓完后直接产出可读摘要与 HTML 可视化看板。

- 支持 `--mode prefill|decode`：prefill 默认 4096 in / 1 out / 5 步，decode 默认 4096 in / 1024 out / 10 步。
- `--start-step` 对应 `/start_profile` 的 `start_step`，是服务端的**绝对** forward 计数
  （服务端取 `max(start_step, forward_ct+1)`），在已经有流量的服务上并不等于“再等 5 步”。
- “等稳定了再抓”由**负载就绪门控**实现：脚本轮询 `/v1/loads` 的 `num_running_reqs`
  （不可用时回退 `/metrics` 的 `sglang:num_running_reqs`），相对起压前基线增长到
  `--load-need`（decode 默认等于 `--bs`，prefill 默认 4）且连续 3 次采样（约 6s）保持后，
  再等 `--wait-sec`（prefill 默认 8s、decode 默认 20s）才发 `/start_profile`。
  驱动启动要先 import sglang，高负载节点上约 40s，固定秒数盲等会抓到空闲步
  或同一服务上其他人的流量；等待上限由 `--ready-timeout`（默认 240s）控制，超时直接报错而不是产出空 trace。
- 内置负载驱动按模式区分：
  - **decode**：`python3 -m sglang.bench_serving`（与 `sglc.sh` 完全一致，`--max-concurrency BS`，
    `random-ids` 冷输入）。decode 没有 prefix cache 命中率概念，不需要 `prefix_hit.py` 的共享前缀逻辑。
  - **prefill**：`sla_bench/prefix_hit.py`，用 `PREFIX_HIT` 控制共享前缀比例（0=冷）。
  驱动预检按模式查具体子模块（decode 查 `sglang.bench_serving`，prefill 查 `sglang.benchmark`），
  避免“装了顶层 `sglang` 但缺 benchmark”时漏过。
  decode 可加 `--fake-prefill` 透传给 `bench_serving`：客户端会往请求体注入
  `bootstrap_host`/`bootstrap_room`，让带 `--disaggregation-transfer-backend fake` 的
  PD decode 实例跳过真实 prefill KV 传输，只压纯 decode。普通单机服务没有该后端，传了不会生效
  （请求仍走真实 prefill），prefill 模式下显式报错。
  已有外部流量时用 `--no-driver` 只负责抓取（同样会等确认到有请求在跑）。
  `--flush-cache` 在每次驱动启动前 `POST /flush_cache` 清空 radix/prefix cache：
  SEED 固定时第二轮驱动会命中第一轮的 prompt 前缀 cache，吞吐虚高且不可复现；
  性能基准轮和抓取轮各清一次。
  `--perf-first` 在抓取前先用同样负载完整跑一遍作为客户端性能基准（输出到
  `<local>/perf_pass/driver.jsonl`，脚本直接打印 Mean TPOT/吞吐等指标），
  再清 cache 进入抓取轮，用于把 prof 步长时间与客户端实测对账：
  `客户端 Mean TPOT ≈ prof 每步耗时 / accept_length`。
  `--no-driver` 时 `--perf-first` 没有内置负载可跑，跳过基准轮并告警。
- `--output-dir` 是服务端可见路径；当共享存储在客户端映射成另一个路径时，用
  `--local-trace-dir` 指向客户端视图，用于轮询和离线分析。
- 轮询 `*.trace.json.gz` 直到 tp_size 个 rank 全部落盘；服务端先建 0 字节文件、抓完才写入，
  所以按 `-size +0c` 计数，再用“大小连续两次不变 + `gzip -t` 全部通过”判定写完，避免读到半截文件。
- `num_steps` 到点只是**停止采集**，写盘（8 个几十 MB 的 gzip）可能比采集本身慢一个量级：
  nmz35 上 decode 采 10 步只用几十秒，8×50MB 的文件却在 `start_profile` 之后约 11 分钟才写齐。
  因此 `--capture-timeout` 默认 1800s，等待期间每 30s 打印一次“有内容 N 个 / 空文件 N 个”的进度；
  驱动负载提前结束时也不会立刻放弃，而是继续等服务端把文件写完。
- 抓完后立即结束驱动负载；已经发到服务端的 decode 请求不会被撤销，`num_running_reqs`
  需要一段时间才回落到 0，共享实例上连续抓多次时注意留出回落时间。
- `prof_analyze.py` 只依赖标准库，输出 `prof_summary.md`、`prof_kernels.csv`、`prof_ranks.csv`：
  每个 rank 的 GPU busy 占比、kernel 累计时长 TopN、GPU memcpy/memset、CPU 顶层 op 独占时长 TopN；
  另输出调度/通信两份扩展 CSV：`prof_sched.csv`（每 rank GPU 空档分桶（<10µs / 10~100µs / 0.1~1ms / ≥1ms
  的次数与合计 ms）、通信聚合（deep_ep dispatch / combine / notify 子类与 allreduce/allgather 等集合通信，
  占比按 span 计）、`head_gap_ms` / `tail_gap_ms` / `span_active_ms`（去窗口边界空档的有效窗口）、
  `comm_exposed_ms` / `comm_hidden_ms`（通信暴露 = 通信区间未与算力/memcpy 重叠的时长，串行流≈100%）、
  `bcc_n` / `bcc_ms`（内部距通信 ≤100µs 的空档））、`prof_gaps_top.csv`（每 rank Top10 大空档：
  距窗口起点偏移 rel_ms、时长、归因 band（B=窗口边界、C=紧邻通信、G=其他）、距通信 µs）。
- **口径说明（重要）**：`GPU busy` 含通信/memcpy、按整窗 span 计，是对账/回归基线，勿与其它口径混用；
  每步 TPOT/吞吐改用 `span_active`（span − 窗口首尾空档，剔除 profiler 起停与上一批调度残留）。
  旧版 `comm_overlap_pct`（空档∩通信）恒为 0 属结构性错误——busy 已含通信，空档是 busy 补集，
  与通信区间永不相交；现由 `comm_exposed`（通信暴露）取代。band=B 的空档不计入每步耗时。
- 分析器按 4 MB 解压文本一批做整体反序列化、解析期间关闭 GC、并对多个 rank 文件多进程并行
  （`--jobs`，默认 min(8, 文件数, 核数)）。实测 nmz35 的 decode 数据 8 个 rank × 69 MB gzip
  （每 rank 37.5 万条事件）`--jobs 8` 总耗时 12.6s；另用 8 个 rank × 69 MB 合成 trace
  （每 rank 1167 万条事件）与逐条 `json.raw_decode` 严格解析对比，span / GPU busy / kernel
  累计时长逐项一致，并行耗时 47s（单文件串行 44s）。
- CPU op 事件数量随 torch.compile 增长，默认每文件只分析前 50 万条并在摘要中标注截断，
  需要完整结果时加 `--cpu-ops-limit 0`。
- **每次 prof 自动产出 HTML 可视化看板**：`prof_capture.sh` 在分析步自动调用 `prof_dashboard.py`
  生成 `<dir>/prof_dashboard.html`（自包含单文件，无第三方依赖，失败只告警不影响摘要）。
  面板含：顶部 chips 自动显示并行方式与运行配置——并行维度从 trace 文件名 `*-TP-x-DP-y-EP-z` 解析
  （如 decode 为 TP8/DP8/EP8），MTP/speculative 算法及步数、CUDA Graph 开关、tbo、overlap 调度
  从目录内 `server_info.json`（prof_capture 抓取后自动保存的服务端参数快照）读取，accept 率取
  `--accept` 或 driver 实测；旧目录缺 server_info.json 时显示「配置未知」，端到端对账
  （prof wall/busy 每 step vs 客户端 `TPOT×accept`（decode）或 `TTFT`（prefill），
  经验签名 +12~17%/+3~8% 判 ✓/△）、kernel 类目构成环图、Top10 kernel、各 rank 对称性
  （busy 差距 >1.5pp 标 ⚠）、空泡 (GPU idle) 诊断与改进建议（判级 <5% ✓ / 5~15% △ / >15% ⚠，
  按 Perfetto 空档形态给出归因与改进：launch gap→开图、通信等待→tbo/负载均衡、memcpy→显存管理、
  MTP 切换→spec 联调、窗口边界空档已自动扣除；并注明 CUPTI/禁图抓取对空泡的本底放大；
  有 prof_sched.csv 时追加「本窗口实测 → 空泡改进优先级」数据驱动建议：窗口边界空档（band=B，
  采集起停+调度残留）自动扣除并重判有效空泡等级（如整窗 8.7% △ → 去边界 0.1% ✓）、
  按有效空档形态给优先级 1 抓手（launch gap 主导→开图/torch.compile/绑核，
  内部紧邻通信空档主导→tbo/EPLB/慢 rank 单查）、通信暴露（未与算力重叠，串行流≈100%）
  给出 tbo 可覆盖上限、deep_ep 占比 ≥5% 提示通算 overlap 收益、rank 空档不均时提示
  「busy 最低 + 内部空档最多」才是慢侧、memcpy/memset 超 3% 提示显存管理）、
  ⑦ 调度与通信（每 rank 空档分桶表：边界空档 ms / 有效空档率 / 通信暴露 ms 与占比 + 形态判读；
  deep_ep 子类表 + 通信/算力比（算力=busy−comm）+ 暴露/隐藏拆分；全 rank Top 大空档表给出
  rel_ms、band 与距通信 µs 供 Perfetto 平移定位。注意通信 kernel 时长含「早到者」自旋等待，
  comm 高不一定是慢 rank。旧产物缺这两个 CSV 时面板给重跑提示，对已有 trace 重跑
  `prof_analyze.py` 即可补齐，无需重抓）
  与 Perfetto 使用提示。对账数据自动读取 `<dir>/driver.jsonl`
  或 `--perf-first` 的 `<dir>/perf_pass/driver.jsonl`。结果目录无写权限时回退写到当前目录
  （容器内 root 创建的目录常见）。也可手动单独生成：

```bash
python3 sglang/prof_dashboard.py /path/to/profiles --mode decode \
  --num-steps 10 --bs 64 --case '4k,1024,0'
```

- 只用核心参数时：

```bash
bash sglang/prof_capture.sh --mode decode  --case '4k,1024,0' --bs 32
bash sglang/prof_capture.sh --mode prefill --case '4096,1,0'
```

- 服务端目录与客户端不一致时：

```bash
bash sglang/prof_capture.sh --mode prefill --case '4k,1,0' \
  --output-dir /mnt11/prof_prefill --local-trace-dir /public/home/guobj/prof_prefill
```

- 只分析已有 trace（不重新抓取）：

```bash
python3 sglang/prof_analyze.py /path/to/profiles --mode decode --top 20
```

- trace 目录由服务端（容器内 root）创建时客户端可能没有写权限，此时用 `--out-dir` 把
  `prof_summary.md` / `prof_kernels.csv` / `prof_ranks.csv` / `prof_sched.csv` / `prof_gaps_top.csv`
  写到可写目录；分析本身只需读权限。

#### nmz35 验证记录（GLM-5.3 TP8/EP8/CP8，`--disable-cuda-graph`）

- prefill `--case '4096,1,0' --num-steps 5`（端到端）：8 个 rank trace 全部落盘，
  每个 191612 条事件、26120 个 kernel；单 rank 时间跨度 10.15s、GPU busy 93.9%～97.1%
  （合计 78.48s / 81.18s，占比 96.7%，空档 2.69s），并行分析 6.6s。Top kernel 为
  `sparse_attn_fwd_kernel`（20.8%）、`deep_ep::intranode::cached_notify_combine_kernel`（16.6%）、
  `m_grouped_fp8_gemm_nt_contiguous_wasp_pair32`（9.6%）。该次起压前基线为 0，
  就绪判定等到 56 并发稳定 3 次采样（约 41s，覆盖驱动 `import sglang` 时间）后才发 `start_profile`。
- decode `--case '4096,1024,0' --bs 32 --num-steps 10`（端到端 `RC=0`）：8 个 rank 全部落盘，
  每个约 59MB gzip、326080 条事件、45350 个 kernel；单 rank 时间跨度 3.94s、GPU busy
  82.8%～84.0%（合计 26.33s / 31.54s，占比 83.5%，空档 5.21s），并行分析 10.2s。Top kernel 为
  `m_grouped_fp8_gemm_nt_contiguous_wasp_pair32`（29.4%）、
  `flash_fwd_headshare8_ropewait_mla_fp8_sparse_kernel`（28.0%）、
  `deep_ep::intranode::combine`（4.7%）、`cached_notify_combine_kernel`（3.2%）。
  该次抓取前 `num_running_reqs` 基线为 6，就绪判定等到 34 并发稳定 3 次采样后才发 `start_profile`；
  抓完驱动负载被正确回收（无残留 `bench_serving` 进程）。
- 若抓取窗口里混入了 prefill 尾部队列（例如上一轮请求还没排空），单 rank 时间跨度会从 4s 涨到
  12s 量级、事件数升到 374697，busy 也会被抬到 95% 左右——这类数据仍可分析，但不再是纯 decode 稳态。
- 同一份 8×70MB 数据在改动前（逐事件 `buf = buf[end:]` 切片）629s 仍未解析完第一个文件，
  现在按 4MB 分批 + 多进程并行后 10～13s 完成。

#### decode 窗口与客户端 TPOT 的口径差异

profile 抓到的是**纯 decode 窗口**：BS32 × 输出 1024 时，队列里其余请求要等前 32 条完成才进入，
`--wait-sec 20` 落在“32 条请求全部 prefill 完成、无新 prefill 插入”的区间
（可用 trace 里 `sparse_attn_fwd_kernel` 计数为 0 佐证）。此时实测单步 394ms
（GPU busy 329ms + 空档 65ms）、按 accept length 3.57 折合每 token 约 110ms。

而客户端长期平均的 TPOT 包含**持续 prefill 抢占**：同参数 512 prompts 实测
Output 138.3 tok/s、Concurrency 31.4、Accept 3.57、Mean TPOT 216.7ms，
反推单步 770～810ms，约为 prof 窗口的 2 倍。两者差的是 prefill 混入，不是分析误差。

要让 profile 代表客户端实际观测到的稳态，把抓取点后移到请求开始周转之后
（例如 `--case 4096,1024,0 --bs 32 --wait-sec 115`），或用 `--no-driver` 直接对
`sglc.sh` 的压测流量取样。反之，想单独看 decode 算子开销就保持默认的纯 decode 窗口。

#### 与郑州 `超节点/prefill_zz/4k` 的一致性

同一模型（GLM-5.3-Channel-FP8-w8a8）、同一 8 卡配置下，两边结果一致：
`sparse_attn_fwd_kernel` 单次调用 6.768ms vs 6.806ms（差 0.6%）、份额同为 20.8%；
按 79 层折算每 forward 2.54s vs 2.47s（差 2.6%）；`m_grouped_fp8_gemm`（9.6%+7.6% vs 16.6%）、
`cached_notify_combine`（16.6% vs 15.2%）、`combine`（7.4% vs 7.4%）、`dispatch`（4.2% vs 4.2%）
份额差均在 1.5 个百分点内。差异有两处：本仓库这次抓到 4 个 forward（kernel 数 316/79）而 zz 那份是 5 个，
以及 GPU 空档 2.9% vs 0.70%（抓取时 nmz35 节点负载较高）。

### vLLM: `vllm/vllmc.sh`

- 使用 Prefix-Salt no-flush 模式；每个 case / BS 使用唯一 `cache_salt`，隔离不同 BS 与不同 run 的 Prefix Cache。
- 通过 `/server_info`、`SERVER_LOG`、`/metrics` 和环境变量提取服务配置与容量信息。
- CSV 前置常用指标与 SGLang 脚本口径对齐。
- 理论 BS = `min(token theory, max_num_seqs)`；如果无法获取 `max_num_seqs`，退回 token theory。
- 支持与 SGLang 相同的 `SLA_RULES` inline 加密扫描。
- 默认测试 `1/8/16/32/64/128/1024K` 输入 case；可用 `VLLM_CASES="8 1,32 1"` 选择指定输入/输出，避免验证时误跑大上下文。
- `MODEL` 使用服务端的 served model name；如果客户端容器不能联网，另设 `TOKENIZER=/本地模型目录`，脚本会将本地 tokenizer 传给 `vllm bench serve`。
- benchmark 只要成功请求数为 0 或存在失败请求，就标记为失败，不会把 vLLM 客户端的 0 退出码当成有效性能结果。
- 直接从 vLLM benchmark 日志提取 RPS、Token 吞吐、请求数、耗时、平均时延和分位数，不保存结果 JSON。

#### vLLM 单侧 Prefill：`vllm/vllm_prefill_bench.sh`

Prefill 脚本使用官方 `vllm bench serve` 的 `random` 数据集，输出长度固定为 `1`，用于测量输入侧
TTFT 容量。`--case` 格式为 `输入,TTFT指标,目标秒数`，同一 case 可以用 `|` 指定多个目标：

```bash
BASE_URL=http://127.0.0.1:8000 \
MODEL=/data/models/Qwen3.8-27B-Channel-INT8-w8a8 \
PREFILL_TARGET_DURATION_S=60 \
bash vllm/vllm_prefill_bench.sh \
  --case '4k,mean_ttft_ms,3|4' \
  --case '64k,p99_ttft_ms,5|6'
```

默认 `TOKEN_UNIT=1024`，输入支持 `0.5k` 等小数；需要十进制 k 时设置
`TOKEN_UNIT=1000`。脚本默认不设置客户端 `--max-concurrency`，避免人为限制 Prefill 吞吐；
确实需要固定并发时才设置 `FIXED_CONCURRENCY=64` 等环境变量。默认流程如下：

1. 先用 `request_rate=inf` 做短时 pilot，按实际耗时调整 `num_prompts` 到目标时长附近。
2. 重复执行 inf baseline，直到输入吞吐不再提升或达到最大 trial 次数，选输入吞吐最高的一次。
3. 以该 baseline 的实际 RPS 为起点，按 `0.1 req/s` 的最小步长从高到低回测，直到每个 TTFT 目标找到满足点。
4. 每个 trial 立即写入 `vllm_prefill_trials.csv`，并刷新 `vllm_prefill_summary.tsv`；退出或中断后已完成的日志仍可续查。

常用参数：

- `PREFILL_TARGET_DURATION_S`：每组请求目标时长，默认 `60` 秒。
- `PREFILL_RATE_STEP`：有限请求率回测步长，默认 `0.1`，不能小于 `0.1`。
- `PREFILL_NUM_PROMPTS`：显式固定请求数；不设置时由 pilot 根据输入长度和实际耗时自动调整。
- `PREFILL_INF_MAX_TRIALS` / `PREFILL_INF_NO_IMPROVEMENT_LIMIT`：inf 稳定性回测上限，默认 `5` / `2`。
- `PREFILL_PLOT=0` 或 `--no-plot`：关闭默认图片；`--cell input_throughput` 可让格子显示输入 token throughput，默认显示实际客户端 RPS。

结果位于 `all_result/bench_vllm_prefill_<RUN_ID>/`，包含：

- `logs/`：每个 case、inf trial 和有限请求率的原始 vLLM benchmark 日志及退出码。
- `vllm_prefill_trials.csv`：所有已完成的 inf、pilot 和降速回测过程记录。
- `vllm_prefill_summary.tsv`：按确认参数整理的最终结果；inf 基线位于同一 case 的首行，有限请求率按实际 RPS 从高到低排列，指标顺序与 `vllmc.sh` 对齐。
- `vllm_prefill_matrix.png`：默认生成的 TTFT 容量矩阵；每格选择满足对应 TTFT SLO 的最大实际 RPS。绘图依赖缺失时只跳过图片，不影响 TSV/日志。

只核对命令、不连接服务时可以使用 `--dry-run`：

```bash
BASE_URL=http://127.0.0.1:8000 MODEL=/path/to/model \
bash vllm/vllm_prefill_bench.sh --dry-run --no-plot \
  --case '4k,mean_ttft_ms,3|4'
```

## 长度口径

脚本中的 `CASES` 默认使用二进制 K，即 `TOKEN_UNIT=1024`。例如 `"4 1"` 表示输入
`4096 tokens`、输出 `1024 tokens`；`"0.5 1"` 表示输入 `512 tokens`、输出 `1024 tokens`。
统一 GSP 入口的 `--case` 支持独立 Prefill（`HIT_RATE=0`，输出长度通常设为 `1`）和带共享前缀的
GSP 测试。需要十进制 k 时设置 `TOKEN_UNIT=1000`，此时 `"1k"` 表示 `1000 tokens`。
小数换算后会四舍五入为整数 token；结果直接根据 `input_len/output_len` 判断实际 token 口径。

## 快速运行

### SGLang

```bash
cd perf_bench
BASE_URL=http://127.0.0.1:30000 \
MODEL=/path/to/model \
bash sglang/sglc.sh
```

### vLLM

```bash
cd perf_bench
BASE_URL=http://127.0.0.1:8000 \
MODEL=/path/to/model \
bash vllm/vllmc.sh
```

单侧 Prefill：

```bash
BASE_URL=http://127.0.0.1:8000 \
MODEL=/path/to/model \
bash vllm/vllm_prefill_bench.sh --case '4k,mean_ttft_ms,3|4'
```

SGLang 的 `sglc.sh` 和 vLLM 的 `vllmc.sh` 也会在各自结果目录生成
`run_metadata.json`，并给每个新 benchmark 日志追加同样的元数据头。这样拿到单个日志时，
仍可以直接确认具体模型、服务地址、节点 hostname、运行时间和脚本版本；容器内 hostname
不正确时可用 `NODE_HOSTNAME` 显式填写宿主机节点。

## 常用环境变量

| 变量 | 说明 |
| --- | --- |
| `BASE_URL` | OpenAI 兼容服务地址。 |
| `MODEL` | benchmark 请求中使用的模型名或模型路径。 |
| `TOKENIZER` | vLLM benchmark 客户端使用的本地 tokenizer 路径；服务名与本地目录不一致或网络受限时必须设置。 |
| `VLLM_CASES` | vLLM 输入/输出 case 覆盖，格式为 `"8 1,32 1"`；不设置时使用脚本默认 case 列表。 |
| `NODE_HOSTNAME` | 覆盖结果元数据中的节点 hostname，适合容器内 hostname 未映射到宿主机节点时使用。 |
| `FIXED_PER_DP_BS` | 固定 per-DP BS 测试点，支持逗号或空格分隔多个值。 |
| `FIXED_CONCURRENCY` | 固定全局并发测试点，必须能被 `DP_SIZE` 整除。 |
| `MAX_TEST_PER_DP_BS` | SGLang 自动理论 BS 扫描的 per-DP 上限；为空时按理论容量扫描，只有自动模式生效，固定 BS/并发模式不截断。 |
| `NUM_PROMPTS_MULTIPLIER` | 请求数量倍率，`num_prompts = max_concurrency * NUM_PROMPTS_MULTIPLIER`。 |
| `SLA_DENSE_ENABLE` | 是否开启 SLA inline 加密扫描，默认 `1`。 |
| `SLA_RULES` | SLA 规则，格式为 `metric:target:near`，多个规则用逗号分隔。 |
| `HARD_STOP_RULES` | SGLang 硬停止规则，格式为 `metric:op:threshold`，多个规则用逗号分隔；默认关闭。 |
| `SGLC_PLOT` | Decode 容量矩阵开关，默认 `1`；设为 `0` 只生成 CSV 和日志。 |
| `SGLC_PLOT_CELL` | Decode 矩阵格子指标，默认 `rps`；可设为 `output_throughput`。 |
| `SGLC_PLOT_BIN` | Decode 作图使用的 Python 命令，默认 `python3`。 |
| `RUN_ID` | 固定结果目录后缀，用于中断后续测；不设置时自动使用当前时间戳。 |
| `DP_SIZE` | 手动指定 DP 数；通常由脚本自动识别。 |
| `KV_TOKENS` / `MAX_TOTAL_TOKENS` | 容量识别失败时的手动兜底值。 |
| `MAX_NUM_SEQS` / `MAX_RUNNING_REQUESTS` | vLLM / SGLang 的请求数上限兜底值。 |
| `TOKEN_UNIT` | K 的 token 定义，默认 `1024`；设为 `1000` 使用十进制 K。 |
| `GSP_NUM_GROUPS_CANDIDATES` | `inf` 命中率校验的 groups 候选，默认 `2,4,8`；自动布局开启时，显式 `GSP_NUM_GROUPS` 也会作为首个候选。 |
| `GSP_PROMPTS_PER_GROUP_CANDIDATES` | 每组 prompt 候选，默认 `16,32,64,96,128,256`；低命中率优先尝试较小组合，高命中率优先尝试较大组合。 |
| `GSP_PROMPTS_PER_GROUP` | 首选每组 prompt 数；自动布局开启且校正失败时，会继续尝试候选组合。设置 `GSP_AUTO_LAYOUT=0` 才固定为该值。 |
| `GSP_CACHE_MISS_REQUESTS_PER_GROUP` | 初始长度计算时每组预估的并发未命中请求数，默认 `4`（按 H20 实测）；最终仍以官方 `cache-report` 实测命中率为准。 |
| `GSP_AUTO_CALIBRATE` | 是否在 `inf` 命中率偏差后自动修正 system/question 长度，默认 `1`。 |
| `GSP_CALIBRATION_MAX_ITERS` | 完整 workload 命中率仍偏离时，每个 GSP 布局最多追加的正式校正次数，默认 `1`。 |
| `GSP_CALIBRATION_STEP_RATIO` | 每次向实测比例修正值移动的步长比例，默认 `0.5`；用于降低 page/cold-miss 波动造成的过校正。 |
| `GSP_AUTO_LAYOUT` | 命中率校正仍不达标时是否自动尝试其他 groups/prompts 布局，默认 `1`；设为 `0` 保持显式布局固定。 |
| `GSP_AUTO_DURATION` | 是否按 pilot RPS 和实测 duration 动态调整每个 case 的请求数，默认 `1`；显式设置 `GSP_NUM_GROUPS` 时不调整 groups。 |
| `GSP_TARGET_DURATION_S` | 每个 case 的目标 benchmark 时长，默认 `60` 秒，允许 `60-120` 秒；按 prompts/group 向上取整请求数。 |
| `GSP_PILOT_INPUT_TOKEN_BUDGET` | 每个自动 case 首轮 `inf` probe 的总输入 token 预算，默认 `1048576`（约 1M）；仅在未显式固定 groups 且开启自动时长时生效。 |
| `GSP_DURATION_FIRST_PASS_S` | 自动时长调整的首轮短测试时长，默认 `30` 秒；之后根据实测 duration 调整到目标时长。 |
| `GSP_DURATION_ADJUST_MAX_ITERS` | 每个 case 的最大动态时长调整次数，默认 `2`。 |
| `GSP_DURATION_TOLERANCE_PCT` | 目标时长容差，默认 `25%`；落在目标上下限内后停止调整。 |
| `GSP_INPUT_IDS` | 是否使用官方 `--gsp-input-ids` 直接发送 token ID，默认 `1`；设为 `0` 关闭。 |
| `GSP_PAGE_SIZE` | GSP KV page 大小；真实执行默认从 `/server_info` 读取，失败回退 `64`。 |
| `GSP_PAGE_ALIGN` | 是否按 KV page 对齐共享前缀，默认 `1`；设为 `0` 保留未对齐理论长度。 |
| `GSP_SYSTEM_LEN_EXTRA_TOKENS` | 在基础 page 对齐长度上追加的共享前缀 token 数，默认 `0`；仅用于已由官方 `cache-report` 证实的稳定偏差校准，建议使用 page 大小的整数倍。 |
| `GSP_HIT_RATE_TOLERANCE_PCT` | page 对齐和兼容逻辑使用的命中率容差，默认 `0.5` 个百分点。 |
| `GSP_BASELINE_HIT_RATE_TOLERANCE_PCT` | `inf` baseline 的严格实测命中率容差，默认 `0.5` 个百分点；需要放宽时可显式设置为 `1.0`。 |
| `GSP_FINITE_HIT_RATE_TOLERANCE_PCT` | 有限请求率回测的命中率容差，默认 `1.5` 个百分点，用于覆盖无并发上限时的冷 miss 漂移。 |
| `GSP_INF_MAX_TRIALS` | 命中率确认后重复 `inf` 的最大次数，默认 `5`。 |
| `GSP_INF_NO_IMPROVEMENT_LIMIT` | `input_throughput` 连续不提升的停止次数，默认 `2`。 |
| `GSP_REQUEST_RATES` | 可选的逗号分隔请求率列表；不设置时按 `inf` 实测 RPS 从高到低回测。 |
| `GSP_RATE_STEP` | 默认有限请求率逐档下降的步长，默认 `0.1 req/s`，不能设置小于 `0.1`；显式请求率仍必须按高到低且满足该间隔。 |
| `GSP_STOP_ON_SLA` | 默认 `1`：每个 case 按高到低请求率测试，所有 `TTFT_TARGETS_S` 都找到满足点且 cache/TPOT 也通过后停止；设为 `0` 才跑完整降速序列。 |

`SLA_RULES` 示例：

```bash
SLA_RULES="mean_tpot_ms:50:5,p99_tpot_ms:75:5,mean_ttft_ms:1000:100"
```

含义是当某个 BS 的指标落入 `target ± near` 区间时，立即连续补测当前基础 BS 到下一个基础 BS 之间的整数 BS。

`HARD_STOP_RULES` 示例：

```bash
HARD_STOP_RULES="mean_tpot_ms:ge:100,p99_tpot_ms:ge:150,mean_ttft_ms:ge:5000" \
bash sglang/sglc.sh
```

规则格式为 `metric:op:threshold`：

- `metric` 使用 `sum_all.csv` 中的指标名，例如 `mean_tpot_ms`、`p99_tpot_ms`、`mean_ttft_ms`、`p99_ttft_ms`。
- `op` 支持 `ge`（greater than or equal，大于等于，`>=`）、`gt`（greater than，大于，`>`）、`le`（less than or equal，小于等于，`<=`）和 `lt`（less than，小于，`<`）。
- 多条规则用逗号分隔，任意一条命中就会触发硬停止。
- 每个 BS 完成并汇总到 `sum_all.csv` 后检查；命中后停止当前 case 的后续 BS，不影响已经完成的结果。
- 默认值为空，表示关闭硬停止；如果同时命中 `SLA_RULES`，会先完成 SLA inline dense 补点，再执行硬停止。

## 续测

脚本默认使用时间戳创建新的结果目录。需要续测时，使用原结果目录的时间戳作为
`RUN_ID`，并保持服务地址、模型、`CASES`、DP 配置、`TOKEN_UNIT` 和 BS 配置一致：

```bash
# 例如原目录为 ./all_result/bench_theory_bs_20261012_143000
RUN_ID=20261012_143000 \
BASE_URL=http://127.0.0.1:30000 \
MODEL=/path/to/model \
bash sglang/sglc.sh
```

vLLM 使用相同方式，目录前缀为 `bench_vllm_prefix_salt_`：

```bash
RUN_ID=20261012_143000 \
BASE_URL=http://127.0.0.1:8000 \
MODEL=/path/to/model \
bash vllm/vllmc.sh
```

续测规则：

- 已有非空日志且对应 `.log.exitcode` 为 `0` 的 case / BS 会输出 `[SKIP]` 并跳过。
- 失败或未完成的 case / BS 会重新执行；`sum_all.csv` 会根据当前日志重新汇总。
- 如果上次因 `HARD_STOP_RULES` 命中而停止，需要清空或调整该规则后再续测，否则会在相同阈值处再次停止。
- 不设置 `RUN_ID` 会创建全新的结果目录，不会读取旧目录中的结果。

## 固定并发示例

固定 per-DP BS：

```bash
FIXED_PER_DP_BS="8,12,16" bash sglang/sglc.sh
FIXED_PER_DP_BS="8,12,16" bash vllm/vllmc.sh
```

固定全局并发：

```bash
FIXED_CONCURRENCY="64 96 128" bash sglang/sglc.sh
FIXED_CONCURRENCY="64 96 128" bash vllm/vllmc.sh
```

SGLang 自动理论 BS 扫描可以单独设置本次测试的最大 per-DP BS。理论容量仍会保留在日志中，
不会因为测试上限而被修改；日志中的 `Effective Theory`、`Auto Test Cap` 和 `Test BS Upper`
分别表示理论容量、配置上限和本次实际测试到的最大 BS：

```bash
# 理论 per-DP BS 为 32 时，本次自动扫描最多验证到 BS16；理论值更小时不扩容。
MAX_TEST_PER_DP_BS=16 bash sglang/sglc.sh
```

## 输出

脚本会在当前目录生成 `all_result/` 子目录，每次运行一个独立结果目录，包含：

- `logs/`：每个 case / BS 的执行日志。
- `sum_all.csv`：汇总后的核心指标表。
`sglc.sh` 默认额外生成 `decode_matrix.png`：每一列是输入/输出/per-DP BS workload，每一行是
`SLA_RULES` 中的一条延迟规则，只保留对应指标小于等于 target 的 BS，格子显示其中实际 RPS
最大的结果；没有满足点显示 `NO MEET`。默认格子显示 RPS，设置
`SGLC_PLOT_CELL=output_throughput` 后显示生成 token throughput。
GSP 的 `gsp_prefix_summary.tsv` 面向复制到表格使用，保留输入/输出、指定和实际 cache 命中率、确认后的 workload 参数、请求率、吞吐和性能指标；执行过程字段如 `page_aligned`、`search_phase`、`log_file`、`backend`、`ttft_metric`、`ttft_targets_s` 等不放入 summary，原始排查信息仍保留在 `gsp_prefix_trials.csv`。`SLO` 将 TTFT/TPOT 要求合并为 JSON，TTFT 使用秒、TPOT 使用毫秒，例如 `{"ttft":{"mean":[5,6],"P90":[8,10]},"tpot":{"mean":[20,30],"P90":[50,60]}}`。
summary 的公共客户端指标严格按 `sglc.sh` 的顺序排列；GSP 的指定/实际 cache 命中率和 `SLO` 紧跟在 case/输入输出/并行配置之后，命中率展示为 `指定/实际`；`system_prompt_len/question_len` 和 `gsp_num_groups/gsp_prompts_per_group` 分别合并为单列，`total_prompt_tokens`、`total_cached_tokens` 放在 GSP 字段末尾。

`all_result/` 是运行产物目录，建议不要提交到仓库。
