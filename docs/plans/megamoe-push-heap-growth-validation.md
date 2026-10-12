# MegaMoE push heap 在线扩容：历史验证记录

本文保留 2026-09-21 的设备验证与测量边界，数据对应 `fa4c5fd0` 接口版本，
基线为 `84519c6f055352d5627688f2a5e0601f306dbde7`。
当前容量接口和重建契约见 [Multicore README](../../hyper_parallel/core/multicore/README.md)，
运行参数见 [Qwen 示例](../../hyper_parallel/core/multicore/examples/mega_moe/README.md)。
下文 static/grow 为历史实验标签；当前可将初始因子设为 EP 大小来预留 static 无损预算。
历史结果不代表 rebase 后版本已经通过设备验证。

## 环境与正确性

单机8张 Ascend 910B3，每卡64 GiB；Torch 2.9.1、Torch-NPU 2.9.1、CANN 9.2。
历史 worktree 的独立 `build/venv` 已 editable 安装，导入路径指向所测目录。
复用与基线一致的 native payload，单独重编译并安装修改后的 SHMEM binding/runtime。
没有以另一个 worktree 的 Python 包代替本次实现。

| 验证 | 结果与范围 |
| --- | --- |
| Multicore UT | 118 passed，535 subtests；容量边界、预算余量回退、未知引用/分配、跨 rank manifest、失败状态、bootstrap 顺序 |
| SHMEM 2卡 lifecycle | 64 → 128 → 64 MiB 重建通过 |
| MegaMoE 2卡 | push/pull 原有 local-capacity 与 ready/checkpoint 回归通过 |
| 增长设备场景 | 单层、共享层、混合 pull 与 lazy push、checkpoint + 交替 stream；两个 forward 后逆序且重复 backward，全量梯度与 common EP 对齐 |
| 4卡子组 | `[0,2]` 扩容时 `[1,3]` 无需进入同一次重建；外部权重、输出及梯度正确 |
| 8卡 Qwen 精度 | 25组初始参数逐元素相同；forward、25组梯度及25组更新参数通过既有 `rtol=0.02, atol=0.002` |
| 8卡连续扩容 | 均衡 → 2.25× → 3.5× → 6× → 均衡，static/grow 各阶段输出及输入、路由权重、专家梯度对齐 |
| Python/C++ 检查 | pylint、lizard、Markdown lint、`git diff --check` 通过；native 编译通过 |

Qwen 首步的最大绝对差：forward 为 `1.91e-6`，梯度为 `3.81e-6`，更新参数为 `0.001953125`。
这只说明所测首步在容差内，不能据此声称后续独立训练轨迹逐步对齐。

发现并修复的 SDK 约束：不能在旧 SHMEM runtime 活动时请求下一次 unique ID。
实际测试中该顺序导致 vendor bootstrap 崩溃；现在严格 finalize 后生成 ID，并在单测中锁定调用顺序。
每个 native 阶段通过原 HCCL root 收敛错误，再进入后续 collective。

## Qwen 固定参数稳态对照

规格：H5120 / I1792 / EP8 / E48 / seq4096 / topk8，2层，BF16。
每种策略在独立进程中仅驻留一个模型；相同 seed、输入和初始参数。
仍执行完整 HyperParallel AdamW 和 FP32 main parameter 路径，但令 learning rate 为0固定参数。
这是容量策略开销对照，不是收敛或正常学习率训练性能结论。
各运行3步 warmup、8步 measurement，计时取每步所有 rank 的最大值，再取中位数。

| 指标 | static 无损 push | grow push |
| --- | ---: | ---: |
| 稳态完整 step 中位数 | 166.54 ms | 166.45 ms |
| 实际 SHMEM heap / rank | 2882 MiB | 802 MiB |
| 峰值 allocator allocated / rank-max | 15988.91 MiB | 15988.91 MiB |
| 峰值 allocator reserved / rank-max | 18008 MiB | 18008 MiB |
| 测量窗口整卡 HBM 采样峰值 / rank-max | 25362 MiB | 23283 MiB |

两者采集到的逐专家 counts、逐 rank 接收量完全一致。
两层 max/mean 分别为1.010620和1.008575；grow 首次遇到这一轻微不均衡时从642扩到802 MiB。
稳态约0.05%的差异属于这次短测的波动范围，不构成加速结论；实际 heap 少2080 MiB。

整卡 HBM 由 DCMI 每200 ms采样，测量窗口各获得6个样本；测量期间没有外来 NPU 进程。
它包含框架外内存，但只是采样峰值，不能证明捕获了瞬时最高值。
进程退出期间监测器曾将正在消失的本任务 PID 标为 foreign，原始退出日志保留此记录；这些不在测量窗口。
allocator peak 与 SHMEM heap 单独报告，不能将 heap 误计为 allocator 已追踪的内存。

## 逐步加重热点

使用同样的专家规格，固定输入、专家权重、上游梯度和显式 top-k IDs，无优化器更新。
每个 token 的8个 expert ID 互不重复；把越来越多 token 从均衡路由切到固定8个专家，构造热点。
每个阶段在首次执行与一次额外 warmup 后计时5次 forward+backward。
先运行 static，再运行 grow；输出和全部梯度在计时窗口之外逐张量比较。
以下是专家层 F+B，不是整网 optimizer step。

| 目的 rank 最大/平均接收量 | static F+B ms | grow F+B ms | grow heap MiB | 此次重建 rank-max ms |
| --- | ---: | ---: | ---: | ---: |
| 1.00× | 29.60 | 29.37 | 642 | — |
| 2.25× | 58.28 | 58.57 | 1042 | 2335 |
| 3.50× | 91.75 | 91.61 | 1442 | 2380 |
| 6.00× | 177.20 | 178.16 | 2242 | 2244 |
| 1.00×（回落） | 29.43 | 29.34 | 2242 | — |

static 的 heap 全程为2882 MiB。grow 的 epoch 为0、1、2、3、3；回落时复用已扩大的 heap。
每次增长后稳态接近 static；扩容停顿约2.2–2.4秒，主要受 vendor finalize 影响。
首次 static/grow 初始化的冷启动时间受运行顺序和算子缓存影响，不作为策略性能比较。

Qwen 精度运行中的一次642 → 802 MiB扩容耗时2623 ms（rank 0），其中finalize约2416 ms、
bootstrap约9 ms、initialize约124 ms。该次发生在 workspace 首次分配前，因此不包含已分配 buffer 的释放/重建。
连续热点实验则覆盖通信和 backward 之后的已分配 heap 重建。

按连续热点实验每次约2.3秒估算，若每100个训练 step 扩一次，平均摊入约23 ms/step；
若每1000个 step 扩一次，则约2.3 ms/step。这是简单摊销估算，非实际长训结果。
初始 factor 和增长 factor 应结合已知路由分布，避免训练早期频繁跨档。

## 边界与历史门禁状态

- 只支持固定 shape、dtype、device、EP 成员和每个 root 串行调用，不支持图捕获或带旧地址的捕获图回放。
- 不承诺重建后降低整网普通激活的负载开销；极端热点仍可能遇到物理 HBM OOM。
- init/finalize/分配异常的状态机使用单测注入；真实 vendor 卡死或 rank 退出依赖底层及 launcher 超时。
  没有进行真实进程失联、跨节点或长时间 soak 验证。
- `autogit check` 总检查仍失败，不能标成绿色。基线同一批文件也复现 docstring、DT描述、UT marker、
  cpplint、clang-format 和 CANN 拼写告警；其80列限制与仓库120列设置冲突，UT marker检查与
  `.agent/rules/unit-test.md` 的 unittest 约定不一致，且 check 在取消暂存后丢失增量行范围。
  本轮新代码通过仓库 pre-commit 对应的 pylint、lizard 和 Markdown 检查；不修改这些无关全局规则。

## 历史配置接口复验

配置已统一为 `dispatch_mode="push" | "pull"`；push 默认
`initial_capacity_factor=1.25`、`capacity_growth_factor=1.25`，也支持显式覆盖。
pull 不接受这两个数值参数。增长因子1.0表示只扩到当前需求，初始因子设为 EP 大小可预留无损上界。
旧 `expert_capacity_factor`、`capacity_policy` 及其 CLI 参数已移除，资源共享兼容性包含两个新因子。

本轮119个UT、545个subtest通过；11个设备回归覆盖 push/pull、共享资源、checkpoint、子组和增长。
Qwen 默认规格以1步 warmup、1步 measurement复验，新默认值均为1.25，实际 heap 为722 MiB，
本次未发生扩容；初始参数、forward、梯度和更新参数比较全部通过。
这是短程接口与精度验证，不作为新的稳态性能数据。
pylint、lizard、Markdown lint、diff检查和AGENTS目录检查通过。

原始记录位于 `build/validation/heap-config/`：`ut-final.log`、`st.log`、`cli-final.log`、
`qwen-default.json`、`qwen-default.log`、`checks-final.log`、`markdownlint.log`。
