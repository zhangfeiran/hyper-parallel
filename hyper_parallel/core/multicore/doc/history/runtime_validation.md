# MegaMoe 重基、构建与存储所有权：历史验证

本文汇总 2026-09-07 至 2026-09-08 的历史实验，保留各阶段的测量配置、失败与修复边界、原始文件及哈希。
结果对应各节列出的旧源码和 native 产物；本次整理没有重跑性能或数值验收，也不将它们改记为合并后结果。
旧报告里的“本轮/当前”均指该历史阶段。后续实现与验收统一由
[当前待办](../megamoe_optimization_backlog.md)跟踪，本文不维护另一份优先级。

以下 output/ 与 /tmp/ 路径是当时的本机证据位置，原始大文件未入 Git；本次未核验其仍然可用。
文末固定 Git 版本可恢复完整旧报告，不保证恢复未入库的原始数据。

| 阶段 | 代码身份 | 历史结论 |
| --- | --- | --- |
| 重基复核 | 0208212598ac96f3ff221be6723167fec80efe50 → fa1a18591c8e110542035c25588f96b2cb7cf344 | shared-B 同配置没有明显退化；发现两个失败，不能判为整分支通过 |
| dX ownership 修复 | 基线 7fe7db653294ce04ea89610db20c66984b8bf17c；修复 d5698b72 | 有效 dX 返回 owned clone；串联图通过，保留额外 HBM 成本 |
| 常规构建与 reference 修复 | e69350fa | 公开 Torch include discovery 与 w1/w3 适配；正式构建后 12 项 NPU 回归通过 |

三阶段环境均按原记录为单机 8 × Ascend 910B3、64 GiB/card、Python 3.11、
Torch/Torch-NPU 2.9.1、CANN 9.2.0；它不是其他轮次 CANN 9.0/9.1 的测量。

## 重基复核与失败边界

设备端 multicore/SHMEM ops 在 A/B 间没有源码差异，12 个相关 patch 的 range-diff 相同。
变化集中在 host 模块组织、公开 Torch adapter、owned SHMEM 和标准 GroupedExperts reference。
Conda 环境的实际头文件位于 prefix/include，原 CMake 硬编码 wheel 内的 torch/include 导致配置失败。
复核时仅在仓外重建诊断 adapters，使用绝对 RUNPATH 并额外链接 torch_python；这不能算常规构建通过。

CPU 回归为 183 passed / 148 subtests；九项真实 NPU launcher 为 7 passed / 2 failed。
失败分别是新 reference 仍被访问 gate_up_weight，以及串联共享 workspace 的 input gradient 不匹配。

### 固定权重的 A/B 直接比较

使用同一个仓外 worker，避免两分支 native reference 和初始化方式不同造成混淆：

- EP2、三层、T=2048、H=512、I=128、E=8、K=4、BF16；capacity factor 显式为 `1.0`。
- B=1、跨层 shared physical weights；每层 seed 为 `2026 + rank + 100 * layer_index`。
- 四个 step，在 balanced / hot-first-half / hot-second-half 之间逐层、逐步换路由。
- 每步先建立三份 forward 图，再逆序 backward；奇数 step 使用 reentrant checkpoint。
- 每步更新 AdamW：lr=1e-4、betas=(0.9, 0.999)、eps=1e-8、weight_decay=0.01，
  foreach=False、fused=False。
- 每步保存输出、dX、route dW、所有参数梯度、更新后参数和 AdamW 的 step / 一阶矩 / 二阶矩。

两卡各 156 个张量，合计 **312/312 逐位相同，全部 finite，max absolute difference = 0**。
这支持已覆盖 shared-B 路由切换和优化器闭环没有新增精度损失，不代表完整训练收敛已验证。
首次诊断启动遇到 SHMEM 端口被占用，未产生数值结果；换用空闲的低端口后完成上述比较。

### 串联 shared-workspace 问题：重基前已经存在

将新验收用例的关键图结构移到同时适用 A/B 公共 API 的 worker：EP2、T=1024、H=512、I=128、
E=4、K=2，三层真正串联，两份完整 forward 保留后逆序反向；比较独立和共享资源。
capacity 显式设为 `2.0`，分别使用一个计算 stream 和两个交替 stream。为排除资源关闭接口差异，
两个场景的所有资源保持存活，固定 512 MiB SHMEM heap，不执行 close shim。

两分支都在相同的十个位置失败：两份 dX、前两层的 route gradients，以及前两层的 expert gradients。
输出和最后一层梯度匹配。A/B 在这项诊断保存的 **128/128 张量逐位相同**，说明失败不是此次
rebase 新引入；但相同的错误不构成精度通过。dX 相对独立资源结果的最大绝对差，rank 0 / 1 为
`7.156372 / 5.187716`，明显不是 BF16 容差边缘。

静态路径显示，one-wave fused backward 返回的 dX 是 `workspace.routed_buffer` 的 view。
串联的上一层把这个 dX 当作 dY；若共享同一 workspace，后续恢复 expert output 的 `copy_`
会在 dY 被消费前覆盖同一 storage。外部 leaf `.grad` 持有该 view 时也会遇到后续复用问题。
需对 autograd 边界返回值的 ownership 做完整修复，而不仅增加计算流等待。

随后仅在新的诊断进程中包装 `_finish_backward`：检测到返回 dX 与 routed buffer 同 storage 时，
返回 owned clone。每个 rank 实际命中 24 次别名；原本失败的单 / 双 stream 场景均通过原容差
`rtol=2e-2, atol=1e-5`。这支持上述 storage ownership 定位，**不是已落地修复**：没有修改
生产函数，没有把该 clone 带入性能 A/B，也没有据此覆盖正式 ST 的失败结果。

### 协议

主比较沿用仓内 `examples/torch/multicore/benchmark_mega_moe_layers.py`：

- EP8、16 个独立 Qwen-shape MoE 层，T=4096、H=5120、I=1792、E=48、K=8。
- B=1、shared-B、显式 capacity factor=1.5，reentrant checkpoint，每步 AdamW。
- 所有 forward 后逆序 backward；没有 attention、router network 或层间连接，不是整网训练 step。
- 每个路由 3 warmup + 7 measured；先 `0.1:42` 偏斜，再 `10000:42` 平衡。
- 分别直接测量 F+B 和 F+B+AdamW 的 wall time；每个样本取 rank max，再取运行中位数。
- A1 → B1 → B2 → A2，全程保持 runtime / worker / example / helper / native 产物冻结。

新版 example 的 `_make_layers` 先构造标准 common MoE，再复制参数给 MegaMoe；旧版直接进行
rank-local Xavier 初始化，seed 也不同。因此主 A/B 仅在仓外替换初始化为旧版的构造方法，保留
两分支原有计时及 optimizer / memory 逻辑；对各 rank 的全部初始参数计算 SHA256，并将该
helper 自身 SHA256 加入运行首尾的 source identity。避免把不同数值或 common 临时分配造成的
allocator 历史误认为 runtime 回退。

### A/B/B/A 实测结果

单位 ms；每行是各自七次测量的 rank-max 中位数：

| 运行 | 偏斜 F+B | 偏斜 full step | 平衡 F+B | 平衡 full step |
| --- | ---: | ---: | ---: | ---: |
| A1：重基前 | 1230.211 | 1361.006 | 671.265 | 802.981 |
| B1：重基后 | 1231.315 | 1361.960 | 673.727 | 804.765 |
| B2：重基后 | 1227.003 | 1357.894 | 672.707 | 803.671 |
| A2：重基前 | 1230.134 | 1361.543 | 671.630 | 802.995 |
| A 两次中位数均值 | 1230.172 | 1361.275 | 671.448 | 802.988 |
| B 两次中位数均值 | 1229.159 | 1359.927 | 673.217 | 804.218 |
| B/A − 1 | -0.082% | -0.099% | +0.264% | +0.153% |

偏斜 full step 略快，平衡略慢，均为亚百分比小差异。本组未观察到有意义的性能回退，也不将其
称为加速收益或统计等价性证明。这里只是两个代表性路由，不替代 55-route 全曲线及完整模型。
两条路由在 B1 placement 后都为 physical W1；残余 multi-wave 仅包含在本轮精度 ST 中，
没有用这两个点宣称残余多 wave 的性能也已全面覆盖。

四次运行的 configuration（含八个 rank 的全部初始权重 hash）、route histogram、physical
placement / wave count、shared pool 和 optimizer-state ledger 完全相同。对应各 rank 内存字段，
排除 runtime free / used / total 后全部逐字节相同，包括 baseline / final allocated、reserved、
peak、SHMEM heap 和 workspace ledger。下面是各次共同的 rank max：

| 字节数 | 偏斜 | 后续平衡 |
| --- | ---: | ---: |
| allocator peak allocated | 24498462720 | 24057401344 |
| allocator peak reserved | 25866272768 | 25866272768 |
| configured SHMEM heap | 872415232 | 872415232 |
| conservative peak = peak allocated + heap | 25370877952 | 24929816576 |

shared weight / gradient pool 仍为每 rank `55050240 / 110100480 bytes`，合计 157.5 MiB。
四次分别有 9 / 10 / 10 / 9 份 NPU census，只出现对应的八个本轮 worker；没有观察到别的 NPU
训练进程。该采样不是连续监测或硬件独占保证。

### 默认 capacity 的变化不应隐藏

`expert_capacity_factor=None` 的默认接口不在上述同配置结论内：该 shape 的 receive capacity
从显式 1.5 的 `49152` 变为最坏全组 routed slots 的 `262144`，接收区更大。不能把保守默认的
HBM 或延迟变化归入相同容量的 rebase A/B。

用当前 `_spec_workspace_bytes` / heap rounding 对一个共享 B1 resource group 计算：有限 1.5
为 `872415232 bytes` heap，None 为 `3087007744 bytes`，增加 **2112 MiB**。这是当前规划公式
的静态结果，不是 None 默认配置的新实测曲线；examples 本身仍显式传 1.5。

## dX ownership 与构建修复

### 根因与最小修复

修复基线为 `megamoe-sun@7fe7db653294ce04ea89610db20c66984b8bf17c`。
one-wave 且两个路由方向均 fused 时，`_finish_backward` 原来直接返回 SHMEM routed buffer 的 view。
三层串联共享 workspace 时，前一层会把这个返回值当作 dY，再在恢复自身 backward 数据时覆盖
同一个 buffer。等待 stream/event 只能保证操作顺序，不能让已经覆盖的数据重新有效。
leaf `.grad` 被 autograd 接管后也可能持有该 storage，受到下一轮复用或 resource close 的影响。

现在只在 `backward_unpermute_fused && permute_fused` 分支，先截取有效的 `[T, hidden]` dX，
再返回 owned `clone()`。复制发生在 workspace claim 内、release event 之前，因此后续复用也必须
等待复制完成。unpermute-only 已有私有输出，不额外复制；外部 route 和 multi-wave 路径不变。
没有增加共享池、改变参数 storage、降低 dW 精度、修改 capacity 默认值或 device schedule。

该方案每次返回只分配有效 dX 大小，不复制整份 receive-capacity SHMEM 区。
独立输出的存活时间交给 autograd / 用户，而不是挂在 module 上的额外 persistent buffer。
这修复的是已有错误，不是一次数值等价的性能优化。

### 回归覆盖

按 add-unit-test 工作流先补 CPU 回归，再修真实 `_finish_backward`：旧代码出现三处 ownership
断言失败，修复后测试文件为 **9 passed / 8 subtests**。覆盖 BF16 / FP32、workspace 覆盖、连续
返回互不别名，以及 unpermute-only 不增加复制。没有用假实现代替被测函数。

新增 EP2 真实 NPU worker `_test_mega_moe_connected.py`：

- 三层真正串联、两个 microbatch，共六次 outstanding forward 后逆序 backward。
- B=1、shared-B，对比独立和共享 execution resource group，保持参数相同。
- capacity=1.5 / 1.0 × 无 checkpoint / reentrant checkpoint，四个 case，每个三步 AdamW。
- 逐层、逐 microbatch、逐步切 balanced / hot-first-half / shifted-hot 路由。
- 观察真实 kernel launch 标量：capacity=1.5 命中 W1；capacity=1.0 同时命中 W1/W2；两者都覆盖
  home-only 和 shared-B launch。观察器不保留 tensor 参数，也不替换 kernel。
- 每步、每 rank 比较 40 个输出、dX、route dW、expert dW、更新后参数和 optimizer-state tensor；
  合计 **960 组比较通过**，沿用 `rtol=2e-2, atol=1e-5`。

原 shared-resource worker 增加 leaf gradient 在最后一个 owner 关闭后仍持有独立 storage 且值不变
的检查。原来的单 / 双 stream、两份串联图、不同 owner 关闭顺序仍全部执行。

| 验证 | 结果 |
| --- | --- |
| MegaMoe / examples / native / SHMEM CPU 回归 | 186 passed，150 subtests，20.40 s |
| 新串联训练 + 原 shared resources NPU launcher | 2 passed，21 deselected，84.36 s |
| 原 level0、hot/deferred replication、lazy/eager compact、residual multi-wave、checkpoint fallback、cycle profile | 8 passed，15 deselected，244.89 s |
| 修改的五个 Python 文件 pylint；`git diff --check` | 通过 |
| 在干净解释器 import ST launcher | 未导入 torch / torch_npu / mindspore / hyper_parallel |

shared-resource 两个 stream 场景，各八步中的 warmup 后五个 allocator 样本都为
`210289152 bytes/rank`，SHMEM allocation count 不增长。新串联对照使用独立 MegaMoe 作为 reference；
与标准 EP 的数值 oracle 由已有 level0 / replication 等 ST 承担。
这些结果不表示长时收敛或任意 checkpoint 变体已经验证。

### dX 修复的匹配 A/B/B/A

沿用上节 16 独立层、同权重初始化和 rank-max median 协议；A 为 7fe7db65，
B 为含 dX 修复的 working tree，两侧仍用诊断 adapters。常规构建在后续阶段独立验收。
CPU 测试、编译和其他本轮 NPU 工作不与 timed run 并行；约十秒的进程采样不是硬件独占保证。

单位 ms；A/B 的最后两行是各自两次运行中位数的均值：

| 运行 | 偏斜 F+B | 偏斜 full step | 平衡 F+B | 平衡 full step |
| --- | ---: | ---: | ---: | ---: |
| A1：修复前 | 1228.833 | 1359.488 | 671.657 | 802.731 |
| B1：owned dX | 1228.471 | 1359.723 | 671.709 | 802.838 |
| B2：owned dX | 1231.580 | 1362.417 | 672.809 | 803.472 |
| A2：修复前 | 1229.649 | 1360.802 | 670.359 | 800.810 |
| A 均值 | 1229.241 | 1360.145 | 671.008 | 801.770 |
| B 均值 | 1230.025 | 1361.070 | 672.259 | 803.155 |
| B/A − 1 | +0.064% | +0.068% | +0.186% | +0.173% |

这两个代表点没有观察到明显的 step 回退，但不能据此证明统计等价或所有路由无开销。
四轮 configuration、全部初始权重、路由及物理 placement、shared pool、optimizer-state ledger
完全相同，17 个 native artifacts 的 hashes 也相同。A/B 的 Python fingerprint 分别固定为
`89957f9ae8f61aff25a39bdcda4d9a35fa70276c9a9d5cabedce33a896ada0c9` /
`17adcfd62f9eda3bd7578ef351a0b64b81f616e976eaeeb4cb2673686feea39f`。
运行内首尾冻结与 finite 检查全部通过；9/9/9/12 份设备 census 只观察到各自八个 worker。

对应 rank-max 内存值如下，同一版本的两次运行一致：

| 指标 | 修复前 | 修复后 | 差值 |
| --- | ---: | ---: | ---: |
| 偏斜 allocator peak allocated | 24498462720 | 24539358208 | +39.001 MiB |
| 偏斜 conservative peak | 25370877952 | 25411773440 | +39.001 MiB |
| 平衡 allocator peak allocated | 24057401344 | 24057401344 | 0 |
| 平衡 conservative peak | 24929816576 | 24929816576 | 0 |
| 两点 allocator peak reserved | 25866272768 | 25904021504 | +36 MiB |
| configured SHMEM heap | 872415232 | 872415232 | 0 |
| 两点 final allocated | 22150163968 | 22150163968 | 0 |

单个 BF16 dX 为 40 MiB；实测峰值差不必恰好等于该大小，因为峰值由其他同时存活张量共同决定。
平衡点在偏斜后运行，保留之前的 allocator 高水位。shared weight/gradient pool 仍为
`55050240 / 110100480 bytes/rank`，没有层数相关的新增持久池；不把 reserved 与 live allocation
相加，也不把该小规模增量说成“零 HBM 成本”。

### 常规构建恢复

multicore 与 SHMEM 的 Torch CMake 改用公开 include discovery，过滤缺失 fallback 并检查必要 header。
configure UT 覆盖 wheel/Conda/缺失 header；CPU 回归为 189 passed / 156 subtests。
shared-weight control-stream reference 改为 GroupedExperts.w1/w3，原有 owner-only cache 失效检查保留。

以下两个命令均完成 configure / compile / install，退出码为 0：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash scripts/build_symmetric_memory.sh --framework torch --soc-list ascend910b --jobs 32
bash scripts/build_multicore.sh --framework torch --soc-list ascend910b --jobs 32
```

multicore 的六个 kernel 本轮重新生成；它们及 vendor host 库、SHMEM kernel 的 SHA256 与第一阶段
产物相同。两个 Torch adapters 和四个重新构建的 SHMEM host / bootstrap 库 hashes 改变。
没有修改这些 C++ / device 源码。普通构建覆盖生成的 component / payload，修复前副本仍保留在
隔离 baseline 中；未删除源码或用户结果。

正式 adapter hashes：

```text
libhyper_parallel_mega_moe_torch.so 6a3e7934b3d995248712f2ddb91d0e6b1611b290127a5fe746755fb6dc88346e
libaclshmem_torch.so               78b4a5aed63a1e0da16052c0e214be311de4c20103595f23f31176b3d7c7cfcb
```

`readelf -d` 确认两个 adapter 不再额外依赖 `torch_python`，SHMEM 的 DT_NEEDED 使用库名而非
诊断构建的 checkout 绝对路径；RUNPATH 分别为：

```text
multicore: $ORIGIN/../../vendors/hyper_parallel_multicore_nn/op_api/lib
SHMEM:     $ORIGIN:$ORIGIN/../..:$ORIGIN/../../shmem
```

这是当前 Conda 环境的常规源码构建验收；wheel 布局由 configure UT 覆盖，不代表完整 wheel
打包、异机分发或其他框架 / SoC 已验证。

### 正式产物的设备验收

正式产物的回归为 **12 passed / 11 deselected，621.66 s**；pytest 另报告六个已有 marker warning。
覆盖 shared-weight lifecycle、shared resources、新串联训练、四卡 default interface、level0、cycle
profile、hot/deferred replication、lazy/eager compact、residual multi-wave 和 checkpoint fallback。
没有失败或 skip；dX 阶段的 960 组 connected tensor 对照与资源关闭检查在正式产物上再次通过。

shared-weight lifecycle 先以 T=128 小配置单独通过：**1 passed / 22 deselected，34.25 s**。
随后 T=2048 同配置复测也通过；两次 rank 日志均报告 21 组数值检查、`checkpoint_modes=[false,true]`，
shared weight / FP32 gradient pool 为 `393216 / 786432 bytes/rank`，均未拒绝本配置的 repeated backward。
每组比较输出、input / route gradient 和两组 expert gradient，保留 owner-only cache 失效与跨流检查。

这里以实际 worker 日志确认 checkpoint 覆盖：历史 `HP_MEGA_MOE_SHARED_REENTRANT_ONLY` 开关只在
直接执行 worker 的 `__main__` 中生效；pytest launcher 会直接调用函数，使用默认的两种模式。
本轮没有依赖该开关缩小验收，也不据此泛化其他 shape 或 retained-scratch 组合。

另完成四卡 default-interface worker：T=4096、H=5120、I=1792、E=32、K=8，覆盖
`capacity=None / 1.5` × `experts / router_experts`，与标准 EP 对比输出、梯度和参数更新；四种组合
全部通过。默认 None 保持不变。该 worker 自带单层计时，不将其与第一阶段的 16 层 B1 step 混算。

### 正式产物的 16 层复测

设备回归结束后再次独立运行第一阶段相同的 matched-weight benchmark，参数、八卡全部初始权重、
Python implementation fingerprint、route / placement、shared pool 和 optimizer-state ledger 与 B1/B2
相同；使用本阶段新建的正式 native payload。未与其他本轮 NPU 测试、编译或 CPU 回归并行。

| 路由 | F+B median | full-step median | conservative peak bytes |
| --- | ---: | ---: | ---: |
| 偏斜 `0.1:42` | 1229.056 ms | 1360.044 ms | 25411773440 |
| 平衡 `10000:42` | 670.697 ms | 801.564 ms | 24929816576 |

对应各 rank 的 allocator / SHMEM ledger 与第一阶段修复版逐字节相同；仅 runtime free / used
等外部观测量不要求相同。首尾 source/native freeze 与 finite 检查通过，九份设备 census 只观察到
本轮八个 worker；结束后八张卡均无残留 NPU 进程。采样不构成硬件独占保证。
该补充运行没有显示额外性能 / HBM 回退，但只有一轮，不用它代替第一阶段的 ABBA 或声称新的加速。
完整 JSON SHA256：`7a9d73c92f009b3fb24642597a174daa78e5cf2e1e68848e1319687f7cd22b27`。
复现脚本为 `/tmp/megamoe-fixes.EJ9KbG/run_canonical.bash`。

## 证据与复现

### 重基复核的原始证据

原始结果在本机 `output/qwen_moe_optimizer/`，不将已有历史结果冒充本轮数据：

- `rebase_sun_ut_20260907.log`、`rebase_sun_st_20260907.log`；ST rank 级 traceback 另在 `logs/`。
- `rebase_sun_shmem_build_20260907.log`、`rebase_sun_diagnostic_adapters_retry_20260907.log`。
- `rebase_{old,sun}_precision_20260907/rank{0,1}.pt`；成功的 B 启动日志带 `precision_retry`。
- `rebase_{old,sun}_connected_20260907/rank{0,1}.pt`。
- `rebase_sun_owned_dx_20260907.log` 及同名目录：隔离 dX alias 的进程内诊断。
- `rebase_{old_a1,sun_b1,sun_b2,old_a2}_20260907.json` 及相邻 `.log`、`*_monitor_20260907.log`。
- `rebase_sun_example_20260907.json` 及相邻运行 / monitor 日志：原样 example 的补充点。

四个主 JSON 的 SHA256：

```text
old_a1 ef6e1b72a39857196723c18b940ddffee190dc04fbc8ceeac1688c4b8009d8d3
sun_b1 e27ee430c0c32c342cda94202f88a780051dc550e48fef63f201c2aa61485386
sun_b2 7b8e1ac5a26564bec7f5e6738837bec83ef174477d862af74331b36a98ec96e3
old_a2 58e1dbd951adde9d6b21231d5bdfa7053ca81d2cfac9fca4c5d3c2ee6bc465ab
```

补充原样 example JSON SHA256：`21b4bcc3f1bd2a467a182d95bf3d7242f50eb13530f54f066f4bf44f5617f1bd`。

主比较 helper SHA256：`0880112e26b197dc348d8e5ab0394b86a0452cafbaa93f528c29520e64f1d593`。
Python implementation fingerprint 的 scope 在 A/B 间为 v2 / v3，不能直接把 hash 不同判为设备
代码变更；运行内冻结检查及 17 项 native hashes 均保留在每个 JSON 中。

本轮工作目录 `/tmp/megamoe-sun-audit.y8Gp0T/` 保留 detached baseline、baseline-venv、
`precision_matched.py`、`connected_shared.py`、`benchmark_matched.py`、`run_abba.bash` 和
`build_adapters.py`。运行时先 source CANN 9.2，再 source 对应 checkout 的
`build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash`，使用该 checkout 的 editable
Python，设置 `HYPER_PARALLEL_PLATFORM=torch`、`OMP_NUM_THREADS=1`、对应 `PYTHONPATH`；
两个分支均使用 `HCCL_NPU_SOCKET_PORT_RANGE=62000-62063`，SHMEM 和 torchrun 使用各自空闲端口。
主 benchmark 参数：

```text
--variant b1 --layers 16 --warmup 3 --measured 7 --pairs 0.1:42,10000:42
```

精度直接比较与性能计时没有并行占卡；编译、CPU tensor 比较不与 timed run 同时执行。
性能运行期间约每十秒保存 NPU 进程采样；不是设备独占锁，不能排除采样间隙或系统竞争。

### 修复与常规构建的原始证据

原始结果位于 `output/qwen_moe_optimizer/`：

- `fix_dx_ownership_red_20260907.log`、`fix_dx_full_ut_20260907.log`、`fix_dx_lint_20260907.log`；
- `fix_dx_connected_st_20260907.log`、`fix_dx_regression_st_20260907.log`；
- `fix_dx_evidence_20260907/`：新串联矩阵与原 shared-resource 验收 JSON；
- `fix_dx_{old_a1,fixed_b1,fixed_b2,old_a2}_20260907.json` 及对应运行 / monitor 日志。
- `fix_cmake_red_20260907.log`、`fix_build_full_ut_20260907.log`、`fix_build_lint_20260907.log`；
- `fix_cmake_{shmem,multicore}_build_20260907.log`：常规完整构建；
- `fix_cmake_shared_default_st_20260907.log`、`fix_cmake_regression_st_20260907.log`；
- `fix_cmake_evidence_20260907/`：正式产物的串联、资源生命周期和默认接口 JSON；
- `fix_cmake_current_c1_20260907.json` 及相邻运行 / monitor 日志：正式产物的 16 层复测。

四个 JSON 的 SHA256：

```text
old_a1   e8bda9f6777e6ce8e57bbd26e049461769b9c6ce12c33d18d799799f208f7b72
fixed_b1 6b6e9e8397ed6f516c5d52b05af9e095b66263aa7022fab139552d98ba70967e
fixed_b2 f11a4341ae5a89b8f23f396a7bd1306347a7e86475f7cef1d3ba1da488dac035
old_a2   8bad6bf35ecf91d633e90815098a61c9f56a2dea05a93f3daf10f931d00bfae5
```

本轮隔离目录为 `/tmp/megamoe-fixes.EJ9KbG/`，包含 baseline、baseline-venv 和 `run_dx_abba.bash`。
matched-initialization helper 仍为 `/tmp/megamoe-sun-audit.y8Gp0T/benchmark_matched.py`，SHA256
`0880112e26b197dc348d8e5ab0394b86a0452cafbaa93f528c29520e64f1d593`。

## 原报告与恢复入口

整理前的完整文档固定保存在提交 5e26dfddc7ba8f6d0ea1fba34fe6149710482aec，路径前缀为 hyper_parallel/core/multicore/doc/。
下表文件已由本文替代；旧阶段待办及完整过程叙述仍可从该提交读取。

| 原文件 | 本文锚点 |
| --- | --- |
| megamoe_sun_rebase_validation_20260907.md | [重基复核与失败边界](#重基复核与失败边界) |
| megamoe_sun_fix_report_20260907.md | [dX ownership 与构建修复](#dx-ownership-与构建修复) |

从仓库根目录查看原文：

```bash
git show 5e26dfddc7ba8f6d0ea1fba34fe6149710482aec:hyper_parallel/core/multicore/doc/megamoe_sun_rebase_validation_20260907.md
```
