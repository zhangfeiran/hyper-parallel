# megamoe-sun 重基复核后的修复与验收

本文跟踪 [重基复核报告](megamoe_sun_rebase_validation_20260907.md) 中确认的问题。
前一报告保留当时的失败结果；这里记录实际修复后的新证据，不把诊断 workaround 当成正式验收。
每个阶段验证后直接本地 commit，不使用 autogit，不 push。

## 1. P0：fused backward 返回 dX 的存储所有权

### 1.1 根因与最小修复

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

### 1.2 回归覆盖

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

### 1.3 性能与 HBM 测量协议

在 8 × Ascend 910B3 上，使用 Qwen shape T=4096、H=5120、I=1792、E=48、K=8，16 个独立 MoE
层，B1/shared-B、显式 capacity=1.5、reentrant checkpoint、每步 AdamW。先偏斜 `0.1:42`，后平衡
`10000:42`，每点 3 warmup + 7 measured。每个样本取 rank max，随后取运行中位数。
这不是 connected transformer 的整网性能；串联图的精度由上一节另行验证。

A 为修复前 detached worktree / editable venv，B 为当前含 dX 修复的 working tree；顺序 A1/B1/B2/A2。
沿用前一复核的 matched-initialization helper，对各 rank 的全部初始参数计算 SHA256。
runtime、example、helper 和 native artifacts 在四轮中冻结，CPU 回归、编译和其他本轮 NPU 测试
不与 timed run 并行。每约十秒保存设备进程采样；这不是设备独占保证。

本阶段两侧仍使用前一复核的同源 device kernels 和相同的 Conda-aware 诊断 adapters。
常规 CMake 构建是否恢复，必须由下一阶段独立验收，不能由这里的性能通过替代。

### 1.4 A/B/B/A 实测结果

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

## 2. 常规 native 构建与标准 EP reference 适配

### 2.1 构建输入探测

multicore 和 symmetric-memory 两个 Torch CMake 都改用公开的
`torch.utils.cpp_extension.include_paths()`，不再假定 frontend 头文件位于 Python package 内。
过滤返回列表中的不存在目录，并检查 `ATen/ATen.h`、`torch/extension.h`、`torch/torch.h` 至少在一个
有效 root 内。ABI 查询保持原样；两个 Torch 查询都设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。
Torch-NPU 仍用 `find_spec` 查安装位置，不在配置阶段主动加载设备后端。

新 UT 使用临时的 wheel / Conda 安装布局，运行仓内真实 CMake configure 并检查生成的 compiler
arguments，不编译假库、不需要设备或实际 Torch 安装。覆盖路径含空格、Conda 返回不存在的 wheel
fallback、两种 C++ ABI、缺失必要 frontend header 时的提前报错。旧代码有四处预期失败，修复后
完整 CPU 回归为 **189 passed / 156 subtests，30.37 s**，新增三个测试共六个 component 子测试通过。
修改的 Python 文件 pylint 和 `git diff --check` 通过。

### 2.2 reference 参数更新

`_check_control_stream` 的 owner-only 更新不再对 `GroupedExperts` 访问旧的 `gate_up_weight`。
candidate 更新 packed gate/up；标准 reference 分别通过已有 `_local_tensor` 更新 `w1` 和 `w3`，
不改 `w2`。保持只在 rank 0 更新版本的原始条件，继续验证全组 cache-miss agreement 与独立 control
stream；没有删掉失效检查、跳过通信或放宽精度容差。

### 2.3 常规源码构建

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

### 2.4 正式产物的设备验收

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

### 2.5 正式产物的 16 层复测

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

## 3. 本地证据与后续优先级

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

P0 已以 `d5698b72` 本地提交；随后两个 Torch CMake 的 Conda-aware include discovery 和
`GroupedExperts.w1/w3` 测试适配也已通过常规构建与设备验收。两阶段没有撤销默认 None 的语义，
没有改 placement / multi-wave device schedule，也没有降低 FP32 physical dW 的聚合精度。

复审后恢复主报告中 P2P payload / wait / overlap 的优化顺序。先定位 weight refresh / gradient
return 的真实传输和不可重叠等待，再决定合批或调度；当前没有证据将 dX 的小幅复制成本提升为
新的首要瓶颈。不为消掉这次正确性修复的复制而重新暴露 workspace alias，也不在没有 profile
证据时扩大 persistent pool。第二阶段验证结果与代码在本报告同一提交中直接本地 commit。
