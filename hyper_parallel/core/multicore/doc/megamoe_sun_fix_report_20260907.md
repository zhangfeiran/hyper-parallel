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

## 2. 本地证据与后续优先级

原始结果位于 `output/qwen_moe_optimizer/`：

- `fix_dx_ownership_red_20260907.log`、`fix_dx_full_ut_20260907.log`、`fix_dx_lint_20260907.log`；
- `fix_dx_connected_st_20260907.log`、`fix_dx_regression_st_20260907.log`；
- `fix_dx_evidence_20260907/`：新串联矩阵与原 shared-resource 验收 JSON；
- `fix_dx_{old_a1,fixed_b1,fixed_b2,old_a2}_20260907.json` 及对应运行 / monitor 日志。

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

P0 完成后，下一项是两个 Torch CMake 的 Conda-aware include discovery，以及 shared-weight
control-stream 测试对 `GroupedExperts.w1/w3` 的正确更新。应以常规 native 构建产物重新跑验收，
再恢复主报告中 P2P payload / wait / overlap 的优化顺序；不为消掉这次正确性修复的复制而重新
暴露 workspace alias，也不在没有 profile 证据时扩大 persistent pool。
