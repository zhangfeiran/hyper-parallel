# megamoe-sun 重基后的精度与性能复核

## 1. 比较对象与边界

本轮检查重基是否引入额外损失，不实施性能优化或修改生产源码。原始设计及历史测量见
[MoonEP + multi-wave 技术报告](moonep_multi_wave_report.md)。

结论：**同配置的 shared-B 主路径未观察到有意义的性能或新增精度损失，HBM 不变；但整个分支
不能判为验收通过。** 常规 Conda 构建和一项 native-reference 测试适配存在回退；另确认了重基前
已经存在的串联 shared-workspace 梯度别名缺陷，应先于下一阶段性能优化处理。

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-09-07 |
| A：重基前 | `megamoe-perf-sun@0208212598ac96f3ff221be6723167fec80efe50` |
| B：重基后 | `megamoe-sun@fa1a18591c8e110542035c25588f96b2cb7cf344` |
| 工作方式 | A 使用 detached worktree 和独立 editable venv；B 使用当前 checkout 的 editable 安装 |
| 硬件 | 单机 8 × Ascend 910B3，64 GiB/card |
| 软件 | Python 3.11、Torch / Torch-NPU 2.9.1、CANN 9.2.0 |
| 远端状态 | 本机 `origin/megamoe-sun` 指向 B；GitCode DNS 解析失败，未声称已核验远端最新提交 |

设备端 `core/multicore/ops` 和 `core/symmetric_memory/ops` 在 A/B 间没有源码差异；限制到
MegaMoe device / placement 的 range-diff 中，12 个相关 patch 均相同。主要变化在 host 侧组织、
Torch adapter、SHMEM 生命周期和测试 reference：

- `platform/torch/mega_moe` 迁到 `modules/mega_moe`，采用 spec / plan / function / module 结构。
- Torch 桥接改为普通共享库及公开 NPU extension 接口；SHMEM tensor 改用带 NPU metadata 的封装。
- 标准 `GroupedExperts + ExpertParallel` 取代旧私有 native reference；部分测试没有完整适配。
- 公共构造默认 capacity 从有限的 `1.5` 改为 `None`。这不是纯重命名，不能混用默认配置比较。

## 2. 构建验证：存在 Conda 头文件探测回退

当前分支常规构建命令：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash scripts/build_symmetric_memory.sh --framework torch --soc-list ascend910b --jobs 32
```

SHMEM device / wrapper 编译完成，但 Torch adapter 的 CMake 配置失败。报错缺少：

```text
.../site-packages/torch/include/torch/csrc/api/include
```

实际 Torch core / frontend 头文件在当前 Conda 环境的 `include` 下。
`torch.utils.cpp_extension.include_paths()` 能返回该目录；新的两个 Torch CMakeLists
却硬编码 `TORCH_PACKAGE_ROOT/include`，只兼容本环境中并不存在的 wheel 布局。
SHMEM CMake 已实测失败；multicore CMake 存在相同硬编码，但前一命令失败后未继续跑完整构建链。

为不把旧 adapter 当成重基结果，本轮在仓外用 `torch.utils.cpp_extension.load`、
`is_python_module=False` 和公开 include / library discovery 编译当前分支未经修改的 C++：

- `libaclshmem_torch.so`；
- `libhyper_parallel_mega_moe_torch.so`。

设备 kernel 和 SHMEM device 库复用 A 的同源产物；A 保留原 adapter，B 加载上述新 adapter。
两侧各记录 17 个 native artifact 的 SHA256，每次性能运行首尾校验源文件与产物没有变化。
诊断构建使用本机绝对 RUNPATH，且 public extension helper 额外链接 `torch_python`；不是
常规 CMake / wheel 的验收，也不代表可分发产物已验证。仓库 CMake 和运行时代码均未修改。

## 3. 精度检查

### 3.1 CPU 与原有 NPU 用例

当前分支 CPU 回归为 **183 passed，148 subtests passed，20.65 s**，覆盖：

```text
tests/ut/core/multicore/mega_moe
tests/ut/core/multicore/test_mega_moe_module.py
tests/ut/examples/torch/multicore
tests/ut/native/test_multicore_native.py
tests/ut/platform/torch/symmetric_memory
```

九项真实 NPU launcher 为 **7 passed，2 failed，285.81 s**。通过的七项：

- level0 输出 / dX / route dW / expert dW 与标准 EP reference 比较；
- hot-expert replication；
- deferred backward hot-expert replication；
- lazy compact 的 home → replica 延迟反向；
- eager compact 的 home → replica 延迟反向；
- replication 后残余 multi-wave；
- checkpointed capacity fallback。

失败项不能合并描述为“ST 通过”：

1. `test_mega_moe_shared_weight_pool_lifecycle`：`_check_control_stream` 仍对新 reference
   `GroupedExperts` 访问旧 `gate_up_weight`，触发 `AttributeError`。这是重基后的测试适配遗漏。
   本轮采用 `HP_MEGA_MOE_SHARED_TOKENS=2048` 和 `HP_MEGA_MOE_SHARED_REENTRANT_ONLY=1`，
   没有扩大 non-reentrant checkpoint 的通过声明。
2. `test_mega_moe_shared_resources`：三层串联、两份 outstanding forward 的 shared 与 independent
   workspace 比较，在 step 0 的第一个 input gradient 就不匹配。这是数值正确性问题，见下节。

### 3.2 固定权重的 A/B 直接比较

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

### 3.3 串联 shared-workspace 问题：重基前已经存在

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

## 4. 性能与 HBM 比较

### 4.1 协议

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

### 4.2 A/B/B/A 实测结果

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

### 4.3 默认 capacity 的变化不应隐藏

`expert_capacity_factor=None` 的默认接口不在上述同配置结论内：该 shape 的 receive capacity
从显式 1.5 的 `49152` 变为最坏全组 routed slots 的 `262144`，接收区更大。不能把保守默认的
HBM 或延迟变化归入相同容量的 rebase A/B。

用当前 `_spec_workspace_bytes` / heap rounding 对一个共享 B1 resource group 计算：有限 1.5
为 `872415232 bytes` heap，None 为 `3087007744 bytes`，增加 **2112 MiB**。这是当前规划公式
的静态结果，不是 None 默认配置的新实测曲线；examples 本身仍显式传 1.5。

### 4.4 原样 examples 的补充运行

额外运行当前分支未经初始化替换的 example，仍为 16 层、B1、capacity=1.5、每步 AdamW，
同样 3 warmup + 7 measured、先偏斜后平衡。源码与 native hashes 和 B1/B2 完全相同，
只有主比较的仓外初始化 helper 未启用：

| 路由 | F+B median | full-step median | conservative peak bytes |
| --- | ---: | ---: | ---: |
| 偏斜 `0.1:42` | 1235.633 ms | 1368.526 ms | 25348348416 |
| 平衡 `10000:42` | 669.875 ms | 800.992 ms | 24910431232 |

前后 finite 检查、optimizer 状态和 source freeze 检查均通过；route / shared pool 与主比较一致，
heap 仍为 `872415232 bytes`，rank-max allocator peak reserved 为 `25495076864 bytes`。
这仅是一次原样 example 的运行；权重数值和初始化分配历史不同，不能把较低的 allocator peak
当成 rebase 节省 HBM 的证据，也不与四轮同权重结果混算。

## 5. 证据与复现位置

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

## 6. 完成后的优先级复审

1. 首先修复串联图的 dX ownership / workspace alias，再补连接层、多个 outstanding graph、
   shared-B、checkpoint、跨流与 optimizer 的组合回归；正式修复后重新测 HBM 和 step。
2. 修复 Conda-aware 的 Torch include discovery，以及 shared-weight control-stream 测试中的
   `GroupedExperts` 参数访问；用常规构建产物重新跑未通过的验收。
3. 然后回到历史报告的 P2P payload / wait / overlap 优化。此次没有性能退化证据支持立即改
   device schedule，也没有放宽精度容差或撤掉失败检查。

本轮生产源码保持 B 不变；仅将复核结果直接本地提交，不使用 autogit，不 push。
