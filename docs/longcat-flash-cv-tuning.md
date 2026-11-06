# LongCat Flash 路由反向与通算重叠调优

本轮在已验收的热副本版本 `ca76a20089d03dc0d9f9f7e8852c93931c5396ef` 上继续调优。
保留 [LongCat 热副本接入](longcat-flash-hot-replicas.md)中的参数、布局、精度和生命周期契约。
本轮正式结果是原宽度 EP4 专家组件，不是完整 LongCat Trainer 或客户 x86 NPU 验收。

## 已合入的路由反向优化

适配器先按全局专家 ID 排序，因此每个 token 的真实专家槽构成前缀、identity 槽构成后缀。
原来的反向 kernel 对全部 12 个槽执行有序 BF16 累加，包括不属于当前分支的零贡献。
新实现只遍历当前分支的有效槽，保持原逆序累加、每个乘积及每次加法的 BF16 舍入。
跳过零贡献时显式保留负零规范化语义；概率梯度仍使用原 FP32 归约。

该优化仅选择 H6144、TopK12 的已有 Triton 路由路径。
混合路由的输入梯度采用已测的 4096 元素 tile；identity 梯度保留 2048。
全真实专家、全 identity 和空输入有各自的安全路径；其他宽度与 TopK 使用原 kernel。
没有新增设备到 CPU 的计数读取，也没有改变专家 GEMM、SwiGLU 或热副本协议。

## 精度与回归

使用冻结旧版路由前端的独立原生 GMM 参考，避免两侧共享新前端掩盖错误。
NPU 验证保留 H6144、专家 I2048、T2048/rank、TopK12，EP4/E32/Z16；每 rank 仍为 8 个真实专家和 4 个 identity 份额。

- B0 Push/Pull，4 rank：输出、dX、dP、Gate/Up dW、Down dW 均为 `max_abs=0`。
- B2 EOI Push、EOI Pull、EIO Push，各 10 步 × 4 rank：每一步全值一致，FP32 SGD master 更新后的完整参数哈希一致。
- 路由覆盖普通路由、热 owner 切换、source 全 identity、全局 identity；全局 identity 的专家梯度保持 None。
- 独立 Torch 路由参考 ST 通过，覆盖 learned、全真实、全 identity、空输入。
- 新专项 ST 直接对照未修改的完整 12 槽 kernel，逐 token 混合 0/1/4/8/12 个真实槽，并包含负零和正负极小值；原始 BF16 位全部一致。
- CPU LongCat 回归 62 项及 105 个子用例通过；修改文件的 pylint 和复杂度检查通过。

全部 rank 正常关闭 SHMEM，`Uninitialized/reference_count=0`，任务退出 0，所属 NPU 进程退出。
严格逐元素门限仍为 `2e-5 + 0.05*abs(reference)`，没有放宽；此处所有有值张量还满足更强的零误差检查。
本轮不重复宣称历史完整 Trainer 验收适用于新版本。

## 正式组件性能

固定完整输入、权重、学习路由及上游梯度；每个进程 10 次预热、30 次正式样本。
每步取 4 rank 最大耗时，每进程保留全部样本取中位数，再对固定进程配对比值取几何平均。
三个旧版本 pilot 在正式测试前冻结 2% 最低耗时下降门限。

A 为使用新路由前端的原生 NPU 大算子路径，B 为旧 MegaMoE，C 为新 MegaMoE。
进程顺序为 `A1 B1 C1 C2 B2 A2 C3 B3 A3 A4 B4 C4`，两组比较均包含正反顺序。
原生使用 GMM、`npu_swiglu`、permute/unpermute 和 AllToAll；两侧使用相同融合激活语义。

| 实现 | 前向 | 前后向 |
| --- | --- | --- |
| 原生大算子，新路由前端 | 18.295 ms | 40.276 ms |
| 旧 MegaMoE | 12.819 ms | 29.704 ms |
| 新 MegaMoE，省略零贡献 | 12.857 ms | 28.885 ms |

新旧 MegaMoE 前后向配对加速为 **1.02835 倍，耗时下降 2.757%**，超过冻结的 2% 门限。
新 MegaMoE 对相同前端的原生大算子路径前后向为 **1.39435 倍**，耗时下降约 28.28%。
前向没有达到改善门限，不宣称前向额外加速。
本表 B0 普通学习路由不包含热副本负载分散收益，也未将历史优化收益相乘。

独立 profiler 中，4 rank 的 identity 梯度 kernel 从 1.438–1.446 ms 降至 0.697–0.709 ms，
输入梯度从 1.040–1.072 ms 降至 0.921–0.944 ms。
这些是插桩单步诊断，正式耗时只采用上表未插桩的交错实验。

所有性能进程核对固定操作数哈希、全部 rank 样本、源码字节哈希和实际加载的 native 库路径。
native payload 与旧控制版本完全相同；该路由优化自身不修改 C++，但必须满足热副本版本已有的 ABI 要求。
逐进程全部 rank-max 样本、精度审计、源码与报告哈希见
[验收数据](longcat-flash-cv-tuning-results.json)。

## 重跑入口

使用与本分支匹配的构建产物和当前工作树的 editable 安装。
在自行预留的 NPU 上设置 `ASCEND_RT_VISIBLE_DEVICES`，沿用原 CANN8.5.0、Python3.11 和客户自建 Torch/PTA 环境。
以下命令不安装或替换 Torch/PTA，不依赖等待器。

```bash
source "$CANN_ENV"
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
python -m pytest -q \
  tests/torch/models/longcat_flash/test_longcat_flash.py::test_longcat_ordered_route_kernels \
  tests/torch/models/longcat_flash/test_longcat_flash.py::test_longcat_prefix_gradient_kernels

export HCCL_DETERMINISTIC=true OMP_NUM_THREADS=1
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  tests/torch/models/longcat_flash/_benchmark_longcat_replicas.py \
  --backend push --budget 0 --fixture normal \
  --ep 4 --tokens 2048 --real 32 --zero 16 \
  --hidden 6144 --intermediate 2048 --top-k 12 \
  --routing-backend triton --swiglu-precision fused \
  --initial-token-capacity 24576 --warmup 10 --samples 30 \
  --output "$BENCHMARK_OUTPUT"
```

原生对照改为 `--backend native`，每次使用独立进程和新输出目录。
原始前端的独立精度参考与新旧性能对照需保留两个独立工作树、各自 editable 环境、隔离的 native 加载路径；
不能在同一个进程中切换 Python 路径后将两次结果视为独立版本比较。
旧客户复现入口冻结了历史源码哈希，不能删除其校验后当作本版本的重跑入口。

## 筛选中排除的方向

- 直接用 BF16 往返 cast 替代显式逐步舍入：输出约 57% 元素不同，已排除。
- 8192 元素的 route tile：部分 kernel 超出 192 KiB UB，已排除；4096 也未普遍更快。
- GateGrad/W1Grad 完整 tile 波次交错：确认队列实际改变，但没有稳定收益，未合入。
- FP32 概率梯度融合归约：逐元素严格门限通过，但无法保留原 FP32 全值结果，本轮未替换。

## 通算与 CV 分叉实验的范围

LongCat 逻辑层在 attention0 后分成 shortcut MoE 与 dense0 两条路径，
后者继续经过 attention1、dense1，最后再与 shortcut 相加。
因此可以先在副流启动 router，再在主流启动 dense0，将 MoE 及其路由/通信放在副流，延迟到最后相加前等待。

原宽度 dense0+MoE 子图中，直接多流改变了 BF16 输入梯度累加顺序并严格失败。
显式维持 dense、expert、router 的累加顺序后，4 rank 的输出、输入及所有参数梯度全值一致。
同一进程筛选观察为 41.163 → 38.671 ms；这不是独立进程 ABBA 验收，也不是完整层速度。
profiler 确认 ordered 路径存在独立 CUBE dense MatMul 与 AIV route kernel 的时间区间交叠。
区间相交不代表硬件占用率，也不能把不同区间的交叠时长相加当作端到端收益。
完整层、激活重算、FSDP 和 Trainer 验证尚未完成，该多流原型尚未进入默认执行路径。
