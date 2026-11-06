# LongCat Flash 分支复现指南

交付分支为 `feat/longcat-flash-repro`，基于客户已经编译并通过 pytest 的 `feat/cann-8.5.0`，
模型基线提交为 `ca3ce419464f32f43b7299b086d5347da2f5c3e3`。
分支只增加复现材料和对应 CPU 测试，不改模型或 native 源码；客户保留与该基线匹配且已验证的编译产物。
入口接受包含基线的后续提交，同时逐文件核对模型、库和测试支持源码哈希。

## 分支内容

- `run.sh`、`launch.py`、`runtime_probe.py`：运行与环境检查入口。
- `templates/`：精度、组件、Trainer、统计和全 rank 审计脚本，以及全部训练配置。
- `assets/`：实验 tokenizer 和确定性回放 JSON 数据，无需下载预训练权重。
- `requirements-no-torch.lock`、`check_deps.py`：参考依赖清单与完整差异检查。
- `reference/`：2026-11-04 实测参考结果、门限、耗时样本和来源说明，历史本机路径已替换为占位符。
- `manifest.json`、`SHA256SUMS`、`checksums.py`：源码前置条件和静态文件校验。

所有必需的复现材料均通过 Git 获取。分支不包含依赖 wheel、虚拟环境、native `.so`、CANN 或原始 checkpoint。
运行入口复用客户现有 Python 环境，不执行 pip，不安装或替换 Torch、Torch-NPU 及其他包。
客户负责预留空闲节点；入口发现节点上有设备进程即失败退出，不等待、不清理其他进程，不依赖资源等待器。

## 运行前提

1. Linux x86_64 或 aarch64、CPython3.11、Torch2.9、客户已验证的 Torch-NPU 和 CANN8.5.0 编译环境。
   参考环境为 Python3.11.15、Torch2.9.1+cpu、Torch-NPU2.9.1、Triton-Ascend3.2.0。
   客户自建 PTA 保留原版本；入口检查实际选中的 adapter、vendor 和 SHMEM 主机库 ELF 架构。
2. 当前 Python 的 editable 安装指向本仓库。切换同一工作树分支可沿用现有安装；若换了目录，需显式更新：

   ```bash
   "$REPRO_PYTHON" -m pip install --no-deps --no-build-isolation -e "$REPRO_REPO"
   ```

3. 客户目标为 x86_64 / 910C / Python3.11。参考实测来自 aarch64 / Ascend910_9382；
   x86 入口和 ELF 检查已有 CPU 验证，x86 NPU 端到端精度与性能仍需在客户机器上实测。
   完整实验需要单机 **16个逻辑 NPU**，组件需要 **4个逻辑 NPU**；数量按 `npu-smi info -m` 的逻辑设备计算。
   设备 ID 由调用方显式指定，入口检查数量、唯一性和可用设备；完整运行的组件设备必须属于 Trainer 设备集合。
4. 完整训练现场生成约98GiB的 step1 全状态 checkpoint，输出磁盘至少需要120GiB可用空间，建议150GiB以上。
   三条路径均从现场同一份 checkpoint 恢复，比较状态、输入、路由和连续更新。
   只有8个逻辑 NPU 的机器可运行4卡组件，不能按本配置复现 EP16 Trainer。

## 拉取分支并检查环境

在客户已编译的仓库中执行；如远端名称不是 `origin`，替换为实际远端：

```bash
git fetch origin feat/longcat-flash-repro
git switch feat/longcat-flash-repro
export REPRO_REPO="$(git rev-parse --show-toplevel)"
export REPRO_PYTHON="$(command -v python)"
```

先激活此前通过编译和 pytest 的 Python/CANN 环境。
将 `REPRO_CANN_ENV` 设置为该 CANN 的环境脚本绝对路径，`REPRO_OUTPUT` 设置为仓库之外尚不存在的绝对输出目录。
再检查现有依赖、导入和 native 库（不启动 torchrun、不执行 NPU kernel、不创建输出目录）：

```bash
bash "$REPRO_REPO/examples/longcat_flash_repro/run.sh" \
  --repo "$REPRO_REPO" \
  --python "$REPRO_PYTHON" \
  --cann-env "$REPRO_CANN_ENV" \
  --out "$REPRO_OUTPUT" \
  --check-only
```

检查一次列出全部缺失包和版本差异。按清单在客户环境中补齐适合 x86_64 / CPython3.11 的依赖后重新检查，
保持客户现有 Torch/PTA。参考依赖、精度门限和性能公式没有放宽；本分支不提供自动依赖安装入口。
`--check-only` 检查导入和算子入口，实际 NPU 兼容性由后续精度阶段验收。

## 运行完整实验

将 `REPRO_COMPONENT_DEVICES` 和 `REPRO_TRAINER_DEVICES` 分别设置为已预留的4个、16个逻辑 NPU ID，逗号分隔、不重复。
完整运行沿用环境检查时的路径参数：

```bash
bash "$REPRO_REPO/examples/longcat_flash_repro/run.sh" \
  --repo "$REPRO_REPO" \
  --python "$REPRO_PYTHON" \
  --cann-env "$REPRO_CANN_ENV" \
  --out "$REPRO_OUTPUT" \
  --component-devices "$REPRO_COMPONENT_DEVICES" \
  --trainer-devices "$REPRO_TRAINER_DEVICES" \
  --phase all
```

只运行4卡组件时使用：

```bash
bash "$REPRO_REPO/examples/longcat_flash_repro/run.sh" \
  --repo "$REPRO_REPO" \
  --python "$REPRO_PYTHON" \
  --cann-env "$REPRO_CANN_ENV" \
  --out "$REPRO_OUTPUT" \
  --component-devices "$REPRO_COMPONENT_DEVICES" \
  --phase component
```

`--out` 必须位于源码仓库之外，现有目录不会覆盖。运行日志、设备信息和生成的 checkpoint 都写入该外部目录。
其他入口为 `--phase precision`、`--phase trainer`；任何实际运行均先完成4-rank精度阶段。
单项入口产生对应阶段结果，完整汇总使用 `--phase all`。
`--prepare-only` 同样需要对应阶段的显式设备参数，只渲染并冻结启动命令和配置，适合运行前审阅。

native 环境脚本默认在仓库的 `build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash`。
若客户编译产物放在其他位置，传入 `--native-env "$REPRO_NATIVE_ENV"`，必要时另传 `--native-root`。
环境脚本、loader 实际选中的 vendor/adapter 和 native 根目录必须一致。

## 实验约定与验收

- 每个逻辑 NPU 保持8个真实专家、4个 identity 份额；H6144、专家I2048、TopK12、2048 tokens。
- 组件 EP4/E32/Z16；完整训练 EP16/E128/Z64，单逻辑层，dense FSDP16 / expert FSDP1。
- 三条路径保留相同 dense MLP 和 HF RMSNorm；BF16计算、FP32主参数/归约、FP32 loss。
- 原生 MoE 调用 `torch_npu.npu_swiglu`；MegaMoE 内嵌 CANN `swi_glu/swi_glu_grad`，默认 fused，HF BF16兼容开关关闭。
- 以原生 GMM 为主精度验收：模块输出及梯度全值误差0；完整训练比较连续10次更新×16rank，
  包括 loss、裁剪前全局梯度范数、更新后的模型/FP32主参数完整哈希。
- 组件：三次强化原生预跑，固定顺序12个正式进程；每阶段10次预热、30个rank-max样本。
- Trainer：新建共同 step1 checkpoint；三条路径数值验收、三次强化原生预跑、六个正式进程；
  每正式进程5次预热更新、10次测量更新。
- 保留所有慢样本；p50/p90取各进程分位数的中位数。加速/尾延迟门限由强化原生预跑按原公式冻结，
  不根据候选性能修改，精度与性能分别判定。

参考结果：组件 Push/Pull 前反向相对强化原生为1.3475x/1.3878x，单层 Trainer Push 为1.0520x。
这些数值来自参考硬件和环境，客户侧以重新生成的报告和门限为准，不代表全28层预训练吞吐或 EP64 跨节点扩展。

完整运行输出 `results.json`、`summary.zh-CN.md`、`raw-evidence-audit.json`、`runtime-preflight.json`、
`frozen-provenance.json`、`aggregation-freeze.json`，以及每个进程/每个rank的日志和原始样本。
汇总前核对全部 rank 退出、设备进程归属、SHMEM释放、参数形状、fused路径、容量变化和rank-max耗时。
只有 `PRECISION_PASS` 与 `PERFORMANCE_PASS` 同时成立才算完整通过；失败时保留报告并返回非零。

## 静态材料校验

在仓库根目录执行：

```bash
python examples/longcat_flash_repro/audit_publication.py
python examples/longcat_flash_repro/checksums.py
```

维护者审阅静态材料修改后，使用 `python examples/longcat_flash_repro/checksums.py --refresh` 更新校验清单。
发布检查范围见 [PUBLICATION.md](PUBLICATION.md)，已有验证范围见 [validation/README.md](validation/README.md)。
