# 参考结果（2026-11-04）

源码 ca3ce4194，PRECISION_PASS / PERFORMANCE_PASS。每卡8个真实专家、4个identity份额；H6144/I2048/TopK12/T2048。组件EP4/E32/Z16，Trainer EP16/E128/Z64单逻辑层。

| 组件实现 | forward p50 ms | forward+backward p50 ms |
|---|---:|---:|
| native_torch | 29.438 | 76.468 |
| native_triton_prefix | 18.188 | 43.754 |
| push_direct_gradient | 12.823 | 32.471 |
| pull_direct_gradient | 11.903 | 31.527 |

| 单层Trainer | p50 s | p90 s | tokens/s |
|---|---:|---:|---:|
| native | 0.473005 | 0.490029 | 69276.2 |
| fast-native | 0.420678 | 0.433373 | 77893.3 |
| megamoe | 0.399879 | 0.409080 | 81944.8 |

模块输出及全部梯度最大误差0；两个候选均通过连续10次更新×16rank，每个候选完整模型/FP32主参数哈希一致8320对。

详见[结构化结果](results.json)、[原始记录审计](raw-evidence-audit.json)、[历史来源说明](README.md)。原始报告以 original-summary.zh-CN.txt 保留，其中旧绝对路径不作为客户依赖。客户实际运行的软件、编译产物、门限及性能均记录在客户报告目录。
