# IC 映射文档（HW/SW Co-Design Track）

> 把这个推理引擎里的每个软件决策映射到数字 IC 前端概念——面试芯片设计岗时的
> "第二语言"：软件效率问题在硬件里对应什么结构、什么代价。
>
> 规范：每完成一个里程碑，更新对应 P 文档——**实测数据 + 自己的推导**。
> ⚠️ 实习提案 PDF 与任何公司内部资料不要放进本公开仓库（NDA 风险）；
> 本文档全部用自己的话写。

## 文档索引

| 文档 | 提案 Phase | 对应里程碑 | 状态 |
|---|---|---|---|
| [P1-gemm-systolic.md](P1-gemm-systolic.md) | Phase 1 | M3 | 骨架就绪，待 M3 实测 |
| [P2-nonlinear-operators.md](P2-nonlinear-operators.md) | Phase 2 | M1 | 骨架就绪，随 M1 补充 |
| [P3-graph-fsm.md](P3-graph-fsm.md) | Phase 3 | M4 | 待 M4 实现 |
| [P4-int8-saturation.md](P4-int8-saturation.md) | Phase 4 | M6 | 待 M6 实现 |
| [P5-flashattention-memory-wall.md](P5-flashattention-memory-wall.md) | Phase 5 | M3 | 骨架就绪，待 M3 实测 |
| [P6-golden-model-testvectors.md](P6-golden-model-testvectors.md) | Phase 6 | M1 | 向量导出已实现 |

## 统一写作模板

1. **软件侧做了什么**（一句话 + 关键数据）
2. **为什么**（瓶颈的物理本质：访存 / 算力 / 控制流）
3. **硬件对应什么**（结构图：mermaid 或手绘拍照）
4. **量化对比**（before/after 表格，或 PPA 估算）
5. **面试一句话**（30 秒讲清的版本）

## 为什么这个 track 值得写

面试芯片岗时，"我用 CUDA 写过 GEMM"只是入场券；"我知道我的 tiling 对应硬件里
scratchpad 与 MAC 阵列的带宽配比"才是差异化。软件优化和硬件结构是同一个物理问题
（数据放在哪、怎么流动）的两种解法——这六篇文档就是把这句话变成证据。
