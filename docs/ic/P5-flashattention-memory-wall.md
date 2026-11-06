# P5 — FlashAttention 与内存墙（Phase 5 ↔ M3）

> 状态：骨架。M3 实现 FlashAttention 后填 Nsight 实测表。

## 1. 内存墙：朴素 attention 的访存灾难

朴素实现的中间量：S = QK^T（N×N）写 HBM → 读回做 softmax → 读回乘 V。
N=4096、14 头时 S 矩阵 ≈ 14 × 4096² × 4B ≈ 0.9 GB，三进三出 ≈ 5.4 GB 流量。
算术强度被访存淹没——典型 memory-bound（对照 M1 朴素 attention kernel 就是这个形态）。

## 2. 软件方案（M3 实现规格）

- 分块：Q/K/V 按 tile 切进 shared memory（4060 每 SM 100KB）
- **online softmax**：维护 running max `m` 和 running sum `l`，新块到达时增量修正
  `m_new = max(m, rowmax(S_i))`，历史部分按 exp(m_old − m_new) 缩放——
  不需要整行分数就位也能正确归一化
- 只写 O(N) 的最终输出，N×N 矩阵从未离开片上

## 3. 硬件映射

### 3.1 Scratchpad（SRAM）带宽与能耗
片上 SRAM：带宽 ~10× HBM、能耗 ~1/100。"分块"的本质就是把数据钉在片上
（与 P1 的 tiling→片上 buffer 是同一原理在 attention 场景的再现）。

### 3.2 Streaming Pipeline（硬件的算子融合）
GEMM → softmax → GEMM 用硬件 FIFO 直连，中间量不出片：

```
QK^T MAC 阵列 ──FIFO──► 比较器树/exp 单元 ──FIFO──► PV MAC 阵列
```

软件的 kernel fusion ↔ 硬件的 streaming pipeline：同一个物理诉求
（中间数据不落慢存储）的两种实现。

## 4. Nsight 实测表（M3 填）

| 指标 | 朴素 attention | FlashAttention | 提升 |
|---|---|---|---|
| DRAM bytes（读+写） | | | |
| L2 sectors | | | |
| kernel 时间 | | | |
| prefill 端到端 | | | |
| 显存峰值 | | | |

## 5. 面试一句话

"FlashAttention 不是更快的 softmax，而是把 attention 从'N² 中间量过 HBM'改成
'分块流水不出片'——软件的算子融合在硬件里就是 streaming pipeline。"
