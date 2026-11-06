# P4 — INT8 量化与饱和逻辑（Phase 4 ↔ M6）

> 状态：待 M6 实现。先定量化数学与硬件映射骨架，实现完填三张实测表。

## 1. PTQ 数学

- 对称量化：`scale = absmax(W) / 127`；`W_q = round(W / scale)`；反量化 `W ≈ W_q · scale`
- **per-tensor vs per-channel**：整张量一个 scale vs 按输出通道各一个。
  LLM 权重按行分布差异大 → per-channel 显著降误差（第一版就做 per-channel）
- 激活的离群值（outlier）问题：LLM.int8() 的动机；本项目折中 = 权重 INT8、激活 FP16
  （weights-only 量化，先不做激活校准——写在 M6 范围界定里）

## 2. INT8 GEMM 与位增长（硬件核心考点）

```
8bit × 8bit → 16bit 积
K=896 次累加，最大 |acc| ≈ 896 × 127 × 127 ≈ 2^28 → 32bit 累加器
反量化前：acc >> s（移位缩放）→ clamp 到 [-127, 127]（饱和单元）
```

饱和单元 = 比较器 + 多路选择器，代价极低但防住灾难性溢出——"硬件安全带"。
对照软件世界：INT8 溢出就是静默的错误输出，饱和逻辑是把 NaN/overflow 扼杀在源头。

## 3. PPA 量级估算（引 Horowitz ISSCC 2014，数值自查填入）

| 单元 | 45nm 能耗量级 | 面积直觉 |
|---|---|---|
| INT8 MAC | ~0.2 pJ | 1× |
| FP32 MAC | ~3.7 pJ | 数倍于 INT8 |
| 访 DRAM（64B） | ~1000× 片上访存 | — |

收益来源两条：每个 MAC 更小更省；权重字节减 4× → 带宽需求降 4×（对 memory-bound
的 decode 是直接提速）。

## 4. 精度验收（M6 实测填表）

- **逐层对拍**：INT8 GEMM vs FP32 的逐层误差（容差放宽到 ~1e-2 相对误差，为何？
  量化误差是系统性的，不是浮点噪声——写进 notes）
- **端到端**：greedy 生成 32 token 的输出与 FP32 引擎的重合率 + top1 命中率
- 三张表：延迟（prefill/decode）、显存（nvidia-smi 进程占用）、精度（重合率）

## 5. 面试一句话

"INT8 的收益来自 MAC 更小更省 + 带宽需求 4× 降，代价是位增长管理（32bit 累加 +
移位饱和）与精度预算（per-channel scale + 离群值处理）。"
