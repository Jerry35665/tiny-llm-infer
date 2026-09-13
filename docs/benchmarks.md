# 性能数据记录

规则：**每次优化后必须新增一行**，数据不好看也要记（负结果同样是面试素材）。
测量要固定条件：同 prompt、同 seq、同 batch、独占 GPU、跑 3 次取中位数。

## 环境

| 项 | 值 |
|---|---|
| GPU | RTX 4060 Laptop 8GB (sm_89) |
| 驱动 | 576.52 |
| CUDA | 13.3 |
| 模型 | Qwen2.5-0.5B-Instruct |
| 精度 | FP32（M1/M2）→ FP16（M3+）|
| 电源模式 | 记录一下（笔记本插电/省电影响巨大）|

## 指标定义

- **prefill tok/s**：处理输入 prompt 的吞吐
- **decode tok/s**：自回归生成速度（单序列 = 1/TPOT）
- **TTFT**：首 token 延迟；**TPOT**：每 token 生成延迟
- 显存：`nvidia-smi` 里进程占用

## 数据表

| 日期 | commit | 配置 | prefill tok/s | decode tok/s | 显存 | 备注 |
|---|---|---|---|---|---|---|
| | | M1 fp32 朴素 | | | | 首个基线 |
| | | M2 +KV cache | | | | |
| | | M3 fp16+融合 | | | | |
| | | M3 vs llama.cpp | | | | 基线对比 |
| | | M4 batch=4/8 | | | | 吞吐曲线 |

## 对比基线（M3 时再跑）

用 llama.cpp 官方 Windows CUDA 预编译包，同一模型转 GGUF（q8_0 或 f16），
`-ngl 99` 全部 offload，同 prompt 同生成长度，记录其 decode tok/s 作为 100% 基线。
