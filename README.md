# tiny-llm-infer

从零实现的轻量级 LLM 推理引擎（C++ / CUDA），目标模型 **Qwen2.5-0.5B-Instruct**，性能对标 llama.cpp。

**一句话**：不是调 API 的 AI 项目，而是把 GEMM、注意力算子、KV Cache、continuous batching、CUDA Graph 这条"推理系统"主线完整走一遍 —— 算子层经历（NVIDIA 实习）之上的系统层拼图。

## 目标能力路线

| 阶段 | 能力 |
|---|---|
| M1 | safetensors 加载 + FP32 前向，逐层与 HuggingFace 对拍 |
| M2 | KV Cache + prefill/decode 分离 + 采样，命令行多轮聊天 |
| M3 | FP16 tensor core + 算子融合 + 自研 GEMM 集成 |
| M4 | continuous batching + CUDA Graph，batch 吞吐曲线 |
| M5 | 选修：PagedAttention / INT8 量化 / 双卡 TP / Triton 算子 |

里程碑详情与验收标准见 [ROADMAP.md](ROADMAP.md)。

## 构建（Windows + CUDA）

```powershell
python scripts/fetch_deps.py
build.bat                      # 内部: vcvars64 + VS自带cmake/ninja
.\build\tli.exe --info
```

> 前提：NVIDIA 驱动 ≥ 580（CUDA 13.3 要求）。详见 docs/milestone0-setup.md。

## 目录结构

```
include/tli/   模块接口（config / weights / model / sampler / kernels）
src/           实现：config 已完成，weights/model/sampler 是 TODO 骨架
kernels/       CUDA kernel：rmsnorm.cu 是参考实现，其余由你实现
scripts/       模型下载、权重检查、金标准生成与逐层对拍
tests/         tli_test：与 HuggingFace 金标准对拍的可执行程序
docs/          里程碑指导文档（按顺序读）
```

## 进度

- [ ] M0 环境与仓库（docs/milestone0-setup.md）
- [ ] M1 FP32 前向跑通（docs/milestone1-forward.md）
- [ ] M2 KV Cache + 聊天
- [ ] M3 性能优化 I（FP16 / 融合 / GEMM）
- [ ] M4 批处理 + CUDA Graph
- [ ] M5 选修

## 基线对比

性能数据统一记录在 [docs/benchmarks.md](docs/benchmarks.md)，每次优化后更新。
