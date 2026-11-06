# tiny-llm-infer

从零实现的轻量级 LLM 推理引擎（C++ / CUDA），目标模型 **Qwen2.5-0.5B-Instruct**，性能对标 llama.cpp。

**一句话**：不是调 API 的 AI 项目，而是把 GEMM、注意力算子、KV Cache、图执行器、continuous batching、CUDA Graph 这条"推理系统"主线完整走一遍——算子层经历（NVIDIA 实习）之上的系统层拼图。

**HW/SW Co-Design 双线**：本项目同时是 NVIDIA 实习推理框架项目（IC 提案）的**开源完整版**。每个软件决策都映射到数字 IC 前端概念（脉动阵列、FSM、定点饱和、片上存储），分析文档见 [docs/ic/](docs/ic/README.md)；FP32 引擎同时作为 Golden Model，自动导出 RTL 风格测试向量。

## 目标能力路线

| 阶段 | 能力 | IC 提案 |
|---|---|---|
| M1 | safetensors 加载 + FP32 前向，逐层与 HuggingFace 对拍 + RTL 测试向量导出 | P2/P6 |
| M2 | KV Cache + prefill/decode 分离 + 采样，命令行多轮聊天 | |
| M3 | FP16 tensor core + 算子融合 + 自研 GEMM + FlashAttention | P1/P5 |
| M4 | 图执行器：JSON DAG + 拓扑排序 + memory pool | P3 |
| M5 | continuous batching + CUDA Graph（软图 → 硬图） | |
| M6 | INT8 PTQ 量化 + 饱和逻辑 + PPA 分析 | P4 |
| M7 | 选修：PagedAttention / 双卡 TP / Triton 算子 | |

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
scripts/       模型下载、权重检查、金标准生成/对拍/测试向量导出
tests/         tli_test：与 HuggingFace 金标准对拍的可执行程序
docs/          里程碑指导文档（按顺序读）+ docs/ic/ 硬件映射分析（Co-Design track）
```

## 进度

- [x] M0 环境与仓库（docs/milestone0-setup.md）
- [ ] M1 FP32 前向 + 对拍 + 测试向量导出（docs/milestone1-forward.md，通俗版导学见 docs/milestone1-walkthrough.md）
- [ ] M2 KV Cache + 聊天
- [ ] M3 性能优化 I（FP16 / 融合 / 自研 GEMM / FlashAttention）
- [ ] M4 图执行器（P3）
- [ ] M5 批处理 + CUDA Graph
- [ ] M6 INT8 量化（P4）
- [ ] M7 选修

## 基线对比

性能数据统一记录在 [docs/benchmarks.md](docs/benchmarks.md)，每次优化后更新。
