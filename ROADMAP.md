# ROADMAP — 里程碑总计划

预估总时长 4~5 周（每天 2~4 小时的业余节奏）。每个里程碑都有明确的**验收标准**，
通过验收才算完成 —— 这也是将来面试时"这个项目做到什么程度"的答案。

## 总览

| 里程碑 | 内容 | 预估 | 核心验收 |
|---|---|---|---|
| M0 | 环境与仓库 | 0.5 天 | `tli --info` 正常输出；模型与依赖就位 |
| M1 | FP32 前向跑通 | 5~7 天 | 两条 prompt 逐层 max\|Δ\|<1e-3，logits top1 与 HF 一致 |
| M2 | KV Cache + 聊天 | 5~7 天 | 命令行多轮对话；512 上下文结果正确；报出 decode tokens/s |
| M3 | 性能优化 I | ~7 天 | FP16 + 算子融合 + 自研 GEMM，相比 M2 ≥2×，给出 vs llama.cpp 数据 |
| M4 | 批处理 + CUDA Graph | ~7 天 | batch 1/4/8 吞吐曲线；TTFT/TPOT 指标 |
| M5 | 选修（见下） | 选做 | 各自定义 |

## 各里程碑说明

### M0 — 环境与仓库（docs/milestone0-setup.md）
工具链验证、模型下载、权重结构盘点。产出：能编译的工程 + 一份"我理解的模型结构"笔记。

### M1 — FP32 前向跑通（docs/milestone1-forward.md）★ 全项目地基
safetensors 加载 → embedding → 24 层 decoder（RMSNorm/QKV/RoPE/attention/SwiGLU）→ lm_head。
**方法论比结果重要**：每实现一小步就和 HuggingFace 金标准逐层对拍。
你将学到：transformer 前向的每一个张量形状、cuBLAS 行主序约定、GPU/CPU 浮点误差的量级感。

### M2 — KV Cache + 采样 + 聊天
prefill/decode 分离；KV Cache 显存布局（自己设计，文档会列设计问题）；
temperature/top-k/top-p 采样；C++ BPE tokenizer（参考 karpathy/minbpe）；命令行聊天。
你将学到：为什么 decode 是 memory-bound、KV Cache 的显存公式、tokenization 的真实成本。

### M3 — 性能优化 I
FP16（cuBLAS tensor core 路径 + 自研 kernel 转换）、算子融合（RMSNorm+RoPE、SiLU+mul、残差）、
把你实习写的 GEMM 集成进来与 cuBLAS 同台对比、pinned memory、kernel launch 合并。
你将学到：roofline 分析、Nsight Systems/Compute 的完整工作流、数据驱动优化。

### M4 — 批处理 + CUDA Graph
连续批处理（continuous batching）调度器、CUDA Graph 捕获 decode 步骤、
TTFT/TPOT/吞吐的系统指标。你将学到：vLLM 论文里的核心思想在代码里长什么样。

### M5 — 选修（任选其一，都是简历强素材）
- **PagedAttention**：分页 KV 管理，长上下文显存碎片优化
- **INT8/FP8 量化**：per-channel 权重量化 + 校准
- **双卡 tensor parallel**（NCCL，需要第二张卡或云）
- **Triton 算子**：用 Triton 重写 2~3 个融合算子与手写 CUDA 对比

## 协作方式（你 & ZCode）

1. 你按当前里程碑文档实现，**小步 commit**（一个算子一个 commit）
2. 卡住了或写完一块 → 在 ZCode 里让我 review / debug（直接说"看看我的 attention kernel"）
3. 通过验收 → 我帮你把成果提炼成简历 bullets，然后写下一份里程碑指导文档
4. 每次性能优化后更新 docs/benchmarks.md，Nsight 结论记进 docs/notes.md
5. **每完成一步就同步 GitHub**：更新 README 进度勾选框（和 benchmarks），
   然后跑 `sync.bat "commit message"`（自动 add + commit + push）。
   一次性前提：装好 gh 并登录（docs/milestone0-setup.md 第 0.6 节）。

## 简历 bullets 模板（做完往里填数）

- 从零实现 C++/CUDA 轻量级 LLM 推理引擎（Qwen2.5，xx kLOC）：支持 GQA/RoPE/SwiGLU
  全算子、KV Cache、continuous batching 与 CUDA Graph，FP16 decode 吞吐 xx tok/s
- 实现 PagedAttention 式分页 KV 管理，长序列显存占用降低 xx%；CUDA Graph 消除
  launch 开销，小 batch decode 提速 xx%
- 基于 Nsight Systems/Compute 全链路 profiling，算子融合 + FP16 tensor core 使
  prefill/decode 分别加速 xx%/xx%，达到 llama.cpp CUDA 后端的 xx%
- 自研 GEMM 集成进引擎：FP16 达 cuBLAS xx%，并通过 layer-wise 数值对拍将
  误差控制在 xx 以内

## 面试考点清单（做完每条你都能脱稿讲）

- RMSNorm 与 LayerNorm 的区别，为什么 LLM 都用 RMSNorm
- RoPE 的数学原理；interleaved 与 rotate-half 两种实现约定
- GQA 为什么省显存，省了多少（算给我看）
- 为什么 prefill 是 compute-bound、decode 是 memory-bound
- KV Cache 显存公式；batch 增大时显存怎么涨
- PagedAttention 解决什么问题；continuous batching 和 static batching 区别
- CUDA Graph 为什么快；什么场景收益最大
- 如何用 Nsight 定位一个 kernel 的瓶颈（给出真实案例）
- 量化的基本原理；per-channel 和 per-tensor 的差别
