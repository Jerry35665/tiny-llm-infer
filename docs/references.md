# 参考资料清单（按需读，不要一开始全读）

## M1 前向
- **karpathy/llama2.c**（GitHub）— 单文件纯 C 的 Llama2 前向，~700 行。
  和本项目 M1 几乎同构，卡住时看它的对应函数（但自己先想 10 分钟）。
- karpathy/minbpe — 最小 BPE 实现，M2 写 tokenizer 前读
- HF 文档：Qwen2.5-0.5B-Instruct 模型页 + transformers 里 modeling_qwen2.py 源码
  （终极对照物，前向有分歧就看它）
- safetensors 格式说明（huggingface/safetensors 仓库 README 的 format 一节）
- cuBLAS 文档：cublasSgemm 的参数含义（重点 lda/ldb/ldc 和 OP_T/OP_N）

## M2 KV Cache / 采样
- vLLM 论文：*Efficient Memory Management for Large Language Model Serving with
  PagedAttention*（SOSP '23）— 先读 §3 就够
- GQA 论文：*GQA: Training Generalized Multi-Query Transformer Language Models
  from Multi-Head Checkpoints*（2023）
- llama.cpp 的 kv cache 实现（ggml-cuda.cu 相关部分，M2 完成后再对照）

## M3 性能优化
- Simon Boehm: *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*
  （博客，GEMM 优化的最佳单篇教程）
- FlashAttention 论文（2022）与 FlashAttention-2（2023）
- CUDA C++ Programming Guide：内存层级、shared memory bank conflict、warp 原语
- Nsight Systems / Nsight Compute 官方文档的 Getting Started

## M4 图执行器（P3）
- llama.cpp 的 ggml compute graph（ggml.c 的 ggml_build_forward_expand / ggml_graph_compute）
  —— "硬编码循环 vs 计算图"的工业界答案
- 拓扑排序（Kahn）与区间图分配教材章节 —— memory plan 的理论基础

## M5 批处理 + CUDA Graph
- vLLM 论文剩余部分（scheduler）；Orca 论文（continuous batching 起源，可选）
- CUDA Graph 官方文档（cudaGraph / Stream Capture）

## M6 INT8 量化（P4）
- LLM.int8() 论文（Dettmers et al. 2022）—— 理解激活离群值对量化的影响
- Horowitz, "Computing's Energy Problem" (ISSCC 2014) —— PPA 能耗经典数据表
- TensorRT / ONNX Runtime 的 INT8 文档 —— 工程实践参照

## 通用
- Qwen2.5 技术报告（arXiv 2412.15115）— 了解模型家族全貌
- 《深入浅出 CUDA》类中文教程仅作入门，以官方文档为准
