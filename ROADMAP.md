# ROADMAP v2 — 里程碑总计划（HW/SW Co-Design 双线版）

> **定位**：本项目是 NVIDIA 实习推理框架项目（IC 提案 Phase 1/2/5）的**开源完整版**——
> 补齐实习未覆盖的系统层（图执行器、KV Cache、batching），并把每个软件决策映射到
> 数字 IC 前端概念（脉动阵列 / FSM / 定点饱和 / 片上存储）。
> 双简历线：**GPU 系统/推理优化岗 + 芯片设计/DSA 岗**。
>
> ⚠️ 实习提案原文与公司内部资料不要放进本公开仓库；docs/ic/ 全部用自己的话写分析。

## 提案 Phase ↔ 里程碑映射

| IC 提案 Phase | 对应里程碑 | 交付物 | IC 映射文档 |
|---|---|---|---|
| P1 CPU GEMM + 脉动阵列映射 | M3（实习已做，开源重做） | 自研 GEMM vs cuBLAS 实测表 | docs/ic/P1-gemm-systolic.md |
| P2 注意力算子 + 非线性硬件化 | M1 | 1e-3 对拍通过的算子库 | docs/ic/P2-nonlinear-operators.md |
| P3 图执行器 + FSM 控制流 | **M4 ★新增** | JSON DAG + 拓扑排序 + memory pool | docs/ic/P3-graph-fsm.md |
| P4 INT8 量化 + 饱和逻辑 | **M6 ★转正** | INT8 GEMM + PPA 分析 | docs/ic/P4-int8-saturation.md |
| P5 FlashAttention + memory wall | M3 | Nsight 显存事务对比报告 | docs/ic/P5-flashattention-memory-wall.md |
| P6 Golden model + 测试向量 | M1 | 逐层对拍 + vectors/ 导出 | docs/ic/P6-golden-model-testvectors.md |

## 总览

预估总时长 5~6 周（每天 2~4 小时业余节奏）。每个里程碑有明确**验收标准**。

| 里程碑 | 内容 | 提案 | 预估 | 核心验收 |
|---|---|---|---|---|
| M0 ✅ | 环境与仓库 | — | 0.5 天 | `tli --info` 正常；模型就位 |
| M1 | FP32 前向 + 金标准对拍 + 测试向量导出 | P2/P6 | 5~7 天 | 逐层 max\|Δ\|<1e-3；logits top1 命中；vectors/ 产出 |
| M2 | KV Cache + 采样 + 聊天 | — | 5~7 天 | 命令行多轮对话；报出 decode tok/s |
| M3 | FP16 + 算子融合 + 自研 GEMM + FlashAttention | P1/P5 | ~7 天 | ≥2× vs M2；vs llama.cpp 数据；Nsight 报告 |
| M4 | **图执行器**：JSON DAG + 拓扑排序 + memory pool | P3 | ~5 天 | 内存池峰值显存省 ≥30%；顶层 FSM 状态图 |
| M5 | 批处理 + CUDA Graph（软图 → 硬图） | — | ~7 天 | batch 1/4/8 吞吐曲线；软图 vs 硬图对比 |
| M6 | **INT8 PTQ 量化** + 饱和逻辑 + PPA 分析 | P4 | ~5 天 | INT8 吞吐/显存/精度三张表 |
| M7 | 选修：PagedAttention / 双卡 TP / Triton 算子 | — | 选做 | 各自定义 |

### M1 — FP32 前向跑通（docs/milestone1-forward.md）★ 全项目地基
safetensors 加载 → 24 层 decoder → lm_head。方法论：每小步与 HF 金标准逐层对拍。
新增交付（P6 后半）：`tli_test` 对拍通过后，`scripts/export_vectors.py` 把金标准导出为
RTL 风格测试向量（vectors/，含 Q1.15 定点 + 饱和统计）。

### M2 — KV Cache + 采样 + 聊天
prefill/decode 分离；KV Cache 显存布局设计；temperature/top-k/top-p；C++ BPE；聊天。
学：为什么 decode 是 memory-bound、KV 显存公式、GQA 省 7× 的账。

### M3 — 性能优化 I（P1 开源重做 + P5）
FP16 tensor core、算子融合、**把实习的自研 GEMM 开源重写**并与 cuBLAS 同台、
FlashAttention（分块 online softmax）+ Nsight 显存事务数对比。
IC 映射：tiling→片上 buffer 复用、SIMD→MAC 阵列（P1 文档）、融合→streaming pipeline（P5 文档）。

### M4 — 图执行器（P3）★ 新增
模型结构变成数据：JSON 节点表 → DAG → Kahn 拓扑排序 → liveness 内存池复用。
这是"硬编码循环"到"真正推理引擎"的分水岭，也是 M5 CUDA Graph 的前置
（同一张图：软件拓扑执行 vs 一次捕获成硬件执行计划——对比本身就是 co-design 叙事）。

### M5 — 批处理 + CUDA Graph（原 M4）
连续批处理调度器、CUDA Graph 捕获、TTFT/TPOT/吞吐指标。

### M6 — INT8 量化（P4）★ 转正
PTQ（scale/zero-point，per-channel）、INT8 GEMM kernel、位增长与移位饱和、
精度对拍 + 端到端命中率、PPA 量级估算（引 Horowitz ISSCC'14 能耗表）。

### M7 — 选修（原 M5，INT8 已移出）
PagedAttention / 双卡 tensor parallel / Triton 算子。

## IC 映射文档写作规范（docs/ic/）

1. 每完成一个里程碑，更新对应 P 文档：**实测数据 + 自己的推导**
2. 统一模板：软件做了什么 → 为什么（瓶颈物理本质）→ 硬件对应什么（图）→ 量化对比 → 面试一句话
3. 篇幅 1~2 页；图（mermaid/手绘照片）胜过文字
4. 实习数据可先填入并标注"实习实测"；M3/M6 用开源代码重测刷新

## 协作方式（你 & ZCode）

1. 按当前里程碑文档实现，**小步 commit**（一个算子一个 commit）
2. 卡住或写完一块 → ZCode review / debug
3. 通过验收 → 提炼简历 bullets + 更新 docs/ic/Px + 出下一份文档
4. 每次优化更新 docs/benchmarks.md；Nsight 结论记 docs/notes.md
5. **每完成一步同步 GitHub**：更新 README 勾选 → `sync.bat "commit message"`

## 简历 bullets 模板（双线，做完填数）

**GPU 系统线**：
- 从零实现 C++/CUDA 轻量级 LLM 推理引擎（Qwen2.5，xx kLOC）：GQA/RoPE/SwiGLU 全算子、
  KV Cache、图执行器、continuous batching 与 CUDA Graph，FP16 decode xx tok/s
- 实现 PagedAttention 式 KV 分页管理 + CUDA Graph，小 batch decode 提速 xx%
- Nsight 全链路 profiling，算子融合 + FP16 使 prefill/decode 加速 xx%，达 llama.cpp 的 xx%

**芯片/DSA 线**：
- 建立推理框架→数字 IC 前端系统映射：GEMM tiling→脉动阵列 dataflow、exp 的 LUT/插值
  硬件化、图执行器→顶层 FSM + valid/ready 握手、INT8 位增长与饱和逻辑，量化 PPA 收益
- 将验证过的 FP32 引擎作为 Golden Model，自动导出 Q1.15 定点测试向量，支撑 RTL
  bit-exact 验证流程

## 面试考点清单

**推理系统**：RMSNorm vs LayerNorm；RoPE 与 rotate-half；GQA 省多少（算给我看）；
prefill 为何 compute-bound、decode 为何 memory-bound；KV 显存公式；PagedAttention 解决
什么；continuous vs static batching；CUDA Graph 为何快；Nsight 定位瓶颈的真实案例。

**IC/体系结构**：脉动阵列为什么"脉动"；weight/output stationary 区别；LUT+插值 vs
CORDIC 取舍；减最大值→比较器树对 T_clk 的影响；valid/ready 握手与反压；8×8 乘法位增长
与饱和；INT8 相对 FP32 的 PPA 收益来源；内存池/liveness 对应硬件的 ping-pong buffer。
