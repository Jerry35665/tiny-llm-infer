# P3 — 图执行器与 FSM 控制流（Phase 3 ↔ M4）

> 状态：待 M4 实现。本文档先定软件设计规格与硬件映射骨架，实现完填实测。

## 1. 为什么需要图执行器

M1~M3 的 forward 是硬编码循环：算子顺序写死、中间张量各配各的显存。图执行器
（ggml 路线）把模型结构变成数据：

- 模型 = JSON 图，引擎 = 通用执行器 → 换模型不用改引擎代码
- 拓扑排序决定执行序 → 调度自由度（如 gate/up 两个投影可并行）
- memory plan：liveness 分析 → 中间张量复用 → 显存峰值大降
- 上游呼应：vLLM 的调度器、CUDA Graph 捕获，都建立在"先有图"之上

## 2. 软件设计规格（M4 实现依据）

- **图格式**：JSON 节点表 `{op, inputs, outputs, attrs}`；
  首批支持 op：linear / rmsnorm / rope / attention / swiglu / add
- **拓扑排序**：Kahn 算法（入度表 + 队列）；检测环报错
- **memory plan**：每个中间张量算 [birth, death] 区间；区间无重叠的复用同一 buffer
  （区间图分配，类似编译器的寄存器分配）
- **验收**：memory pool 峰值显存 vs 朴素分配的比值，目标省 ≥30%（24 层 Qwen 上实测）

## 3. 硬件映射

### 3.1 图执行 = 顶层 FSM 的"程序"

```mermaid
stateLR
    [*] --> IDLE
    IDLE --> FETCH_CFG: 启动
    FETCH_CFG --> DISPATCH: 图解析完成
    DISPATCH --> WAIT_READY: 发算子配置
    WAIT_READY --> RUN_OP: valid&&ready
    RUN_OP --> DISPATCH: 还有节点
    RUN_OP --> DONE: 图执行完
    DONE --> [*]
```

硬件控制器按图逐节点激活算子块；软件的拓扑排序在硬件里就是控制器的状态转移序列。

### 3.2 valid/ready 握手（AXI-Stream 风格）

算子块之间：上游 valid + 下游 ready 才传输——软件"生产者-消费者队列"的硬件对应物，
反压（backpressure）天然实现：下游算得慢，上游自动等待。

### 3.3 ping-pong buffer + DMA

memory pool 的复用在硬件 = 双 buffer 交替：DMA 搬第 N+1 块输入的同时，计算单元
处理第 N 块——计算与搬运重叠。软件里的双缓冲/prefetch 是同一个思想的两种实现。

## 4. M5 呼应：软图 → 硬图

M5 的 CUDA Graph 把这张软件图一次性捕获成 GPU 端固定执行计划：
- 软件图：每次执行都要拓扑遍历 + 节点调度开销
- CUDA Graph：一次提交整图，消除逐 kernel launch 开销
同一抽象横跨软件调度与硬件提交——本项目 co-design 叙事的中心。M5 实测两者差异填表。

## 5. 面试一句话

"图执行器是推理引擎的'指令集'：软件里它换来内存复用与调度自由，硬件里它就是顶层
FSM 的控制流 + valid/ready 握手 + ping-pong 重叠搬运。"
