# P1 — GEMM 优化与脉动阵列映射（Phase 1 ↔ M3）

> 状态：骨架。M3 用开源代码实测填表；实习数据（CPU SIMD/分块 3×、GPU 10×、cuBLAS 70%）
> 可先填入并标注"实习实测"。

## 1. 软件侧优化阶梯（M3 实测填表）

| 版本 | 优化点 | GFLOPS | vs cuBLAS | L1 miss 率 |
|---|---|---|---|---|
| v0 | naive O(N³) 三重循环（基线） | | | |
| v1 | B 转置 / 连续访存 | | | |
| v2 | 循环展开 4×/8× | | | |
| v3 | 分块 tiling（块大小扫参） | | | |
| v4 | SIMD（AVX2 FMA） | | | |
| v5 | v3+v4 组合 | | | |

测量：1000×1000 与 4096×4096 两档，perf/VTune 或 Nsight；3 次取中位数。

## 2. 为什么有效（物理本质）

- **连续访存**：cache line 64B；stride 访存让每次读浪费 7/8 带宽
- **分块**：把 working set 塞进 L1/L2 —— 对应硬件"把数据放到离计算单元最近的地方"
- **SIMD/展开**：一指令多 MAC —— 对应硬件的空间并行（同一拍多个乘法器同时工作）

## 3. 硬件映射

### 3.1 SIMD → MAC 阵列（空间并行）
软件 8-lane AVX 乘加 ↔ 硬件一拍 8 个乘法器并行。区别：软件靠编译器调度寄存器，
硬件数据流是布线固定的——这就是"软件调优化顺序、硬件调 dataflow"。

### 3.2 tiling → Register File / 片上 buffer 的数据复用
分块大小 ≈ cache 容量的配比问题；硬件里就是 scratchpad(SRAM) 容量与 MAC 阵列
输入带宽的配比。dataflow 三分类（对照自己的 kernel 分类）：

| dataflow | 驻留数据 | 极端优势 | 代表 |
|---|---|---|---|
| Weight Stationary | 权重 | 权重复用最大 | TPU v1 |
| Output Stationary | 部分和 | 累加不搬运 | 许多 DSA |
| Row Stationary | 行数据 | 三者平衡 | Eyeriss |

物理根据：一次 DRAM 访问能耗 ≈ 片上 SRAM 的 100× 以上、寄存器的 1000× 量级
（Horowitz ISSCC 2014 能耗表——M6 的 PPA 也引用它，数值自己去查证填入）。

### 3.3 脉动阵列
2D PE 网格，数据节拍式流向四邻（"脉动"得名于心脏泵血）：
- N×N×N 乘加 O(N) 拍完成——数据复用率最大化的极端形态
- 代价：N² 个 PE 的面积、dataflow 固定不灵活
- 对照：TPU v1 = 256×256 权重固定脉动阵列；NVIDIA tensor core = 4×4×4 小块 MAC，
  调度更灵活——两种"复用 vs 灵活"的取舍

## 4. Roofline 分析（填实测）

- RTX 4060 Laptop：FP32 峰值 ~11 TFLOPS、带宽 256 GB/s → 脊点 AI ≈ 43 FLOP/B
- 现代 CPU：AVX2 峰值百 GFLOPS 级、DDR 带宽 ~50 GB/s → 脊点 ≈ 4 FLOP/B
- GEMM 理论 AI = O(N)，天然 compute-bound；但实现差的 GEMM AI < 1 ——
  **优化前先测 AI，才知道自己在屋顶的哪一侧、优化访存还是优化计算**

## 5. 面试一句话

"GEMM 优化的本质是把数据组织到离计算最近的存储层级；软件的 tiling/SIMD 在硬件里
分别对应 scratchpad 配比与 MAC 空间并行，极端形态就是脉动阵列。"
