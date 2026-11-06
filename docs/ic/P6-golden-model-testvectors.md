# P6 — Golden Model 与测试向量（Phase 6 ↔ M1）

> 状态：**向量导出已实现**（scripts/export_vectors.py）；对拍方法论见
> docs/milestone1-forward.md §6。

## 1. Golden Model 定义

- **金标准** = HuggingFace CPU fp32 前向：高精度、确定性、与训练生态同源
- **本引擎** = 被测物（DUT）的软件孪生；逐层对拍 = 软件世界的 reference model 对分
- 误差预算体系：L0 <1e-6（精确）→ 深层 <1e-4 正常 → >1e-3 必有 bug；
  第一处超差层 = 案发现场，二分定位

这套方法论（golden 对拍 + 误差预算 + 二分定位）就是 M1 的灵魂，也是将来
RTL/UVM 验证里 scoreboard 的思想预演。

## 2. 测试向量导出（已实现）

```powershell
python scripts/golden_test.py --model models/qwen2.5-0.5b --out golden
python scripts/export_vectors.py --golden golden --out vectors --fixed
```

产出 `vectors/`：

| 文件 | 内容 |
|---|---|
| `p{i}_stimulus.txt` | 输入 token id 序列（每行一个）|
| `p{i}_expected_L{k}.txt` | fp32 十进制逐层期望输出 |
| `p{i}_expected_L{k}.vec` | Q1.15 定点十六进制（`--fixed` 时生成）|
| `manifest.json` | 格式自述 + 张量形状 + 饱和 clip 统计 |

## 3. Q1.15 定点格式（与 P4 呼应）

- 1 符号位 + 15 小数位：范围 [-1, 1)，步进 2⁻¹⁵ ≈ 3.05e-5
- 编码：`v_int = clamp(round(v × 32767), -32768, 32767)`，负数补码 4 位十六进制
- **饱和逻辑**：超界值 clip 而非报错——正是 P4 要设计的 saturation 单元的
  软件演示；被 clip 的数量统计在 manifest 里（量化前先看分布，这就是校准的雏形）

## 4. RTL 对接姿势（概念，不写 RTL）

```
stimulus.txt → DUT（未来的 RTL/仿真器）→ 定点输出
                                          ↔ 逐 token 对比 .vec 期望
             （fp32 golden 引擎 = 行为级参考，.vec = bit-exact 目标）
```

两级验证：行为级对拍容差（1e-3）+ 定点位级比对（bit-exact）——这正是工业界
"float golden → fixed RTL"的标准流程缩影。

## 5. 面试一句话

"我把验证过的 FP32 引擎当 Golden Model，自动导出 Q1.15 定点激励/期望向量对，
饱和 clip 统计顺带完成了校准预分析——为 RTL bit-exact 验证铺好了数据链路。"
