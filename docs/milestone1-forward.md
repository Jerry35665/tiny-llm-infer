# M1 — FP32 前向跑通，与 HuggingFace 逐层对拍（5~7 天）

> 这是全项目的地基。目标不是"能出字"，而是**每一个中间张量都被证明正确**。
> 这套逐层对拍的方法论本身就会成为你面试里的亮点。

## 1. 目标与验收

**目标**：加载 Qwen2.5-0.5B safetensors，用纯 CUDA + cuBLAS 跑通单序列 FP32 前向。

**验收**：
- [ ] `tli_test models/qwen2.5-0.5b golden out` 两条 prompt 逐层 max|Δ| < 1e-3
- [ ] logits 的 top1/top5 与 HuggingFace 金标准一致
- [ ] `python scripts/export_vectors.py --golden golden --out vectors --fixed` 产出 vectors/ 测试向量（IC 提案 P6 交付）
- [ ] 记录第一版端到端耗时（是"朴素但正确"的基线，M3 来打败它）
- [ ] 全过程有小步 commit

## 2. 数据流全景（你要实现的东西）

```
tokens[seq]
   │ embedding 查表
   ▼
x [seq, 896] ──────────────────────── 24 层循环 ───────────────────────┐
   │                                                                   │
   │  ┌─ RMSNorm ──► QKV 投影(GEMM+bias) ──► RoPE(q,k) ─┐              │
   │  │                                                ▼              │
   │  │            朴素 attention（causal + GQA）──► o_proj ──► (+)残差│
   │  │                                                               │
   │  └─ RMSNorm ──► gate/up 投影 ──► silu(gate)*up ──► down ──► (+)残差
   ▼                                                                   │
   └───────────────────────────────────────────────────────────────────┘
   │ final RMSNorm
   ▼
h_last [896] ──► lm_head(=embed_tokens 转置 GEMM) ──► logits [151936]
```

## 3. 模型结构（用 inspect_model.py 的输出核实！）

Qwen2.5-0.5B-Instruct 的公开配置（**以下载到的 config.json 为准**）：

| 参数 | 值 | 备注 |
|---|---|---|
| hidden_size | 896 | |
| num_hidden_layers | 24 | |
| num_attention_heads | 14 | head_dim = 64 |
| num_key_value_heads | 2 | GQA：1 个 KV head 服务 7 个 Q head |
| intermediate_size | 4864 | |
| vocab_size | 151936 | embed_tokens [151936, 896] ≈ 544MB(FP32) |
| rms_norm_eps | 1e-6 | |
| rope_theta | 1e6 | |
| tie_word_embeddings | true | 没有 lm_head.weight，复用 embed_tokens |
| attention_bias | 看张量清单 | 小尺寸 Qwen 通常 q/k/v 有 bias，o_proj 没有 |

每层的权重（`i` = 0..23）：
```
model.layers.{i}.input_layernorm.weight         [896]
model.layers.{i}.self_attn.q_proj.weight        [896, 896]   (+bias [896]?)
model.layers.{i}.self_attn.k_proj.weight        [128, 896]   (+bias [128]?)
model.layers.{i}.self_attn.v_proj.weight        [128, 896]   (+bias [128]?)
model.layers.{i}.self_attn.o_proj.weight        [896, 896]
model.layers.{i}.post_attention_layernorm.weight [896]
model.layers.{i}.mlp.gate_proj.weight           [4864, 896]
model.layers.{i}.mlp.up_proj.weight             [4864, 896]
model.layers.{i}.mlp.down_proj.weight           [896, 4864]
model.norm.weight                               [896]        (final RMSNorm)
model.embed_tokens.weight                       [151936, 896]
```
⚠️ safetensors 里的形状是 PyTorch 线性层约定 `[out_features, in_features]`，
即 `y = x @ W^T`。这个 `^T` 和 cuBLAS 的列主序缠在一起是 M1 第一大坑（见 4.3）。

## 4. 逐算子规范

### 4.1 Embedding
`x[r, :] = table[tokens[r], :]`，纯查表。完成后对拍：与 golden `L00` 比。

### 4.2 RMSNorm
```
out = x / sqrt( mean(x^2) + eps ) * w      eps = 1e-6
```
- 均值是对整个 hidden 维度（896 个数）求的
- HF 实现内部先转 FP32 再算（我们全程 FP32，无此问题，但要记住 M3 转 FP16 时的坑）
- `kernels/rmsnorm.cu` 已给参考实现，直接用

### 4.3 QKV 投影 —— cuBLAS 行主序大坑（先读懂再写）
线性层是 `y = x @ W^T`（行主序，x:[n,h]，W:[o,h]，y:[n,o]）。
cuBLAS 是列主序。行主序的 `A(m×k)` 在内存里等于列主序的 `A^T(k×m)`，代入化简后
**一次都不用显式转置**，直接这样调：

```cpp
// y[n,o] = x[n,h] @ W[o,h]^T   （全部行主序）
// 等价的 cuBLAS 调用：
//   C^T(o×n) = W(o×h) · x^T(h×n)
//   内存里 W 是列主序的 W^T(h×o, lda=h) -> 取 OP_T
//   内存里 x 是列主序的 x^T(h×n, ldb=h) -> 取 OP_N
cublasSgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            o, n, h,
            &alpha,          // 1.0f
            W, h,            // A(=W 存储)，lda = h
            x, h,            // B(=x 存储)，ldb = h
            &beta,           // 0.0f
            y, o);           // C，ldc = o
```
**推导过程亲手在纸上过一遍**（面试真问）。验证方法：拿 2×3 的矩阵手算对比。

bias（若有）：GEMM 之后逐行加，或者写个小 kernel 把 bias 广播加上（也可并入 4.6 的残差融合，M1 先分开写）。

### 4.4 RoPE（rotate-half 约定，与 Llama 相同）
```
inv_freq[j] = theta^(-2j/64)，j = 0..31        theta = 1e6
angle[p, j] = p * inv_freq[j]                   p = 0..seq-1
配对 (j, j+32)：
  q'_j      =  q_j * cos - q_{j+32} * sin
  q'_{j+32} = q_{j+32} * cos + q_j * sin
```
- k 同样处理；v 不转
- 建议：cos/sin 两张表 `[max_seq, 32]` 在 Engine 构造时离线算好传到显存
- 只对 attention 内的 q/k 做，且 Qwen 全部层用同一份 RoPE（无 layer 偏移）

### 4.5 朴素 attention（causal + GQA）
对每个 (query 位置 i，q head h)：
```
kv_head g = h / 7                            // 14 个 q head → 2 个 kv head
s_j = dot(q[i,h,:], k[j,g,:]) / sqrt(64)     // j = 0..i
s_j = -inf                                   // j > i（causal）
p = softmax(s over j<=i)
out[i,h,:] = Σ_j p_j * v[j,g,:]
```
建议 grid=(seq, 14)，block 128 线程；分数存 shared memory，两遍 softmax（先求 max 再求 exp 和）。
M1 不要求在线 softmax（FlashAttention 是 M3 的故事）。

### 4.6 o_proj + 残差
`x = x + attn_out @ Wo^T`。注意 o_proj 的输入是 14 个 head 拼起来的 [seq, 896]
（head 排布是 `h*64` 连续，与 Qwen 权重一一对应，不用重排）。

### 4.7 MLP（SwiGLU）
```
h_gate = x @ Wg^T        [n, 4864]
h_up   = x @ Wu^T        [n, 4864]
h_mlp  = silu(h_gate) * h_up          silu(v) = v * sigmoid(v)
x = x + h_mlp @ Wd^T     [n, 896]
```
gate/up 都没有 bias。`swiglu` kernel 是纯逐元素操作，几分钟写完。

### 4.8 final norm + lm_head
最后一层之后：RMSNorm（用 `model.norm.weight`）→ `logits = h @ embed_tokens^T`。
- tie_word_embeddings：lm_head 就是 embed_tokens 的转置 GEMM，同一个指针
- 只需要**最后一个位置**的 logits（M1 只取 h 的最后一行去做 GEMM 即可，[1,896]@[896,151936]）

## 5. 推荐实现顺序（每步对拍，别攒大招）

| 步骤 | 内容 | 对拍点 |
|---|---|---|
| S0 | `GpuWeights::load_safetensors` | 打印形状/数值抽样，和 inspect_model.py 输出对 |
| S1 | 缓冲区 + embedding | 层 0 对拍（应该精确相等，<1e-6）|
| S2 | RMSNorm 接入（已有参考实现）| 单层内先对拍 norm 后的中间量 |
| S3 | QKV GEMM（cuBLAS）| 对拍 q/k/v 投影输出（可临时只跑 1 层）|
| S4 | RoPE | 同上 |
| S5 | attention | 同上，重点查 causal mask 和 GQA 映射 |
| S6 | o_proj + 残差 + MLP | 整层对拍，1 层通过后再放开 24 层 |
| S7 | final norm + lm_head + logits | 全模型对拍 |

**调试期建议只跑第 1 层**（加个环境变量或参数控制层数），1 层全对了再跑 24 层 ——
否则 24 层的误差混在一起根本没法定位。

## 6. 对拍工作流

```powershell
# 一次性生成金标准（CPU fp32，确定性）
python scripts/golden_test.py --model models/qwen2.5-0.5b --out golden

# 实现 tli_test 后（把每层 hidden dump 到 out/）
.\build\Release\tli_test.exe models\qwen2.5-0.5b golden out

# 逐层对比，输出每层 max|delta| 和第一处超差位置
python scripts/compare_golden.py --golden golden --mine out

# 通过后：导出 RTL 风格测试向量（IC 提案 P6）
python scripts/export_vectors.py --golden golden --out vectors --fixed
```

**层的下标约定**（两边必须一致，已在 golden_test.py 里对齐）：
- `L00` = embedding 输出
- `L01..L23` = 第 1..23 个 decoder 层的输出（0 起数）
- `L24` = 第 24 层输出 + final RMSNorm 之后
- 这与 HF `output_hidden_states` 的下标一一对应

**误差判读**（GPU fp32 vs CPU fp32）：
- L00/L01：<1e-6（精确）
- 深层：<1e-4 正常，1e-4~1e-3 可接受（求和顺序差异），>1e-3 有 bug
- 定位方法：第一处超差的层 k → 该层四个算子（attn 链 / mlp 链）里二分：
  把 RMSNorm 后的 x、QKV 输出、RoPE 输出、attn 输出、MLP 输出分别 dump 出来，
  和 HF 里对应的中间量比（golden_test.py 可加 output_attentions 或自改）
- logits 误差天然偏大（vocab 大、数值大），以 top1/top5 一致为准

## 7. 常见 bug 清单（90% 的时间会栽在这）

1. **cuBLAS 行主序**：结果"大体对但数值错" → 99% 是转置/lda 问题（见 4.3）
2. RoPE 配对用成了相邻对 (2j, 2j+1)（那是 GPT-J 约定，Qwen 是 rotate-half）
3. GQA 映射写反：q head h 应该用 kv head `h / (H/Kv)` = `h/7`
4. causal mask 写成 `j >= i` 或忘了把被 mask 位置从 softmax 分母排除
5. bias 加错地方（q/k/v 有 bias，gate/up/down/o_proj 没有 —— 以实际张量清单为准）
6. RMSNorm 的 eps 加在 mean 之前还是之后（之后：`mean + eps`）
7. lm_head 忘了是 tie 的，去找不存在的 lm_head.weight
8. 层与层的 norm 用错：attention 前用 `input_layernorm`，MLP 前用 `post_attention_layernorm`
9. 残差结构搞错：正确的是 `x = x + attn(ln1(x))`，不是 `x = ln1(x) + attn(...)`
10. dump 时忘了 `cudaDeviceSynchronize`/stream 同步，拷出来的是旧数据

## 8. 完成后

1. `benchmarks.md` 记录第一版性能（哪怕很慢）：prompt 处理耗时 + 单次前向耗时
2. commit 规范示例：`feat(weights): safetensors loader` / `feat(model): attention naive`
3. 在 ZCode 里喊我 review，通过后我出 M2 文档（KV Cache + 采样 + 聊天）
