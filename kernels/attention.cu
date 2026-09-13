// TODO(M1): 朴素 attention（无 KV Cache，M2 再改造）
//
// 输入（已过 RoPE）：q [seq, 14, 64]，k/v [seq, 2, 64]
// 输出：attn_out [seq, 14, 64]（之后 reshape 成 [seq, 896] 进 o_proj）
//
// 对每个 query 位置 i、每个 q head h：
//   kv_head g = h / (14 / 2) = h / 7          （GQA 映射）
//   s_j = dot(q[i,h,:], k[j,g,:]) / sqrt(64)   对所有 j <= i
//   s_j = -inf                                 对 j > i（causal mask）
//   p = softmax(s)                             只在 j <= i 上归一化
//   out[i,h,:] = sum_j p_j * v[j,g,:]
//
// 动手前想清楚：
//   1. grid/block 划分建议：(seq, n_heads) 二维 grid，一个 block 算一个
//      (i, h)；分数存 shared memory（seq <= 512 时最多 2KB）。
//   2. softmax 要两遍还是在线（online softmax）？M1 允许两遍。
//   3. 除以 i+1 还是只在有效位上归一化？被 mask 的 -inf 参与归一化会怎样？
//   4. 这个 kernel 访存模式为什么差（k/v 被重复读 14/2 次）？
//      记下来，M2 做 KV Cache、M3 做优化时的出发点。
