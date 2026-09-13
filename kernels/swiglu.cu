// TODO(M1): SwiGLU 激活
//
// 数学：out[i] = silu(gate[i]) * up[i]，其中 silu(x) = x * sigmoid(x)
//       gate/up 形状 [seq, intermediate=4864]，out 同形状
// 注意：Qwen2.5 的 MLP 是 down( silu(gate(x)) * up(x) )，
//       gate/up 两个投影没有 bias。
//
// 动手前想清楚：
//   1. 这是纯逐元素操作 —— M3 优化时它是最适合和相邻 GEMM/残差融合的候选，
//      为什么？（提示：arithmetic intensity）
//   2. sigmoid 用精确 expf 还是 fast math 近似？对拍容差会受影响吗？
