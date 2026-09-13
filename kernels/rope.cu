// TODO(M1): RoPE（旋转位置编码）
//
// Qwen2.5 与 Llama 同款 rotate-half 约定，head_dim=64：
//   inv_freq[j] = theta^(-2j/64)，j = 0..31，theta = 1e6
//   angle[p, j] = p * inv_freq[j]          （p = token 位置）
//   对每对 (j, j+32)：
//     q'_j     =  q_j * cos(p,j) - q_{j+32} * sin(p,j)
//     q'_{j+32} = q_{j+32} * cos(p,j) + q_j * sin(p,j)
// k 同样处理。v 不转。
//
// 动手前想清楚：
//   1. cos/sin 表什么时候算？建议构造时离线生成 [max_seq, 32] 两张表存显存，
//      forward 里只做查表旋转 —— 想想为什么比每次现算好。
//   2. 一共要转 q (n*14*64) 和 k (n*2*64) 两块，怎么复用同一个 kernel？
//   3. 旋转是"就地"还是"另开 buffer"？注意后面 attention 还要读原始 k 吗？
//
// 对照 HF 验证时的等价写法：q*cos + rotate_half(q)*sin，
// 其中 rotate_half(x) = (-x_{32..63}, x_{0..31})，cos/sin 每行重复两半。
// 两种写法数值等价，选一种实现即可。
