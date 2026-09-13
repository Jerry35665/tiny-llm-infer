// TODO(M1): embedding 查表
//
// 数学：out[r*hidden + j] = table[tokens[r]*hidden + j]
//
// 动手前想清楚：
//   1. grid/block 怎么划分？（一个 thread 一个元素？一个 block 一行？）
//   2. tokens 是 int32，从显存读它和读 float 有什么区别？
//   3. vocab_size=151936 * hidden=896 * 4B ≈ 544MB，这次查表是
//      compute-bound 还是 memory-bound？对 M3 优化意味着什么？
//
// 写完后：在 include/tli/kernels.h 补声明，并在 src/model.cpp 的 S1 接上。
// 本文件保持可编译（只有注释也能过），不要留会改变语义的半成品代码。
