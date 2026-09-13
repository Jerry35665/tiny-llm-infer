#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace tli {

// ===== 参考实现（已写好，见 kernels/rmsnorm.cu）=====
// out[r,:] = x[r,:] / sqrt(mean(x[r,:]^2) + eps) * w[:]
void rmsnorm(float* out, const float* x, const float* w, int rows, int hidden,
             float eps, cudaStream_t stream = nullptr);

// ===== TODO(M1)：以下算子由你实现。接口签名自己设计，实现完在这里补声明 =====
// 设计建议和数学规范见 docs/milestone1-forward.md 第 4 节。
//
// void embedding(...)   // 查表：out[r,:] = table[tokens[r],:]
// void rope(...)        // 建议：cos/sin 表在构造时离线算好存显存
// void attention_naive(...)  // causal mask + GQA（q head i -> kv head i/(H/Kv)）
// void swiglu(...)      // out = silu(gate) * up，逐元素
// void add(...)         // 残差 out = a + b（也可以并入其他 kernel 做融合）

}  // namespace tli
