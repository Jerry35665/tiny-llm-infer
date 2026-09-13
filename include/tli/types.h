#pragma once
#include <cstdint>
#include <vector>

#include "tli/cuda_check.h"

namespace tli {

enum class DataType { F32, F16 };

inline int64_t numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

// 显存张量的最小封装：一个 device 指针 + 形状。刻意保持简单，
// M4 做 batching 时再演进成带 allocator 的版本。
// M1 阶段全部 fp32，所以 data 直接用 float*。
struct Tensor {
  float* data = nullptr;  // device pointer
  std::vector<int64_t> shape;
  int64_t numel() const { return tli::numel(shape); }
};

// 简单的显存 RAII buffer（容量按 float 个数计）
struct DeviceBuffer {
  float* data = nullptr;
  int64_t cap = 0;

  void alloc(int64_t n_floats) {
    if (n_floats <= cap) return;
    if (data) CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaMalloc(&data, n_floats * sizeof(float)));
    cap = n_floats;
  }
  ~DeviceBuffer() {
    if (data) cudaFree(data);
  }
};

}  // namespace tli
