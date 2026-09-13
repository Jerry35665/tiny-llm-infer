#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tli/types.h"

namespace tli {

// safetensors 文件头里的一条张量记录
struct TensorMeta {
  std::string name;
  DataType dtype = DataType::F32;
  std::vector<int64_t> shape;
  uint64_t data_begin = 0;  // 相对数据区起点的字节偏移（含）
  uint64_t data_end = 0;    // （不含）
};

// 把 safetensors 里的全部权重上传到显存（M1 统一转成 F32；M3 再换 F16 路线）。
//
// TODO(M1): 实现 load_safetensors，参考实现见 scripts/inspect_model.py
//   1. 读文件前 8 字节 -> uint64 header_len（小端）
//   2. 读 header_len 字节 -> JSON，形如
//      {"model.layers.0.mlp.down_proj.weight":
//         {"dtype":"F16","shape":[896,4864],"data_offsets":[0,1734656]}, ...}
//   3. 对每个张量：host 读出 -> (F16 则先转 F32) -> cudaMalloc + cudaMemcpy
//   4. tie_word_embeddings=true 的模型没有 lm_head.weight：
//      让 "lm_head.weight" 这条记录直接复用 embed_tokens 的 device 指针
class GpuWeights {
 public:
  GpuWeights() = default;
  GpuWeights(const GpuWeights&) = delete;             // 独占显存，禁止拷贝
  GpuWeights& operator=(const GpuWeights&) = delete;
  GpuWeights(GpuWeights&& other) noexcept;
  GpuWeights& operator=(GpuWeights&& other) noexcept;
  ~GpuWeights();

  static GpuWeights load_safetensors(const std::string& path);

  // 取权重 device 指针；找不到返回 nullptr（想想：什么时候会找不到？）
  const float* get(const std::string& name) const;
  const std::unordered_map<std::string, float*>& all() const { return tensors_; }

 private:
  std::unordered_map<std::string, float*> tensors_;
};

}  // namespace tli
