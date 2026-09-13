#include "tli/weights.h"

#include <stdexcept>
#include <string>

#include "tli/cuda_check.h"

namespace tli {

// ---------------------------------------------------------------------------
// 已实现：移动语义 + 析构（显存归属管理）
// TODO(M1): load_safetensors —— 见 docs/milestone1-forward.md 第 5 节步骤 S0
// ---------------------------------------------------------------------------

GpuWeights::GpuWeights(GpuWeights&& other) noexcept
    : tensors_(std::move(other.tensors_)) {
  other.tensors_.clear();
}

GpuWeights& GpuWeights::operator=(GpuWeights&& other) noexcept {
  if (this != &other) {
    for (auto& [name, ptr] : tensors_) {
      (void)name;
      if (ptr) cudaFree(ptr);
    }
    tensors_ = std::move(other.tensors_);
    other.tensors_.clear();
  }
  return *this;
}

GpuWeights::~GpuWeights() {
  // 注意：tie_word_embeddings 时 lm_head 与 embed_tokens 指向同一块显存。
  // 你在实现加载时要想清楚怎么避免 double free（提示：只登记一次）。
  for (auto& [name, ptr] : tensors_) {
    (void)name;
    if (ptr) cudaFree(ptr);
  }
}

const float* GpuWeights::get(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : it->second;
}

GpuWeights GpuWeights::load_safetensors(const std::string& path) {
  // TODO(M1)：
  //  1) 读文件头 8 字节 (uint64, 小端) 得到 header_len
  //  2) 读 header_len 字节 JSON -> unordered_map<string, TensorMeta>
  //     （nlohmann/json 已配好；解析逻辑照抄 scripts/inspect_model.py）
  //  3) 逐张量：fseek 到 8 + header_len + data_begin，读出字节
  //     - dtype F32 直接上传
  //     - dtype F16 先在 host 转 F32（_f16_to_f32 半精度转换自己写，练手）
  //  4) tie_word_embeddings: 把 "lm_head.weight" 指向 embed_tokens 的显存
  //  5) 返回 GpuWeights。加载失败抛 std::runtime_error
  (void)path;
  throw std::runtime_error("TODO(M1): GpuWeights::load_safetensors 未实现");
}

}  // namespace tli
