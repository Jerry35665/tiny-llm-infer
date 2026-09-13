#pragma once
#include <cstdint>

namespace tli {

struct SamplerConfig {
  float temperature = 1.0f;  // 0 视为 greedy
  int top_k = 0;             // 0 = 不启用
  float top_p = 1.0f;        // 1.0 = 不启用
  uint64_t seed = 0;
};

// M2 实现。建议顺序：
//   1. greedy（argmax）—— M1 末尾就会用到
//   2. temperature + top-k
//   3. top-p (nucleus)
// 注意 logits 在 device 上：先想清楚哪一步必须在 GPU 做、哪一步拷回 CPU 更划算，
// 这是 M2 的一个设计讨论点。
class Sampler {
 public:
  explicit Sampler(SamplerConfig cfg) : cfg_(cfg) {}
  // 返回采样出的 token id
  int32_t sample(const float* logits_dev, int vocab_size);

 private:
  SamplerConfig cfg_;
};

}  // namespace tli
