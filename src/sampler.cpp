#include "tli/sampler.h"

#include <stdexcept>

namespace tli {

int32_t Sampler::sample(const float* logits_dev, int vocab_size) {
  // TODO(M2)：
  //   1) greedy: argmax（可以先 cudaMemcpy 拷回 CPU 再找，之后换 GPU kernel）
  //   2) temperature / top-k / top-p
  (void)logits_dev;
  (void)vocab_size;
  throw std::runtime_error("TODO(M2): Sampler::sample 未实现");
}

}  // namespace tli
