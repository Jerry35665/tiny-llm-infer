#include "tli/model.h"

#include <stdexcept>

#include "tli/cuda_check.h"
#include "tli/kernels.h"

namespace tli {

// ---------------------------------------------------------------------------
// TODO(M1)：这是整个项目最核心的文件。
// 强烈建议按 docs/milestone1-forward.md 第 5 节的 S1~S7 顺序小步实现，
// 每完成一步就用金标准对拍一次，不要一口气写完再调。
// ---------------------------------------------------------------------------

Engine::Engine(const ModelConfig& cfg, GpuWeights weights)
    : cfg_(cfg), weights_(std::move(weights)) {
  CUBLAS_CHECK(cublasCreate(&cublas_));
  // TODO(M1): 分配缓冲区；离线计算 rope 的 cos/sin 表
}

Engine::~Engine() {
  if (cublas_) cublasDestroy(cublas_);
}

const float* Engine::forward(const std::vector<int32_t>& tokens,
                             const LayerHook& hook) {
  // TODO(M1) 主体流程：
  //   [S1] embedding 查表 -> x
  //   [S2..S6] 循环 num_layers 层：
  //       RMSNorm -> QKV 投影 -> RoPE -> 朴素 attention(causal+GQA)
  //       -> o_proj -> 残差
  //       RMSNorm -> gate/up 投影 -> SiLU*up -> down 投影 -> 残差
  //   [S7] final RMSNorm -> lm_head -> 取最后一行 logits
  //  对拍：每层结束后调用 hook（下标约定见 include/tli/model.h），
  //  与 scripts/golden_test.py 生成的金标准逐层对比。
  (void)tokens;
  (void)hook;
  throw std::runtime_error("TODO(M1): Engine::forward 未实现");
}

}  // namespace tli
