#pragma once
#include <cublas_v2.h>
#include <cstdint>
#include <functional>
#include <vector>

#include "tli/config.h"
#include "tli/weights.h"

namespace tli {

// 每层 hidden dump 回调（对拍用，可为空）。
// 约定与 HF output_hidden_states 的下标一一对应：
//   layer=0            -> embedding 输出
//   layer=1..L-1       -> 第 layer-1 个 decoder 层的输出（0 起数）
//   layer=L            -> 最后一层输出 + final RMSNorm 之后的结果
// 返回 false 可提前终止。
using LayerHook =
    std::function<bool(int layer, const float* hidden, int rows, int hidden_size)>;

// M1 的核心类：无 KV Cache 的朴素前向。
class Engine {
 public:
  Engine(const ModelConfig& cfg, GpuWeights weights);
  ~Engine();

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // 输入 tokens[seq]，返回最后一个位置的 logits（device 指针，[vocab] 个 F32）。
  // M1 不做自回归生成（那是 M2 的事），单次调用单条序列。
  const float* forward(const std::vector<int32_t>& tokens, const LayerHook& hook = {});

  const ModelConfig& config() const { return cfg_; }

 private:
  ModelConfig cfg_;
  GpuWeights weights_;
  cublasHandle_t cublas_{};

  // TODO(M1): 在构造函数里分配中间缓冲区。至少需要：
  //   x            [seq, hidden]        层间流动的 hidden state
  //   q/k/v        [seq, n_heads*head_dim] / [seq, n_kv*head_dim] x2
  //   attn_out     [seq, hidden]
  //   mlp 中间量   gate/up [seq, intermediate]
  //   logits       [vocab]
  //   rope 的 cos/sin 表 [max_seq, head_dim/2]（建议构造时离线算好）
  // 提示：把每层的权重名拼出来（"model.layers.3.self_attn.q_proj.weight"），
  //       写个小工具函数省得满屏字符串拼接。
};

}  // namespace tli
