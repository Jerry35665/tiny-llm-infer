#pragma once
#include <cstdint>
#include <string>

namespace tli {

// 模型结构参数。默认值是 Qwen2.5-0.5B-Instruct 的公开配置，
// 加载时会被 <model_dir>/config.json 覆盖。
// ⚠️ 务必用 scripts/inspect_model.py 核对你下载到的模型的真实值。
struct ModelConfig {
  std::string model_dir;

  int32_t vocab_size = 151936;
  int32_t hidden_size = 896;
  int32_t num_layers = 24;
  int32_t num_heads = 14;
  int32_t num_kv_heads = 2;     // GQA：2 个 KV head 服务 14 个 Q head
  int32_t head_dim = 64;        // = hidden_size / num_heads
  int32_t intermediate_size = 4864;

  float rms_norm_eps = 1e-6f;
  float rope_theta = 1000000.0f;
  bool tie_word_embeddings = true;  // lm_head 与 embed_tokens 共享权重
};

// 解析 <model_dir>/config.json（已实现，直接用）
ModelConfig load_config(const std::string& model_dir);

}  // namespace tli
