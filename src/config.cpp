#include "tli/config.h"

#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace tli {

ModelConfig load_config(const std::string& model_dir) {
  ModelConfig cfg;
  cfg.model_dir = model_dir;

  const std::string path = model_dir + "/config.json";
  std::ifstream f(path);
  if (!f) throw std::runtime_error("无法打开 " + path);

  const auto j = nlohmann::json::parse(f);
  cfg.vocab_size = j.at("vocab_size").get<int32_t>();
  cfg.hidden_size = j.at("hidden_size").get<int32_t>();
  cfg.num_layers = j.at("num_hidden_layers").get<int32_t>();
  cfg.num_heads = j.at("num_attention_heads").get<int32_t>();
  cfg.num_kv_heads = j.value("num_key_value_heads", cfg.num_heads);
  if (j.contains("head_dim")) {
    cfg.head_dim = j.at("head_dim").get<int32_t>();
  } else {
    cfg.head_dim = cfg.hidden_size / cfg.num_heads;
  }
  cfg.intermediate_size = j.at("intermediate_size").get<int32_t>();
  cfg.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
  cfg.rope_theta = j.value("rope_theta", 1000000.0f);
  cfg.tie_word_embeddings = j.value("tie_word_embeddings", true);
  return cfg;
}

}  // namespace tli
