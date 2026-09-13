// tli - Tiny LLM Inference
// M0: --info 打印 GPU 信息，验证 CUDA 工具链可用
// M1: 加载 Qwen2.5-0.5B，fp32 前向与 HF 金标准对拍（见 docs/milestone1-forward.md）
#include <cuda_runtime.h>

#include <iostream>
#include <string>

#include "tli/cuda_check.h"

namespace {

void print_gpu_info() {
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::cout << "device      : " << prop.name << "\n"
            << "sm arch     : " << prop.major << "." << prop.minor << "\n"
            << "sm count    : " << prop.multiProcessorCount << "\n"
            << "global mem  : " << prop.totalGlobalMem / 1073741824.0 << " GiB\n"
            << "l2 cache    : " << (prop.l2CacheSize >> 20) << " MiB\n"
            << "mem bus     : " << prop.memoryBusWidth << " bit\n";
}

void usage() {
  std::cout << "tli - tiny llm inference\n"
            << "  tli --info                                   GPU info\n"
            << "  tli_test <model_dir> <golden_dir> <out_dir>   M1 golden check\n"
            << "  tli --model <dir> --prompt <s> [--steps n]    M1/M2 pipeline\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir, prompt;
  int steps = 32;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--info") {
      print_gpu_info();
      return 0;
    } else if (arg == "--model" && i + 1 < argc) {
      model_dir = argv[++i];
    } else if (arg == "--prompt" && i + 1 < argc) {
      prompt = argv[++i];
    } else if (arg == "--steps" && i + 1 < argc) {
      steps = std::stoi(argv[++i]);
    }
  }

  if (model_dir.empty()) {
    usage();
    return 1;
  }

  // TODO(M1)：接通推理主流程
  //   ModelConfig cfg = tli::load_config(model_dir);
  //   GpuWeights w = GpuWeights::load_safetensors(model_dir + "/model.safetensors");
  //   Engine engine(cfg, std::move(w));
  //   tokenizer：M1 先不实现 C++ BPE，token ids 从 golden/meta.json 拿
  //              （M2 再写 BPE，参考 karpathy/minbpe）
  (void)prompt;
  (void)steps;
  std::cout << "TODO(M1): pipeline not wired yet, see docs/milestone1-forward.md\n";
  return 0;
}
