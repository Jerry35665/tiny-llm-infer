// M1 金标准对拍程序。
//
// 用法（M1 实现完后）：
//   tli_test <model_dir> <golden_dir> <out_dir>
//
// 流程：
//   1) 读 <golden_dir>/meta.json（由 scripts/golden_test.py 生成），
//      里面有每条测试 prompt 的 token_ids —— M1 阶段不需要实现 tokenizer
//   2) 对每条 prompt 跑 Engine::forward，通过 LayerHook 把每层 hidden
//      dump 成 <out_dir>/p{i}_L{k}.bin（fp32 raw，行主序 [seq, hidden]），
//      把 logits dump 成 <out_dir>/p{i}_logits.bin（fp32 raw [vocab]）
//   3) python scripts/compare_golden.py --golden golden --mine out
//
// dump 帮助函数提示：cudaMemcpy 拷回 host 后，用 std::ofstream 以二进制
// 模式写入即可；文件名下标规则必须和 golden 完全一致（k 从 0 到 num_layers）。
//
// TODO(M1): 实现 main（当前版本只打印提示，保证工程可编译）
#include <cstdio>
#include <iostream>

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  std::cout << "TODO(M1): golden check not implemented.\n"
            << "See docs/milestone1-forward.md section 6 for the workflow.\n";
  return 1;
}
