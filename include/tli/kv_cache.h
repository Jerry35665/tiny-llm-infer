#pragma once
// M2 里程碑再实现。先想清楚三个设计问题再动手（M2 文档会展开）：
//
//   1. 显存布局：KV Cache 用 [layer][kv_head][max_seq][head_dim] 这种
//      静态预分配够吗？max_seq 定多少、8GB 卡的显存账怎么算？
//
//   2. prefill 和 decode 阶段分别怎么读写它？
//      （prefill 一次写入整段 prompt 的 KV；decode 每步只追加 1 个 token）
//
//   3. M4 做 batching 时多条序列怎么共存？
//      如果现在按"每序列独立大数组"设计，M4 会遇到什么问题？
//      —— 这正是 PagedAttention 要解决的，M5 可选做。
namespace tli {
class KvCache {
  // TODO(M2)
};
}  // namespace tli
