#!/usr/bin/env python3
"""生成 M1 金标准（CPU fp32，结果确定、可复现）。

产出（写入 --out 目录）：
  p{i}_L{k}.bin    第 i 条 prompt 的第 k 层 hidden states（fp32 raw，[seq, hidden]）
                   k=0 是 embedding 输出，k=num_layers 是最后一层+final RMSNorm
  p{i}_logits.bin  最后一个位置的 logits（fp32 raw，[vocab]）
  meta.json        prompt 原文、token_ids、greedy 下一个 token 等元信息

依赖: pip install "transformers>=4.45" numpy（torch 用现有的 CPU 版即可）
用法: python scripts/golden_test.py --model models/qwen2.5-0.5b --out golden
"""
import argparse
import json
import pathlib

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 短 prompt：seq 小，M1 的朴素 attention 实现起来最简单
PROMPTS = ["The capital of France is", "1, 2, 3, 4, 5,"]


def dump(path: pathlib.Path, arr) -> None:
    path.write_bytes(np.ascontiguousarray(arr, dtype=np.float32).tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen2.5-0.5b")
    ap.add_argument("--out", default="golden")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"加载模型（CPU fp32，确定性）: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model)

    meta = {"model": args.model, "dtype": "fp32", "prompts": []}
    with torch.no_grad():
        for pi, text in enumerate(PROMPTS):
            enc = tok(text, return_tensors="pt")
            ids = enc.input_ids[0].tolist()
            res = model(input_ids=enc.input_ids, output_hidden_states=True)

            # hidden_states: tuple，长度 num_layers+1
            for k, h in enumerate(res.hidden_states):
                dump(out / f"p{pi}_L{k:02d}.bin", h[0].numpy())

            logits = res.logits[0, -1]  # [vocab]
            dump(out / f"p{pi}_logits.bin", logits.numpy())
            top5 = torch.topk(logits, 5).indices.tolist()

            meta["prompts"].append({
                "index": pi,
                "text": text,
                "token_ids": ids,
                "seq_len": len(ids),
                "greedy_next": int(logits.argmax()),
                "top5_ids": [int(t) for t in top5],
            })
            print(f"  p{pi}: {text!r} -> {len(ids)} tokens, "
                  f"greedy next id = {meta['prompts'][-1]['greedy_next']}")

    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"金标准已写入 {out.resolve()}")


if __name__ == "__main__":
    main()
