#!/usr/bin/env python3
"""逐层对比 C++ dump 与金标准。

用法:
  python scripts/compare_golden.py --golden golden --mine out

C++ 侧（tli_test）需要产出同样命名的文件：
  out/p{i}_L{k}.bin    fp32 raw [seq, hidden]，k 与 HF hidden_states 下标一致
  out/p{i}_logits.bin  fp32 raw [vocab]
  out/meta.json        {"prompts": [{"index": 0, "token_ids": [...]}, ...]}

判读标准（fp32 GPU vs CPU，24 层累积后）：
  每层 max|Δ| < 1e-4      正常
  1e-4 ~ 1e-3             可接受（浮点求和顺序不同）
  > 1e-3                  有 bug —— 第一处超差的层就是嫌疑人
"""
import argparse
import json
import pathlib

import numpy as np


def load(path: pathlib.Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"缺少文件: {path}")
    return np.fromfile(path, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="golden")
    ap.add_argument("--mine", default="out")
    args = ap.parse_args()

    golden_dir = pathlib.Path(args.golden)
    mine_dir = pathlib.Path(args.mine)

    gmeta = json.loads((golden_dir / "meta.json").read_text(encoding="utf-8"))
    mmeta = json.loads((mine_dir / "meta.json").read_text(encoding="utf-8"))

    gp = {p["index"]: p for p in gmeta["prompts"]}
    mp = {p["index"]: p for p in mmeta.get("prompts", [])}
    if set(gp) != set(mp):
        print(f"prompt 集合不一致: golden={sorted(gp)} mine={sorted(mp)}")
        return 1

    first_bad_layer = None
    ok = True
    for pi in sorted(gp):
        if gp[pi]["token_ids"] != mp[pi].get("token_ids"):
            print(f"p{pi}: token_ids 与金标准不一致！")
            ok = False
            continue
        print(f"--- p{pi}: {gp[pi]['text']!r} (seq={gp[pi]['seq_len']}) ---")
        for k in range(0, 25):
            try:
                g = load(golden_dir / f"p{pi}_L{k:02d}.bin")
                m = load(mine_dir / f"p{pi}_L{k:02d}.bin")
            except FileNotFoundError:
                print(f"  L{k:02d}: 文件缺失，跳过")
                continue
            if g.shape != m.shape:
                print(f"  L{k:02d}: 形状不一致 golden={g.shape} mine={m.shape}")
                ok = False
                continue
            diff = float(np.max(np.abs(g - m)))
            flag = ""
            if diff > 1e-3:
                flag = "  <-- 第一处超差从这层开始排查"
                if first_bad_layer is None:
                    first_bad_layer = k
                ok = False
            elif diff > 1e-4:
                flag = "  (可接受)"
            print(f"  L{k:02d}: max|delta| = {diff:.3e}{flag}")

        g_logits = load(golden_dir / f"p{pi}_logits.bin")
        m_logits = load(mine_dir / f"p{pi}_logits.bin")
        if g_logits.shape == m_logits.shape:
            g_top5 = np.argsort(-g_logits)[:5]
            m_top5 = np.argsort(-m_logits)[:5]
            l1 = float(np.max(np.abs(g_logits - m_logits)))
            match = int(g_top5[0]) == int(m_top5[0])
            print(f"  logits: max|delta| = {l1:.3e}, "
                  f"top1 gold={gp[pi]['greedy_next']} mine={int(m_top5[0])} "
                  f"{'OK' if match else 'MISMATCH'}, "
                  f"top5 重合 {len(set(g_top5) & set(m_top5))}/5")
            if not match:
                ok = False
        print()

    if first_bad_layer is not None:
        print(f">>> 从第 {first_bad_layer} 层开始超差：bug 大概率在该层的某个算子，"
              "用二分法把中间量 dump 出来定位（见 milestone1 文档第 6 节）")
    print("结果:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
