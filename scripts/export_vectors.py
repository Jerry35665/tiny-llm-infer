#!/usr/bin/env python3
"""把 golden 对拍数据导出为 RTL 仿真风格的标准测试向量（IC 提案 P6）。

输入: scripts/golden_test.py 产出的 golden/ 目录（必须在仓库内）
输出: vectors/ 目录
  p{i}_stimulus.txt       输入 token id 序列（每行一个 id）
  p{i}_expected_L{k}.txt  期望输出（fp32 十进制，每行一个数）
  p{i}_expected_L{k}.vec  定点十六进制（--fixed 时生成，Q1.15，含饱和）
  manifest.json           格式自述 + 形状 + 饱和 clip 统计

Q1.15: v_int = clamp(round(v * 32767), -32768, 32767)，负数补码 4 位十六进制。
被饱和 clip 的值会统计进 manifest —— 这正是 P4 的 saturation 设计预演。

用法:
  python scripts/golden_test.py            # 先生成金标准
  python scripts/export_vectors.py --fixed
"""
import argparse
import json
import os
import pathlib

import numpy as np

Q_SCALE = 32767.0  # Q1.15 正满量程
Q_MIN, Q_MAX = -32768, 32767


def q1_15(v: float) -> str:
    """fp32 -> Q1.15 补码十六进制（4 位），饱和处理。"""
    i = int(round(v * Q_SCALE))
    i = max(Q_MIN, min(Q_MAX, i))
    return f"{i & 0xFFFF:04x}"


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parent.parent
    default_golden = repo / "golden"
    default_out = repo / "vectors"

    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=str(default_golden))
    ap.add_argument("--out", default=str(default_out))
    ap.add_argument("--fixed", action="store_true", help="额外生成 Q1.15 .vec 文件")
    args = ap.parse_args()

    golden = pathlib.Path(args.golden).resolve()
    out = pathlib.Path(args.out).resolve()
    # 路径安全：输入输出都限制在仓库目录内
    repo_str = str(repo)
    if os.path.commonpath([str(golden), repo_str]) != repo_str or \
       os.path.commonpath([str(out), repo_str]) != repo_str:
        print("非法路径：golden/out 必须在本仓库目录内")
        return 1
    if not (golden / "meta.json").is_file():
        print(f"缺少 {golden}/meta.json，先运行 scripts/golden_test.py")
        return 1

    meta = json.loads((golden / "meta.json").read_text(encoding="utf-8"))
    num_layer_files = sum(
        1 for f in golden.glob("p0_L*.bin") if "logits" not in f.name)
    out.mkdir(parents=True, exist_ok=True)

    satur_total = 0
    vectors = []
    for p in meta["prompts"]:
        pi = p["index"]
        seq_len = p["seq_len"]
        vec = {"index": pi, "text": p["text"], "seq_len": seq_len,
               "token_ids": p["token_ids"], "layers": []}

        (out / f"p{pi}_stimulus.txt").write_text(
            "\n".join(str(t) for t in p["token_ids"]) + "\n", encoding="utf-8")

        # 层 hidden states: L00 .. L{num_layer_files-1}
        for k in range(num_layer_files):
            src = golden / f"p{pi}_L{k:02d}.bin"
            if not src.is_file():
                continue
            arr = np.frombuffer(src.read_bytes(), dtype=np.float32)
            (out / f"p{pi}_expected_L{k:02d}.txt").write_text(
                "\n".join(f"{v:.6e}" for v in arr) + "\n", encoding="utf-8")

            layer_info = {"index": k, "shape": [seq_len, len(arr) // seq_len]}
            if args.fixed:
                qints = np.clip(np.round(arr * Q_SCALE), Q_MIN, Q_MAX).astype(np.int64)
                satur = int(np.sum(qints != np.round(arr * Q_SCALE)))
                satur_total += satur
                hexlines = "\n".join(f"{int(q) & 0xFFFF:04x}" for q in qints)
                (out / f"p{pi}_expected_L{k:02d}.vec").write_text(
                    hexlines + "\n", encoding="utf-8")
                layer_info["fixed"] = "Q1.15 hex"
                layer_info["saturated"] = satur
            vec["layers"].append(layer_info)

        # logits
        lg = golden / f"p{pi}_logits.bin"
        if lg.is_file():
            arr = np.frombuffer(lg.read_bytes(), dtype=np.float32)
            (out / f"p{pi}_expected_logits.txt").write_text(
                "\n".join(f"{v:.6e}" for v in arr) + "\n", encoding="utf-8")
            vec["logits"] = {"shape": [len(arr)], "greedy_next": p["greedy_next"]}

        vectors.append(vec)
        print(f"p{pi}: 导出 {len(vec['layers'])} 层 + logits, seq={seq_len}")

    manifest = {
        "format": {
            "stimulus": "token id, one per line",
            "expected_txt": "fp32 decimal, one per line, row-major",
            "expected_vec": "Q1.15 two's-complement hex (4 digits), saturated",
            "q1_15": {"range": "[-1.0, 1.0)", "step": 3.0517578125e-05},
        },
        "golden": "HuggingFace CPU fp32 (see scripts/golden_test.py)",
        "saturated_total": satur_total,
        "vectors": vectors,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n测试向量已写入 {out}")
    print(f"Q1.15 饱和 clip 总数: {satur_total}（校准预分析线索）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
