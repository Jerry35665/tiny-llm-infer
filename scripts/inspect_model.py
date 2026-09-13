#!/usr/bin/env python3
"""打印模型 config.json 与 safetensors 文件头（张量名/形状/dtype），不加载权重。

零依赖（纯标准库）。这份解析逻辑同时是你 M1 里 C++ 加载器的参考实现。

用法: python scripts/inspect_model.py [models/qwen2.5-0.5b]

安全说明：只允许读取本仓库 models/ 目录下的文件（含目录包含性校验）。
"""
import json
import os
import pathlib
import struct
import sys

DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4}


def read_header(path: pathlib.Path) -> dict:
    data = path.read_bytes()
    (header_len,) = struct.unpack("<Q", data[:8])
    return json.loads(data[8:8 + header_len])


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parent.parent
    allowed_base = str((repo / "models").resolve())

    default = os.path.join(allowed_base, "qwen2.5-0.5b")
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else default).resolve()
    if os.path.commonpath([str(root), allowed_base]) != allowed_base:
        print(f"非法路径：只允许读取 {allowed_base} 目录下的模型")
        return 1
    if not root.is_dir():
        print(f"目录不存在: {root}")
        return 1
    cfg_path = root / "config.json"
    st = root / "model.safetensors"
    if not cfg_path.is_file() or not st.is_file():
        print("缺少 config.json 或 model.safetensors，先运行 scripts/download_model.py")
        return 1

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    keep = [
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size", "vocab_size",
        "rms_norm_eps", "rope_theta", "tie_word_embeddings",
        "attention_bias", "max_position_embeddings",
    ]
    print("=== config.json（关键字段）===")
    for k in keep:
        if k in cfg:
            print(f"  {k:24s} = {cfg[k]}")

    print(f"\n=== {st.name} 张量清单 ===")
    header = read_header(st)
    total = 0
    has_bias = False
    for name in sorted(header):
        if name == "__metadata__":
            continue
        meta = header[name]
        shape = meta["shape"]
        n = 1
        for d in shape:
            n *= d
        total += n
        if name.endswith(".bias"):
            has_bias = True
        print(f"  {name:52s} {meta['dtype']:5s} {shape}")
    print(f"\n参数总量: {total / 1e9:.3f} B  (fp16 显存约 {total * 2 / 2**30:.2f} GiB)")
    print(f"QKV bias : {'存在（前向要加 bias！）' if has_bias else '不存在'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
