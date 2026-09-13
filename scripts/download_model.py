#!/usr/bin/env python3
"""下载 Qwen2.5-0.5B-Instruct 权重到 models/qwen2.5-0.5b/

- 默认走 hf-mirror.com（国内可达），可用环境变量 HF_ENDPOINT 覆盖
- 依赖: pip install -U huggingface_hub
- 备选: pip install modelscope，然后用 modelscope.snapshot_download
"""
import os
import pathlib
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
ALLOW = [
    "config.json",
    "generation_config.json",
    "model.safetensors",  # ~988MB
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]


def main() -> int:
    dest = pathlib.Path(__file__).resolve().parent.parent / "models" / "qwen2.5-0.5b"
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("请先安装: pip install -U huggingface_hub")
        return 1

    print(f"endpoint: {os.environ['HF_ENDPOINT']}")
    print(f"repo    : {REPO}")
    print(f"dest    : {dest}")
    path = snapshot_download(REPO, local_dir=str(dest), allow_patterns=ALLOW)
    print(f"完成: {path}")
    print("下一步: python scripts/inspect_model.py models/qwen2.5-0.5b")
    return 0


if __name__ == "__main__":
    sys.exit(main())
