#!/usr/bin/env python3
"""下载唯一的第三方依赖：nlohmann/json 单头文件 -> third_party/

安全说明：下载地址限定为 https + 域名白名单（jsdelivr / github raw），
不读取任何用户输入的 URL。
"""
import pathlib
import sys
import urllib.request
from urllib.parse import urlparse

ALLOWED_HOSTS = {"cdn.jsdelivr.net", "raw.githubusercontent.com"}

URLS = [
    "https://cdn.jsdelivr.net/gh/nlohmann/json@v3.11.3/single_include/nlohmann/json.hpp",
    "https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp",
]

DEST = pathlib.Path(__file__).resolve().parent.parent / "third_party" / "nlohmann" / "json.hpp"


def fetch(url: str) -> bytes:
    parts = urlparse(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
        raise ValueError(f"域名不在白名单内: {url}")
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def main() -> int:
    if DEST.exists() and DEST.stat().st_size > 100_000:
        print(f"已存在: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        try:
            print(f"下载 {url} ...")
            data = fetch(url)
            DEST.write_bytes(data)
            print(f"完成: {DEST} ({len(data) / 1024:.0f} KiB)")
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"  失败: {e}")
    print("两个源都失败了。可手动下载 json.hpp 放到 third_party/nlohmann/ 下。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
