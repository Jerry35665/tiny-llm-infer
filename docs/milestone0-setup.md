# M0 — 环境与仓库（0.5 天）

目标：工具链全部验证可用，模型权重就位，工程能编译运行。

## 0.1 本机环境现状（已探测确认，无需重装）

| 组件 | 版本 | 说明 |
|---|---|---|
| GPU | RTX 4060 Laptop 8GB | sm_89，CMake 里已配好 |
| 显卡驱动 | ⚠️ 576.52，**必须升级到 580+** | 见下方"驱动升级" |
| CUDA Toolkit | 13.3 | nvcc 可用 |
| C++ 编译器 | MSVC 14.51（VS 18 Community）| Windows 下 nvcc 必须配 MSVC ✅ |
| Python | 3.11.5 + torch 2.2.2 CPU | 金标准生成够用，**不要**装 GPU 版 torch |
| 磁盘 | C: 剩余 ~70GB | 模型约 1GB |

注意：Git Bash 里的 gcc（w64devkit）**不能**用于 CUDA，编译一律走 CMake + MSVC。

### ⚠️ 驱动升级（M0 唯一的手工步骤，必须做）

当前驱动 576.52 只支持到 CUDA 12.9，而 toolkit 是 13.3 —— 编译能过、运行时
报 `err=35 cudaErrorInsufficientDriver`。两种解法：

1. **升级 NVIDIA 驱动（推荐）**：到 nvidia.com 或 NVIDIA App 下载 RTX 4060 Laptop
   的最新 Game Ready / Studio 驱动（580 或更新），装完重启即可。
2. 备选：改装 CUDA Toolkit 12.9 匹配现有驱动（不推荐，白折腾）。

已验证的事实（2026-09-13 探测）：编译链 Ninja+MSVC+nvcc 13.3 全通，
仅运行时被驱动卡住 —— 升级驱动后无需改任何代码。

### 构建方式（重要：用 build.bat，不要直接敲 cmake）

- PATH 里的 `cmake` 是 w64devkit 的 MinGW 版，**对 VS+CUDA 项目检测会失败**；
- VS 生成器还需要 CUDA 的 VS 集成文件，当前也没装好。
- 所以统一用根目录的 `build.bat`：它内部调用 VS 自带的 cmake + ninja，
  并先跑 vcvars64 准备好 MSVC 环境（已实测编译通过）。

## 0.2 步骤

```powershell
# 0) 进入项目目录（在 PowerShell 或 cmd 里做，别在 Git Bash 里编译）
cd C:\Users\Jerry\.zcode\workspace\default\tiny-llm-infer

# 1) 第三方依赖（nlohmann/json 单头文件，走国内 CDN）
python scripts/fetch_deps.py

# 2) 下载模型（~1GB，走 hf-mirror.com；如失败见 0.4）
pip install -U huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple
python scripts/download_model.py

# 3) 金标准所需的 python 库（transformers 用现有 CPU torch 即可）
pip install "transformers>=4.45" numpy -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4) 盘点权重结构 —— 把输出记到 docs/notes.md 里
python scripts/inspect_model.py models/qwen2.5-0.5b

# 5) 编译 + 验证（用 build.bat；先完成上面的驱动升级）
build.bat
.\build\tli.exe --info
```

## 0.3 验收清单

- [ ] **显卡驱动已升级到 580+**（`nvidia-smi` 左上角看版本）
- [ ] `tli --info` 打印出 RTX 4060（sm 8.9）
- [ ] `inspect_model.py` 列出全部张量；确认三个问题：
  1. q/k/v_proj 有没有 `.bias`？（影响 M1 前向要不要加 bias）
  2. `lm_head.weight` 在不在文件里？（tie_word_embeddings 的影响）
  3. 张量 dtype 是 F32 还是 F16/BF16？（影响加载器要不要做类型转换）
- [ ] `git log` 有第一次 commit

## 0.4 常见问题

- **模型下载 404/超时**：确认 HF_ENDPOINT=https://hf-mirror.com 没被覆盖；
  或者改用 modelscope：`pip install modelscope`，然后
  `from modelscope import snapshot_download; snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", local_dir="models/qwen2.5-0.5b")`
- **运行时报 err=35 / (null)**：驱动太旧，见 0.1 的驱动升级
- **改了 CMakeLists 后重新配置**：删掉 build\ 目录再跑 `build.bat` 即可
- **想省事**：llama.cpp 的 Windows CUDA 预编译包可以先从 GitHub Releases 下载备用
  （M3 做性能对比时才用），不用现在自己编译

## 0.5 完成后

建 `docs/notes.md`，把 inspect_model 的输出和你对下面问题的理解写进去（以后翻简历面试就靠它）：
1. 这个模型有多少参数？显存放不下时会发生什么？
2. 哪些权重形状你有预感会踩坑（转置？行/列主序？）
然后告诉我"M0 完成"，我把 M1 文档要点过一遍，你开始写代码。

## 0.6 GitHub 推送（一次性设置，已完成安装，差登录）

gh CLI 已装好（`C:\Program Files\GitHub CLI\gh.exe`）。你只需做一次登录，
在自己开的终端（PowerShell）里运行：

```powershell
gh auth login
# 交互选择: GitHub.com -> HTTPS -> Login with a web browser
# 复制屏幕上的 8 位设备码，浏览器里粘贴授权
```

登录完成后告诉 ZCode，之后的工作流：
- ZCode 帮你建远端仓库 `tiny-llm-infer` 并完成首次推送（建公开仓库，简历项目要可见；
  想先私密就说一声）
- 以后每完成一步：更新 README 进度勾选 → `sync.bat "commit message"` 一键提交推送
