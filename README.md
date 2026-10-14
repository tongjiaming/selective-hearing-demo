# Selective Hearing：选择听谁说话

Windows 本地实验程序：发现声音后生成匿名人物按钮，选择「全部听 / 全部静音 / 只听说话人 1 / 只听说话人 2」。目标设备链路为 **AirPods 麦克风 → Windows 推理 → AirPods 播放**，不经过 iPhone。

目前已实现桌面界面、公开录音处理、真实双人波形分离、跨块身份匹配及设备收放音代码。**这是有明显缓冲延迟的原型；AirPods 实机收音与端到端延迟尚未验收。** 测试结论见 [实现与实测](docs/implementation.md)。

## 启动

需要 Windows、Python 3.12（含 Tcl/Tk）、约 2 GB 可用磁盘空间。当前使用 CPU，不要求 NVIDIA 显卡。

```powershell
# 首次安装；模型首次下载需联网，之后推理在本机进行。
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Models

# 打开程序，不会自动打开麦克风或播放声音。
powershell -ExecutionPolicy Bypass -File run.ps1
```

界面中点「加载人声模型」，然后：

安装完成后也可双击项目中的 `run.cmd`。它按脚本所在位置定位项目和环境，不依赖当前工作目录。若启动失败，运行 `run.cmd -Check` 可查看实际项目路径、运行环境和 Python 版本；“项目环境缺失”不等于电脑未安装 Python。不要仅下载仓库中的单个启动文件运行，它需要同目录的项目文件。

- **先试录音：**「导入并处理」选择 WAV / FLAC / OGG；确认人物后选择输出设备，点「循环试听」，切换人物。可导出分离音轨和 JSON 报告。
- **连接耳机后：**「刷新设备」，明确选择 AirPods 输入和输出，点「开始收音」。首次确认人物至少需要两个 4 秒音频块。
- **排查设备：**取消「自动识别人声并分离」，可先测试原声与静音。若提示 16 kHz 格式不支持，尝试同名 MME 设备。程序不会悄悄换成笔记本麦克风。

「停止」关闭输入与播放。切换按钮在软件下一次输出回调生效，已经送入蓝牙/声卡的声音无法撤回。默认播放增益为 40%；使用耳机测试以避免扬声器反馈。

短于两块的录音可能只出现灰色「确认中」按钮。最多保存两个匿名身份，重新开始收音或导入文件会重置。全部静音时仍分析新人；新人不会解除静音。

## 下载与排错

基础界面可单独安装 `scripts/setup.ps1`（不加 `-Models`），但人物分离必须安装模型依赖。约 20 MB 分离权重和 83 MB 声纹权重存放在 `models/`；下载和加载均验证 SHA-256。

若官方网络不可达，可自行选择镜像重试：

```powershell
.venv\Scripts\python.exe scripts/download_models.py --hf-endpoint https://hf-mirror.com
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Models -IndexUrl https://pypi.tuna.tsinghua.edu.cn/simple -HfEndpoint https://hf-mirror.com
```

PyTorch CPU 包仍取自官方源。参数只对本次安装生效，不修改全局配置。

安装环境默认位于用户文档目录的 `Codex\sh-runtime-<项目哈希>`，项目 `.venv` 链接到该目录。新建环境时可传 `-RuntimeRoot D:\SelectiveHearing\venv` 指定其他短路径。请确保桌面用户和安装进程都能访问该路径；执行工具能读取目录，并不一定代表桌面启动器也能读取。

```powershell
# 只枚举设备，不收音
.venv\Scripts\python.exe -m selective_hearing --devices

# 离线处理，不播放
.venv\Scripts\python.exe -m selective_hearing --file mixed.wav --output results/my-test

# 控制逻辑测试，不使用真实麦克风或扬声器
.venv\Scripts\python.exe -m pip install pytest==8.3.5
.venv\Scripts\python.exe -m pytest -q
```

## 模型与限制

首版用 **Conv-TasNet Libri2Mix → Silero VAD → ECAPA-TDNN → 会话身份匹配**。先从同一混合波形分离两路，再判断各路是谁；不是根据转写时间戳静音，也不是用干净音轨冒充分离。

diart、WeSep、Step 模型均未接入首版。选择现成 CPU 权重先验证完整流程，后续可替换 `ModelBackend`。结构、依据和实测边界见 [实现与实测](docs/implementation.md)。

- 4 秒分块、非因果模型，包含收集音频的等待时间和计算时间；尚无跨块交叠融合，边界可能产生音质问题。
- 分离模型在 8 kHz 工作，输出重采样为 16 kHz 不会恢复高频。
- 身份匹配有可能漏识别或认错；未匹配目标输出静音，绝不自动回退混合声。
- 暂只支持最多两人和单目标选择，不能承诺第三人、中文、嘈杂环境、远场或 AirPods 收音效果。
- 软件静音只控制程序输出，不能消除耳机外部声音或物理漏音。

## 文档与数据

| 文档 | 内容 |
| --- | --- |
| [实现与实测](docs/implementation.md) | 当前架构、已测结果和未测项目 |
| [方案与边界](docs/design.md) | 原始需求与目标链路 |
| [模型调研](docs/model-research.md) | 早期候选和公开依据 |
| [验证计划](docs/validation.md) | 设备和现场验收计划 |
| [讨论与决策记录](docs/discussion-log.md) | 方案演变和本次实现决定 |

不自动保存或上传麦克风音频。显式导出仅写入用户选择的本地文件夹。模型、音频、运行环境、缓存、声纹和凭据不提交 GitHub。测试脚本只下载公开样本用于本地验证。

[2026 Step 黑客松手册](https://wvixbzgc0u7.feishu.cn/wiki/HAQfwzM09ii5QQkap5YcuiwMnoc)（访问受限；本仓库不转载正文）。
