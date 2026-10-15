# Selective Hearing：选择听谁说话

Windows 本地实验程序：首次识别到人声后生成人物控制，每个人独立选择「听他 / 不听他」，可同时听多个人，也可选择「全部听（原声）/ 全部静音」。目标设备链路为 **AirPods 麦克风 → Windows 推理 → AirPods 播放**，不经过 iPhone。

目前已实现桌面界面、公开录音处理、真实双人波形分离、跨块身份匹配及设备收放音代码。**这是有明显缓冲延迟的原型；AirPods 实机收音与端到端延迟尚未验收。** 测试结论见 [实现与实测](docs/implementation.md)。

## 启动

需要 Windows、Python 3.12（含 Tcl/Tk）、约 2 GB 可用磁盘空间。当前使用 CPU，不要求 NVIDIA 显卡。

```powershell
# 首次安装；模型首次下载需联网，之后推理在本机进行。
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Models

# 主程序：自动加载模型，并使用默认设备开始收音和播放。
powershell -ExecutionPolicy Bypass -File run.ps1

# 独立的音频文件测试程序：自动加载模型，不打开麦克风。
powershell -ExecutionPolicy Bypass -File run.ps1 -FileTest
```

安装完成后也可双击项目中的 `run.cmd` 打开主程序，或 `test-audio.cmd` 打开文件测试程序。启动器按脚本所在位置定位项目和环境，不依赖当前工作目录。若启动失败，运行 `run.cmd -Check` 可查看实际项目路径、运行环境和 Python 版本；“项目环境缺失”不等于电脑未安装 Python。不要仅下载仓库中的单个启动文件运行，它需要同目录的项目文件。

- **主程序：**打开后自动加载本地模型，成功后开始收音。启动时选择 Windows 系统默认输入和输出，并标注「系统默认」；加载期间仍可调整设备，或点「停止」取消自动收音。连接或切换耳机后，先停止、刷新设备，再点「开始收音」。手动选择的设备在刷新后仍可用时会保留。设备或模型启动失败时显示原因，不会擅自换用其他设备。
- **音频文件测试：**在独立程序中等模型就绪，点「导入并处理」选择 WAV / FLAC / OGG；生成说话人后点「循环试听」，切换人物。可导出分离音轨和 JSON 报告。此程序没有收音入口，不打开麦克风。
- **说话人：**实时模式每 1.5 秒分析一次，首位人物至少需要 1 秒有效语音，首次识别后就显示可用的「听他 / 不听他」。后续新声音先暂存为候选，需在不同片段重复出现且明显不同于已知人物，才新增按钮；不再因一次声纹波动直接建人。名字可在人物卡片上直接编辑，回车或移开焦点保存；后续识别更新不会覆盖名字。名字限本次收音或当前文件，重新开始收音、导入另一个文件或重开程序会重置；文件导出的报告保留自定义名字及其音轨 ID。需要采样和计算时间，太短或不清晰的人声可能无法识别。
- **排查设备：**列表使用 WASAPI 音频端点，过滤不支持本程序音频格式的设备，不重复列出 MME、DirectSound 和 WDM-KS 接口。使用 Windows 共享模式转换采样率和声道；枚举只检查能力，不启动收音。设备被拔出、权限或驱动状态变化仍可能导致启动失败。取消「自动识别人声并分离」，可先测试原声与静音。

「停止」关闭输入与播放，也取消尚未完成的自动收音启动；模型可以继续加载，但不会在加载完成后自行收音。切换按钮在软件下一次输出回调生效，已经送入蓝牙/声卡的声音无法撤回。默认播放增益为 40%；使用耳机测试以避免扬声器反馈。

只有一次识别的首位人物标注「确认中」，此时也可选择；后续块继续确认身份。其他新声音未达到建人条件时，状态栏提示「新声音确认中」，不会累积为人物按钮；候选会过期，未确认片段不补播。会话身份数量不限制为两人，人物多时可滚动列表；重新开始收音或导入文件会重置。每个人的开关独立保存，点 A 的「听他」不会改变 B、C。全部静音后新人默认不听；从「全部听」开始调整时，新人默认听。绿色按钮表示当前选择。

按人控制只叠加已识别且开启的分离音轨，缺失或未知轨道输出静音。「全部听（原声）」恢复未经人物筛选的原始混合音，包括背景声音。当前模型没有独立背景音输出，暂不提供背景音开关；不能把混合音减去人声的残差直接标成纯背景，否则被关闭的人声可能泄漏。

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

- 实时模式 1.5 秒分块，离线文件默认 4 秒；非因果模型包含收集音频和计算时间，尚无跨块交叠融合。短窗口和初次声纹可能降低匹配稳定性。
- 分离模型在 8 kHz 工作，输出重采样为 16 kHz 不会恢复高频。
- 身份匹配有可能漏识别或认错；未匹配目标输出静音，绝不自动回退混合声。
- 会话可陆续登记多个人，但当前分离模型每块只有两路输出，不支持可靠分离三人及以上同时说话。人数放开后仍可能误新增或误合并身份，不能承诺中文、嘈杂环境、远场或 AirPods 收音效果。
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
