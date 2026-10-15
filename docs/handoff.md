# Selective Hearing 项目交接

更新日期：2026-10-15。仓库：[tongjiaming/selective-hearing-demo](https://github.com/tongjiaming/selective-hearing-demo)，私有。无需 Codex 即可运行或继续修改。

## 明天怎么打开

当前电脑项目目录：

```text
C:\Users\JYXC-DZ-0100805\Documents\Codex\2026-10-09\demo-2026-step-https-wvixbzgc0u7-feishu\work\selective-hearing-demo
```

双击目录中的 `run.cmd`。主程序自动加载模型并收音；离线文件测试用 `test-audio.cmd`。`run.cmd -Check` 仅检查环境，不打开麦克风。所有程序都可以从资源管理器或 PowerShell 启动，不经过 Codex。

`.venv` 是指向 `C:\Users\JYXC-DZ-0100805\Documents\Codex\sh-runtime` 的目录链接。虽然目录名包含 Codex，它是普通 Python 虚拟环境，不要求 Codex 登录。**不要删除该 runtime、项目目录、Python 3.12 安装目录或本地 models。** 用户资料在 `Documents\SelectiveHearing\profiles`。

当前保存的实时分析窗口为 **1.5 秒**，文件测试为 **1.0 秒**。界面可设置 1–10 秒。新电脑没有这些设置时默认实时 3 秒、文件 4 秒。启动器不会自动替你把缺失环境安装好。

新电脑安装：

```powershell
git clone git@github.com:tongjiaming/selective-hearing-demo.git
cd selective-hearing-demo
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Models
powershell -ExecutionPolicy Bypass -File run.ps1
```

需自行安装 Python 3.12；第一次下载依赖和模型需要网络。现有电脑本地模型可离线使用。模型文件由 `selective_hearing/model_assets.py` 的固定 SHA-256 校验，不需要 Hugging Face 账号；当前未继续处理用户的 Hugging Face 连通性问题。

## 需求与行为

- Windows 直接连接耳机，不依赖 iPhone。真实可用输入/输出设备列表，启动选系统默认，热插拔刷新。
- 每个人分别“听他 / 不听他”，可自定义名字；发言显示绿框。
- 有“未分类（含背景音）”独立开关，默认听。所有已知人物不听不等于全静音。“全部听”恢复原声，“全部静音”关闭全部输出。
- 首个人物尽快出按钮，后续人物要求相似片段确认；可陆续登记多人，没有五人上限。当前分离器同窗只有两路，三人以上同时讲话不可保证。
- SQLite 持久化名字、声纹、学习样本、别名；不保存原始麦克风录音。
- 用户明确要求：**每次程序更新清空已有学习资料，保留分析窗口；普通重启保留资料。** 由 `revision.py` 的程序指纹自动执行。不要为了保留资料而擅自关掉这个约定。

## 代码地图

| 文件 | 责任 |
|---|---|
| `ui.py` / `file_ui.py` | Tk 主界面、独立文件测试入口、人物控件 |
| `audio.py` | WASAPI 设备、收放音线程、缓冲、离线文件处理 |
| `models.py` | Conv-TasNet、Silero ONNX VAD、ECAPA、过去上下文 |
| `core.py` | 匹配、候选建档、人物状态、独立收听及静音规则 |
| `feature_bank.py` | 有界特征库、可靠近邻、历史归属修正 |
| `profiles.py` / `revision.py` | SQLite、资料改名删除、更新清理 |
| `model_assets.py` / `scripts/download_models.py` | 模型来源、校验和、下载 |
| `scripts/validate_*.py` | 公开样本及算法/性能验证，不替代现场验收 |
| `tests/` | 自动回归，包括静音恢复、持久化、候选与上下文 |

上表中的 Python 文件均在 `selective_hearing/` 目录，除非明确写了 `scripts/`。

推理链路：输入音频 → 最多 3 秒过去上下文 → 8 kHz 双路分离 → VAD → 192 维 ECAPA 声纹 → 已有人物匹配或候选聚类 → 当前窗口的独立音轨。历史上下文不重复播放；模型推理在工作线程，音频回调不运行模型。

分类不是只保存一个均值：初始参照、自适应模板、近邻样本共同参与，另有防误学习和不可合并约束。候选有期限和数量限制，正式人物没有两人/五人上限。当前仍会出错，不能承诺“使用越久必然越准”。

## 本轮修改与已知问题

见 [五人测试与剩余问题](five-speaker-validation.md)。两组五人长轮换各维持五个档案；快速 2.4 秒轮换仍漏建或重复建档。157 项自动测试通过，公共样本不等同于用户现场效果。

其他限制：耳机输入降噪可能已经抹掉远处人声，软件不能恢复未进入输入流的声音；蓝牙、缓冲和非因果推理有延迟；未分类开启时，漏认的被屏蔽人声可能仍听到。diart 是可选研究入口，未成为主程序后端。

## 下一步建议

1. 用五位真实说话人的授权录音固定验收集，包括长句、短句、停顿后返回、换距离、重叠说话；明确人物标签并保留未分类统计。
2. 优先定位短轮换中分离输出的声纹不稳定，评估说话人变化检测与分段；不要只调低阈值或硬设人数上限。
3. 新方案必须同时复跑 `validate_five_speakers.py` 和现有一至三人回归，防止“少建人”变成“把所有人认成一个”。
4. 获取标准分段模型后再评估 diart；若采用分段方案，仍需解决静音所需的人声波形提取。

## 验证、提交与回退

```powershell
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe -B scripts\validate_five_speakers.py
git status --short
git log -5 --oneline
```

修改后提交源码、测试、说明和匿名公共测试结果。`.venv/`、`models/`、`work/`、`results/`、SQLite 人物资料、私人录音/特征不要提交。旧版本 `b4ad4de` 可用作本轮对照；需回退时先保留当前代码，使用新的工作目录检出旧提交，避免覆盖别人的未提交修改。切换代码版本也会触发学习清理。

主程序启动会自动收音。离线开发和无人值守验证使用测试脚本或文件测试入口。本次交接不需要让麦克风整夜运行。
