# 模型调研与证据

资料核查截至 2026-10-10。以下为公开文档和代码核查，不代表本项目的实测结果。权重链接、接口和依赖可能变化，实施前需重新检查。

2026-10-13 补充：首版实际采用 Conv-TasNet + ECAPA + Silero，尚未接入下文建议的 diart / WeSep。当前依据、实现和实验结果见 [实现与实测](implementation.md)。

2026-10-15 补充：diart 0.9.2 可选依赖已在本机安装，独立文件验证入口和流式程序测试已接通；标准分段权重需要 Hugging Face 授权及登录，真实模型尚未运行，主程序尚未切换为 diart。下方“尚未安装”是早期调研状态，当前状态以 [实现与实测](implementation.md) 为准。

## 1. 两类不同任务

| 任务 | 需要的输出 | 对应产品行为 |
| --- | --- | --- |
| 说话人分段与聚类（diarization） | 时间段、匿名身份、说话概率 | 新增按钮、复用按钮、显示谁在讲话 |
| 目标说话人提取（TSE） | 从混合音频中提取的目标波形 | 只播放选中的人 |

标记出重叠发言不等于拆出了独立声音。ASR 转写、TTS 音色克隆也不能替代目标说话人提取。

## 2. 非 Step 候选

### diart + pyannote

- [官方仓库](https://github.com/juanmc2005/diart)
- 结合说话人分段模型、声音嵌入与增量聚类，面向实时说话人识别。
- 提供麦克风输入和流式处理示例；默认模型需要在 Hugging Face 接受相应使用条件。
- 可用于本次会话内区分新人及再次出现的人；准确性并非保证，跨会话持久身份不是本项目已验证的能力。
- 不输出每个人独立的音频。
- 论文/实现讨论的延迟配置范围为 0.5–5 秒，不能当作本机端到端结果。
- 本项目尚未安装依赖、取得权重或跑通 Windows 麦克风流。

### NVIDIA Nemotron-3-Diarization

- [官方模型卡与权重](https://huggingface.co/nvidia/Nemotron-3-Diarization)
- 支持流式和离线推理，最多 8 名说话人，按首次出现顺序组织输出。
- 使用说话人缓存维护音频块之间的身份；输出说话活动概率和标签，不是分离波形。
- 推荐流式输入缓冲档位为 0.32、0.64、1.04 秒，不含计算时间。
- 官方主要支持 Linux / NVIDIA GPU，亦提供 Transformers 用法与原生运行时链接；不能据此保证当前 Intel Windows PC 实时可用。
- 是自动发现模块的另一候选，尚未实测。

### WeSep / REAL-TSE

- [官方仓库](https://github.com/REAL-TSE/wesep-real-tse)
- [推理入口 evaluate.py](https://github.com/REAL-TSE/wesep-real-tse/blob/main/evaluate.py)
- 接收混合音频和目标参考声音，输出提取音频。
- README 提供 Google Drive 预训练权重链接，包含 BSRNN 因果和非因果候选；尚未下载或验证权重加载。
- 推理入口有 CPU 和 CUDA 选项，默认 CPU。
- 当前核查的入口是文件推理。因果模型存在，不代表流式缓存、连续分块与低延迟播放已经实现。
- 建议优先用同一段混合录音配不同参考声音，验证能否正确提取 A/B，再评估速度和连续处理。

### Hearpoint

- [项目仓库](https://github.com/Hearpoint-ai/hearpoint_realtime)
- 采用登记声音、流式因果 TFGridNet、提取/原声切换，和本项目目标接近。
- 实现面向 Apple Silicon / CoreML，16 kHz 双声道输入输出，原型使用额外双耳麦克风。
- 可参考完整链路，不能直接视为 Windows + AirPods 麦克风方案。
- 不沿用 README 的延迟或医疗/监管相关结论；本项目没有复现这些结果。

### Look Once to Hear / Target Speech Hearing

- [研究主页](https://tsh.cs.washington.edu/)
- [代码](https://github.com/vb000/LookOnceToHear)
- 已有通过声音登记后保留目标人声的耳机研究，不能将“指定听某人”宣称为首次提出。
- 前期只读过 README、依赖和测试代码；未训练、未推理。README 中未确认直接可用的预训练权重下载入口。
- 前期代码检查发现：缺少 checkpoint 时测试可能仅警告后继续。使用时必须核验真实权重存在并成功加载，不能把随机权重输出当作结果。

## 3. Step 模型核查

### StepAudio 2.5 ASR：有离线说话人标签

[音频文件识别官方接口](https://platform.stepfun.com/docs/zh/api-reference/audio/asr)

- 推荐模型 `stepaudio-2.5-asr`。
- `enable_speaker_info=true` 和 `show_utterances=true` 返回每句的 `speaker.id`、文本与时间戳。
- 单任务最多区分 10 人，同一任务内 ID 稳定，跨任务不保证。
- 异步文件处理；不返回分离后的音频，不能直接承担实时按人选择收听。

### StepAudio 2.5 ASR Stream

[双向流式识别官方接口](https://platform.stepfun.com/docs/zh/api-reference/audio/asr-stream)

公开文档包含实时音频、VAD、转写和时间戳，未列出持续说话人 ID 字段。不能将实时转写等同于实时发现新人。

### StepAudio 3 ASR / Realtime

- [ASR 官方说明](https://platform.stepfun.com/docs/zh/guides/models/stepaudio-3-asr)：主要输出增量及最终文本。
- [Realtime 官方说明](https://platform.stepfun.com/docs/zh/guides/models/stepaudio-3-realtime)：全双工对话、声音理解和工具调用。
- [Realtime 事件接口](https://platform.stepfun.com/docs/zh/api-reference/realtime/chat)：音频输出事件描述的是模型生成的音频，未找到指定输入说话人原始音轨的输出能力。

### Step-Audio-EditX

[官方模型卡](https://huggingface.co/stepfun-ai/Step-Audio-EditX)

主要为情绪、风格、副语言编辑及 TTS/音色克隆，公开能力列表未包含重叠人声分离。

### Step 当前结论

已找到相关能力，但未找到能直接完成“流式发现新人 + 身份持续稳定 + 输出指定人声波形”的公开一体方案。此结论不排除 Step 内部或尚未公开的能力。

若向音频团队咨询，应明确询问：流式声纹识别/说话人聚类，以及能输出波形的目标说话人提取；不能只问“能否识别语音”。

## 4. 当前建议

优先验证 diart + WeSep / REAL-TSE，Nemotron 作为身份识别模块的另一候选。这是实验起点，不是确定可交付的栈。

两个模块通过应用内身份和自动采集的参考声音连接；不能假定不同模型的 embedding 可以直接互换。

## 5. 设备资料

- [Apple：AirPods 与非 Apple 设备配对](https://support.apple.com/zh-cn/guide/airpods/dev499c9718b/27/web/27)
- [Microsoft：Windows 经典蓝牙音频](https://learn.microsoft.com/en-us/windows-hardware/drivers/bluetooth/bluetooth-classic-audio)

经典蓝牙的 A2DP 用于播放，HFP 支持同时单声道收音和播放；实际设备格式、音质和延迟需实测。

历史 iPhone 路线的参考（当前不实施）：

- [Apple：bluetoothHighQualityRecording](https://developer.apple.com/documentation/avfaudio/avaudiosession/categoryoptions-swift.struct/bluetoothhighqualityrecording)
- [WWDC25：AirPods 高质量录音](https://developer.apple.com/videos/play/wwdc2025/251/)

高质量录音可用性不等于低延迟全双工已经验证，也不能默认 Windows 提供相同能力。
