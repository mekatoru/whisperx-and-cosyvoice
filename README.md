# 🎙️ 端到端音频分离与零样本声音克隆 (Audio Separation & Voice Cloning)

本项目实现了一套完整的音频处理流水线：从“包含嘈杂背景音的多人对话”视频/音频中，自动剥离纯净人声，进行说话人识别与切片，最后利用大模型进行 Zero-shot（零样本）音色克隆。

## ⚠️ 核心避坑指南：多环境隔离运行

本项目集成了目前业界最前沿的三个音频大模型（SAM-Audio, WhisperX, CosyVoice）。**这三个模型的底层依赖（如 PyTorch 版本、CUDA 编译库、torchaudio 等）存在严重的版本冲突。** 强烈建议**不要**试图将它们安装在同一个 Python 虚拟环境中！请为以下三个步骤分别创建独立的 Conda 环境，以保证各模块稳定运行。

---

## 📁 核心模块说明

* **`sam_inference.py`**：**背景音去噪**。基于 SAM-Audio，利用文本 Prompt（如 "clean speech"）零样本提取纯净人声，剥离 BGM。
* **`test.py`**：**转录与说话人分离**。基于 WhisperX，输出带时间戳和说话人标签（Diarization）的 `.json` 结果。
* **`wclip.py`**：**自动化切片与文本挂载**。根据 WhisperX 的 JSON，精准裁剪目标人物的音频，**并自动导出对应的 `.txt` 参考文本**。
* **`cosy_inference.py`**：**纯本地音色克隆**。基于 CosyVoice-300M-Instruct，纯命令行离线推理，支持直接传入音频文件路径进行生成，彻底告别 WebUI。
* **`aclip.py`**：**备用切片方案**。根据 AssemblyAI API 返回的 JSON 数据，切分特定说话人的音频。

---

## 🚀 运行流水线 (分步操作指南)

假设我们的测试文件为 `example/street_inter.wav`，目标是提取里面 `SPEAKER_00` 的声音并克隆新台词。

### 🟢 Step 1: 复杂背景音去噪 (环境 A: SAM-Audio)

先用 SAM-Audio 剥离纯净人声，防止背景音干扰后续的说话人识别。
*(需提前将 `checkpoint.pt` 和 `config.json` 放在项目根目录)*

```bash
python sam_inference.py \
  --input "example/street_inter.wav" \
  --output "example/speech_clean.wav" \
  --prompt "clean speech"
```

### 🔵 Step 2: 识别、分离与切片 (环境 B: WhisperX)

切换到 WhisperX 环境，将去噪后的纯净人声拆开，并精准切出目标人物的音频和台词。

```bash
# 1. 识别并生成带有说话人标签的 JSON
python test.py \
  --audio "example/speech_clean.wav" \
  --output "example/test_result.json"

# 2. 提取 SPEAKER_00 的音频和对应的 TXT 文本
python wclip.py \
  --audio "example/speech_clean.wav" \
  --json "example/test_result.json" \
  --speaker "SPEAKER_00" \
  --output "example/only_SPEAKER_00.wav"
```

### 🔴 Step 3: 一键声音克隆 (环境 C: CosyVoice)

切换到 CosyVoice 环境。读取上一步切出来的干净原声和自动生成的 `.txt` 文本，直接在命令行完成 Zero-shot 克隆！

```bash
python cosy_inference.py \
  --model_dir "pretrained_models/CosyVoice-300M-Instruct" \
  --prompt_wav "example/only_SPEAKER_00.wav" \
  --prompt_text "$(cat example/only_SPEAKER_00.txt)" \
  --tts_text "你好，我是通过本地自动化流水线克隆出来的声音！" \
  --output "example/final_clone_output.wav"
```

---

## 📌 Git 提交与克隆注意事项

如果你要 Fork 或 Clone 本项目，请注意以下文件**不会**被提交到远程仓库，你需要自行准备或生成：

1. `pretrained_models/` 目录下的 CosyVoice 模型权重。
2. 项目根目录的 `checkpoint.pt` (SAM-Audio 权重)。
3. 流水线生成的 `.wav`、`.json` 和 `.txt` 等中间产物。

---