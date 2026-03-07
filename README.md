# 音频分离与声音克隆测试汇报

这个项目主要是跑通了一套流水线：从“多人混音”的文件里提取纯净单人声音，再拿这个纯净声音去做零样本（Zero-shot）的音色克隆。

## 📁 文件说明与核心流程

* `test.py`：**WhisperX 核心推理脚本**。调用大模型对音频进行转录、词级对齐以及说话人分离（Diarization），最终输出带有时间戳和说话人标签的 `.json` 文件。
* `wclip.py`：**基于 WhisperX 的音频切片工具**。读取 `test.py` 生成的 JSON 文件，利用 `pydub` 精准提取指定说话人的纯净音频片段。
* `aclip.py`：**基于 AssemblyAI 的音频切片工具**。读取 AssemblyAI API 返回的 JSON 格式数据，切分并提取指定说话人的音频片段（作为另一套对比方案）。
* `street inter voiceonly.wav`：测试用的原始素材（一段包含多个人说话的英文街头采访）。
* `only_SPEAKER_00.wav`：用 WhisperX 成功单独切出来的目标人物的 10 秒纯净干音。
* `en93343873.wav`：用 CosyVoice 克隆生成的纯英文测试音频。
* `ch93343873.wav`：用 CosyVoice 克隆生成的跨语种测试音频（用原声的外国人音色说中文）。

## 🚀 跑通的流程记录

### 第一步：把混杂的声音拆开（音频分离）
因为克隆声音必须要有纯净的单人干音，所以我主要测试了怎么把多个人说话的音频准确切开。
* **尝试了两种方案**：一开始试了调 AssemblyAI 的接口（用 `aclip.py` 处理结果），对比后也在本地跑通了开源的 WhisperX（用 `wclip.py` 提取）。
* **提取成果**：成功切出了仅包含目标人物单独说话的短音频（如 `only_SPEAKER_00.wav`），供下一步克隆当素材用。
* **运行命令**：终端执行 `python wclip.py` 即可。

### 第二步：一键声音克隆（音色复刻）
拿到那几秒钟的干净原声后，我直接在本地启动了阿里开源的 CosyVoice 网页端进行测试。
* **部署方式**：直接启动官方的 WebUI，加载了 `CosyVoice-300M-Instruct` 预训练模型：
  ```bash
  python webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M-Instruct