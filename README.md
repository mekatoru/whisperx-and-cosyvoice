# 音频分离与声音克隆测试汇报

这个项目主要是跑通了一套流水线：从“多人混音”的文件里提取纯净单人声音，再拿这个纯净声音去做零样本（Zero-shot）的音色克隆。

## 📁 文件说明与核心流程

* `sam_inference.py`：**[新增] SAM-Audio 音轨分离脚本**。基于文本指令（Prompt）的零样本声音分离工具，用于在送入识别前，剥离复杂的背景音乐（BGM）或环境噪音。
* `test.py`：**WhisperX 核心推理脚本**。调用大模型对去噪后的音频进行转录、词级对齐以及说话人分离（Diarization），最终输出带有时间戳和说话人标签的 `.json` 文件。
* `wclip.py`：**基于 WhisperX 的音频切片工具**。读取 `test.py` 生成的 JSON，利用 `pydub` 精准提取指定说话人的纯净音频片段。
* `aclip.py`：**基于 AssemblyAI 的音频切片工具**。读取 API 返回的数据进行切分（作为备用对比方案）。
* `example/`：**[新增] 测试素材与结果收纳目录**。
  * `street inter voiceonly.wav`：原始素材（一段包含多个人说话的英文街头采访）。
  * `speech.wav`：经过 SAM-Audio 去除背景音后提取的纯净人声。
  * `only_SPEAKER_00.wav`：用 WhisperX 成功单独切出来的目标人物的纯净干音。
  * `en93343873.wav` / `ch93343873.wav`：用 CosyVoice 克隆生成的纯英文/跨语种（老外配音说中文）测试音频。

## 🚀 跑通的流程记录

### 🟢 第一步：复杂背景音去噪与音轨分离 (SAM-Audio)
由于网络限制，脚本已支持完全离线加载。只需将 checkpoint.pt 和 config.json 放在项目根目录即可直接运行。
**痛点**：如果原始素材（如漫威电影）包含巨大的 BGM 和特效音，直接做说话人分离（Diarization）效果不佳。
**解法**：先用 SAM-Audio 剥离纯净人声。
* **运行命令**：
  ```bash
  python sam_inference.py --input example/原始视频.mp4 --output example/speech.wav --prompt "clean speech"

### 第二步：把混杂的声音拆开（音频分离）
拿到第一步去噪后的纯净人声（speech.wav）后，开始把多个人说话的音频准确切开。
test.py 用来获取给wclip.py使用的json文件
* **尝试了两种方案**：一开始试了调 AssemblyAI 的接口（用 `aclip.py` 处理结果），对比后也在本地跑通了开源的 WhisperX（用 `wclip.py` 提取）。
* **提取成果**：成功切出了仅包含目标人物单独说话的短音频（如 `only_SPEAKER_00.wav`），供下一步克隆当素材用。
* **运行命令**：终端执行 `python wclip.py` 即可。

### 第三步：一键声音克隆（音色复刻）
拿到那几秒钟的干净原声后，我直接在本地启动了阿里开源的 CosyVoice 网页端进行测试。
* **部署方式**：直接启动官方的 WebUI，加载了 `CosyVoice-300M-Instruct` 预训练模型：
  ```bash
  python webui.py --port 50000 --model_dir pretrained_models/CosyVoice-300M-Instruct