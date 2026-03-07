import os
import json
import argparse
import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

def main(args):
    """
    WhisperX 核心推理流程：音频转录 -> 词级对齐 -> 说话人分离 (Diarization)
    """
    # 1. 稳健性校验：检查输入的音频文件是否存在
    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"未找到音频文件: {args.audio}")

    # 2. 安全读取 HuggingFace Token (避免将密钥硬编码在代码中泄露)
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("⚠️ 警告: 未检测到 HF_TOKEN 环境变量。若本地无缓存，模型下载可能会报错。")

    # 配置国内镜像源加速下载
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    device = "cuda"           # 使用 GPU 加速
    compute_type = "float16"  # 使用半精度降低显存占用

    # 3. 加载大语言模型并执行基础转录
    print(f"[*] 正在加载 WhisperX 模型处理: {args.audio}")
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    audio = whisperx.load_audio(args.audio)
    
    print("[*] 1/3 开始转录 (Transcription)...")
    result = model.transcribe(audio, batch_size=16)

    # 4. 时间戳精确对齐 (解决 Whisper 原生时间戳不准的问题)
    print("[*] 2/3 开始时间戳对齐 (Alignment)...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # 5. 基于 Pyannote 的说话人分离
    print("[*] 3/3 开始说话人分离 (Diarization)...")
    diarize_model = DiarizationPipeline(device=device)
    diarize_segments = diarize_model(audio)

    # 6. 将说话人标签合并回转录文本中，并落盘保存
    print("[*] 正在合并结果并导出 JSON...")
    result = assign_word_speakers(diarize_segments, result)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, ensure_ascii=False, indent=4)
    print(f"[+] 处理完成！结果已保存至 {args.output}")

if __name__ == "__main__":
    # 配置命令行参数解析器，提升脚本的通用性
    parser = argparse.ArgumentParser(description="WhisperX 说话人分离与转录脚本")
    parser.add_argument("--audio", type=str, required=True, help="输入的原始音频/视频文件路径")
    parser.add_argument("--output", type=str, default="test_result.json", help="输出的 JSON 文件名")
    args = parser.parse_args()
    main(args)