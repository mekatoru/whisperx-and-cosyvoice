import os
import json
import argparse
from pydub import AudioSegment

def main(args):
    """
    根据 WhisperX 输出的 JSON 时间戳，精准裁剪并拼接特定说话人的音频。
    """
    # 1. 路径校验
    if not os.path.exists(args.audio) or not os.path.exists(args.json):
        raise FileNotFoundError("音频文件或 JSON 配置文件不存在，请检查路径。")

    # 2. 载入原始音频 (pydub 支持直接从 mp4 提取音频轨)
    print(f"[*] 载入媒体文件: {args.audio}")
    audio = AudioSegment.from_file(args.audio)

    with open(args.json, "r", encoding="utf-8") as f:
        segments = json.load(f)

    print(f"[*] 正在提取说话人 [{args.speaker}] 的音频片段...")
    speaker_audio = AudioSegment.empty()  # 初始化一个空的音频容器
    text_transcript = []

    # 3. 遍历 JSON，进行时间戳匹配与音频切片
    for seg in segments:
        if seg.get("speaker") == args.speaker:
            text_transcript.append(seg.get("text", ""))
            
            # 提取时间戳 (pydub 要求的单位是毫秒，因此乘以 1000)
            start_ms = int(seg.get("start", 0) * 1000)
            end_ms = int(seg.get("end", 0) * 1000)
            
            # 切片并拼接到总音频中
            speaker_audio += audio[start_ms:end_ms]

    # 4. 导出最终干音
    if len(speaker_audio) > 0:
        speaker_audio.export(args.output, format="wav")
        print(f"[+] 提取成功！保存路径: {args.output}")
        print(f"[*] 提取总时长: {len(speaker_audio) / 1000:.2f} 秒")
    else:
        print(f"[-] 提取失败：JSON 数据中未找到说话人 {args.speaker}。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据 WhisperX JSON 提取特定说话人音频")
    parser.add_argument("--audio", type=str, required=True, help="原始音频/视频路径")
    parser.add_argument("--json", type=str, required=True, help="WhisperX 生成的 JSON 文件路径")
    parser.add_argument("--speaker", type=str, required=True, help="目标说话人标签 (如 SPEAKER_00)")
    parser.add_argument("--output", type=str, default="output.wav", help="输出音频的保存路径")
    args = parser.parse_args()
    main(args)