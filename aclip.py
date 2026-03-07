import os
import json
import argparse
from pydub import AudioSegment

def main(args):
    """
    根据 AssemblyAI API 返回的 JSON 数据，切分特定说话人的音频。
    """
    # 1. 路径校验
    if not os.path.exists(args.audio) or not os.path.exists(args.json):
        raise FileNotFoundError("音频文件或 JSON 配置文件不存在，请检查路径。")

    print(f"[*] 载入媒体文件: {args.audio}")
    audio = AudioSegment.from_file(args.audio)

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[*] 正在提取 Speaker [{args.speaker}] 的音频片段...")
    speaker_audio = AudioSegment.empty()
    match_count = 0

    # 2. 解析 AssemblyAI 的特定数据结构 ('utterances' 列表)
    if "utterances" in data:
        for utterance in data["utterances"]:
            if utterance.get("speaker") == args.speaker:
                match_count += 1
                
                # AssemblyAI 返回的时间戳单位已经是毫秒，直接使用
                start_ms = utterance.get("start", 0)
                end_ms = utterance.get("end", 0)
                
                # 执行切片并拼接
                speaker_audio += audio[start_ms:end_ms]
    else:
        print("[-] JSON 文件解析错误：缺少 'utterances' 字段。")
        return

    # 3. 导出音频
    if match_count > 0:
        speaker_audio.export(args.output, format="wav")
        print(f"[+] 提取成功！共匹配 {match_count} 段发言。")
        print(f"[+] 保存路径: {args.output} (总时长: {len(speaker_audio)/1000:.2f} 秒)")
    else:
        print(f"[-] 提取失败：未找到目标说话人 {args.speaker}。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据 AssemblyAI JSON 提取特定说话人音频")
    parser.add_argument("--audio", type=str, required=True, help="原始音频/视频路径")
    parser.add_argument("--json", type=str, required=True, help="AssemblyAI 生成的 JSON 文件路径")
    parser.add_argument("--speaker", type=str, required=True, help="目标说话人标签 (如 A, B, C)")
    parser.add_argument("--output", type=str, default="assembly_output.wav", help="输出音频的保存路径")
    args = parser.parse_args()
    main(args)