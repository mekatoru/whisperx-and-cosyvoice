import os
import torch
import torchaudio
import argparse
from cosyvoice.cli.cosyvoice import CosyVoice
# 注意：我们不再需要 load_wav，因为我们要把路径直接给模型

def main(args):
    print(f"[*] 正在加载 CosyVoice 模型: {args.model_dir} ...")
    # 初始化模型
    cosyvoice = CosyVoice(args.model_dir)
    
    print(f"[*] 正在加载参考音频 (Prompt Audio): {args.prompt_wav}")
    print(f"[*] 目标合成文本: '{args.tts_text}'")
    print(f"[*] 正在执行零样本 (Zero-shot) 音色克隆推理...")
    
    # 第三个参数直接传 args.prompt_wav (路径字符串)，而不是读取后的 tensor
    output = cosyvoice.inference_zero_shot(args.tts_text, args.prompt_text, args.prompt_wav)
    
    # 提取生成的音频张量并保存
    for i, j in enumerate(output):
        out_wav = j['tts_speech']
        # CosyVoice 默认生成的音频采样率为 22050Hz
        torchaudio.save(args.output, out_wav, 22050)
        print(f"\n[+] 音色克隆完成！结果已保存至: {args.output}")
        break 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CosyVoice 本地命令行推理脚本 (Zero-shot)")
    parser.add_argument('--model_dir', type=str, default='pretrained_models/CosyVoice-300M-Instruct', help='模型路径')
    parser.add_argument('--prompt_wav', type=str, required=True, help='参考音频路径')
    parser.add_argument('--prompt_text', type=str, required=True, help='参考音频里的文字内容')
    parser.add_argument('--tts_text', type=str, required=True, help='你想让它说的新台词')
    parser.add_argument('--output', type=str, default='cloned_output.wav', help='输出文件名')
    
    args = parser.parse_args()
    main(args)