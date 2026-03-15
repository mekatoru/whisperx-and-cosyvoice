import os
import sys
import argparse
import subprocess

def run_cmd(cmd):
    """执行命令，附带日志打印和错误检查"""
    print(f"\n[*] 正在执行: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main(args):
    """
    端到端自动化流水线：SAM-Audio 去噪 -> WhisperX 识别 -> wclip 切片 -> CosyVoice 克隆
    """
    # 自动获取当前激活的 Python 解释器路径 (彻底告别硬编码绝对路径)
    python_exec = sys.executable

    # 1. 统一管理中间文件的输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 提取输入文件名（不带扩展名）用于生成对应的中间文件
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    
    clean_audio = os.path.join(args.out_dir, f"{base_name}_clean.wav")
    json_result = os.path.join(args.out_dir, f"{base_name}_result.json")
    prompt_wav = os.path.join(args.out_dir, f"{base_name}_{args.speaker}.wav")
    prompt_txt = os.path.join(args.out_dir, f"{base_name}_{args.speaker}.txt")

    try:
        # ==========================================
        # 步骤 1：SAM-Audio 剥离背景音
        # ==========================================
        if not args.skip_sam:
            run_cmd(f'{python_exec} sam_inference.py --input "{args.input}" --output "{clean_audio}" --prompt "clean speech"')
        else:
            print("[*] 选项触发: 已跳过 SAM-Audio 去噪，直接使用原始音频。")
            clean_audio = args.input

        # ==========================================
        # 步骤 2：WhisperX 识别转录与说话人分离
        # ==========================================
        run_cmd(f'{python_exec} test.py --audio "{clean_audio}" --output "{json_result}"')
        
        # ==========================================
        # 步骤 3：wclip.py 提取特定说话人的干净音频和文本
        # ==========================================
        run_cmd(f'{python_exec} wclip.py --audio "{clean_audio}" --json "{json_result}" --speaker "{args.speaker}" --output "{prompt_wav}"')
        
        # ==========================================
        # 步骤 4：读取参考文本，调用 CosyVoice 进行克隆
        # ==========================================
        if not os.path.exists(prompt_txt):
            raise FileNotFoundError(f"未找到参考文本 {prompt_txt}，请检查 wclip.py 是否正确导出了 txt 文件。")
            
        with open(prompt_txt, "r", encoding="utf-8") as f:
            prompt_text = f.read().strip()
            
        run_cmd(f'{python_exec} cosy_inference.py '
                f'--model_dir "{args.model_dir}" '
                f'--prompt_wav "{prompt_wav}" '
                f'--prompt_text "{prompt_text}" '
                f'--tts_text "{args.tts_text}" '
                f'--output "{args.output}"')
                
        print(f"\n[+] 全链路流水线完美收官！最终克隆音频已生成: {args.output}")

    except subprocess.CalledProcessError:
        print(f"\n[-] 流水线执行失败！程序在执行某一条命令时返回错误码，请往上翻看具体报错。")
    except Exception as e:
        print(f"\n[-] 发生意外错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="端到端音频分离与音色克隆流水线")
    parser.add_argument("--input", type=str, required=True, help="输入的原始音频/视频文件路径")
    parser.add_argument("--tts_text", type=str, required=True, help="想要克隆生成的新台词")
    parser.add_argument("--speaker", type=str, default="SPEAKER_00", help="目标说话人标签 (默认: SPEAKER_00)")
    parser.add_argument("--model_dir", type=str, default="pretrained_models/CosyVoice-300M-Instruct", help="CosyVoice 模型相对路径")
    parser.add_argument("--output", type=str, default="final_clone_output.wav", help="最终生成的音频路径")
    parser.add_argument("--out_dir", type=str, default="example", help="中间文件的存放目录 (默认: example)")
    parser.add_argument("--skip_sam", action="store_true", help="是否跳过 SAM-Audio 去噪步骤 (适合本身就很干净的音频)")
    
    args = parser.parse_args()
    main(args)