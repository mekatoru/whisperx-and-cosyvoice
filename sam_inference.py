# -*- coding: utf-8 -*-
import os
import torch
import torchaudio
import json
import argparse
from sam_audio import SAMAudio, SAMAudioProcessor
from sam_audio.model.config import SAMAudioConfig

def clean_state_dict(state_dict):
    """
    清洗权重字典 (Large模型专用):
    移除无关的训练参数和DDP前缀，确保严格加载兼容性
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("criterion.") or k.startswith("optimizer") or k.startswith("best_"): continue
        if k.startswith("module."): k = k[7:]
        new_state_dict[k] = v
    return new_state_dict

def main(args):
    # 显存碎片优化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    print(f"[*] 正在初始化 SAM-Audio (LARGE FP32) 推理引擎...")

    # ==========================================
    # 1. 模型配置与加载
    # ==========================================
    with open(args.config, 'r') as f: 
        config_dict = json.load(f)
    
    cfg = SAMAudioConfig()
    
    # --- Transformer Config (Large 修正版) ---
    if "transformer" in config_dict:
        t_cfg = config_dict["transformer"]
        cfg.transformer.dim = t_cfg.get("dim", 1536)
        cfg.transformer.n_layers = t_cfg.get("n_layers", 24) 
        cfg.transformer.n_heads = t_cfg.get("n_heads", 16)   
        cfg.transformer.context_dim = 2816                   

    print("[*] 正在构建模型并加载权重...")
    model = SAMAudio(cfg)
    
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"找不到权重文件: {args.ckpt}")
        
    state = torch.load(args.ckpt, map_location="cpu")
    if "model" in state: state = state["model"]
    
    clean_state = clean_state_dict(state)
    
    try:
        model.load_state_dict(clean_state, strict=True)
        print("[+] 模型权重加载成功.")
    except RuntimeError as e:
        print(f"[-] 模型加载失败: {str(e)[:500]}")
        return

    model.to(device).eval()

    # ==========================================
    # 2. 音频预处理
    # ==========================================
    TARGET_SR = 48000
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"找不到输入文件: {args.input}")

    wav, sr = torchaudio.load(args.input)
    
    if sr != TARGET_SR:
        print(f"[*] 重采样: {sr} -> {TARGET_SR} Hz")
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    
    # 转单声道
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    # 显存保护机制
    if args.duration > 0:
        print(f"[*] 为防止显存溢出，音频已截取前 {args.duration} 秒...")
        wav = wav[:, : int(TARGET_SR * args.duration)]
    
    wav = wav.to(device)

    # ==========================================
    # 3. 推理分离
    # ==========================================
    processor = SAMAudioProcessor.from_pretrained("facebook/sam-audio-large")
    inputs = processor(audios=[wav], descriptions=[args.prompt]).to(device)
    ode_opt = {"method": "rk4", "options": {"step_size": 2 / 64}}

    print(f"[*] 正在针对提示词 '{args.prompt}' 进行音频分离...")
    with torch.inference_mode():
        result = model.separate(inputs, ode_opt=ode_opt, predict_spans=False, reranking_candidates=1)

    # ==========================================
    # 4. 保存结果
    # ==========================================
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_wav = result.target[0].unsqueeze(0).cpu().float()
    
    torchaudio.save(args.output, out_wav, TARGET_SR)
    print(f"[+] 处理完成！分离结果已保存至: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM-Audio 文本驱动音频分离与去噪工具")
    parser.add_argument("--input", type=str, required=True, help="输入音频/视频文件路径")
    parser.add_argument("--output", type=str, default="output/separated.wav", help="输出的音频文件路径")
    parser.add_argument("--prompt", type=str, default="clean speech", help="要提取的声音文本提示词 (如 'only speech', 'background music')")
    parser.add_argument("--ckpt", type=str, default="checkpoint.pt", help="模型权重文件路径")
    parser.add_argument("--config", type=str, default="config.json", help="模型配置文件路径")
    parser.add_argument("--duration", type=float, default=15.0, help="截取前 N 秒处理防止爆显存 (设为 0 则处理全长)")
    
    args = parser.parse_args()
    main(args)