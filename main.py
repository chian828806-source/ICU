import sys
import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
import openslide

# --- 路径设置 ---
# 将 src 目录加入系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# --- 模块导入 ---
from src.models.uni_loader import load_uni_components
from src.models.mlp_loader import load_mlp_components
from src.utils.pre_filter import WSIFilter
from src.utils.post_process import HeatmapSmoother, visualize_and_save

# --- 全局常量 ---
CLASS_NAMES = ['Normal', 'Benign', 'InSitu', 'Invasive']
NORMAL_CLASS_INDEX = 0
PATCH_SIZE = 224
TILE_LEVEL = 0

def setup_environment(config):
    """配置 OpenSlide DLL """
    dll_path = config.get("openslide_bin_path")
    if dll_path and os.path.exists(dll_path):
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(dll_path)
        elif dll_path not in os.environ['PATH']:
            os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']
    else:
        print(f" Warning: OpenSlide bin path not configured or not found.")

def process_wsi(wsi_path, uni_model, uni_transform, mlp_classifier, scaler, device):
    """核心推理流程"""
    if not os.path.exists(wsi_path):
        print(f"!!Error: 输入文件不存在: {wsi_path}")
        return None

    try:
        slide = openslide.OpenSlide(wsi_path)
    except Exception as e:
        print(f"!!Error: 无法打开 WSI 文件: {e}")
        return None

    w, h = slide.dimensions
    grid_w, grid_h = w // PATCH_SIZE, h // PATCH_SIZE
    
    print(f"\n处理文件: {os.path.basename(wsi_path)}")
    print(f"尺寸: {w}x{h} | 网格: {grid_w}x{grid_h}")

    # 1. 背景过滤
    print("Step 1/3: 正在过滤背景...")
    filter_proc = WSIFilter(slide, tile_size=PATCH_SIZE)
    valid_patches, _ = filter_proc.get_valid_patches()

    # 2. 推理
    print(f"Step 2/3: 开始推理 ({len(valid_patches)} 个图块)...")
    prediction_grid = np.full((grid_h, grid_w), NORMAL_CLASS_INDEX, dtype=np.int32)
    
    mlp_classifier.eval()
    
    for (gx, gy, x, y) in tqdm(valid_patches, desc="Inference", unit="patch"):
        try:
            tile = slide.read_region((x, y), TILE_LEVEL, (PATCH_SIZE, PATCH_SIZE)).convert('RGB')
            img_tensor = uni_transform(tile).unsqueeze(0).to(device)
            
            with torch.no_grad():
                feats = uni_model(img_tensor).cpu().numpy()
            
            feats_scaled = scaler.transform(feats)
            feats_tensor = torch.from_numpy(feats_scaled).float().to(device)
            
            with torch.no_grad():
                logits = mlp_classifier(feats_tensor)
                pred = torch.argmax(logits, dim=1).item()
            
            prediction_grid[gy, gx] = pred
        except Exception as e:
            continue # 跳过损坏的块

    slide.close()
    return prediction_grid

def main():
    # --- 1. 命令行参数定义 ---
    parser = argparse.ArgumentParser(description="ICU WSI 比赛推理脚本")
    
    # 必须参数：--input 
    parser.add_argument('--input', type=str, required=True, 
                        help="输入的 WSI 图像路径 (.svs, .ndpi 等)")
    
    # 可选参数：--config 
    parser.add_argument('--config', type=str, default='./configs/inference_settings.json', 
                        help="配置文件路径")
    
    args = parser.parse_args()

    # --- 2. 加载配置 ---
    if not os.path.exists(args.config):
        print(f"!!Error: 找不到配置文件: {args.config}")
        print("请检查 configs/inference_settings.json 是否存在。")
        return

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # --- 3. 初始化 ---
    setup_environment(config)
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备: {device}")

    # --- 4. 加载模型 ---
    try:
        uni_model, uni_transform = load_uni_components(config['uni_checkpoint_dir'], device)
        mlp_model, scaler = load_mlp_components(config['mlp_weights_path'], config['scaler_path'], device)
    except Exception as e:
        print(f"!!Error: 模型加载失败: {e}")
        return

    # --- 5. 执行处理 ---
    wsi_name = os.path.splitext(os.path.basename(args.input))[0]
    prediction_grid = process_wsi(args.input, uni_model, uni_transform, mlp_model, scaler, device)

    # --- 6. 后处理与保存 ---
    if prediction_grid is not None:
        print("Step 3/3: 后处理与结果保存...")
        smoother = HeatmapSmoother(median_ksize=5, close_ksize=5, open_ksize=3)
        smoothed_grid = smoother.process(prediction_grid)
        
        output_dir = config.get("output_dir", "./output")
        visualize_and_save(prediction_grid, smoothed_grid, wsi_name, output_dir, CLASS_NAMES)

if __name__ == "__main__":
    main()