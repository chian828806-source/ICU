# File: module_03_wsi_processor.py

import os
# 1. 限制 CPU 占用 (必须在最前面)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import openslide

# 导入自定义模块
from model_loader import load_uni_components, DEVICE
from mlp_loader import load_mlp_components
from wsi_filter import WSIFilter, OPENSLIDE_DLL_PATH
# !!! 导入新写的去噪模块 !!!
from heatmap_postprocess import HeatmapSmoother 

# 确保 DLL 路径
if OPENSLIDE_DLL_PATH not in os.environ['PATH']:
    os.environ['PATH'] = OPENSLIDE_DLL_PATH + os.pathsep + os.environ['PATH']

# 配置
CLASS_NAMES = ['Normal', 'Benign', 'InSitu', 'Invasive']
NORMAL_CLASS_INDEX = 0
PATCH_SIZE = 224
TILE_LEVEL = 0

def process_wsi_and_visualize(wsi_path, uni_model, uni_transform, mlp_classifier, scaler):
    slide = openslide.OpenSlide(wsi_path)
    w, h = slide.dimensions
    grid_w, grid_h = w // PATCH_SIZE, h // PATCH_SIZE
    
    print(f"\n处理: {os.path.basename(wsi_path)}")
    print(f"尺寸: {w}x{h} | 网格: {grid_w}x{grid_h}")

    # --- 1. 背景过滤 ---
    print("Step 1: 背景过滤...")
    filter_proc = WSIFilter(slide, tile_size=PATCH_SIZE)
    valid_patches, _ = filter_proc.get_valid_patches()

    # --- 2. 推理循环 ---
    print(f"Step 2: 开始推理 (有效块数: {len(valid_patches)})...")
    prediction_grid = np.full((grid_h, grid_w), NORMAL_CLASS_INDEX, dtype=np.int32)
    
    mlp_classifier.eval()
    
    for (gx, gy, x, y) in tqdm(valid_patches, desc="Progress"):
        try:
            # 读取 & 预处理
            tile = slide.read_region((x, y), TILE_LEVEL, (PATCH_SIZE, PATCH_SIZE)).convert('RGB')
            img_tensor = uni_transform(tile).unsqueeze(0).to(DEVICE)
            
            # 模型推理
            with torch.no_grad():
                feats = uni_model(img_tensor).cpu().numpy()
            
            # 分类
            feats_scaled = scaler.transform(feats)
            feats_tensor = torch.from_numpy(feats_scaled).float().to(DEVICE)
            with torch.no_grad():
                logits = mlp_classifier(feats_tensor)
                pred = torch.argmax(logits, dim=1).item()
            
            # 填入网格
            prediction_grid[gy, gx] = pred
            
            # 休息一下降低CPU温度
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Error: {e}")

    slide.close()

    # --- 3. 后处理 (平滑去噪) ---
    print("Step 3: 热力图平滑去噪...")
    # 这里实例化类，你可以根据测试结果调整 parameters
    # median_ksize=5: 去除孤立噪点
    # close_ksize=5: 填补内部空洞
    smoother = HeatmapSmoother(median_ksize=5, close_ksize=5, open_ksize=3)
    smoothed_grid = smoother.process(prediction_grid)

    # --- 4. 可视化对比 ---
    print("Step 4: 生成结果图...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8 * grid_h / grid_w))
    
    # 设置颜色映射
    cmap = plt.cm.get_cmap('jet', len(CLASS_NAMES))
    
    # 左图：原始结果
    im1 = axes[0].imshow(prediction_grid, cmap=cmap, vmin=0, vmax=len(CLASS_NAMES)-1)
    axes[0].set_title(f"原始预测 (Raw Output)\n{os.path.basename(wsi_path)}")
    axes[0].axis('off')
    
    # 右图：平滑结果
    im2 = axes[1].imshow(smoothed_grid, cmap=cmap, vmin=0, vmax=len(CLASS_NAMES)-1)
    axes[1].set_title(f"平滑处理后 (Post-Processed)\nMedian=5, Close=5")
    axes[1].axis('off')
    
    # 共用 Colorbar
    cbar = fig.colorbar(im2, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_ticks(np.arange(len(CLASS_NAMES)))
    cbar.set_ticklabels(CLASS_NAMES)
    
    plt.show()

def main(wsi_path):
    uni, transform = load_uni_components()
    mlp, scaler = load_mlp_components()
    process_wsi_and_visualize(wsi_path, uni, transform, mlp, scaler)

if __name__ == "__main__":
    WSI_PATH = r"archive\ICIAR2018_BACH_Challenge\ICIAR2018_BACH_Challenge\WSI\A01.svs"
    main(WSI_PATH)