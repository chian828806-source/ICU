# File: module_03_wsi_processor.py (最终修正版 - 使用 PATH 注入)

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
# 导入模块一和模块二的加载接口
from model_loader import load_uni_components, DEVICE
from mlp_loader import load_mlp_components


# ----------------------------------------------------------------------
# 1. 配置：OpenSlide DLL 路径 (必须是包含 DLL 文件的文件夹路径，无引号!)
# 请确保这是 'bin' 文件夹的路径，而不是 'libopenslide-1.dll' 文件路径!
# 修正后的路径示例: r'S:\Openslide\openslide-win64\openslide-bin-4.0.0.8-windows-x64\bin'
OPENSLIDE_DLL_PATH = r'S:\Openslide\openslide-win64\openslide-bin-4.0.0.8-windows-x64\bin' 
# ----------------------------------------------------------------------


# --- 2. WSI 处理和分类配置 ---
CLASS_NAMES = ['Normal', 'Benign', 'InSitu', 'Invasive']
PATCH_SIZE = 224  # UNI 模型输入尺寸
TILE_LEVEL = 0    # 提取图块的 WSI 级别 (0为最高分辨率)
TILE_STEP = 224   # 分块步长 (不重叠)


# ----------------------------------------------------
# 核心功能：WSI 全部分割、分类与可视化
# ----------------------------------------------------
def process_wsi_and_visualize(wsi_path, uni_model, uni_transform, mlp_classifier, scaler):
    """
    对 WSI 进行全部分块、特征提取、标准化、MLP分类并生成热图。
    """
    print(f"\n--- 模块 03: 正在处理 WSI 图像: {os.path.basename(wsi_path)} ---")
    
    # 强制链接 OpenSlide DLL 文件 (使用 os.environ['PATH'] 更稳定)
    try:
        import openslide  # 尝试直接导入
    except ImportError:
        # 如果直接导入失败，尝试在导入前添加路径
        try:
            # 仅当路径尚未在 PATH 中时才添加，并使用 os.pathsep (;) 分隔
            if OPENSLIDE_DLL_PATH not in os.environ['PATH']:
                os.environ['PATH'] = OPENSLIDE_DLL_PATH + os.pathsep + os.environ['PATH']
            import openslide  # 再次尝试导入
        except Exception as e:
            # 如果仍失败，输出错误信息并返回
            print(f"致命错误：无法加载 OpenSlide 库。请检查路径: {OPENSLIDE_DLL_PATH}。")
            print(f"详细错误: {e}")
            return
            
    # 打开 WSI 文件
    try:
        slide = openslide.OpenSlide(wsi_path)
    except Exception as e:
        print(f"错误：无法打开 WSI 文件。请检查文件路径和格式。{e}")
        return

    width, height = slide.level_dimensions[TILE_LEVEL]
    grid_w = width // TILE_STEP
    grid_h = height // TILE_STEP
    prediction_grid = np.zeros((grid_h, grid_w), dtype=np.int32)
    
    print(f"WSI 尺寸: {width}x{height}. 网格尺寸: {grid_w}x{grid_h}")

    # --- 遍历分块和推理 ---
    for i in tqdm(range(grid_w), desc="WSI 分块处理"):
        for j in range(grid_h):
            x = i * TILE_STEP
            y = j * TILE_STEP

            # 1. 提取分块、预处理
            tile = slide.read_region((x, y), TILE_LEVEL, (PATCH_SIZE, PATCH_SIZE)).convert('RGB')
            image_tensor = uni_transform(tile).unsqueeze(0).to(DEVICE)

            # 2. 特征提取 (UNI 模型)
            with torch.no_grad():
                feature_vector = uni_model(image_tensor)
            
            # 3. 特征标准化 (StandardScaler)
            feature_vector_np = feature_vector.cpu().numpy()
            feature_vector_scaled = scaler.transform(feature_vector_np)
            
            # 4. 转换为 Tensor 供 MLP 使用
            feature_tensor_scaled = torch.from_numpy(feature_vector_scaled).float().to(DEVICE)

            # 5. MLP 分类器预测
            with torch.no_grad():
                mlp_classifier.eval()
                logits = mlp_classifier(feature_tensor_scaled)
                predicted_index = torch.argmax(logits, dim=1).item()
            
            # 6. 存储结果
            prediction_grid[j, i] = predicted_index
            
    slide.close()
    
    # --- 可视化 ---
    plt.figure(figsize=(10, 10 * grid_h / grid_w))
    cmap = plt.cm.get_cmap('jet', len(CLASS_NAMES)) 
    plt.imshow(prediction_grid, cmap=cmap, vmin=0, vmax=len(CLASS_NAMES)-1)
    
    # 添加颜色棒和标签
    cbar = plt.colorbar(ticks=np.arange(len(CLASS_NAMES)))
    cbar.set_ticklabels(CLASS_NAMES)
    plt.title(f"WSI 预测热图 ({os.path.basename(wsi_path)})")
    plt.xlabel(f"分块 x 坐标 ({TILE_STEP}px/块)")
    plt.ylabel(f"分块 y 坐标 ({TILE_STEP}px/块)")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------
# 主函数：运行整个系统
# ----------------------------------------------------
def main_run_system(wsi_file_path):
    # 1. 加载所有组件 (通过模块一和模块二的接口)
    uni_model, uni_transform = load_uni_components() 
    mlp_classifier, scaler = load_mlp_components()   
    
    # 2. 处理并可视化 WSI
    process_wsi_and_visualize(wsi_file_path, uni_model, uni_transform, mlp_classifier, scaler)


if __name__ == "__main__":
    # 3. 配置：WSI 文件路径
    # 替换为您 WSI 图像的实际路径 (.svs, .tif 等)
    WSI_FILE_PATH = r"archive\ICIAR2018_BACH_Challenge\ICIAR2018_BACH_Challenge\WSI\A01.svs" 
    
    main_run_system(WSI_FILE_PATH)