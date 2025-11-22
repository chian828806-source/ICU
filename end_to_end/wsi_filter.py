# File: wsi_filter.py
import os
import numpy as np
import cv2  # pip install opencv-python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
# ----------------------------------------------------------------------
# 配置：OpenSlide DLL 路径 (确保独立运行时也能找到DLL)
OPENSLIDE_DLL_PATH = r'S:\Openslide\openslide-win64\openslide-bin-4.0.0.8-windows-x64\bin'

if hasattr(os, 'add_dll_directory'):
    # Python 3.8+ Windows
    try:
        os.add_dll_directory(OPENSLIDE_DLL_PATH)
    except Exception:
        if OPENSLIDE_DLL_PATH not in os.environ['PATH']:
             os.environ['PATH'] = OPENSLIDE_DLL_PATH + os.pathsep + os.environ['PATH']

import openslide
# ----------------------------------------------------------------------

class WSIFilter:
    def __init__(self, slide, tile_size=224, sat_thresh=15, tissue_pct=0.1):
        """
        初始化过滤器
        :param slide: OpenSlide 对象
        :param tile_size: 切片尺寸 (224)
        :param sat_thresh: 饱和度阈值 (0-255)，越大过滤越严格(只留颜色深的)
        :param tissue_pct: 单个Patch中组织占比阈值 (0.0-1.0)，比如0.1表示只要有10%是组织就保留
        """
        self.slide = slide
        self.tile_size = tile_size
        self.sat_thresh = sat_thresh
        self.tissue_pct = tissue_pct

    def get_valid_patches(self):
        """
        核心功能：计算并返回有效Patch的坐标
        :return: 
            valid_patches: List [(grid_x, grid_y, abs_x, abs_y), ...]
            tissue_mask_grid: (H, W) 浮点数矩阵，用于后续可视化
        """
        w, h = self.slide.dimensions
        grid_w = w // self.tile_size
        grid_h = h // self.tile_size

        # 1. 获取缩略图 (使用下采样来加速计算)
        # 下采样32倍通常能在保留足够细节的同时极快地处理
        downsample = 32
        thumb_w = w // downsample
        thumb_h = h // downsample
        
        thumbnail = self.slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
        thumb_np = np.array(thumbnail)

        # 2. HSV 空间转换 (S通道提取)
        hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1]

        # 3. 二值化 (Tissue Mask)
        # 饱和度 > 阈值 设为 1 (组织)，否则为 0 (背景)
        _, binary_mask = cv2.threshold(s_channel, self.sat_thresh, 1, cv2.THRESH_BINARY)

        # 4. 映射回 Grid (核心步骤)
        # 我们将缩略图的Mask缩放到 (grid_w, grid_h) 的尺寸
        # 使用 INTER_AREA 插值，这样缩放后的像素值代表“区域内组织的平均比例”
        grid_ratios = cv2.resize(binary_mask.astype(np.float32), (grid_w, grid_h), interpolation=cv2.INTER_AREA)

        # 5. 筛选
        valid_patches = []
        # 找出比例 > 阈值的坐标
        ys, xs = np.where(grid_ratios > self.tissue_pct)
        
        for i in range(len(xs)):
            gx, gy = xs[i], ys[i]
            # 计算回 level 0 的绝对坐标
            abs_x = gx * self.tile_size
            abs_y = gy * self.tile_size
            valid_patches.append((gx, gy, abs_x, abs_y))

        return valid_patches, grid_ratios

# ==============================================================================
# 单元测试 / 可视化模块
# 只有直接运行此文件时才会执行，被import时不会执行
# ==============================================================================
if __name__ == "__main__":
    # 测试用的 WSI 路径
    TEST_WSI_PATH = r"archive\ICIAR2018_BACH_Challenge\ICIAR2018_BACH_Challenge\WSI\A01.svs"
    
    print(f"--- 开始测试 WSI 筛选模块 ---")
    print(f"加载: {TEST_WSI_PATH}")
    
    try:
        slide = openslide.OpenSlide(TEST_WSI_PATH)
        
        # 实例化过滤器 (可以在这里调整参数看效果)
        # 如果背景还是太多，调高 sat_thresh (比如 20)
        # 如果组织漏了，调低 tissue_pct (比如 0.05)
        wsi_filter = WSIFilter(slide, tile_size=224, sat_thresh=15, tissue_pct=0.1)
        
        valid_patches, grid_map = wsi_filter.get_valid_patches()
        
        print(f"筛选结果: 总 Grid 数 {grid_map.size}, 有效 Patch 数 {len(valid_patches)}")
        print(f"保留比例: {len(valid_patches)/grid_map.size:.2%}")
        
        # --- 可视化 ---
        plt.figure(figsize=(12, 6))
        
        # 左图：原始缩略图
        plt.subplot(1, 2, 1)
        thumb = slide.get_thumbnail((slide.dimensions[0]//64, slide.dimensions[1]//64))
        plt.imshow(thumb)
        plt.title("原始 WSI (缩略图)")
        plt.axis('off')
        
        # 右图：生成的二值 Mask
        plt.subplot(1, 2, 2)
        # 将 grid_map 二值化以便显示黑白
        display_mask = grid_map > wsi_filter.tissue_pct
        plt.imshow(display_mask, cmap='gray')
        plt.title(f"筛选 Mask (白色=保留)\nS_Thresh={wsi_filter.sat_thresh}, Pct={wsi_filter.tissue_pct}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        slide.close()
        print("测试完成。如果在右图中看到白色区域覆盖了所有组织，说明参数合适。")
        
    except Exception as e:
        print(f"测试出错: {e}")