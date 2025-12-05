import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

class HeatmapSmoother:
    def __init__(self, median_ksize=5, close_ksize=5, open_ksize=3):
        """
        初始化平滑处理器
        :param median_ksize: 中值滤波核大小 (必须是奇数, 如 3, 5, 7)。去除椒盐噪声。
        :param close_ksize: 闭运算核大小。填补内部空洞。
        :param open_ksize: 开运算核大小。去除孤立噪点。
        """
        self.median_ksize = median_ksize
        self.close_ksize = close_ksize
        self.open_ksize = open_ksize
        
        # 预先生成结构元素，提高处理效率
        if self.close_ksize > 1:
            self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_ksize, self.close_ksize))
        if self.open_ksize > 1:
            self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_ksize, self.open_ksize))

    def process(self, grid):
        """
        执行平滑处理
        :param grid: (H, W) 的二维 numpy 数组
        :return: 处理后的 grid (uint8)
        """
        # 确保数据类型为 uint8 (OpenCV 要求)
        processed_grid = grid.astype(np.uint8)

        # 中值滤波 
        if self.median_ksize > 1:
            processed_grid = cv2.medianBlur(processed_grid, self.median_ksize)

        # 3. 形态学闭运算 - 填补空洞
        if self.close_ksize > 1:
            processed_grid = cv2.morphologyEx(processed_grid, cv2.MORPH_CLOSE, self.kernel_close)

        # 4. 形态学开运算- 去除噪点
        if self.open_ksize > 1:
            processed_grid = cv2.morphologyEx(processed_grid, cv2.MORPH_OPEN, self.kernel_open)

        return processed_grid


def visualize_and_save(original_grid, smoothed_grid, wsi_name, output_dir, class_names):
    """
    可视化原始结果与平滑结果，并保存对比图。
    
    Args:
        original_grid: 原始预测网格 (numpy array)
        smoothed_grid: 平滑后的网格 (numpy array)
        wsi_name: WSI 文件的名称 (用于标题和文件名)
        output_dir: 保存目录
        class_names: 类别名称列表 (用于 Colorbar)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    grid_h, grid_w = original_grid.shape
    # 防止宽度过小导致除零或报错
    grid_w = max(1, grid_w)
    
    # 动态计算图像比例
    fig, axes = plt.subplots(1, 2, figsize=(18, 8 * grid_h / grid_w))
    
    # 获取颜色映射
    cmap = plt.cm.get_cmap('jet', len(class_names))
    
    # --- 左图：原始预测 ---
    axes[0].imshow(original_grid, cmap=cmap, vmin=0, vmax=len(class_names)-1)
    axes[0].set_title(f"原始预测 (Raw Prediction)\n{wsi_name}")
    axes[0].axis('off')
    
    # --- 右图：平滑后 ---
    im2 = axes[1].imshow(smoothed_grid, cmap=cmap, vmin=0, vmax=len(class_names)-1)
    axes[1].set_title(f"平滑处理后 (Post-Processed)\n{wsi_name}")
    axes[1].axis('off')
    
    # --- 公共 Colorbar ---
    cbar = fig.colorbar(im2, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_ticks(np.arange(len(class_names)))
    cbar.set_ticklabels(class_names)
    
    # --- 保存 ---
    save_path = os.path.join(output_dir, f"{wsi_name}_result.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"结果图已保存至: {save_path}")
    plt.close() #


# 单元测试模块
if __name__ == "__main__":
    print("--- 单元测试: 热力图去噪与绘图 ---")
    
    # 1. 生成模拟数据 (圆+噪声)
    H, W = 100, 100
    y, x = np.ogrid[:H, :W]
    mask = (x - 50)**2 + (y - 50)**2 <= 30**2
    test_grid = np.zeros((H, W), dtype=int)
    test_grid[mask] = 1 # 假设 1 是肿瘤
    
    # 添加随机噪声
    noise = np.random.random((H, W))
    test_grid[noise > 0.95] = 1  # 噪点
    test_grid[(noise < 0.05) & mask] = 0 # 内部空洞

    # 2. 测试平滑类
    smoother = HeatmapSmoother(median_ksize=5, close_ksize=5, open_ksize=3)
    result = smoother.process(test_grid)
    
    # 3. 测试绘图函数
    print("正在测试绘图保存功能...")
    dummy_classes = ['Background', 'Tumor']
    visualize_and_save(test_grid, result, "Test_Unit_Image", "./test_output", dummy_classes)
    
    print("单元测试完成，请检查 ./test_output 文件夹。")