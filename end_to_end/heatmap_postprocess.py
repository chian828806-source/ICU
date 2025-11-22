# File: heatmap_postprocess.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
class HeatmapSmoother:
    def __init__(self, median_ksize=5, close_ksize=5, open_ksize=3):
        """
        初始化平滑处理器
        :param median_ksize: 中值滤波核大小 (必须是奇数, 如 3, 5, 7)。越大去噪越强，但细节丢失越多。
        :param close_ksize: 闭运算核大小 (填充空洞)。越大能填补越大的空白缝隙。
        :param open_ksize: 开运算核大小 (去除孤立小点)。越大能去除越大的孤立噪点。
        """
        self.median_ksize = median_ksize
        self.close_ksize = close_ksize
        self.open_ksize = open_ksize

    def process(self, grid):
        """
        执行平滑处理
        :param grid: (H, W) 的二维 numpy 数组，存储类别索引 (int)
        :return: 处理后的 grid
        """
        # 1. 确保数据类型为 uint8 (OpenCV 要求)
        processed_grid = grid.astype(np.uint8)

        # 2. 中值滤波 (Median Blur) - 去除椒盐噪声
        if self.median_ksize > 1:
            processed_grid = cv2.medianBlur(processed_grid, self.median_ksize)

        # 3. 形态学闭运算 (Closing) - 连接断裂区域，填补内部空洞
        if self.close_ksize > 1:
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_ksize, self.close_ksize))
            processed_grid = cv2.morphologyEx(processed_grid, cv2.MORPH_CLOSE, kernel_close)

        # 4. 形态学开运算 (Opening) - 消除细小的孤立毛刺
        if self.open_ksize > 1:
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_ksize, self.open_ksize))
            processed_grid = cv2.morphologyEx(processed_grid, cv2.MORPH_OPEN, kernel_open)

        return processed_grid

# ==============================================================================
# 单元测试：生成随机噪声图来测试平滑效果
# ==============================================================================
if __name__ == "__main__":
    print("--- 测试热力图去噪模块 ---")
    
    # 1. 生成模拟数据 (创建一个圆，然后撒盐)
    H, W = 100, 100
    y, x = np.ogrid[:H, :W]
    mask = (x - 50)**2 + (y - 50)**2 <= 30**2
    test_grid = np.zeros((H, W), dtype=int)
    test_grid[mask] = 1 # 假设 1 是肿瘤
    
    # 添加噪声 (随机撒点)
    noise = np.random.random((H, W))
    test_grid[noise > 0.95] = 1  # 背景里的噪点
    test_grid[(noise < 0.05) & mask] = 0 # 肿瘤里的空洞

    # 2. 处理
    # 调整这里的参数看看效果变化
    smoother = HeatmapSmoother(median_ksize=5, close_ksize=5, open_ksize=3)
    result = smoother.process(test_grid)
    
    # 3. 可视化对比
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(test_grid, cmap='jet')
    ax[0].set_title("原始噪声图")
    ax[1].imshow(result, cmap='jet')
    ax[1].set_title("平滑处理后")
    plt.show()