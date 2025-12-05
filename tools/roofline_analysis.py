"""
一站式分析 UNI-ViT + MLP 的 Roofline 模型
用于评估模型的计算强度 (Arithmetic Intensity)
"""

import sys
import os
import torch
import torch.nn as nn
import timm
from thop import profile
import matplotlib.pyplot as plt
import numpy as np

# --- 核心：将项目根目录加入路径，以便导入 src ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# 导入规范化后的模型加载器
from src.models.mlp_loader import MLPClassifier 
from src.models.mlp_model import MLPClassifier 


# 工具函数
def count_weight_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())

class ActivationMemoryProfiler:
    def __init__(self):
        self.total_read_bytes = 0
        self.total_write_bytes = 0
        self.hooks = []

    def _hook_fn(self, module, input, output):
        if isinstance(input, tuple):
            for inp in input:
                if isinstance(inp, torch.Tensor) and inp.is_floating_point():
                    self.total_read_bytes += inp.numel() * inp.element_size()
        elif isinstance(input, torch.Tensor) and input.is_floating_point():
            self.total_read_bytes += input.numel() * inp.element_size()

        if isinstance(output, torch.Tensor) and output.is_floating_point():
            self.total_write_bytes += output.numel() * output.element_size()
        elif isinstance(output, (tuple, list)):
            for out in output:
                if isinstance(out, torch.Tensor) and out.is_floating_point():
                    self.total_write_bytes += out.numel() * out.element_size()

    def register_hooks(self, model):
        for name, module in model.named_modules():
            if (len(list(module.children())) == 0 
                and not isinstance(module, (nn.Sequential, nn.ModuleList))
                and hasattr(module, 'forward')):
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_total_bytes(self):
        return self.total_read_bytes + self.total_write_bytes

def build_uni_vit():
    # 为了 Roofline 分析，不需要加载真实权重，只需要结构一致即可
    return timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=False
    )


# 主分析函数
def analyze_models():
    device = "cpu"
    print("=" * 70)
    print("正在分析模型 FLOPs 与带宽...")
    print("=" * 70)

    # --- ViT ---
    print("\n[1/2] 分析 UNI-ViT (结构)...")
    vit_model = build_uni_vit().to(device).eval()
    vit_input = torch.randn(1, 3, 224, 224).to(device)
    
    # 计算 FLOPs
    vit_flops, _ = profile(vit_model, inputs=(vit_input,), verbose=False)
    
    # 计算字节数
    vit_w_bytes = count_weight_bytes(vit_model)
    profiler = ActivationMemoryProfiler()
    profiler.register_hooks(vit_model)
    with torch.no_grad():
        _ = vit_model(vit_input)
    vit_a_bytes = profiler.get_total_bytes()
    profiler.remove_hooks()
    vit_i_bytes = vit_input.numel() * vit_input.element_size()
    vit_bytes = vit_w_bytes + vit_a_bytes + vit_i_bytes
    vit_ai = vit_flops / vit_bytes

    # --- MLP ---
    print("[2/2] 分析 MLP (结构)...")
    mlp_model = MLPClassifier(input_dim=1024, num_classes=4).to(device).eval()
    mlp_input = torch.randn(1, 1024).to(device)
    
    mlp_flops, _ = profile(mlp_model, inputs=(mlp_input,), verbose=False)
    
    mlp_w_bytes = count_weight_bytes(mlp_model)
    profiler = ActivationMemoryProfiler()
    profiler.register_hooks(mlp_model)
    with torch.no_grad():
        _ = mlp_model(mlp_input)
    mlp_a_bytes = profiler.get_total_bytes()
    profiler.remove_hooks()
    mlp_i_bytes = mlp_input.numel() * mlp_input.element_size()
    mlp_bytes = mlp_w_bytes + mlp_a_bytes + mlp_i_bytes
    mlp_ai = mlp_flops / mlp_bytes

    return {
        'vit': {'flops': vit_flops, 'bytes': vit_bytes, 'ai': vit_ai},
        'mlp': {'flops': mlp_flops, 'bytes': mlp_bytes, 'ai': mlp_ai}
    }


# 绘制 Roofline 图
def plot_roofline(results, save_path):
    # 昇腾 910B 参数 (参考值)
    peak_compute = 320  # TFLOPS (FP16)
    peak_bandwidth = 0.4  # TB/s
    ridge_point = peak_compute / peak_bandwidth

    vit_ai = results['vit']['ai']
    mlp_ai = results['mlp']['ai']

    # 理论性能上限
    vit_perf = min(peak_compute, vit_ai * peak_bandwidth)
    mlp_perf = min(peak_compute, mlp_ai * peak_bandwidth)

    # 绘图设置
    plt.figure(figsize=(10, 6))
    
    # 绘制 Roofline 线
    ai_range = np.logspace(-1, 3, 500)
    roofline = np.minimum(peak_compute, ai_range * peak_bandwidth)
    plt.plot(ai_range, roofline, 'k-', linewidth=2, label='Roofline (Ascend 910B)')
    plt.axvline(x=ridge_point, color='red', linestyle='--', label=f'Ridge Point (AI={ridge_point:.0f})')

    # 绘制模型点
    plt.scatter([vit_ai], [vit_perf], s=150, color='blue', edgecolor='black', zorder=5)
    plt.scatter([mlp_ai], [mlp_perf], s=150, color='green', edgecolor='black', zorder=5)

    plt.text(vit_ai * 1.2, vit_perf * 0.9, 'UNI-ViT', fontsize=10, color='blue', fontweight='bold')
    plt.text(mlp_ai * 1.5, mlp_perf * 1.1, 'MLP', fontsize=10, color='green', fontweight='bold')

    plt.xscale('log')
    plt.yscale('log') # 性能轴通常也可以用对数坐标，如果差异很大的话
    plt.xlabel('Arithmetic Intensity (FLOPs / Byte)', fontsize=12)
    plt.ylabel('Performance (TFLOPS)', fontsize=12)
    plt.title('Roofline Analysis: UNI-ViT vs MLP', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nRoofline 图已保存至: {save_path}")


# 主程序
def main():
    # 设置输出目录
    output_dir = os.path.join(PROJECT_ROOT, "output", "analysis")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    results = analyze_models()

    # 计算百分比
    total_flops = results['vit']['flops'] + results['mlp']['flops']
    total_bytes = results['vit']['bytes'] + results['mlp']['bytes']

    vit_flops_percent = (results['vit']['flops'] / total_flops) * 100
    mlp_flops_percent = (results['mlp']['flops'] / total_flops) * 100

    vit_bytes_percent = (results['vit']['bytes'] / total_bytes) * 100
    mlp_bytes_percent = (results['mlp']['bytes'] / total_bytes) * 100

    # 格式化函数
    def fmt(x, unit='B'):
        if unit == 'B':
            if x < 1e6: return f"{x/1e3:.1f} KB"
            elif x < 1e9: return f"{x/1e6:.2f} MB"
            else: return f"{x/1e9:.3f} GB"
        else:
            if x < 1e6: return f"{x/1e3:.1f} K"
            elif x < 1e9: return f"{x/1e6:.2f} M"
            elif x < 1e12: return f"{x/1e9:.2f} G"
            else: return f"{x/1e12:.2f} T"

    print("\n" + "="*70)
    print("最终分析报告")
    print("="*70)
    print(f"UNI-ViT:\n  FLOPs = {fmt(results['vit']['flops'], 'FLOPs')} ({vit_flops_percent:.2f}%)\n"
          f"  Bytes = {fmt(results['vit']['bytes'])} ({vit_bytes_percent:.2f}%)\n"
          f"  AI    = {results['vit']['ai']:.2f}")
    print("-" * 40)
    print(f"MLP:\n  FLOPs = {fmt(results['mlp']['flops'], 'FLOPs')} ({mlp_flops_percent:.2f}%)\n"
          f"  Bytes = {fmt(results['mlp']['bytes'])} ({mlp_bytes_percent:.2f}%)\n"
          f"  AI    = {results['mlp']['ai']:.2f}")
    
    # 保存路径
    save_path = os.path.join(output_dir, "roofline_analysis.png")
    plot_roofline(results, save_path)

if __name__ == "__main__":
    main()