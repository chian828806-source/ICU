import os
import torch
import timm
from torchvision import transforms

# ----------------------------------------------------
# 1. 配置
# ----------------------------------------------------
# 替换为您的 UNI 模型权重文件 pytorch_model.bin 所在的目录
UNI_MODEL_DIR = r"C:\Users\Lenovo\Desktop\ICU" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------
# 2. 核心接口：加载模型和预处理器
# ----------------------------------------------------
def load_uni_components(uni_model_dir=UNI_MODEL_DIR, device=DEVICE):
    """
    加载 UNI 模型和相应的图像预处理 Transform。
    """
    print(f"--- 模块 01: 正在加载 UNI 特征提取器到 {device} ---")

    # --- A. 模型架构定义 (ViT-L/16) ---
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,    # 移除分类头
    )

    # --- B. 加载本地权重 ---
    checkpoint_path = os.path.join(uni_model_dir, "pytorch_model.bin")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"UNI 模型权重文件未找到: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # 使用非严格加载以应对 UNI 的定制层（如 LayerScale）
    try:
        model.load_state_dict(state_dict, strict=True) 
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)

    # --- C. 设置评估模式和设备 ---
    model.eval()
    model.to(device)

    # --- D. 定义预处理器 ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    print("✅ 模块 01: UNI 模型和预处理器加载完成。")
    return model, transform

if __name__ == "__main__":
    load_uni_components()