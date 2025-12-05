import os
import json
import torch
import timm
from torchvision import transforms

def load_uni_components(checkpoint_dir, device=None):
    """
    加载 UNI 模型和相应的图像预处理 Transform。
    
    Args:
        checkpoint_dir (str): 包含 pytorch_model.bin 和 config.json 的文件夹路径
        device (str, optional): 运行设备 ('cuda' 或 'cpu')
        
    Returns:
        model: 加载好权重的 timm 模型
        transform: 对应的预处理转换
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"--- [UNI Loader] 正在从 {checkpoint_dir} 加载模型 ---")
    
    #路径定义
    config_path = os.path.join(checkpoint_dir, "config.json")
    weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"错误: 在 {checkpoint_dir} 中未找到权重文件 pytorch_model.bin")

    # 确定模型参数 (优先读取 config.json，如果不存在则使用默认值)
    model_kwargs = {
        "model_name": "vit_large_patch16_224",
        "img_size": 224,
        "patch_size": 16,
        "init_values": 1e-5, # 默认值，会被json覆盖
        "num_classes": 0,
        "dynamic_img_size": True
    }

    if os.path.exists(config_path):
        print(f"--- [UNI Loader] 发现配置文件 config.json，正在读取参数 ---")
        with open(config_path, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
            # 将 json 中的参数映射到 timm 的参数
            if 'architecture' in file_config:
                model_kwargs['model_name'] = file_config['architecture']
            if 'img_size' in file_config:
                model_kwargs['img_size'] = file_config['img_size']
            if 'init_values' in file_config:
                model_kwargs['init_values'] = file_config['init_values']
            if 'dynamic_img_size' in file_config:
                model_kwargs['dynamic_img_size'] = file_config['dynamic_img_size']
            # 注意：mean 和 std 通常也在 config 里的 pretrained_cfg 中
    
    # 创建模型骨架
    print(f"--- [UNI Loader] 构建模型架构: {model_kwargs['model_name']} ---")
    model = timm.create_model(**model_kwargs)

    # 加载权重
    print(f"--- [UNI Loader] 加载权重文件... ---")
    state_dict = torch.load(weights_path, map_location="cpu")
    
    # 处理权重加载 (尝试严格模式，失败则回退)
    try:
        msg = model.load_state_dict(state_dict, strict=True)
        print(f"--- [UNI Loader] 权重加载成功 (Strict mode) ---")
    except RuntimeError as e:
        print(f"--- [UNI Loader] Strict加载失败，尝试非严格模式 (通常是因为LayerScale等层) ---")
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"--- [UNI Loader] 非严格模式加载结果: {msg} ---")

    model.eval()
    model.to(device)

    #定义图像预处理 Transform (与训练时保持一致)
    transform = transforms.Compose([
        transforms.Resize((model_kwargs['img_size'], model_kwargs['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    print("[UNI Loader] 模型与预处理器准备就绪。")
    return model, transform