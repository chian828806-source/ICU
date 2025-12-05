import os
import torch
import joblib


try:
    from .mlp_model import MLPClassifier
except ImportError:
    try:
        from .mlp_model import MLPClassifier
    except ImportError:
        raise ImportError("无法在 src/models/ 下找到 mlp_model.py 或 mlp_architecture.py，请检查文件名。")

def load_mlp_components(weights_path, scaler_path, device=None):
    """
    加载 MLP 分类器和 StandardScaler。
    
    Args:
        weights_path (str): .pth 权重文件的路径
        scaler_path (str): .pkl scaler 文件的路径
        device (str): 'cuda' or 'cpu'
        
    Returns:
        mlp_classifier: 加载好权重的 PyTorch 模型
        scaler: 加载好的 sklearn scaler
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- [MLP Loader] 正在加载分类器组件 ---")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"!!MLP 权重文件未找到: {weights_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"!!Scaler 文件未找到: {scaler_path}")

    # 加载 StandardScaler 
    try:
        scaler = joblib.load(scaler_path)
        print(f"!!Scaler 加载成功")
    except Exception as e:
        raise RuntimeError(f"加载 Scaler 失败: {e}")

    # 加载 MLP 模型 
    print(f"--- [MLP Loader] 初始化模型架构 (In:1024, Out:4) ---")
    mlp_classifier = MLPClassifier(input_dim=1024, num_classes=4)
    
    try:
        state_dict = torch.load(weights_path, map_location=device)
        mlp_classifier.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"加载 MLP 权重失败，请确认模型定义与权重匹配。错误: {e}")

    # 设置模式与设备
    mlp_classifier.eval()
    mlp_classifier.to(device)

    print(f"[MLP Loader] 分类器加载完成 (Device: {device})")
    return mlp_classifier, scaler