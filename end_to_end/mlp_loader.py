import os
import torch
import joblib
import sys

# --- 导入路径修复 ---
# 1. 获取当前文件的目录路径 (baseline/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 获取父目录路径 (ICU/)
parent_dir = os.path.dirname(current_dir)

# 3. 构造并行目录的路径
PARALLEL_DIR_NAME = 'MLP_UNI_Classifier' 
parallel_dir = os.path.join(parent_dir, PARALLEL_DIR_NAME)

# 4. 将并行目录添加到 sys.path
if parallel_dir not in sys.path:
    sys.path.insert(0, parallel_dir)

# 5. 打印调试信息以确认路径
print(f"添加到sys.path的路径: {parallel_dir}")
print(f"sys.path: {sys.path}")

# 现在可以直接导入模块
try:
    from mlp_architecture import MLPClassifier 
    print("✅ 成功导入 MLPClassifier")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print(f"请检查 {parallel_dir} 目录中是否存在 mlp_architecture.py 文件")
    raise


# ----------------------------------------------------
# 1. 配置
# ----------------------------------------------------
# 替换为您的分类器组件所在的目录
CLASSIFIER_DIR = r"C:\Users\Lenovo\Desktop\ICU\MLP_UNI_Classifier" 
MLP_WEIGHTS_PATH = os.path.join(CLASSIFIER_DIR, "mlp_uni_classifier_weights.pth")
SCALER_PATH = os.path.join(CLASSIFIER_DIR, "mlp_uni_scaler.pkl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------
# 2. 核心接口：加载 MLP 分类器和 Scaler
# ----------------------------------------------------
def load_mlp_components(weights_path=MLP_WEIGHTS_PATH, scaler_path=SCALER_PATH, device=DEVICE):
    """
    加载 MLP 分类器 (PyTorch) 和特征缩放器 (StandardScaler)。

    返回: mlp_classifier (torch.nn.Module), scaler (StandardScaler object)
    """
    print("--- 模块 02: 正在加载 MLP 分类器和 StandardScaler ---")

    # --- A. 加载 MLP 分类器 (PyTorch 方式) ---
    # ⚠️ 实例化时可能需要根据 config.json 传入参数
    # 假设特征维度是 1024 (来自 UNI)，类别数是 4
    mlp_classifier = MLPClassifier(input_dim=1024, num_classes=4) 
    
    try:
        state_dict = torch.load(weights_path, map_location=device)
        mlp_classifier.load_state_dict(state_dict)
        mlp_classifier.eval()
        mlp_classifier.to(device)
    except FileNotFoundError:
        raise FileNotFoundError(f"MLP 权重文件未找到: {weights_path}")
    except Exception as e:
        print(f"MLP 权重加载失败，请检查 mlp_architecture.py 定义是否与权重文件匹配: {e}")
        exit()

    # --- B. 加载特征缩放器 ---
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"特征缩放器文件未找到: {scaler_path}")

    print("✅ 模块 02: MLP 分类器和 StandardScaler 加载完成。")
    return mlp_classifier, scaler

if __name__ == "__main__":
    load_mlp_components()