import torch
import numpy as np
import pickle
from mlp_architecture import MLPClassifier # 从同目录导入模型定义

# -------------------------------
# 1. 配置和加载
# -------------------------------
MODEL_WEIGHTS_PATH = "model_weights.pth"
SCALER_PATH = "standard_scaler.pkl"
CLASS_NAMES = ['Normal', 'Benign', 'InSitu', 'Invasive'] # 从 config.json 或训练时确认

def load_model_and_scaler():
    print(f"加载模型权重 from {MODEL_WEIGHTS_PATH}...")
    model = MLPClassifier() # 实例化模型结构
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location='cpu', weights_only=True)) # 加载权重
    model.eval() # 设置为评估模式

    print(f"加载 StandardScaler from {SCALER_PATH}...")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    print("模型和预处理器加载完成")
    return model, scaler

def predict(model, scaler, features):
    print("执行推理...")
    # features: numpy array (1024,)
    features_scaled = scaler.transform(features.reshape(1, -1)) # reshape for scaler: (1, 1024)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32) # (1, 1024)

    with torch.no_grad():
        logits = model(features_tensor) # (1, 4)
        probabilities = torch.softmax(logits, dim=1).squeeze().numpy() # (4,)
        predicted_class_idx = np.argmax(probabilities)
        predicted_class_name = CLASS_NAMES[predicted_class_idx]

    return predicted_class_name, probabilities[predicted_class_idx], probabilities

# -------------------------------
# 2. 示例用法 (可选，也可以做成命令行接口)
# -------------------------------
if __name__ == "__main__":
    # 加载模型和预处理器
    model, scaler = load_model_and_scaler()

    # 假设你有一个UNI特征向量 (例如，从 .npy 文件加载)
    # example_features = np.load("path/to/your/uni_feature_vector.npy")
    # 为了演示，我们生成一个随机向量
    example_features = np.random.rand(1024)

    # 预测
    pred_class, pred_conf, all_probs = predict(model, scaler, example_features)

    # 输出结果
    print("\n--- 预测结果 ---")
    print(f"输入特征形状: {example_features.shape}")
    print(f"预测类别: {pred_class}")
    print(f"预测置信度: {pred_conf:.4f}")
    print(f"所有类别概率: {dict(zip(CLASS_NAMES, all_probs))}")