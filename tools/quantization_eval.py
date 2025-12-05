import sys
import os
import argparse
import json
import time
import numpy as np
import torch
import torch.quantization as tq
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

# --- 核心：将项目根目录加入路径，以便导入 src ---
# 获取当前脚本所在目录 (tools/) 的上一级目录 (项目根目录)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# 导入模块 
from src.models.uni_loader import load_uni_components
from src.models.mlp_loader import load_mlp_components

# 全局设置
CLASS_NAMES = ['Normal', 'Benign', 'InSitu', 'Invasive']
DEVICE_CPU = "cpu" # 量化通常在 CPU 上进行评估

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_features_and_labels(image_dir, model, transform, device, max_per_class=20):
    """通用特征提取函数"""
    all_features = []
    all_labels = []
    label_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    print(f"--- 开始提取特征 (Max {max_per_class}/class) ---")
    model.eval()
    model.to(device)

    for class_name in CLASS_NAMES:
        class_path = os.path.join(image_dir, class_name)
        if not os.path.exists(class_path):
            print(f"⚠️ 警告: 类别目录不存在 {class_path}")
            continue

        # 获取图片文件
        img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.tif','.png','.jpg','.jpeg'))]
        img_files = img_files[:max_per_class]

        for img_name in tqdm(img_files, desc=f"Extracting {class_name}"):
            img_path = os.path.join(class_path, img_name)
            try:
                # 读取并预处理
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # 推理
                with torch.no_grad():
                    features = model(image_tensor)
                
                all_features.append(features.squeeze().cpu().numpy())
                all_labels.append(label_map[class_name])
            except Exception as e:
                print(f"❌ Error {img_name}: {e}")

    return np.array(all_features), np.array(all_labels)

def evaluate_mlp_performance(features, labels, mlp_model, scaler):
    """评估 MLP 分类性能"""
    # 标准化特征
    features_scaled = scaler.transform(features)
    # 转为 Tensor
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(DEVICE_CPU)
    
    # MLP 推理
    with torch.no_grad():
        logits = mlp_model(features_tensor)
        preds = torch.argmax(logits, dim=1).numpy()
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return acc, f1

def measure_inference_time(model, transform, image_path, device, repeat=20):
    """测量单张图片推理耗时"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    # 预热
    with torch.no_grad():
        for _ in range(5): model(image_tensor)
        
    start = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(image_tensor)
    total_time = time.time() - start
    return total_time / repeat

def main():
    parser = argparse.ArgumentParser(description="UNI 模型量化评估脚本")
    parser.add_argument('--config', type=str, default='./configs/quantization_config.json', help='配置路径')
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    output_dir = config.get('output_dir', './output/quantization')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("=== 步骤 1: 加载模型 (FP32) ===")
    # 复用 uni_loader，强制使用 CPU (因为量化对比需要在同设备下才有意义，且 int8 需 cpu)
    uni_fp32, uni_transform = load_uni_components(config['uni_checkpoint_dir'], device=DEVICE_CPU)
    
    # 复用 mlp_loader
    mlp_model, scaler = load_mlp_components(config['mlp_weights_path'], config['scaler_path'], device=DEVICE_CPU)

    print("\n=== 步骤 2: 生成量化模型 (INT8) ===")
    # 动态量化: 仅量化 Linear 层
    uni_int8 = tq.quantize_dynamic(uni_fp32, {torch.nn.Linear}, dtype=torch.qint8)
    uni_int8.to(DEVICE_CPU)
    print("INT8 模型转换完成")

    print("\n=== 步骤 3: 提取特征与对比 ===")
    dataset_dir = config['dataset_dir']
    
    # 提取 FP32 特征
    print(">>> 正在提取 FP32 特征...")
    feats_fp32, labels = extract_features_and_labels(
        dataset_dir, uni_fp32, uni_transform, DEVICE_CPU, config['max_images_per_class']
    )
    
    # 提取 INT8 特征
    print(">>> 正在提取 INT8 特征...")
    feats_int8, _ = extract_features_and_labels(
        dataset_dir, uni_int8, uni_transform, DEVICE_CPU, config['max_images_per_class']
    )

    # 保存特征 (可选)
    np.save(os.path.join(output_dir, 'feats_fp32.npy'), feats_fp32)
    np.save(os.path.join(output_dir, 'feats_int8.npy'), feats_int8)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)

    print("\n=== 步骤 4: 计算相似度与精度 ===")
    # 4.1 特征相似度
    l2_dist = np.linalg.norm(feats_fp32 - feats_int8, axis=1).mean()
    cos_sim = np.diag(cosine_similarity(feats_fp32, feats_int8)).mean()
    
    # 4.2 下游任务精度 (MLP)
    acc_fp32, f1_fp32 = evaluate_mlp_performance(feats_fp32, labels, mlp_model, scaler)
    acc_int8, f1_int8 = evaluate_mlp_performance(feats_int8, labels, mlp_model, scaler)

    print("\n" + "="*40)
    print(f"特征保真度:")
    print(f"   平均 L2 距离 (越小越好): {l2_dist:.6f}")
    print(f"   平均余弦相似度 (越大越好): {cos_sim:.6f}")
    print("-" * 40)
    print(f"分类性能 (MLP):")
    print(f"   FP32 (原始): ACC={acc_fp32:.4f}, F1={f1_fp32:.4f}")
    print(f"   INT8 (量化): ACC={acc_int8:.4f}, F1={f1_int8:.4f}")
    print(f"   精度损失: {acc_fp32 - acc_int8:.4f}")
    print("="*40)

    print("\n=== 步骤 5: 推理速度测试 ===")
    # 找一张图测试
    first_class_dir = os.path.join(dataset_dir, CLASS_NAMES[0])
    test_img = os.path.join(first_class_dir, os.listdir(first_class_dir)[0])
    
    t_fp32 = measure_inference_time(uni_fp32, uni_transform, test_img, DEVICE_CPU)
    t_int8 = measure_inference_time(uni_int8, uni_transform, test_img, DEVICE_CPU)

    print(f"速度对比 (单张图, CPU):")
    print(f"   FP32 耗时: {t_fp32*1000:.2f} ms")
    print(f"   INT8 耗时: {t_int8*1000:.2f} ms")
    print(f"   加速比: {t_fp32 / t_int8:.2f}x")
    print("="*40)

if __name__ == "__main__":
    main()