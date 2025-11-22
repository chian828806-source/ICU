# File: extract_uni_features_split.py

import os
import numpy as np
from PIL import Image
import torch
import timm
from torchvision import transforms
from tqdm import tqdm

# -------------------------------
# 1. 配置参数
# -------------------------------
# UNI 模型相关路径
UNI_MODEL_DIR = r"E:\py_ai\python-code\ICT_related\UNI"  # 包含 pytorch_model.bin 的目录
# 训练集图像目录 (增强后的)
TRAIN_IMAGE_DIR = r"E:\py_ai\python-code\ICT_related\UNI_Photos_Split_Enhanced"
# 验证集图像目录 (原始的)
VAL_IMAGE_DIR = r"E:\py_ai\python-code\ICT_related\UNI_Photos_Split\val"
# 特征和标签保存目录
FEATURES_OUTPUT_DIR = r"E:\py_ai\python-code\ICT_related\Data_train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------
# 2. 加载本地 UNI 模型和预处理器
# -------------------------------
def load_local_uni_model():
    """加载本地的 UNI 模型权重 (基于 ViT-L/16)"""
    print("正在加载本地 UNI 模型...")

    # 1. 定义模型架构 (ViT-L/16) - 移除 dynamic_img_size 参数
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,  # 移除分类头，只提取特征
        # dynamic_img_size=True # 移除此参数，避免错误
    )

    # 2. 加载本地权重
    checkpoint_path = os.path.join(UNI_MODEL_DIR, "pytorch_model.bin")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"UNI 模型权重文件未找到: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    # 3. 设置为评估模式
    model.eval()
    model.to(DEVICE)

    # 4. 定义预处理器 (根据官方文档，使用 ImageNet 归一化参数)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    print("✅ 本地 UNI 模型加载完成")
    return model, transform


# -------------------------------
# 3. 提取图像特征
# -------------------------------
def extract_features(image_dir, model, transform, description=""):
    """从指定目录下的所有图像提取特征"""
    print(f"开始提取 {description} 特征...")

    all_features = []
    all_labels = []

    class_names = ['Normal', 'Benign', 'InSitu', 'Invasive']
    label_map = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_path = os.path.join(image_dir, class_name)
        if not os.path.exists(class_path):
            print(f"⚠️  跳过不存在的类别目录: {class_path}")
            continue

        img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.tif', '.png', '.jpg', '.jpeg'))]
        print(f"处理类别 {class_name} ({description}): {len(img_files)} 张图像")

        for img_name in tqdm(img_files, desc=f"提取 {class_name} ({description}) 特征"):
            img_path = os.path.join(class_path, img_name)

            try:
                # 加载图像
                image = Image.open(img_path).convert('RGB')
                # 预处理
                image_tensor = transform(image).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]

                # 提取特征
                with torch.no_grad():
                    features = model(image_tensor)  # [1, 1024]

                all_features.append(features.squeeze().cpu().numpy())  # [1024,]
                all_labels.append(label_map[class_name])

            except Exception as e:
                print(f"⚠️  处理 {img_name} 时出错: {e}")
                continue

    features_array = np.array(all_features)  # [N, 1024]
    labels_array = np.array(all_labels)  # [N,]

    print(f"✅ {description} 特征提取完成: {features_array.shape}, {labels_array.shape}")
    return features_array, labels_array


# -------------------------------
# 4. 保存特征和标签
# -------------------------------
def save_features_and_labels(features, labels, features_path, labels_path):
    """保存特征和标签数组"""
    os.makedirs(os.path.dirname(features_path), exist_ok=True)

    np.save(features_path, features)
    np.save(labels_path, labels)

    print(f"✅ 特征已保存至: {features_path}")
    print(f"✅ 标签已保存至: {labels_path}")


# -------------------------------
# 5. 主函数
# -------------------------------
def main():
    # 1. 加载模型
    model, transform = load_local_uni_model()

    # 2. 提取增强训练集特征
    train_features, train_labels = extract_features(
        TRAIN_IMAGE_DIR, model, transform, description="增强训练集"
    )

    # 3. 提取原始验证集特征
    val_features, val_labels = extract_features(
        VAL_IMAGE_DIR, model, transform, description="原始验证集"
    )

    # 4. 保存特征和标签
    # 训练集
    save_features_and_labels(
        train_features, train_labels,
        os.path.join(FEATURES_OUTPUT_DIR, "X_train_enhanced.npy"),
        os.path.join(FEATURES_OUTPUT_DIR, "y_train_enhanced.npy")
    )
    # 验证集
    save_features_and_labels(
        val_features, val_labels,
        os.path.join(FEATURES_OUTPUT_DIR, "X_val_total.npy"),
        os.path.join(FEATURES_OUTPUT_DIR, "y_val_total.npy")
    )

    print("\n🎉 UNI 特征提取与保存完成！")
    print(f"增强训练集特征形状: {train_features.shape}")
    print(f"原始验证集特征形状: {val_features.shape}")
    print(f"增强训练集标签形状: {train_labels.shape}")
    print(f"原始验证集标签形状: {val_labels.shape}")
    print(f"标签分布 (训练集): {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    print(f"标签分布 (验证集): {dict(zip(*np.unique(val_labels, return_counts=True)))}")


# -------------------------------
# 6. 程序入口
# -------------------------------
if __name__ == "__main__":
    main()