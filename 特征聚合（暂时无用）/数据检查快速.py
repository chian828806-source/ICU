import numpy as np
import matplotlib
# 设置matplotlib后端为TkAgg（适合Windows环境）
matplotlib.use('TkAgg')
# 添加中文字体设置
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
# 移除交互模式设置，使用非交互模式

# 定义一个确保图表显示的函数
def ensure_plot_shows():
    """确保图表显示"""
    plt.show(block=True)  # 使用block=True确保图表窗口不会立即关闭

def correct_data_analysis(features_path, labels_path):
    """修正的数据分析 - 针对已聚合的特征"""
    
    print("🚀 修正版数据检查开始...")
    
    # 加载数据
    features = np.load(features_path)  # 形状: (400, 1024)
    labels = np.load(labels_path)      # 形状: (400,)
    
    print(f"📊 特征形状: {features.shape}")
    print(f"🏷️  标签形状: {labels.shape}")
    
    # 基本验证
    if len(features) != len(labels):
        print("❌ 错误: 特征和标签数量不匹配!")
        return None, None
    
    # 数据统计
    n_wsis, feature_dim = features.shape
    print(f"✅ 数据格式: 已聚合的特征")
    print(f"📈 WSI数量: {n_wsis}")
    print(f"🎯 特征维度: {feature_dim}")
    
    # 标签分布
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"📋 标签分布:")
    for label, count in zip(unique_labels, counts):
        print(f"  类别 {label}: {count} 个样本 ({count/len(labels)*100:.1f}%)")
    
    # 特征质量检查
    print(f"🔍 特征值范围: [{features.min():.3f}, {features.max():.3f}]")
    print(f"📏 特征均值: {features.mean():.3f} ± {features.std():.3f}")
    
    # 检查NaN和无限值
    nan_count = np.sum(np.isnan(features))
    inf_count = np.sum(np.isinf(features))
    print(f"🧹 数据清洁度 - NaN: {nan_count}, 无限值: {inf_count}")
    
    return features, labels

def visualize_aggregated_features(features, labels):
    """可视化已聚合的特征"""
    
    print("\n📊 开始特征可视化...")
    
    # 创建新的图形，确保不会与其他图形冲突
    plt.figure(figsize=(15, 12))
    
    # 1. 标签分布饼图
    plt.subplot(2, 2, 1)
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.pie(counts, labels=[f'类别 {l}' for l in unique_labels], autopct='%1.1f%%')
    plt.title('WSI标签分布')
    
    # 2. 特征值分布
    plt.subplot(2, 2, 2)
    # 随机选择一些特征维度
    sample_dims = np.random.choice(features.shape[1], 5, replace=False)
    for dim in sample_dims:
        plt.hist(features[:, dim], bins=30, alpha=0.6, label=f'维度{dim}')
    plt.xlabel('特征值')
    plt.ylabel('频次')
    plt.title('特征值分布 (随机5个维度)')
    plt.legend()
    
    # 3. PCA可视化
    plt.subplot(2, 2, 3)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # 关键修复：将字符串标签转换为数值类型
    # 创建标签映射字典
    label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
    # 将字符串标签转换为数字
    numeric_labels = np.array([label_to_num[label] for label in labels])
    
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                        c=numeric_labels, cmap='viridis', alpha=0.7)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('特征空间分布 (PCA)')
    
    # 创建自定义颜色条，显示原始标签名称
    cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)))
    cbar.set_ticklabels(unique_labels)
    
    # 4. 类别间特征差异
    plt.subplot(2, 2, 4)
    from sklearn.metrics import pairwise_distances
    intra_dists = []
    inter_dists = []
    
    for label in unique_labels:
        class_features = features[labels == label]
        other_features = features[labels != label]
        
        if len(class_features) > 1:
            intra_dist = pairwise_distances(class_features).mean()
            intra_dists.append(intra_dist)
        
        if len(other_features) > 0:
            inter_dist = pairwise_distances(class_features, other_features).mean()
            inter_dists.append(inter_dist)
    
    separation_ratio = np.mean(inter_dists) / np.mean(intra_dists)
    
    plt.bar(['类内距离', '类间距离'], [np.mean(intra_dists), np.mean(inter_dists)])
    plt.ylabel('平均距离')
    plt.title(f'特征分离度: {separation_ratio:.3f}')
    
    # 在图上添加分离度评估
    if separation_ratio > 1.5:
        evaluation = "优秀"
    elif separation_ratio > 1.2:
        evaluation = "良好"
    elif separation_ratio > 1.0:
        evaluation = "一般"
    else:
        evaluation = "较差"
    
    plt.text(0.5, 0.9, f'评估: {evaluation}', 
             transform=plt.gca().transAxes, ha='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    # 保存图表到文件
    plt.savefig('aggregated_features_analysis.png', dpi=300, bbox_inches='tight')
    
    # 确保图表显示
    print("图表已生成，按任意键关闭图表继续...")
    plt.show(block=True)  # 这会阻塞程序直到关闭图表窗口
    
    print(f"🎯 特征分离度: {separation_ratio:.3f} ({evaluation})")
    return separation_ratio

def analyze_class_separation(features, labels):
    """分析类别间的分离程度"""
    
    print("\n🔬 深度分析类别分离...")
    
    from sklearn.metrics import pairwise_distances
    unique_labels = np.unique(labels)
    
    # 计算每个类别的中心
    class_centers = {}
    for label in unique_labels:
        class_centers[label] = np.mean(features[labels == label], axis=0)
    
    # 计算类别中心之间的距离
    print("类别中心之间的距离:")
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i+1:]:
            dist = np.linalg.norm(class_centers[label1] - class_centers[label2])
            print(f"  类别 {label1} ↔ 类别 {label2}: {dist:.3f}")
    
    # 计算每个类别的紧密度
    print("\n各类别紧密度 (类内平均距离):")
    for label in unique_labels:
        class_features = features[labels == label]
        if len(class_features) > 1:
            intra_dist = pairwise_distances(class_features).mean()
            print(f"  类别 {label}: {intra_dist:.3f}")
    
    return class_centers

# 主执行函数
def main():
    """主执行函数"""
    
    # 1. 数据检查
    features, labels = correct_data_analysis('features.npy', 'labels.npy')
    
    if features is not None:
        # 2. 可视化分析
        separation_ratio = visualize_aggregated_features(features, labels)
        
        # 3. 深度分析
        class_centers = analyze_class_separation(features, labels)
        
        # 4. 保存分析结果
        analysis_report = {
            'data_shape': features.shape,
            'label_distribution': dict(zip(*np.unique(labels, return_counts=True))),
            'feature_stats': {
                'min': float(features.min()),
                'max': float(features.max()),
                'mean': float(features.mean()),
                'std': float(features.std())
            },
            'separation_ratio': float(separation_ratio)
        }
        
        print("\n✅ 数据分析完成!")
        print("📋 下一步建议:")
        
        if separation_ratio > 1.5:
            print("   🎉 特征质量优秀，可以直接开始分类器优化")
        elif separation_ratio > 1.2:
            print("   👍 特征质量良好，建议先做特征选择再优化分类器")
        else:
            print("   ⚠️  特征分离度一般，建议探索特征增强方法")
        
        return features, labels, analysis_report
    
    return None, None, None

if __name__ == "__main__":
    features, labels, report = main()

'''
📊 特征形状: (400, 1024)
📊 特征形状: (400, 1024)
🏷️  标签形状: (400,)
✅ 数据格式: 已聚合的特征
📈 WSI数量: 400
🎯 特征维度: 1024
📋 标签分布:
🎯 特征维度: 1024
📋 标签分布:
  类别 Benign: 100 个样本 (25.0%)
  类别 InSitu: 100 个样本 (25.0%)
  类别 Invasive: 100 个样本 (25.0%)
  类别 Normal: 100 个样本 (25.0%)
🔍 特征值范围: [-11.308, 15.353]
📏 特征均值: 0.008 ± 1.514
🧹 数据清洁度 - NaN: 0, 无限值: 0
 特征分离度: 1.182 (一般)

🔬 深度分析类别分离...
类别中心之间的距离:
  类别 Benign ↔ 类别 InSitu: 6.717
  类别 Benign ↔ 类别 Invasive: 12.881
  类别 Benign ↔ 类别 Normal: 7.575
  类别 InSitu ↔ 类别 Invasive: 17.038
  类别 InSitu ↔ 类别 Normal: 3.741
  类别 Invasive ↔ 类别 Normal: 19.104

各类别紧密度 (类内平均距离):
  类别 Benign: 17.772
  类别 InSitu: 18.039
  类别 Invasive: 21.426
  类别 Normal: 17.911
'''