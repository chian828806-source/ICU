import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
import os

# ---- 配置 ----
FEATURE_IN = "features.npy"   # 或已处理的 X_corr 文件，如果你用的是 X_corr.npy 改名
LABEL_IN = "labels.npy"
OUT_DIR = "pca_candidates"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 读数据（如果你已有 X_corr，直接载入它并注释下面两行） ----
X = np.load(FEATURE_IN)   # 若用 X_corr.npy 请改为 np.load("X_corr.npy")
y_raw = np.load(LABEL_IN)

# ---- 若需要用之前的标准化/方差过滤/相关过滤结果，请直接改 X = np.load("fe_torch_outputs/X_corr.npy") ----
# X = np.load("fe_torch_outputs/X_corr.npy")

le = LabelEncoder()
y = le.fit_transform(y_raw)
n_classes = len(le.classes_)

# 标准化（若之前已做过可跳过）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 全量 PCA（用于累计解释方差曲线）
pca_all = PCA(n_components=min(X_scaled.shape[0], X_scaled.shape[1]), random_state=42)
pca_all.fit(X_scaled)
cumvar = np.cumsum(pca_all.explained_variance_ratio_)

# 打印累计解释方差前 50 成分（数值）
print("累计解释方差（前 50 主成分）：")
for i in [5,10,15,18,20,30,40,50]:
    if i <= len(cumvar):
        print(f"  top-{i}: {cumvar[i-1]:.4f}")

# 寻找拐点（可选）：主成分数对应 80%, 90%, 95% 的索引
for target in [0.80, 0.90, 0.95, 0.99]:
    idx = np.searchsorted(cumvar, target) + 1
    print(f"需要 {idx} 个主成分以解释 {int(target*100)}% 方差")

# 候选维度（你可以修改）
candidates = [18, 32, 64, 128, 256]

results = []
for k in candidates:
    k_use = min(k, X_scaled.shape[1])
    pca = PCA(n_components=k_use, random_state=42)
    Xp = pca.fit_transform(X_scaled)
    # explained variance
    ev = np.sum(pca.explained_variance_ratio_)
    # silhouette (需要样本数 > 类数)
    try:
        sil = silhouette_score(Xp, y) if Xp.shape[0] > n_classes else np.nan
    except Exception:
        sil = np.nan
    # separation: mean inter-centroid distance / mean intra-class distance
    classes = np.unique(y)
    centroids = []
    intra = []
    for c in classes:
        Xc = Xp[y == c]
        cent = Xc.mean(axis=0)
        centroids.append(cent)
        intra.append(np.mean(np.linalg.norm(Xc - cent, axis=1)))
    centroids = np.vstack(centroids)
    pdist = pairwise_distances(centroids)
    n = pdist.shape[0]
    inter_mean = pdist[np.triu_indices(n, k=1)].mean()
    intra_mean = np.mean(intra) if len(intra) > 0 else np.nan
    separation = inter_mean / (intra_mean + 1e-12)

    results.append({
        "k": k_use,
        "explained_variance": ev,
        "silhouette": sil,
        "separation": separation,
        "shape": Xp.shape
    })

    # save features
    np.save(os.path.join(OUT_DIR, f"PCA_{k_use}.npy"), Xp)
    print(f"PCA_{k_use}: ev={ev:.4f}, sil={sil:.4f}, sep={separation:.4f}, shape={Xp.shape}")

# summary table
df = pd.DataFrame(results).sort_values("k")
print("\nSummary:")
print(df)
df.to_csv(os.path.join(OUT_DIR, "pca_candidates_summary.csv"), index=False)
print("Saved PCA candidates and summary to", OUT_DIR)
