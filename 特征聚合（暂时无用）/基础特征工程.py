import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# --------------------------
# 1. 读取数据
# --------------------------
X = np.load('features.npy')
y = np.load('labels.npy')

print("特征数据形状：", X.shape)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("类别编码映射：", dict(zip(le.classes_, le.transform(le.classes_))))

# --------------------------
# 2. 标准化
# --------------------------
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# --------------------------
# 3. 方差过滤
# --------------------------
var_thresh = VarianceThreshold(threshold=1e-3)
X_var = var_thresh.fit_transform(X_std)
print("方差过滤后维度：", X_var.shape)

# --------------------------
# 4. PCA 降维（使用过滤后的数据）
# --------------------------
pca = PCA(n_components=128, random_state=42)
X_pca = pca.fit_transform(X_var)
print("PCA 后维度：", X_pca.shape)

le = LabelEncoder()
y_enc = le.fit_transform(y)
print("类别编码：", dict(zip(le.classes_, le.transform(le.classes_))))

# --- 划分数据 ---
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# --- XGBoost（更适合你数据的参数） ---
model = XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.04,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.5,
    reg_alpha=0.5,
    min_child_weight=1,
    objective="multi:softmax",
    num_class=4,
    tree_method="hist",     # 非常重要！能稳定结果
    eval_metric="mlogloss",
)

# --- 训练 ---
model.fit(X_train, y_train)

# --- 预测 ---
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
