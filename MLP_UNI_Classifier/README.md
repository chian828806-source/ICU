# MLP UNI 分类器

这是一个使用 UNI 特征进行 BACH 数据集四分类的 MLP 模型。

## 文件结构

- `model_weights.pth`: 训练好的 MLP 模型权重。
- `standard_scaler.pkl`: 用于特征标准化的 StandardScaler。
- `mlp_architecture.py`: MLP 模型的 PyTorch 定义。
- `inference_script.py`: 推理脚本，封装了加载和预测逻辑。
- `config.json`: 模型配置信息。
- `requirements.txt`: Python 依赖项。
- `README.md`: 本说明文件。

## 环境要求

Python 3.8+

## 安装依赖

```bash
pip install -r requirements.txt