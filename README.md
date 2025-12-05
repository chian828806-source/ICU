# ICU WSI End-to-End Analysis System

这是一个基于深度学习的病理全切片图像（WSI）端到端分析系统。项目集成了 **UNI 模型** 进行特征提取，使用 **MLP** 进行分类，并包含完整的预处理流程。

## 项目结构

```text
ICU_WSI_Project/
│
├── main.py                    # 推理主程序入口
├── requirements.txt           # Python 依赖列表
├── README.md                  # 说明文档
│
├── configs/                   # 配置文件目录
│   └── inference_settings.json # 核心运行配置 (路径、设备等)
│
├── checkpoints/               # 模型权重目录
│   ├── uni_model/             # UNI 模型相关
│   │   ├── pytorch_model.bin
│   │   └── config.json
│   └── mlp_classifier/        # MLP 分类器相关
│       ├── weights.pth
│       └── scaler.pkl
│
├── src/                       # 源代码目录
│   ├── models/                # 模型加载与定义
│   │   ├── uni_loader.py
│   │   ├── mlp_loader.py
│   │   └── mlp_model.py
│   └── utils/                 # 工具类
│       ├── pre_filter.py      # WSI 背景过滤
│       └── post_process.py    # 热力图后处理与可视化
│
├── tools/                      # [工具箱]
│   ├── quantization_eval.py    # [新增] 量化与速度评估
│   └── roofline_analysis.py    # [新增] 性能瓶颈分析
│
└── output/
    ├── quantization/           # 量化脚本的输出
    └── analysis/               # Roofline图的输出
```

## 环境安装与配置

### 1. 安装 Python 依赖
建议使用 Python 3.8 或更高版本。
```bash
pip install -r requirements.txt
```

### 2. 配置 OpenSlide (Windows 用户必读) ⚠️
本系统依赖 `OpenSlide` 读取 WSI 图像。在 Windows 系统下，除了安装 python 库外，必须下载二进制核心文件。

1.  下载 OpenSlide Windows Binaries: [下载链接](https://openslide.org/download/) (下载 `windows-x64` 版本)
2.  解压下载的压缩包（例如解压到 `C:\openslide-bin`）。
3.  **关键步骤**：打开本项目的 `configs/inference_settings.json` 文件，修改 `"openslide_bin_path"` 字段，将其指向解压后的 `bin` 文件夹。

   ```json
   {
     "openslide_bin_path": "S:\\Openslide\\openslide-win64\\openslide-bin-4.0.0.8-windows-x64\\bin"
   }
   ```

## 运行方法

### 基础用法
在终端（Terminal/CMD）中运行以下命令，指定 WSI 文件路径即可启动分析：

```bash
python main.py --input "path/to/your/image.svs"
```

### 进阶用法
如果你想使用自定义的配置文件（例如切换模型路径或更改运行设备）：

```bash
python main.py --input "path/to/your/image.svs" --config "configs/custom_settings.json"
```

### 运行参数说明
*   `--input`: (必填) 输入的 WSI 图像路径，支持 `.svs`, `.ndpi` 等格式。
*   `--config`: (选填) 配置文件路径，默认为 `./configs/inference_settings.json`。

## 算法流程

1.  **预处理 (Preprocessing)**: 使用 `pre_filter.py` 对 WSI 进行背景剔除，通过 OTSU 阈值法筛选出包含组织的有效图块 (Patch)。
2.  **特征提取 (Feature Extraction)**: 使用预训练的 **UNI (ViT-Large)** 模型提取每个图块的高维特征。
3.  **分类推理 (Classification)**: 将特征输入 MLP 分类器，输出 4 类诊断结果 (Normal, Benign, InSitu, Invasive)。
4.  **后处理 (Post-processing)**:
    *   将分类结果映射回空间网格。
    *   使用中值滤波、形态学闭运算/开运算对热力图进行平滑去噪，消除孤立噪点并填补空洞。
5.  **可视化**: 生成原始预测与平滑处理后的对比图，并保存至 `output/` 目录。

## 配置文件说明 (`inference_settings.json`)

```json
{
  "uni_checkpoint_dir": "./checkpoints/uni_model",       // UNI 模型文件夹路径
  "mlp_weights_path": "./checkpoints/mlp_classifier/weights.pth", // MLP 权重路径
  "scaler_path": "./checkpoints/mlp_classifier/scaler.pkl",       // 特征缩放器路径
  "openslide_bin_path": "...",                           // OpenSlide 二进制路径
  "output_dir": "./output",                              // 结果保存位置
  "device": "cuda"                                       // 运行设备 (cuda 或 cpu)
}
```

## 📝 注意事项
*   请确保显存 (VRAM) 足够加载 ViT-Large 模型（建议 >8GB）。
*   若遇到 `DLL load failed` 错误，请再次检查 `openslide_bin_path` 是否配置正确。
```