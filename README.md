
# OGP Afterbonding Training & Inference (YOLOv8 + PatchCore)

本项目用于 **After Bonding Wire 异常检测**（OGP 工业相机 BMP 图像），包含：

- **YOLOv8**：wire 区域检测（目标检测）
    
- **PatchCore（Anomalib + OpenVINO）**：wire patch 异常检测（无监督/单类）
    
- **PDF 自动报告**：将异常整图按 **cell 编号** 汇总成 PDF 供生产复查
    

仓库同时提供：

- 训练 YOLO 的脚本：`train_YOLO/`
    
- 训练 PatchCore 并导出 ONNX 的脚本：`train_AB/`
    
- 推理+生成报告脚本：`predict_pdf.py`
    

---

## 0. Quick Start

```bash
conda env create -f environment.yml
conda activate yolov8

python predict_pdf.py
```

---

## 1. Train Models

### 1.1 Train YOLOv8 (wire detector)

**脚本位置**

- `train_YOLO/train_yolov8.py`
    
- `train_YOLO/train_yolov8.sh`
    
- `train_YOLO/data.yaml`
    

**数据准备**  
在 `train_YOLO/data.yaml` 里配置你的 YOLO 数据集路径（Ultralytics 格式），通常包含：

- `train:` 训练图片路径
    
- `val:` 验证图片路径
    
- `names:` 类别名称（wire/glue 等）
    


**训练命令（Python 方式）**

```bash
cd train_YOLO
sbatch train_yolov8.sh
```

**训练输出**  
Ultralytics 默认输出到 `runs/`，脚本中配置了：

```python
project='runs/afterbonding'
name='train_ab_addbkg'
```

最终权重一般在：

```
runs/afterbonding/train_ab_addbkg/weights/best.pt
```

将该 `best.pt` 复制/链接到推理配置使用的位置，例如：

```
models/YOLO/best.pt
```

---

### 1.2 Train PatchCore (wire anomaly model) + Export ONNX

**脚本位置**

- `train_AB/complete_pipeline.py`
    
- `train_AB/train.sh`
    

这个脚本使用 `anomalib` 的 `Folder` datamodule 进行训练，并最后导出 ONNX。

#### 数据目录约定（Folder datamodule）

脚本里写的是：

```python
datamodule = Folder(
    root=Path("/cms/user/huangsuyun/YOLOAB/dataset"),
    normal_dir="afterbondingnew_0105_1000",
)
```

因此目录应该类似：

```
/cms/user/huangsuyun/YOLOAB/dataset/
└── afterbondingnew_0105_1000/
    ├── xxx1.jpg
    ├── xxx2.jpg
    └── ...
```

> PatchCore/PaDiM 这类方法通常只需要 **normal 数据训练**（单类/无监督）

#### 训练

```bash
cd train_AB
sbatch train.sh
```

训练输出目录由脚本指定：

```python
default_root_dir="patchcore0105_yolo_noresize_1000"
```

会保存 checkpoint、日志等。

#### 推理评估/预测

脚本中也包含：

```python
predictions = engine.predict(model=model, dataset=test_data)
```

并将分数导出到 Excel：

```
prediction_scores.xlsx
```

#### 导出 ONNX

脚本末尾：

```python
engine.export(
    export_root=Path("exported_model_0105_noresize_1000"),
    export_type=ExportType.ONNX,
)
```

导出产物一般在：

```
exported_model_0105_noresize_1000/weights/onnx/model.onnx
```

将其复制/链接到推理配置使用的位置，例如：

```
models/ANOMALIB/model.onnx
```

---

## 2. Inference & PDF Report (YOLO + PatchCore)

### 2.1 Pipeline Overview

```
原始图像 (BMP / JPG / PNG)
        │
        ▼
YOLOv8 检测 wire 区域
        │
        ├─ 保存 YOLO 可视化整图
        │
        ├─ 裁剪 wire patch
        │
        ▼
PatchCore (OpenVINO)
        │
        ├─ 正常 patch → 忽略
        │
        └─ 异常 patch
             ├─ 标注异常框 + 分数
             └─ 画回原图
        │
        ▼
保存异常整图 (JPG)
        │
        ▼
PDF 报告生成（5 列网格 + cell 编号）
```

---

## 3. Image Naming Conventions

你的图片命名类似：

```
/cms/user/.../320MLF3WCIH0350_after_bonding_front/
└── module_after_bonding_front_check-36-1.BMP
```

解析规则：

- 文件名按 `-` 分割
    
- **倒数第二段为 cell index（例：36）**
    
- 通过 `CELL_PHOTO_MAP` 映射为实际 cell 编号（例：1）
    
---

## 4. Environment Setup (environment.yml)

在项目根目录执行：

### 3.1 way1

在项目根目录下执行：
```bash

cd src/
conda env create -f environment.yml
```
激活环境：
```bash
conda activate yolov8
```
### 3.2 way2

```bash

conda create -n yoloab python=3.10

conda activate yoloab

pip install ultralytics opencv-python pandas tqdm fpdf

pip install anomalib openvino

```
---

## 5. Running the Pipeline

```bash
python predict_pdf.py
```

---

## 6. Output

运行完成后将生成：

- `yolo_origin/`  
    YOLO 检测可视化整图（中间产物）
    
- `patch/`  
    裁剪的 wire patch（中间产物）
    
- `patchcore_scores.csv`  
    PatchCore 异常评分结果
    
- `yolo_abnormal/`  
    异常整图（JPG，已标注）
    
- `wire_anomaly_report.pdf`  
    最终 PDF 报告
    

> 注意：`fpdf` 不支持 BMP，因此异常图保存为 JPG 再写入 PDF。
