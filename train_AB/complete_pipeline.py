# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import os
import numpy as np
from anomalib.data import Folder,PredictDataset
from anomalib.engine import Engine
from anomalib.models import Padim,Patchcore
from anomalib.deploy import ExportType
from anomalib.loggers import AnomalibTensorBoardLogger
from anomalib.pre_processing import PreProcessor
from torchvision.transforms.v2 import Compose, Resize, ToTensor,Normalize
# ===== 关键修复：限制 OpenBLAS 线程 =====
os.environ["OPENBLAS_NUM_THREADS"] = "4"
# TARGET_IMAGE_SIZE = (256, 256) #标准的预处理流程：调整尺寸 -> 转换为 Tensor -> 归一化
# # 注意：Folder 数据集加载的已经是 PIL Image，所以这里从 Resize 开始。
# # 在 Anomalib 内部，Compose 会自动将 PIL Image 转换为 Tensor。
# transforms = Compose([
#     Resize(size=TARGET_IMAGE_SIZE, antialias=True), # <-- 强制调整图像尺寸
#     ToTensor(), # 必须转换为 Tensor
#     # 推荐的 ImageNet 归一化 (Anomalib 默认使用此设置)
#     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# pre_processor = PreProcessor(transform=transforms)
# =====================================================
# 1. 创建 PaDiM 模型 + 加入评估指标（**这是你缺少的关键部分**）
# =====================================================
from anomalib.metrics import (
    AUROC, AUPR,
    F1Max
    
)
from anomalib.metrics import Evaluator

# model = Padim(
#     backbone="resnet18",
#     layers=["layer1", "layer2", "layer3"],
#     pre_trained=True,
#     n_features=100,
# )

model = Patchcore(
    backbone="resnet18",
    # layers=["layer1", "layer2", "layer3"],
    pre_trained=True,
    # n_features=100,
    coreset_sampling_ratio=0.01,
    # pre_processor=pre_processor


)

# --- ⭐ 添加测试指标（推荐组合） ---
model.evaluator = Evaluator(
    test_metrics=[
        AUROC(fields=["pred_score", "gt_label"]),
        AUPR(fields=["pred_score", "gt_label"]),
        F1Max(fields=["pred_score", "gt_label"]),
        # F1AdaptiveThreshold(fields=["pred_score", "gt_label"]),
    ]
)


# =====================================================
# 2. 数据模块 —— 正确的目录结构是必须的
# =====================================================
# dataset/
# ├── train/
# │    └── normal/
# └── test/
#      ├── normal/
#      └── anomaly/

datamodule = Folder(
    name="my_dataset",
    root=Path("/cms/user/huangsuyun/YOLOAB/dataset"),
    normal_dir="afterbondingnew_0105_1000",
    # abnormal_dir="abnormal",       
    train_batch_size=8,
    eval_batch_size=32,
    num_workers=4,
    # image_size=(256, 256),

)

# test_datamodule = Folder(
#     name="my_dataset_test",
#     root=Path("/cms/user/huangsuyun/ANOMALIB/dataset/test"),
#     # normal_dir="normal",
#     abnormal_dir="selected_wire",   
#     eval_batch_size=32,
#     num_workers=4,
# )
test_data = PredictDataset(
    path=Path("/cms/user/huangsuyun/YOLOAB/dataset/test/"),
    # image_size=(560, 480),
    # image_size=(128, 128),
)



# =====================================================
# 3. 训练
# =====================================================
engine = Engine(
    logger=AnomalibTensorBoardLogger(save_dir="logs/tensorboard"),
    max_epochs=100,
    # accelerator="gpu",
    # devices=1,
    enable_checkpointing=True,
    default_root_dir="patchcore0105_yolo_noresize_1000",
    # normalization="none", # 如果你不想让它自动缩放到 0-1

)

print("Starting Training...")
engine.train(model=model, datamodule=datamodule)


# =====================================================
# 4. 测试（metrics 自动计算）
# =====================================================
print("\nRunning Test Evaluation...")
# engine.test(model=model, datamodule=test_datamodule, 
# )

# engine.test(model=model, dataset=test_data, 
# )
# # =====================================================
# 5. 推理（如果你还希望逐张打印）
# =====================================================
print("\nRunning Inference...")
# predictions = engine.predict(model=model, datamodule=test_datamodule)
predictions = engine.predict(model=model, dataset=test_data)


print("\nProcessing Results...")

if predictions is not None:
    for batch_or_pred in predictions:
        # 兼容 batch 和 single prediction
        if isinstance(batch_or_pred, (list, tuple)):
            preds = batch_or_pred
        else:
            preds = [batch_or_pred]

        for pred in preds:
            print("\n------------------")
            print(f"Image: {pred.image_path}")
            print("Pred Score Tensor:")
            print(pred.pred_score)   # ← 直接打印张量（可能是 heatmap 或 vector）
            print("Pred Label Tensor:")
            print(pred.pred_label)   # ← 同样直接打印张量
            print("------------------\n")

import pandas as pd
from pathlib import Path

# ====== 收集结果 ======
records = []
MANUAL_THRESHOLD = 0.9

for batch_or_pred in predictions:
    if isinstance(batch_or_pred, (list, tuple)):
        preds = batch_or_pred
    else:
        preds = [batch_or_pred]

    for pred in preds:
        img_path = pred.image_path[0]  # list → string
        score = float(pred.pred_score[0].cpu().numpy())
        label = bool(pred.pred_label[0].cpu().numpy())
        manual_label = score > MANUAL_THRESHOLD

        # 提取所在文件夹名称（如 anomaly1 / anomal / selected_wire）
        folder = Path(img_path).parent.name

        records.append({
            "folder": folder,
            "image": Path(img_path).name,
            "full_path": img_path,
            "score": score,
            "label": label,
            "manual_label": manual_label,

        })

# ====== 保存成 Excel ======
df = pd.DataFrame(records)

excel_path = "prediction_scores.xlsx"
df.to_excel(excel_path, index=False)

print(f"\n保存成功：{excel_path}")
print(df.head())



# =====================================================
# 6. 导出 ONNX
# =====================================================
print("\nExporting ONNX...")
engine.export(
    model=model,
    export_root=Path("exported_model_0105_noresize_1000"),
    # input_size=(560, 480),
    # input_size=TARGET_IMAGE_SIZE, 
    export_type=ExportType.ONNX,
)
