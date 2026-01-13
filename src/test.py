# import os
# from pathlib import Path
# import matplotlib.pyplot as plt
# from anomalib.deploy import OpenVINOInferencer

# # ================== 模型 ==================
# model = OpenVINOInferencer(
#     path="/cms/user/huangsuyun/exported_model/weights/onnx/model.onnx",
#     device="CPU"
# )

# # ================== 数据目录 ==================
# folders = {
#     "normal": Path("/cms/user/huangsuyun/YOLOAB/dataset/test/normal_patches"),
#     "wire": Path("/cms/user/huangsuyun/YOLOAB/dataset/test/wire_patches"),
# }

# # ================== 开始预测 ==================
# all_scores = {}

# for name, img_dir in folders.items():
#     print(f"\n>>> Processing {name}: {img_dir}")

#     image_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
#     scores = []

#     for img_path in image_paths:
#         predictions = model.predict(str(img_path))
#         if predictions is None:
#             continue

#         for batch_or_pred in predictions:
#             preds = batch_or_pred if isinstance(batch_or_pred, (list, tuple)) else [batch_or_pred]
#             for pred in preds:
#                 score = float(pred.pred_score)
#                 scores.append(score)

#     print(f"{name}: 收集到 {len(scores)} 个 score")
#     all_scores[name] = scores

# # ================== 画图 ==================

# # 1️⃣ 叠加对比图
# plt.figure(figsize=(8, 6))
# for name, scores in all_scores.items():
#     plt.hist(scores, bins=50, alpha=0.5, label=name)

# plt.title("Prediction Score Distribution (Normal vs Wire)")
# plt.xlabel("Anomaly Score")
# plt.ylabel("Count")
# plt.legend()
# plt.tight_layout()
# plt.savefig("score_distribution_compare.png", dpi=200)
# plt.close()

# # 2️⃣ 各自单独画
# for name, scores in all_scores.items():
#     plt.figure(figsize=(6, 4))
#     plt.hist(scores, bins=50)
#     plt.title(f"Score Distribution - {name}")
#     plt.xlabel("Anomaly Score")
#     plt.ylabel("Count")
#     plt.tight_layout()
#     plt.savefig(f"score_hist_{name}.png", dpi=200)
#     plt.close()

# print("\n✅ 已生成：")
# print(" - score_distribution_compare.png")
# print(" - score_hist_normal.png")
# print(" - score_hist_wire.png")



import os
# import cv2
# import numpy as np
# import pandas as pd
from pathlib import Path
# from tqdm import tqdm
# import onnxruntime as ort
from anomalib.deploy import OpenVINOInferencer

from pathlib import Path
import pandas as pd

model = OpenVINOInferencer(
    # path="/cms/user/huangsuyun/exported_model/weights/onnx/model.onnx",
    path="/cms/user/huangsuyun/ANOMALIB/src/exported_model_0104_noresize_afbnew/weights/onnx/model.onnx",
    device="CPU"
)

image_dir = Path("/cms/user/huangsuyun/YOLOAB/dataset/test/normal_patches")
image_paths = sorted(image_dir.glob("*.jpg"))

# 用来存所有结果
rows = []

for img_path in image_paths:
    predictions = model.predict(str(img_path))

    if predictions is None:
        continue

    for batch_or_pred in predictions:
        preds = batch_or_pred if isinstance(batch_or_pred, (list, tuple)) else [batch_or_pred]

        for pred in preds:
            # print("\n------------------")
            # print(f"Image: {pred.image_path}")
            # print("Pred Score Tensor:")
            # print(pred.pred_score)
            # print("Pred Label Tensor:")
            # print(pred.pred_label)
            # print("------------------\n")

            # 转成 python list / 标量，避免 tensor 不能存表格
            score = pred.pred_score
            label = pred.pred_label

            try:
                score = score.tolist()
            except:
                pass

            try:
                label = label.tolist()
            except:
                pass

            rows.append({
                "image_path": str(pred.image_path),
                "pred_score": score,
                "pred_label": label
            })

# 转成 DataFrame
df = pd.DataFrame(rows)

# 保存为 CSV 或 Excel
df.to_csv("openvino_predictions_normal_afnew.csv", index=False)
# 或
# df.to_excel("openvino_predictions.xlsx", index=False)

print(f"✅ 已保存 {len(df)} 条结果到 openvino_predictions_wire.csv / xlsx")


# model = OpenVINOInferencer(
#     path="/cms/user/huangsuyun/exported_model/weights/onnx/model.onnx",
#     device="CPU"
# )

# image_dir = Path("/cms/user/huangsuyun/YOLOAB/dataset/test/normal_patches")

# image_paths = sorted(image_dir.glob("*.jpg"))  # 或 *.png

# for img_path in image_paths:
#     predictions = model.predict(str(img_path))

#     if predictions is None:
#         continue

#     for batch_or_pred in predictions:
#         preds = batch_or_pred if isinstance(batch_or_pred, (list, tuple)) else [batch_or_pred]

#         for pred in preds:
#             print("\n------------------")
#             print(f"Image: {pred.image_path}")
#             print("Pred Score Tensor:")
#             print(pred.pred_score)
#             print("Pred Label Tensor:")
#             print(pred.pred_label)
#             print("------------------\n")



# predictions = model.predict("/cms/user/huangsuyun/YOLOAB/dataset/normal_patches_1222/2_0.jpg")
# if predictions is not None:
#     for batch_or_pred in predictions:
#         # 兼容 batch 和 single prediction
#         if isinstance(batch_or_pred, (list, tuple)):
#             preds = batch_or_pred
#         else:
#             preds = [batch_or_pred]

#         for pred in preds:
#             print("\n------------------")
#             print(f"Image: {pred.image_path}")
#             print("Pred Score Tensor:")
#             print(pred.pred_score)   # ← 直接打印张量（可能是 heatmap 或 vector）
#             print("Pred Label Tensor:")
#             print(pred.pred_label)   # ← 同样直接打印张量
#             print("------------------\n")







# # ========== 1. 路径与参数配置 ==========
# IMAGE_DIR = Path("/cms/user/huangsuyun/YOLOAB/dataset/normal_patches_1222")
# # 替换为你导出的 ONNX 模型实际路径
# ONNX_MODEL_PATH = Path("/cms/user/huangsuyun/YOLOAB/models/model.onnx") 
# OUTPUT_DIR = Path("/cms/user/huangsuyun/YOLOAB/onnx_results")

# # 分类文件夹
# GOOD_DIR = OUTPUT_DIR / "Good"
# BAD_DIR = OUTPUT_DIR / "Bad"
# for d in [GOOD_DIR, BAD_DIR]: d.mkdir(parents=True, exist_ok=True)

# # PatchCore 默认通常是 224 或 256，请参考你导出时的配置
# # 如果导出时指定了 input_size，这里必须匹配
# INPUT_SIZE = (256, 256) 
# ANOMALY_THRESHOLD = 0.8  # 需要根据 ONNX 输出的得分范围进行调整

# # ========== 2. 初始化 ONNX Session ==========
# print(f"Loading ONNX model from: {ONNX_MODEL_PATH}")
# providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
# session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=providers)

# input_node = session.get_inputs()[0]
# input_name = input_node.name
# input_shape = input_node.shape  # 例如 [1, 3, 256, 256]
# print(f"Model expects input: {input_name} with shape {input_shape}")

# # ========== 3. 预处理函数 ==========
# def preprocess(image_path, target_size):
#     # 读取并转为 RGB
#     img = cv2.imread(str(image_path))
#     if img is None:
#         return None, None
#     original_shape = img.shape[:2]
    
#     # 调整大小
#     img_resized = cv2.resize(img, target_size)
#     img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
#     # 归一化 (ImageNet 标准)
#     img_float = img_rgb.astype(np.float32) / 255.0
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     img_norm = (img_float - mean) / std
    
#     # HWC -> BCHW
#     img_tensor = img_norm.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
#     return img_tensor, img

# # ========== 4. 执行推理 ==========
# img_paths = list(IMAGE_DIR.glob("*.[jJ][pP][gG]")) + list(IMAGE_DIR.glob("*.[pP][nN][gG]"))
# records = []

# print(f"Processing {len(img_paths)} images...")
# for img_path in tqdm(img_paths):
#     input_tensor, original_bgr = preprocess(img_path, INPUT_SIZE)
#     if input_tensor is None:
#         continue
    
#     # 推理
#     # 注意：Anomalib 导出的 ONNX 通常返回 [anomaly_map, pred_score]
#     outputs = session.run(None, {input_name: input_tensor})
    
#     # 获取异常得分 (通常是第二个输出，或者输出中的最大值)
#     # 不同版本的 Anomalib 导出顺序可能不同，通常 outputs[1] 是全局得分
#     score = float(outputs[1]) 
    
#     is_bad = score > ANOMALY_THRESHOLD
#     result_label = "NG" if is_bad else "OK"
    
#     # 保存结果
#     save_path = (BAD_DIR if is_bad else GOOD_DIR) / img_path.name
#     # 在图上标记分数
#     color = (0, 0, 255) if is_bad else (0, 255, 0)
#     cv2.putText(original_bgr, f"Score: {score:.3f}", (50, 50), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
#     cv2.imwrite(str(save_path), original_bgr)
    
#     records.append({"filename": img_path.name, "score": score, "result": result_label})

# # 保存 CSV
# pd.DataFrame(records).to_csv(OUTPUT_DIR / "results.csv", index=False)
# print(f"Done! Results saved to {OUTPUT_DIR}")