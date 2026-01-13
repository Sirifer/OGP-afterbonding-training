import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from anomalib.deploy import OpenVINOInferencer

# =========================================================
# 0. 参数配置
# =========================================================
ANOMALY_THRESHOLD = 0.95

IMAGE_DIR = Path(
    # "/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/af_wire"
    "/cms/user/huangsuyun/YOLOAB/samples/320MLF3WCIH0350_after_bonding_front"
)

YOLO_MODEL_PATH = Path(
    "/cms/user/huangsuyun/YOLOAB/models/YOLO/best.pt"
)

PATCHCORE_MODEL = Path(
    # "/cms/user/huangsuyun/exported_model/weights/onnx/model.onnx"
    # "/publicfs/cms/user/huangsuyun/ANOMALIB/src/exported_model_0105_noresize_1000/weights/onnx/model.onnx"
    "/cms/user/huangsuyun/YOLOAB/models/ANOMALIB/model.onnx"
)

TMP_PATCH_DIR = Path("/publicfs/cms/user/huangsuyun/YOLOAB/tmp_95/af_wire_new_0104/patch")
YOLO_TMP_DIR = Path("/publicfs/cms/user/huangsuyun/YOLOAB/tmp_95/af_wire_new_0104/yolo_origin")
ABNORMAL_DIR = Path("/publicfs/cms/user/huangsuyun/YOLOAB/tmp_95/af_wire_new_0104/yolo_abnormal")

for d in [TMP_PATCH_DIR, YOLO_TMP_DIR, ABNORMAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1. 初始化模型
# =========================================================
print("Loading YOLO...")
yolo = YOLO(str(YOLO_MODEL_PATH))

print("Loading PatchCore...")
patchcore = OpenVINOInferencer(
    path=str(PATCHCORE_MODEL),
    device="CPU"
)

# =========================================================
# 2. Stage 1：YOLO 检测 + 裁剪 patch + 保存可视化
# =========================================================
print("Stage 1: YOLO detect & crop patches")

patch_records = []

img_paths = sorted(
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.png")) +
    list(IMAGE_DIR.glob("*.bmp")) +
    list(IMAGE_DIR.glob("*.BMP"))
)

for img_path in tqdm(img_paths, desc="YOLO Processing"):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    results = yolo.predict(
        source=img,
        conf=0.05,
        iou=0.5,
        verbose=False
    )

    result = results[0]

    # 保存 YOLO 框图
    if len(result.boxes) > 0:
        yolo_vis = result.plot()
        cv2.imwrite(str(YOLO_TMP_DIR / img_path.name), yolo_vis)

    # 裁剪 wire patch
    for idx, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        if cls_id != 0:  # wire class
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        patch = img[y1:y2, x1:x2]
        # patch = cv2.resize(patch, (128, 128))

        if patch.size == 0:
            continue

        patch_name = f"{img_path.stem}_wire{idx}.jpg"
        patch_path = TMP_PATCH_DIR / patch_name
        cv2.imwrite(str(patch_path), patch)

        patch_records.append({
            "patch_name": patch_name,
            "source_image": img_path.name,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })

patch_df = pd.DataFrame(patch_records)
print(f"Total wire patches: {len(patch_df)}")

# =========================================================
# 3. Stage 2：PatchCore 评分 + abnormal 画回整图
# =========================================================
print("Stage 2: PatchCore scoring")

score_records = {}
abnormal_images = {}

for patch_path in tqdm(TMP_PATCH_DIR.glob("*.jpg"), desc="PatchCore"):
    preds = patchcore.predict(str(patch_path))
    if preds is None:
        continue

    patch_name = patch_path.name

    for batch in preds:
        batch = batch if isinstance(batch, (list, tuple)) else [batch]

        for pred in batch:
            score = float(pred.pred_score)
            label = int(pred.pred_label)

            score_records[patch_name] = {
                "score": score,
                "label": label
            }

            if score <= ANOMALY_THRESHOLD:
                continue

            # 找回 YOLO 信息
            row = patch_df[patch_df["patch_name"] == patch_name]
            if row.empty:
                continue
            row = row.iloc[0]

            src_img_name = row["source_image"]
            src_img_path = IMAGE_DIR / src_img_name

            if src_img_name not in abnormal_images:
                img = cv2.imread(str(src_img_path))
                if img is None:
                    continue
                abnormal_images[src_img_name] = img

            img = abnormal_images[src_img_name]

            x1, y1, x2, y2 = map(int, [row.x1, row.y1, row.x2, row.y2])

            # 画框 + 分数
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img,
                f"{score:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

# =========================================================
# 4. 保存 abnormal 整图
# =========================================================
for name, img in abnormal_images.items():
    cv2.imwrite(str(ABNORMAL_DIR / name), img)

# =========================================================
# 5. 保存 PatchCore CSV
# =========================================================
rows = []
for patch_name, v in score_records.items():
    rows.append({
        "patch": patch_name,
        "score": v["score"],
        "label": v["label"],
        "abnormal": int(v["score"] > ANOMALY_THRESHOLD)
    })

df = pd.DataFrame(rows)
df.to_csv(TMP_PATCH_DIR / "patchcore_scores.csv", index=False)

# =========================================================
# 6. Done
# =========================================================
print("\n========== DONE ==========")
print(f"YOLO vis      : {YOLO_TMP_DIR}")
print(f"Patch crops   : {TMP_PATCH_DIR}")
print(f"Abnormal imgs : {ABNORMAL_DIR}")
