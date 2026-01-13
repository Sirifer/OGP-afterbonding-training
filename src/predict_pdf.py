import os
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from fpdf import FPDF
from ultralytics import YOLO
from anomalib.deploy import OpenVINOInferencer

# =========================================================
# 0. 参数配置
# =========================================================
ANOMALY_THRESHOLD = 0.98

IMAGE_DIR = Path(
    # "/cms/user/huangsuyun/dataset/samples/afterbonding/afterbondingall/af_wire"
    "/cms/user/huangsuyun/YOLOAB/samples/320MLF3WCIH0350_after_bonding_front"
)

YOLO_MODEL_PATH = Path(
    "/cms/user/huangsuyun/YOLOAB/models/YOLO/best.pt"
)

PATCHCORE_MODEL = Path(
    "/cms/user/huangsuyun/YOLOAB/models/ANOMALIB/model.onnx"
)

TMP_PATCH_DIR = Path("/publicfs/cms/user/huangsuyun/YOLOAB/tmp_wire_pdf_run/af_wire_new_0104/patch")
YOLO_TMP_DIR = Path("/publicfs/cms/user/huangsuyun/YOLOAB/tmp_wire_pdf_run/af_wire_new_0104/yolo_origin")
ABNORMAL_DIR = Path("/publicfs/cms/user/huangsuyun/YOLOAB/tmp_wire_pdf_run/af_wire_new_0104/yolo_abnormal")

# 输出 PDF
PDF_OUT = Path("/publicfs/cms/user/huangsuyun/YOLOAB/tmp_wire_pdf_run/af_wire_new_0104/wire_anomaly_report.pdf")

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
        conf=0.2,
        iou=0.5,
        verbose=False
    )

    result = results[0]

    # 保存 YOLO 框图（保持原文件名；若原图是 bmp，这里会写出 bmp）
    if len(result.boxes) > 0:
        yolo_vis = result.plot()
        cv2.imwrite(str(YOLO_TMP_DIR / img_path.name), yolo_vis)

    # 裁剪 wire patch
    for idx, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        if cls_id != 0:  # wire class
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        # 防止越界/空 patch
        H, W = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        patch = img[y1:y2, x1:x2]
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

for patch_path in tqdm(sorted(TMP_PATCH_DIR.glob("*.jpg")), desc="PatchCore"):
    preds = patchcore.predict(str(patch_path))
    if preds is None:
        continue

    patch_name = patch_path.name

    for batch in preds:
        batch = batch if isinstance(batch, (list, tuple)) else [batch]

        for pred in batch:
            score = float(pred.pred_score)
            label = int(pred.pred_label)

            score_records[patch_name] = {"score": score, "label": label}

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
                img, f"{score:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2
            )

# =========================================================
# 4. 保存 abnormal 整图（强制写 JPG，避免 FPDF 不支持 BMP）
# =========================================================
for name, img in abnormal_images.items():
    stem = Path(name).stem
    out_path = ABNORMAL_DIR / f"{stem}.jpg"
    cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

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
# 6. 生成 PDF：把异常整图粘贴进去（5列网格 + cell编号）
# =========================================================

CELL_PHOTO_MAP = {
    36 : "1", 38 : "2", 40 : "30", 42 : "13", 44 : "24", 46 : "34", 48 : "5", 50 : "25",
    52 : "82", 54 : "22", 56 : "70", 58 : "83", 60 : "98", 62 : "58", 64 : "48", 66 : "141",
    68 : "61", 70 : "60", 72 : "142", 74 : "129", 76 : "130", 78 : "156", 80 : "180", 82 : "171",
    84 : "136", 86 : "94", 88 : "138", 90 : "163", 92 : "190", 94 : "192", 96 : "126", 98 : "176",
    100 : "154", 102 : "177", 104 : "3", 106 : "4", 108 : "7", 110 : "27", 112 : "28", 114 : "51",
    116 : "63", 118 : "74", 120 : "104", 122 : "105", 124 : "91", 126 : "77", 128 : "93", 130 : "64",
    132 : "80", 134 : "111", 136 : "140", 138 : "139", 140 : "168", 142 : "153", 144 : "179", 146 : "189",
    148 : "150", 150 : "149", 152 : "174", 154 : "161", 156 : "172", 158 : "184", 160 : "196", 162 : "186",
    164 : "198", 166 : "169", 168 : "132", 170 : "133", 172 : "120", 174 : "112", 176 : "99", 178 : "116",
    180 : "102", 182 : "86", 184 : "118", 186 : "85", 188 : "71", 190 : "87", 192 : "57", 194 : "41",
    196 : "31", 198 : "corner_9", 200 : "corner_18", 202 : "corner_95", 204 : "corner_197", 206 : "corner_191",
    208 : "corner_81", 210 : "66", 212 : "52", 214 : "67", 216 : "54", 218 : "55", 220 : "47", 222 : "8",
    224 : "124", 226 : "122", 228 : "185", 230 : "81",
}

def parse_cell_from_filename(img_path: Path) -> str:
    """
    你的文件名示例：
    module_after_bonding_front_check-36-1.BMP
    或保存后：
    module_after_bonding_front_check-36-1.jpg

    解析逻辑：取倒数第二段 = 36
    """
    try:
        parts = img_path.stem.split("-")
        num = int(parts[-2])
        return str(CELL_PHOTO_MAP.get(num, "?"))
    except Exception:
        return "?"

def create_pdf():
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=10)
    try:
        pdf.add_font("SourceHanSansSC", style="", fname="SourceHanSansSC-Regular.otf", uni=True)
        pdf.add_font("SourceHanSansSC", style="B", fname="SourceHanSansSC-Bold.otf", uni=True)
        pdf.set_font("SourceHanSansSC", "B", 16)
    except Exception:
        pdf.set_font("Arial", "B", 16)
    return pdf

def add_images_to_pdf(pdf, image_paths, title):
    pdf.add_page()
    font_name = "SourceHanSansSC" if "SourceHanSansSC" in pdf.fonts else "Arial"
    pdf.set_font(font_name, "B", 14)
    pdf.cell(0, 10, title, ln=True, align="C")

    margin = 10
    cols = 5
    gap = 2
    width = (210 - 2 * margin - (cols - 1) * gap) / cols
    img_h = width * 0.75
    row_height = img_h + 10
    y = 25

    for idx, img_path in enumerate(image_paths):
        x = margin + (idx % cols) * (width + gap)
        if idx % cols == 0 and idx != 0:
            y += row_height
            if y > 260:
                pdf.add_page()
                y = 25

        pdf.image(str(img_path), x=x, y=y, w=width, h=img_h)

        cell = parse_cell_from_filename(Path(img_path))
        pdf.set_font(font_name, "", 8)
        pdf.text(x + 2, y + img_h + 5, f"cell {cell}")

abnormal_imgs = sorted(
    list(ABNORMAL_DIR.glob("*.jpg")) +
    list(ABNORMAL_DIR.glob("*.jpeg")) +
    list(ABNORMAL_DIR.glob("*.png"))
)

if not abnormal_imgs:
    print("⚠️ No abnormal images found. PDF not generated.")
else:
    pdf = create_pdf()
    add_images_to_pdf(pdf, abnormal_imgs, title=f"Wire Anomaly Report - {IMAGE_DIR.name}")
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(PDF_OUT))
    print(f"✅ PDF generated: {PDF_OUT}")

# =========================================================
# 7. Done
# =========================================================
print("\n========== DONE ==========")
print(f"YOLO vis      : {YOLO_TMP_DIR}")
print(f"Patch crops   : {TMP_PATCH_DIR}")
print(f"Abnormal imgs : {ABNORMAL_DIR}")






# import cv2
# import shutil
# import tempfile
# from pathlib import Path
# from tqdm import tqdm
# from fpdf import FPDF
# from ultralytics import YOLO
# from anomalib.deploy import OpenVINOInferencer
# from PyPDF2 import PdfMerger

# # =======================
# # 路径配置（按你生产流程）
# # =======================
# ROOT_DIR = Path("/cms/user/huangsuyun/YOLOAB/samples")   # 大文件夹：下面很多模块文件夹
# FINAL_PDF = Path("/cms/user/huangsuyun/YOLOAB/report/all_wire_anomaly.pdf")

# YOLO_MODEL_PATH = Path("/cms/user/huangsuyun/YOLOAB/models/YOLO/best.pt")
# PATCHCORE_MODEL = Path("/cms/user/huangsuyun/YOLOAB/models/ANOMALIB/model.onnx")

# ANOMALY_THRESHOLD = 0.95
# YOLO_CONF = 0.05
# YOLO_IOU = 0.5
# WIRE_CLASS_ID = 0

# # 只为了插入 PDF 临时保存“标注后的整图”
# TMP_ANNOT_DIR = Path("/tmp/wire_anom_annot_for_pdf")

# # =======================
# # 你的 cell 映射表（原样粘贴进来）
# # =======================
# CELL_PHOTO_MAP = {  # cell号映射表
#     36 : "1",
#     38 : "2",
#     40 : "30",
#     42 : "13",
#     44 : "24",
#     46 : "34",
#     48 : "5",
#     50 : "25",
#     52 : "82",
#     54 : "22",
#     56 : "70",
#     58 : "83",
#     60 : "98",
#     62 : "58",
#     64 : "48",
#     66 : "141",
#     68 : "61",
#     70 : "60",
#     72 : "142",
#     74 : "129",
#     76 : "130",
#     78 : "156",
#     80 : "180",
#     82 : "171",
#     84 : "136",
#     86 : "94",
#     88 : "138",
#     90 : "163",
#     92 : "190",
#     94 : "192",
#     96 : "126",
#     98 : "176",
#     100 : "154",
#     102 : "177",
#     104 : "3",
#     106 : "4",
#     108 : "7",
#     110 : "27",
#     112 : "28",
#     114 : "51",
#     116 : "63",
#     118 : "74",
#     120 : "104",
#     122 : "105",
#     124 : "91",
#     126 : "77",
#     128 : "93",
#     130 : "64",
#     132 : "80",
#     134 : "111",
#     136 : "140",
#     138 : "139",
#     140 : "168",
#     142 : "153",
#     144 : "179",
#     146 : "189",
#     148 : "150",
#     150 : "149",
#     152 : "174",
#     154 : "161",
#     156 : "172",
#     158 : "184",
#     160 : "196",
#     162 : "186",
#     164 : "198",
#     166 : "169",
#     168 : "132",
#     170 : "133",
#     172 : "120",
#     174 : "112",
#     176 : "99",
#     178 : "116",
#     180 : "102",
#     182 : "86",
#     184 : "118",
#     186 : "85",
#     188 : "71",
#     190 : "87",
#     192 : "57",
#     194 : "41",
#     196 : "31",
#     198 : "corner_9",
#     200 : "corner_18",
#     202 : "corner_95",
#     204 : "corner_197",
#     206 : "corner_191",
#     208 : "corner_81",
#     210 : "66",
#     212 : "52",
#     214 : "67",
#     216 : "54",
#     218 : "55",
#     220 : "47",
#     222 : "8",
#     224 : "124",
#     226 : "122",
#     228 : "185",
#     230 : "81",
# }

# # =======================
# # PDF 相关
# # =======================
# def create_pdf():
#     pdf = FPDF(unit="mm", format="A4")
#     pdf.set_auto_page_break(auto=True, margin=10)
#     try:
#         pdf.add_font("SourceHanSansSC", style="", fname="SourceHanSansSC-Regular.otf", uni=True)
#         pdf.add_font("SourceHanSansSC", style="B", fname="SourceHanSansSC-Bold.otf", uni=True)
#         pdf.set_font("SourceHanSansSC", "B", 16)
#     except Exception:
#         pdf.set_font("Arial", "B", 16)
#     return pdf

# def parse_cell_from_filename(img_path: Path) -> str:
#     """
#     完全沿用你 leakage 脚本的解析方式：
#     num = int(stem.split("-")[-2]) -> CELL_PHOTO_MAP[num]
#     """
#     try:
#         num = int(img_path.stem.split("-")[-2])
#         return str(CELL_PHOTO_MAP.get(num, "?"))
#     except Exception:
#         return "?"

# def add_images_to_pdf(pdf, image_paths, module_id):
#     """
#     布局与你 leakage 报告一致：5列网格，图片下方写 cell xx
#     """
#     pdf.add_page()
#     font_name = "SourceHanSansSC" if "SourceHanSansSC" in pdf.fonts else "Arial"
#     pdf.set_font(font_name, "B", 14)
#     pdf.cell(0, 10, f"Wire Anomaly Report - {module_id}", ln=True, align="C")

#     margin = 10
#     cols = 5
#     gap = 2
#     width = (210 - 2 * margin - (cols - 1) * gap) / cols
#     img_h = width * 0.75
#     row_height = img_h + 10
#     y = 25

#     for idx, img_path in enumerate(image_paths):
#         x = margin + (idx % cols) * (width + gap)
#         if idx % cols == 0 and idx != 0:
#             y += row_height
#             if y > 260:
#                 pdf.add_page()
#                 y = 25

#         pdf.image(str(img_path), x=x, y=y, w=width, h=img_h)

#         cell = parse_cell_from_filename(Path(img_path))
#         pdf.set_font(font_name, "", 8)
#         pdf.text(x + 2, y + img_h + 5, f"cell {cell}")

# # =======================
# # 推理：给 patch 打分
# # =======================
# def infer_patch_score(patchcore, patch_bgr):
#     if patch_bgr is None or patch_bgr.size == 0:
#         return None

#     # OpenVINOInferencer 通常吃文件路径：用临时文件避免存大量 patch
#     with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
#         cv2.imwrite(tmp.name, patch_bgr)
#         preds = patchcore.predict(tmp.name)

#     if preds is None:
#         return None

#     best_score = None
#     for batch in preds:
#         batch = batch if isinstance(batch, (list, tuple)) else [batch]
#         for pred in batch:
#             score = float(pred.pred_score)
#             best_score = score if best_score is None else max(best_score, score)
#     return best_score

# # =======================
# # 单个模块处理：只输出“异常整图（带框和分数）”
# # =======================
# def process_module(module_dir: Path, yolo, patchcore):
#     img_paths = sorted(
#         list(module_dir.glob("*.jpg")) +
#         list(module_dir.glob("*.png")) +
#         list(module_dir.glob("*.bmp")) +
#         list(module_dir.glob("*.BMP"))
#     )
#     if not img_paths:
#         return []

#     module_id = module_dir.name.split("_after_")[0]  # 沿用你原来写法
#     out_imgs = []

#     for img_path in tqdm(img_paths, desc=f"YOLO+PatchCore {module_id}", leave=False):
#         img = cv2.imread(str(img_path))
#         if img is None:
#             continue

#         results = yolo.predict(source=img, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
#         r = results[0]
#         if r.boxes is None or len(r.boxes) == 0:
#             continue

#         abnormal_this_image = False

#         for box in r.boxes:
#             cls_id = int(box.cls[0])
#             if cls_id != WIRE_CLASS_ID:
#                 continue

#             x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

#             # 防越界
#             H, W = img.shape[:2]
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(W - 1, x2), min(H - 1, y2)
#             if x2 <= x1 or y2 <= y1:
#                 continue

#             patch = img[y1:y2, x1:x2]
#             score = infer_patch_score(patchcore, patch)
#             if score is None:
#                 continue

#             if score > ANOMALY_THRESHOLD:
#                 abnormal_this_image = True
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(
#                     img, f"{score:.2f}",
#                     (x1, max(y1 - 10, 20)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6,
#                     (0, 0, 255), 2
#                 )

#         if abnormal_this_image:
#             TMP_ANNOT_DIR.mkdir(parents=True, exist_ok=True)
#             # 不管原图后缀是什么，都把“标注图”统一存成 jpg
#             save_path = TMP_ANNOT_DIR / f"{module_id}__{img_path.stem}.jpg"
#             cv2.imwrite(str(save_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
#             out_imgs.append(save_path)


#     return out_imgs, module_id

# # =======================
# # 主流程：遍历 ROOT_DIR 下所有模块文件夹，生成一个总 PDF
# # =======================
# def process_all_modules(root_dir: Path, final_pdf_path: Path):
#     # 清理旧临时目录
#     if TMP_ANNOT_DIR.exists():
#         shutil.rmtree(TMP_ANNOT_DIR)

#     print("Loading YOLO...")
#     yolo = YOLO(str(YOLO_MODEL_PATH))

#     print("Loading PatchCore...")
#     patchcore = OpenVINOInferencer(path=str(PATCHCORE_MODEL), device="CPU")

#     pdf = create_pdf()
#     any_found = False

#     for folder in sorted(root_dir.iterdir()):
#         if not folder.is_dir():
#             continue

#         print(f"\n🔍 Processing module folder: {folder.name}")
#         abnormal_imgs, module_id = process_module(folder, yolo, patchcore)

#         if not abnormal_imgs:
#             print(f"✅ No abnormal in {module_id}, skip")
#             continue

#         any_found = True
#         print(f"⚠️ Found {len(abnormal_imgs)} abnormal images in {module_id}")
#         add_images_to_pdf(pdf, abnormal_imgs, module_id)

#     if not any_found:
#         print("⚠️ No abnormal found in all modules. PDF not generated.")
#         return

#     final_pdf_path.parent.mkdir(parents=True, exist_ok=True)
#     pdf.output(str(final_pdf_path))
#     print(f"\n✅ Final PDF generated: {final_pdf_path}")

#     # 如果你生产环境不想留任何中间产物：打开这行
#     # shutil.rmtree(TMP_ANNOT_DIR)

# if __name__ == "__main__":
#     process_all_modules(ROOT_DIR, FINAL_PDF)
