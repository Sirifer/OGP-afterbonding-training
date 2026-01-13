from ultralytics import YOLO

model = YOLO('/cms/user/huangsuyun/dataset/yolov8n.pt') 

results = model.train(
    data='data.yaml', 
    epochs=100,
    imgsz=640,
    batch=16,
    device=[0],      
    lr0=0.01,
    iou=0.5,        
    conf=0.05,         
    project='runs/afterbonding',  # 自定义主目录
    name='train_ab_addbkg',                # 自定义子目录名
    exist_ok=True,
)




# from ultralytics import YOLO

# model = YOLO('/cms/user/huangsuyun/dataset/yolov8n.pt') 

# results=model.train(
#     data='data.yaml',
#     epochs=100,
#     imgsz=640,
#     batch=16,
#     lr0=0.01,
#     iou=0.5,        
#     conf=0.1,         
# )


# from ultralytics import YOLO

# model = YOLO('/cms/user/huangsuyun/dataset/yolov8n.pt') 

# results = model.train(
#     data='data.yaml', 
#     epochs=100,
#     imgsz=640,
#     batch=16,
#     device=[0,1],      
#     lr0=0.01,
#     iou=0.5,        
#     conf=0.05,         
#     # augment=True,      # 开启数据增强
#     # hsv_h=0.015,      # 色调变换 ±1.5%
#     # hsv_s=0.3,        # 饱和度变换 ±70%
#     # hsv_v=0.2,        # 亮度变换 ±40%
#     # degrees=90,        # 随机旋转 ±90°
#     # # scale=0.5,         # 随机缩放 ±50%
#     # # shear=2.0,         # 随机剪切 ±2°
#     # fliplr=0.5,        # 左右翻转概率1
#     # flipud=0.5,        # 上下翻转概率1
#     # # 高级增强
#     # # mosaic=1.0,        # 使用 Mosaic 增强
#     # # mixup=0.2          # 使用 MixUp 增强
#     project='runs/afterbonding',  # 自定义主目录
#     name='train_1000bg_noaug',                # 自定义子目录名
#     exist_ok=True,
# )

#  name='train_1_105_1000',               #1类，105signal，1000bkg
