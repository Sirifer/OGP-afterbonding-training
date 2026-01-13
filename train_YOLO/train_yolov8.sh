#! /bin/bash

######## Part 1: 资源申请 #########
#SBATCH --partition=gpu              # 使用 GPU 分区
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu            # 账户 (按你集群要求)
#SBATCH --job-name=yolo_glue_train   # 作业名
#SBATCH --ntasks=1                   # 任务数 (只跑 1 个训练任务)
#SBATCH --cpus-per-task=8            # 每任务 CPU 核数
#SBATCH --mem-per-cpu=4096          # 每 CPU 内存 (MB)，8*4096=32G
#SBATCH --gpus=v100:1                # 申请 1 张 V100
#SBATCH -t 2:00:00                  # 最长运行时间
#SBATCH -o /cms/user/huangsuyun/dataset/runs/slurm-%j.out   # 输出日志文件



# 加载环境
source /cvmfs/lhcbdev.cern.ch/conda/miniconda/linux-64/prod/etc/profile.d/conda.sh
conda activate /cms/user/huangsuyun/conda/envs/YOLOv8

# 配置 YOLOv8 的用户目录，避免权限报错
export YOLO_CONFIG_DIR=/cms/user/huangsuyun/ultralytics_cfg
export ULTRALYTICS_CACHE_DIR=/cms/user/huangsuyun/ultralytics_cache
mkdir -p "$YOLO_CONFIG_DIR" "$ULTRALYTICS_CACHE_DIR"

export MPLCONFIGDIR=/cms/user/huangsuyun/matplotlib_cache
mkdir -p $MPLCONFIGDIR

# export YOLO_CONFIG_DIR=/cms/user/huangsuyun/ultralytics_cfg
# export ULTRALYTICS_CACHE_DIR=/cms/user/huangsuyun/ultralytics_cache
# mkdir -p "$YOLO_CONFIG_DIR" "$ULTRALYTICS_CACHE_DIR" /cms/user/huangsuyun/dataset/runs

# 启动训练
srun python /cms/user/huangsuyun/dataset/train_yolov8.py
