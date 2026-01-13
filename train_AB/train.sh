#! /bin/bash

######## Part 1: 资源申请 #########
#SBATCH --partition=gpu              # 使用 GPU 分区
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu           
#SBATCH --job-name=yolo_glue_train   
#SBATCH --ntasks=1                 
#SBATCH --cpus-per-task=2       
#SBATCH --mem-per-cpu=40GB         # 每 CPU 内存 (MB)，8*4096=32G
#SBATCH --gpus=v100:1              
#SBATCH -t 00:30:00                 
#SBATCH -o /cms/user/huangsuyun/ANOMALIB/runs/slurm-%j.out 


export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 
# export CUDA_VISIBLE_DEVICES=0
# 加载环境
source /cms/user/huangsuyun/app/miniconda3/etc/profile.d/conda.sh
conda activate /cms/user/huangsuyun/conda/envs/ab

# 配置 YOLOv8 的用户目录，避免权限报错
# export YOLO_CONFIG_DIR=/cms/user/huangsuyun/ultralytics_cfg
# export ULTRALYTICS_CACHE_DIR=/cms/user/huangsuyun/ultralytics_cache
# mkdir -p "$YOLO_CONFIG_DIR" "$ULTRALYTICS_CACHE_DIR"

export MPLCONFIGDIR=/cms/user/huangsuyun/matplotlib_cache
mkdir -p $MPLCONFIGDIR
export HF_HOME=/publicfs/cms/user/huangsuyun/.cache/huggingface
export HF_HUB_OFFLINE=1

# export YOLO_CONFIG_DIR=/cms/user/huangsuyun/ultralytics_cfg
# export ULTRALYTICS_CACHE_DIR=/cms/user/huangsuyun/ultralytics_cache
# mkdir -p "$YOLO_CONFIG_DIR" "$ULTRALYTICS_CACHE_DIR" /cms/user/huangsuyun/dataset/runs

# 启动训练
# srun python /cms/user/huangsuyun/ANOMALIB/src/padim.py
srun python /cms/user/huangsuyun/ANOMALIB/src/complete_pipeline.py