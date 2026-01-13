#!/bin/bash

######## Part 1: 资源申请 #########
#SBATCH --partition=gpu
#SBATCH --qos=cmsnormal
#SBATCH --account=cmsgpu
#SBATCH --job-name=yolo_ab_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10GB
#SBATCH --gpus=v100:0
#SBATCH -t 1:00:00
#SBATCH -o /cms/user/huangsuyun/YOLOAB/runs/slurm-%j.out

######## Part 2: TMPDIR 修复 #########
export TMPDIR=/cms/user/huangsuyun/tmp
export TEMP=/cms/user/huangsuyun/tmp
export TMP=/cms/user/huangsuyun/tmp
mkdir -p /cms/user/huangsuyun/tmp

######## Part 3: 环境 #########
source /cms/user/huangsuyun/app/miniconda3/etc/profile.d/conda.sh
conda activate /cms/user/huangsuyun/conda/envs/yolov8

######## Part 4: 运行 #########
# srun python /cms/user/huangsuyun/YOLOAB/src/test.py
srun python /cms/user/huangsuyun/YOLOAB/src/pipeline.py

