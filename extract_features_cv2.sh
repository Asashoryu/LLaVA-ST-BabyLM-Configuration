#!/bin/bash
#SBATCH --job-name=extract_cv2
#SBATCH --partition=gpusL40
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=extract_features_cv2_%j.out

# Load conda
source ~/.bashrc
conda activate llava-st

# Run extraction with OpenCV
cd /data1/ososovskyy/LLaVA-ST-BabyLM-Configuration
python extract_video_features_cv2.py
