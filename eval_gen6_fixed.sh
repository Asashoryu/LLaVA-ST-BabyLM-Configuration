#!/bin/bash
#SBATCH --job-name=eval_gen6_fix
#SBATCH --output=logs/eval_gen6_fix_%j.out
#SBATCH --error=logs/eval_gen6_fix_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --partition=normal

source /shared_software/anaconda3/bin/activate llava-st

MODEL="/data1/ososovskyy/LLaVA-ST-BabyLM-Configuration/output/ckpt_mixed_6_tim/checkpoint-1258-9Mwords"
TASK="blimp"

echo "============================================"
echo "🧪 EVAL GEN 6 WITH FIXED BLACK IMAGE CACHE"
echo "Model: $MODEL"
echo "============================================"

cd /data1/ososovskyy/babylm_eval

export MM_PERCEIVER_LATENTS=64
export MM_PERCEIVER_LATENTS_FAST=32

python evaluation_pipeline/sentence_zero_shot/run.py \
    --task_type $TASK \
    --batch_size 64 \
    --model_path "$MODEL" \
    --backend causal \
    --image_template append_image_token
