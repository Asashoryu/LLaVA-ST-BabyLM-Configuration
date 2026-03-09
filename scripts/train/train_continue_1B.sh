#!/bin/bash
#SBATCH --job-name=baby_llama_1B
#SBATCH --output=logs/train_1B_%j.out
#SBATCH --error=logs/train_1B_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --partition=gpusL40

source ~/.bashrc
conda activate llava-st
mkdir -p logs

# ============================================================
# FINAL WORKING APPROACH: Use original output_dir directly
# ============================================================

# ✅ Use the ORIGINAL base model
MODEL_PATH="./baby_llama_baseline"

# ✅ CRITICAL: Use the SAME output_dir where Gen 23 checkpoints are
# This allows trainer to automatically resume from the latest checkpoint
OUTPUT_DIR="output/ckpt_mixed_23_tim"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Base model path '$MODEL_PATH' does not exist."
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: Output directory '$OUTPUT_DIR' does not exist."
    exit 1
fi

echo "------------------------------------------------"
echo "CONTINUING TRAINING TO 1 BILLION WORDS"
echo "Base model: $MODEL_PATH"
echo "Output dir: $OUTPUT_DIR (contains Gen 23 checkpoints)"
echo "Will resume from latest checkpoint automatically"
echo "Expected new checkpoints: 200M, 300M, 400M, 500M, 600M, 700M, 800M, 900M, 1000M"
echo "------------------------------------------------"

SUFFIX="_tim"
STAGE="2"

# Training parameters
MM_TUNABLE_PARTS="mm_vision_resampler,mm_mlp_adapter,mm_language_model"
MM_VISION_SELECT_LAYER=-2
MM_PROJECTOR_TYPE="mlp2x_gelu"
MM_RESAMPLER_TYPE="fast_slow_resampler"
MM_PERCEIVER_LATENTS=64
MM_PERCEIVER_LATENTS_FAST=32
MM_PERCEIVER_DEPTH=2
USE_DOWNSAMPLE_IMAGE=False
FP16=True
ATTN_IMPLEMENTATION="sdpa"
NUM_TRAIN_EPOCHS=10
PER_DEVICE_TRAIN_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=5e-4
MM_PROJECTOR_LR=5e-4
SAVE_STRATEGY="steps"
SAVE_STEPS=2000
SAVE_TOTAL_LIMIT=50
WEIGHT_DECAY=0.
WARMUP_RATIO=0.1
LR_SCHEDULER_TYPE="cosine"
LOGGING_STEPS=2
MODEL_MAX_LENGTH=32768
GRADIENT_CHECKPOINTING=True
DATALOADER_NUM_WORKERS=2
LAZY_PREPROCESS=True
REPORT_TO="wandb"
IMAGE_ASPECT_RATIO="pad"
VISION_MODEL="facebook/dinov2-large"
DEEPSPEED_SCRIPT="scripts/zero2.json"
BASE_DATA_DIR="data/localized_narratives/llava_datasets"
DATA_PATH="${BASE_DATA_DIR}/all_shards${SUFFIX}_merged.json"

[ -f "$DATA_PATH" ] || { echo "ERROR: data path $DATA_PATH not found"; exit 1; }

# Environment Variables
export OMP_NUM_THREADS=8
export WANDB_PROJECT="baby_llama_1B"
export BABYLM_ENABLED=true
unset BABYLM_WORD_LIMIT
export USE_PLAIN_LM=1
export MM_PERCEIVER_LATENTS=64
export MM_PERCEIVER_LATENTS_FAST=32

# ✅ Use base model + original output_dir
# Trainer will auto-detect checkpoint-14378-100Mwords and resume from there
python -u llava/train/train_mem.py \
    --model_name_or_path ${MODEL_PATH} \
    --version llama_v2 \
    --data_path "${DATA_PATH}" \
    --image_folder "." \
    --vision_tower ${VISION_MODEL} \
    --image_aspect_ratio ${IMAGE_ASPECT_RATIO} \
    --output_dir ${OUTPUT_DIR} \
    --mm_tunable_parts "${MM_TUNABLE_PARTS}" \
    --mm_vision_select_layer ${MM_VISION_SELECT_LAYER} \
    --mm_projector_type ${MM_PROJECTOR_TYPE} \
    --mm_resampler_type ${MM_RESAMPLER_TYPE} \
    --mm_perceiver_latents ${MM_PERCEIVER_LATENTS} \
    --mm_perceiver_latents_fast ${MM_PERCEIVER_LATENTS_FAST} \
    --mm_perceiver_depth ${MM_PERCEIVER_DEPTH} \
    --use_downsample_image ${USE_DOWNSAMPLE_IMAGE} \
    --fp16 ${FP16} \
    --attn_implementation ${ATTN_IMPLEMENTATION} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --mm_projector_lr ${MM_PROJECTOR_LR} \
    --stage "${STAGE}" \
    --save_strategy "${SAVE_STRATEGY}" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --logging_steps ${LOGGING_STEPS} \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
    --lazy_preprocess ${LAZY_PREPROCESS} \
    --report_to ${REPORT_TO}

echo "✅ Training completed: $(date)"
