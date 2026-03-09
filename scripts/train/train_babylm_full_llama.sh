#!/bin/bash
#SBATCH --job-name=full_llama_multi
#SBATCH --output=logs/train_full_llama_%j.out
#SBATCH --error=logs/train_full_llama_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --partition=gpusH100

source ~/.bashrc
conda activate llava-st
mkdir -p logs

# ============================================================
# FULL LLAMA2-7B TRAINING
# Same hyperparameters as baby_llama but with full 7B model
# 1 = _t, 2 = _ti, 3 = _tim
# ============================================================
EXP_ID=3
PROJECT_VERSION=full_7b

# Use the full Llama2-7B model prepared by prepare_full_llama2.py
MODEL_PATH="./full_llama2_7b_baseline"

if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model path '$MODEL_PATH' does not exist."
    echo "Run: python prepare_full_llama2.py first!"
    exit 1
fi

# Removed dataset shards.
# Using merged global dataset

case $EXP_ID in
    1)
        SUFFIX="_t"
        STAGE="1"
        ;;
    2)
        SUFFIX="_ti"
        STAGE="1"
        ;;
    3)
        SUFFIX="_tim"
        STAGE="2"
        ;;
    *)
        echo "ERROR: Unknown EXP_ID ($EXP_ID) or experiment type (insert 1, 2, or 3)."
        exit 1
        ;;
esac

# ============================================================
# SAME HYPERPARAMETERS AS BABY_LLAMA EXPERIMENTS
# ============================================================
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
NUM_TRAIN_EPOCHS=5
PER_DEVICE_TRAIN_BATCH_SIZE=4   # Can use 4 with H100 (80GB each)
GRADIENT_ACCUMULATION_STEPS=8   # 2 GPUs × 4 batch × 8 accum = 64 effective
LEARNING_RATE=5e-4
MM_PROJECTOR_LR=5e-4
SAVE_STRATEGY="steps"
SAVE_STEPS=2000
SAVE_TOTAL_LIMIT=50
WEIGHT_DECAY=0.
WARMUP_RATIO=0.1
LR_SCHEDULER_TYPE="cosine"
LOGGING_STEPS=2
MODEL_MAX_LENGTH=16384  # Can use 16K with H100s
GRADIENT_CHECKPOINTING=True
DATALOADER_NUM_WORKERS=2
LAZY_PREPROCESS=True
REPORT_TO="wandb"
IMAGE_ASPECT_RATIO="pad"

VISION_MODEL="facebook/dinov2-large"
DEEPSPEED_SCRIPT="scripts/zero2.json"
BASE_DATA_DIR="data/localized_narratives/llava_datasets"
OUTPUT_DIR="output/ckpt_full_7b_${PROJECT_VERSION}${SUFFIX}"

# Data selection: use the merged global dataset file
DATA_PATH="${BASE_DATA_DIR}/all_shards${SUFFIX}_merged.json"

[ -f "$DATA_PATH" ] || { echo "ERROR: data path $DATA_PATH not found"; exit 1; }

# Environment Variables (same as baby_llama)
export OMP_NUM_THREADS=8
export WANDB_PROJECT="full_llama_7b_multi"
export BABYLM_ENABLED=true
export BABYLM_WORD_LIMIT=100000000
export USE_PLAIN_LM=1

echo "================================================"
echo "FULL LLAMA2-7B TRAINING - EXPERIMENT $EXP_ID ($SUFFIX)"
echo "================================================"
echo "Model:          $MODEL_PATH"
echo "Dataset:        $DATA_PATH"
echo "Output:         $OUTPUT_DIR"
echo "Batch Size:     $PER_DEVICE_TRAIN_BATCH_SIZE per GPU (grad_accum=$GRADIENT_ACCUMULATION_STEPS)"
echo "Effective BS:   $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 2)) (2×4×8=64)"
echo "GPUs:           2x H100 (80GB each = 160GB total VRAM)"
echo "Max Length:     16384 (2x longer than baby model context)"
echo "DeepSpeed:      $DEEPSPEED_SCRIPT (ZeRO-2)"
echo "Epochs:         $NUM_TRAIN_EPOCHS"
echo "LR:             $LEARNING_RATE"
echo "Stage:          $STAGE"
echo "Vision:         $VISION_MODEL"
echo "Perceiver:      $MM_PERCEIVER_LATENTS latents (fast=$MM_PERCEIVER_LATENTS_FAST)"
echo "================================================"

# FORCE ENVIRONMENT VARIABLES FOR TOKEN PACKER
export MM_PERCEIVER_LATENTS=64
export MM_PERCEIVER_LATENTS_FAST=32

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
    --report_to ${REPORT_TO} \
    --deepspeed ${DEEPSPEED_SCRIPT}

echo "================================================"
echo "Training completed: $(date)"
echo "================================================"
echo "🎯 Compare with baby_llama results:"
echo "   Baby model (~195M params): 79.05% BLIMP"
echo "   Full model (7B params):    Check results/"
echo "================================================"
