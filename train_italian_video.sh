#!/bin/bash
#SBATCH --job-name=babby_italian
#SBATCH --output=logs/train_italian_%j.out
#SBATCH --error=logs/train_italian_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gpusL40

source ~/.bashrc
conda activate llava-st
mkdir -p logs

# ============================================================
# Italian Video-Language Model Training
# 1 = _t_it (text-only Italian), 2 = _vi_it (video+italian)
# ============================================================
EXP_ID=2
PROJECT_VERSION=3

# Model: Use BAMBI with random weights (training from scratch)
# Created with: python create_random_weights_model.py
MODEL_PATH="./babylm_italian/bambi_random_init"

if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model path '$MODEL_PATH' does not exist."
    echo "Run: python create_random_weights_model.py"
    exit 1
fi

# Dataset selection based on experiment ID
case $EXP_ID in
    1)
        SUFFIX="_t_it"
        STAGE="1"
        echo "📝 EXP 1: Text-only Italian training"
        ;;
    2)
        SUFFIX="_vi_it"
        STAGE="1"  # Note: Stage 1 for video, not stage 2
        echo "🎬 EXP 2: Video + Italian text training"
        ;;
    *)
        echo "ERROR: Unknown EXP_ID ($EXP_ID). Use 1 for text-only, 2 for video+text"
        exit 1
        ;;
esac

# ============================================================
# Training Hyperparameters
# ============================================================
# Set to path of pretrained weights to resume training, or leave empty to start from scratch
PRETRAIN_MM_MLP_ADAPTER=""
# Example: PRETRAIN_MM_MLP_ADAPTER="./output/ckpt_italian_1_vi_it/mm_projector.bin"

# Vision encoder (for video processing)
VISION_MODEL="facebook/dinov2-large"
# VISION_MODEL="google/siglip-so400m-patch14-384"

# Multimodal components to train
MM_TUNABLE_PARTS="mm_vision_resampler,mm_mlp_adapter,mm_language_model"
MM_VISION_SELECT_LAYER=-2
MM_PROJECTOR_TYPE="mlp2x_gelu"
MM_RESAMPLER_TYPE="fast_slow_resampler"
# Original LLaVA-ST uses 81/9, but with 32K context
# GPT-2 has 2048 limit, so we keep 81/9 but set fast_frame_num=slow_frame_num=4
# This gives: fast=(9×4-2)=34, slow=(81×4-2)=322, total=356 tokens ✓
MM_PERCEIVER_LATENTS=81
MM_PERCEIVER_LATENTS_FAST=9
MM_PERCEIVER_DEPTH=2
USE_DOWNSAMPLE_IMAGE=False

# Training configuration
FP16=True
ATTN_IMPLEMENTATION="eager"  # GPT-2 doesn't support sdpa yet
NUM_TRAIN_EPOCHS=80
PER_DEVICE_TRAIN_BATCH_SIZE=4  # Appropriate for single GPU
GRADIENT_ACCUMULATION_STEPS=4  # Effective batch size = 16
LEARNING_RATE=5e-4
MM_PROJECTOR_LR=5e-4
SAVE_STRATEGY="steps"
SAVE_STEPS=230  # Save every 230 steps (~30 checkpoints for 6960 total steps = 80 epochs)
SAVE_TOTAL_LIMIT=10
WEIGHT_DECAY=0.
WARMUP_RATIO=0.1
LR_SCHEDULER_TYPE="cosine"
LOGGING_STEPS=2
MODEL_MAX_LENGTH=2048  # Shorter for Italian (avg 16 words/caption)
GRADIENT_CHECKPOINTING=True
DATALOADER_NUM_WORKERS=2  # Match working train_ln_t.sh; llava-st handles decord properly with workers
LAZY_PREPROCESS=True
REPORT_TO="wandb"
IMAGE_ASPECT_RATIO="pad"

# Video processing parameters
VIDEO_FOLDER=""  # JSON contains full paths from project root
VIDEO_FPS=1  # Extract 1 frame per second
FRAMES_UPBOUND=4  # Max frames to extract - GPT-2 has 2048 token limit (256 tokens/frame * 4 = ~1024 for video)
NUM_FRAMES=4  # Actual frames used (subsampled from extracted)
FORCE_SAMPLE=False
ADD_TIME_INSTRUCTION=False

# DeepSpeed configuration
DEEPSPEED_SCRIPT="scripts/zero2.json"

# Data and output paths
BASE_DATA_DIR="data/webvid_italian/processed"
DATA_PATH="${BASE_DATA_DIR}/webvid${SUFFIX}.json"
OUTPUT_DIR="output/ckpt_italian_${PROJECT_VERSION}${SUFFIX}"

# Verify data file exists
[ -f "$DATA_PATH" ] || {
    echo "ERROR: data path $DATA_PATH not found"
    echo "Run: python scripts_italian/generate_webvid_for_llava_italian.py"
    exit 1
}

# ============================================================
# Environment Variables
# ============================================================
export OMP_NUM_THREADS=8
export WANDB_PROJECT="baby_llama_italian"
export BABYLM_ENABLED=false  # Not using BabyLM word limits for now

# Critical for video loading over NFS - allows decord to retry on network issues
export DECORD_EOF_RETRY_MAX=2048000
export USE_PLAIN_LM=1  # Use plain LM mode (no instruction format)

# Force perceiver latent values (bypass HfArgumentParser issues)
export MM_PERCEIVER_LATENTS=81
export MM_PERCEIVER_LATENTS_FAST=9

echo "================================================"
echo "STARTING ITALIAN TRAINING - EXPERIMENT $EXP_ID"
echo "================================================"
echo "Model:   $MODEL_PATH"
echo "Dataset: $DATA_PATH"
echo "Output:  $OUTPUT_DIR"
echo "Stage:   $STAGE"
if [ -n "$PRETRAIN_MM_MLP_ADAPTER" ] && [ -f "$PRETRAIN_MM_MLP_ADAPTER" ]; then
    echo "Pretrained weights: $PRETRAIN_MM_MLP_ADAPTER"
fi
echo "================================================"

# ============================================================
# Training Command
# ============================================================
python -u llava/train/train_mem.py \
    --model_name_or_path ${MODEL_PATH} \
    --version llama_v2 \
    --data_path "${DATA_PATH}" \
    --image_folder "." \
    --video_folder "${VIDEO_FOLDER}" \
    --vision_tower ${VISION_MODEL} \
    --image_aspect_ratio ${IMAGE_ASPECT_RATIO} \
    --output_dir ${OUTPUT_DIR} \
    --mm_tunable_parts "${MM_TUNABLE_PARTS}" \
    $([ -n "$PRETRAIN_MM_MLP_ADAPTER" ] && [ -f "$PRETRAIN_MM_MLP_ADAPTER" ] && echo "--pretrain_mm_mlp_adapter ${PRETRAIN_MM_MLP_ADAPTER}") \
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
    --video_fps ${VIDEO_FPS} \
    --frames_upbound ${FRAMES_UPBOUND} \
    --num_frames ${NUM_FRAMES} \
    --force_sample ${FORCE_SAMPLE} \
    --add_time_instruction ${ADD_TIME_INSTRUCTION}

echo "Training completed: $(date)"
echo "Checkpoint saved to: ${OUTPUT_DIR}"
