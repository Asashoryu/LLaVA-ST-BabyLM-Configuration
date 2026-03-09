#!/bin/bash
#SBATCH --job-name=eval_entity_track
#SBATCH --output=logs/eval_entity_track_%j.out
#SBATCH --error=logs/eval_entity_track_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --partition=gpusH100
#SBATCH --gres=gpu:1

# ======================================================
# 🔧 ENTITY TRACKING EVALUATION
# ======================================================
# Specify model by version and checkpoint by words (in millions).
MODEL_VERSION="23"
WORDS_M="3"

TASK="entity_tracking"
DATA_PATH="/data1/ososovskyy/babylm_eval/evaluation_data/fast_eval/entity_tracking_fast"
BATCH_SIZE=64
BACKEND="causal"

# ======================================================
# Attiva ambiente
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llava-st
mkdir -p logs

echo "============================================"
echo "🧪 ENTITY TRACKING EVALUATION"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "Task: $TASK"
echo "Batch Size: $BATCH_SIZE"
echo "============================================"

# Build model directory path
OUTPUT_ROOT="/data1/ososovskyy/LLaVA-ST-BabyLM-Configuration/output"
MODEL_DIR="$OUTPUT_ROOT/ckpt_mixed_${MODEL_VERSION}_tim"

WORDS_NUM="${WORDS_M%[Mm]}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "Model directory $MODEL_DIR does not exist. Exiting."
  exit 1
fi

# Create results directory
RESULTS_DIR="$MODEL_DIR/results"
mkdir -p "$RESULTS_DIR"

# Look for checkpoint
MATCH=$(ls -d "$MODEL_DIR"/checkpoint-*"-${WORDS_NUM}"Mwords 2>/dev/null | sort | tail -n1 || true)
if [ -z "$MATCH" ]; then
  echo "ERROR: No checkpoint matching '*-${WORDS_NUM}Mwords' found in $MODEL_DIR"
  exit 1
fi

CHECKPOINT_DIR="$MATCH"
CKPT_NAME=$(basename "$CHECKPOINT_DIR")

echo "Checkpoint: $CHECKPOINT_DIR"

# Set output directory
CHECKPOINT_OUTPUT_DIR="$RESULTS_DIR/$CKPT_NAME"
mkdir -p "$CHECKPOINT_OUTPUT_DIR"

# Export PYTHONPATH
export PYTHONPATH=/data1/ososovskyy/babylm_eval:${PYTHONPATH:-}

# Match training mode (plain LM, no instruction wrapper)
export USE_PLAIN_LM=1

# Run evaluation
python -m evaluation_pipeline.sentence_zero_shot.run \
  --model_path_or_name $CHECKPOINT_DIR \
  --backend $BACKEND \
  --task $TASK \
  --data_path $DATA_PATH \
  --batch_size $BATCH_SIZE \
  --output_dir $CHECKPOINT_OUTPUT_DIR

EXIT_CODE=$?

echo "============================================"
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
