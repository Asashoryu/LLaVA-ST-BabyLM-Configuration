#!/bin/bash
#SBATCH --job-name=eval_batch_%A_%a
#SBATCH --output=logs/eval_batch_%A_%a.out
#SBATCH --error=logs/eval_batch_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpusH100
#SBATCH --gres=gpu:1
#SBATCH --array=0-18  # 19 checkpoints (1M-10M by 1M, 10M-100M by 10M)

# ======================================================
# 🔧 BATCH EVALUATION - Entity Tracking + WUG Tasks
# ======================================================
# Configure which model variant to evaluate
MODEL_SUFFIX="tim"  # Options: t, ti, tim

# Define checkpoint milestones (in millions)
CHECKPOINTS=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100)

# Get checkpoint for this array task
WORDS_M=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}

BATCH_SIZE=64
BACKEND="causal"

# ======================================================
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llava-st
mkdir -p logs

echo "============================================"
echo "🧪 BATCH EVALUATION - Additional Metrics"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Model: ckpt_mixed_23_${MODEL_SUFFIX}"
echo "Checkpoint: ${WORDS_M}Mwords"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================"

# Build model directory path  
OUTPUT_ROOT="/data1/ososovskyy/LLaVA-ST-BabyLM-Configuration/output"
MODEL_DIR="$OUTPUT_ROOT/ckpt_mixed_23_${MODEL_SUFFIX}"

# Normalize WORDS_M
WORDS_NUM="${WORDS_M%[Mm]}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: Model directory $MODEL_DIR does not exist."
  exit 1
fi

# Look for checkpoint matching the pattern
MATCH=$(ls -d "$MODEL_DIR"/checkpoint-*"-${WORDS_NUM}"Mwords 2>/dev/null | sort | tail -n1 || true)
if [ -z "$MATCH" ]; then
  echo "WARNING: No checkpoint matching '*-${WORDS_NUM}Mwords' found in $MODEL_DIR"
  echo "Skipping this checkpoint."
  exit 0
fi

CHECKPOINT_DIR="$MATCH"
CKPT_NAME=$(basename "$CHECKPOINT_DIR")

echo "Checkpoint: $CHECKPOINT_DIR"

# Create results directory
RESULTS_DIR="$MODEL_DIR/results"
CHECKPOINT_OUTPUT_DIR="$RESULTS_DIR/$CKPT_NAME"
mkdir -p "$CHECKPOINT_OUTPUT_DIR"

# Export PYTHONPATH
export PYTHONPATH=/data1/ososovskyy/babylm_eval:${PYTHONPATH:-}

# Match training mode (Gen 23+ uses plain LM, no instruction wrapper)
export USE_PLAIN_LM=1

# ======================================================
# 1. Entity Tracking
# ======================================================
echo ""
echo "=== [1/3] Evaluating Entity Tracking ==="
python -m evaluation_pipeline.sentence_zero_shot.run \
  --model_path_or_name $CHECKPOINT_DIR \
  --backend $BACKEND \
  --task entity_tracking \
  --data_path "/data1/ososovskyy/babylm_eval/evaluation_data/fast_eval/entity_tracking_fast" \
  --batch_size $BATCH_SIZE \
  --output_dir $CHECKPOINT_OUTPUT_DIR

ENTITY_EXIT=$?
echo "Entity Tracking exit code: $ENTITY_EXIT"

# ======================================================
# 2. WUG Adjective
# ======================================================
echo ""
echo "=== [2/3] Evaluating WUG_ADJ ==="
python -m evaluation_pipeline.sentence_zero_shot.run \
  --model_path_or_name $CHECKPOINT_DIR \
  --backend $BACKEND \
  --task wug_adj \
  --data_path "/data1/ososovskyy/babylm_eval/evaluation_data/fast_eval/wug_adj_nominalization" \
  --batch_size $BATCH_SIZE \
  --output_dir $CHECKPOINT_OUTPUT_DIR

WUG_ADJ_EXIT=$?
echo "WUG_ADJ exit code: $WUG_ADJ_EXIT"

# ======================================================
# 3. WUG Past Tense
# ======================================================
echo ""
echo "=== [3/3] Evaluating WUG_PAST ==="
python -m evaluation_pipeline.sentence_zero_shot.run \
  --model_path_or_name $CHECKPOINT_DIR \
  --backend $BACKEND \
  --task wug_past \
  --data_path "/data1/ososovskyy/babylm_eval/evaluation_data/fast_eval/wug_past_tense" \
  --batch_size $BATCH_SIZE \
  --output_dir $CHECKPOINT_OUTPUT_DIR

WUG_PAST_EXIT=$?
echo "WUG_PAST exit code: $WUG_PAST_EXIT"

echo "============================================"
echo "✅ CHECKPOINT ${WORDS_M}M COMPLETE"
echo "End: $(date)"
echo "Exit codes: Entity=$ENTITY_EXIT, WUG_ADJ=$WUG_ADJ_EXIT, WUG_PAST=$WUG_PAST_EXIT"
echo "============================================"

# Exit with error if any evaluation failed
if [ $ENTITY_EXIT -ne 0 ] || [ $WUG_ADJ_EXIT -ne 0 ] || [ $WUG_PAST_EXIT -ne 0 ]; then
  exit 1
fi

exit 0
