#!/bin/bash
#SBATCH --job-name=eval_blimp_fixed
#SBATCH --output=logs/eval_blimp_fixed_%j.out
#SBATCH --error=logs/eval_blimp_fixed_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
# #SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpusH100
#SBATCH --gres=gpu:1

# ======================================================
# 🔧 CONFIGURAZIONE - Con Image Template FIX
# ======================================================
# Specify model by version and checkpoint by words (in millions).
# Edit these two variables directly in the script before running.
MODEL_VERSION="22"
# Example: 90 -> will match checkpoints with "90Mwords" suffix
WORDS_M="100"

TASK="blimp"
DATA_PATH="/data1/ososovskyy/babylm_eval/evaluation_data/fast_eval/blimp_fast"
BATCH_SIZE=64
BACKEND="causal"

# ======================================================
# Attiva ambiente
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llava-st
mkdir -p logs


echo "============================================"
echo "🧪 BABYLM EVALUATION - $TASK (CON IMAGE TEMPLATE FIX)"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "Model: $CHECKPOINT_DIR"
echo "Task: $TASK"
echo "Batch Size: $BATCH_SIZE"
echo "⚡ TEST: REMOVED --image_template to match training distribution (pure text)"
echo "============================================"


# Optional: prefer an explicit fp32 model file if present — do not fail if missing
if [ -f "$CHECKPOINT_DIR/model-fp32.safetensors" ]; then
  echo "Found model-fp32.safetensors — switching model.safetensors to fp32 variant"
  mv -f "$CHECKPOINT_DIR/model.safetensors" "$CHECKPOINT_DIR/model-bak.safetensors" 2>/dev/null || true
  mv -f "$CHECKPOINT_DIR/model-fp32.safetensors" "$CHECKPOINT_DIR/model.safetensors"
else
  echo "No model-fp32.safetensors found in $CHECKPOINT_DIR — using existing model.safetensors"
fi

# Export PYTHONPATH
export PYTHONPATH=/data1/ososovskyy/babylm_eval:$PYTHONPATH

# ⚠️ CRITICAL: Match training mode (Gen 23+ uses plain LM, no instruction wrapper)
export USE_PLAIN_LM=1

# Build model directory path from version and search for an exact checkpoint
OUTPUT_ROOT="/data1/ososovskyy/LLaVA-ST-BabyLM-Configuration/output"
MODEL_DIR="$OUTPUT_ROOT/ckpt_mixed_${MODEL_VERSION}_tim"

# normalize WORDS_M in case someone leaves a trailing 'M' or 'm'
WORDS_NUM="${WORDS_M%[Mm]}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "Model directory $MODEL_DIR does not exist. Exiting."
  exit 1
fi

# Look for checkpoint directories that include the exact "${WORDS_NUM}Mwords" suffix
MATCH=$(ls -d "$MODEL_DIR"/checkpoint-*"-${WORDS_NUM}"Mwords 2>/dev/null | sort | tail -n1 || true)
if [ -z "$MATCH" ]; then
  echo "ERROR: No checkpoint matching '*-${WORDS_NUM}Mwords' found in $MODEL_DIR"
  exit 1
fi

CHECKPOINT_DIR="$MATCH"

# provare srun python
# Run evaluation WITHOUT image_template for pure text tasks (matches training distribution)
python -m evaluation_pipeline.sentence_zero_shot.run \
  --model_path_or_name $CHECKPOINT_DIR \
  --backend $BACKEND \
  --task $TASK \
  --data_path $DATA_PATH \
  --batch_size $BATCH_SIZE

EXIT_CODE=$?

echo "============================================"
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
