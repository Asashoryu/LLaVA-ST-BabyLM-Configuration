#!/bin/bash
#SBATCH --job-name=eval_flamingo_baseline
#SBATCH --output=logs/eval_flamingo_baseline_%j.out
#SBATCH --error=logs/eval_flamingo_baseline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpusH100
#SBATCH --gres=gpu:1

# ======================================================
# 🔧 BABYLM BASELINE EVALUATION - FLAMINGO MODEL
# ======================================================
# Evaluate BabyLM Challenge Flamingo baseline on same tasks as Gen 23 models
# For fair comparison with ckpt_mixed_23_{t,ti,tim}

MODEL_PATH="/data1/ososovskyy/babylm_baselines/flamingo"
TASK="blimp"
DATA_PATH="/data1/ososovskyy/babylm_eval/evaluation_data/fast_eval/blimp_fast"
BATCH_SIZE=32  # Adjust if needed
BACKEND="causal"

# ======================================================
# Activate environment
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llava-st
mkdir -p logs

echo "============================================"
echo "🧪 BABYLM BASELINE EVALUATION - FLAMINGO"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "Model: $MODEL_PATH"
echo "Task: $TASK"
echo "Batch Size: $BATCH_SIZE"
echo "============================================"

# Export PYTHONPATH
export PYTHONPATH=/data1/ososovskyy/babylm_eval:${PYTHONPATH:-}

# Flamingo is a pure model (not instruction-tuned), so use plain LM mode
export USE_PLAIN_LM=1

# Create results directory
OUTPUT_ROOT="/data1/ososovskyy/babylm_baselines"
RESULTS_DIR="$OUTPUT_ROOT/flamingo_results"
mkdir -p "$RESULTS_DIR"

# Run evaluation
echo "🚀 Starting Flamingo baseline evaluation on $TASK..."
python -m evaluation_pipeline.sentence_zero_shot.run \
  --model_path_or_name $MODEL_PATH \
  --backend $BACKEND \
  --task $TASK \
  --data_path $DATA_PATH \
  --batch_size $BATCH_SIZE \
  --output_dir $RESULTS_DIR

EXIT_CODE=$?

echo "============================================"
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
#!/bin/bash
#SBATCH --job-name=eval_git_baseline
#SBATCH --output=logs/eval_git_baseline_%j.out
#SBATCH --error=logs/eval_git_baseline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpusH100
#SBATCH --gres=gpu:1

# ======================================================
# 🔧 BABYLM BASELINE EVALUATION - GIT MODEL
# ======================================================
# Evaluate BabyLM Challenge GIT baseline on same tasks as Gen 23 models
# For fair comparison with ckpt_mixed_23_{t,ti,tim}

MODEL_PATH="/data1/ososovskyy/babylm_baselines/git"
TASK="blimp"
DATA_PATH="/data1/ososovskyy/babylm_eval/evaluation_data/fast_eval/blimp_fast"
BATCH_SIZE=32  # GIT is smaller, can use larger batch
BACKEND="causal"

# ======================================================
# Activate environment
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llava-st
mkdir -p logs

echo "============================================"
echo "🧪 BABYLM BASELINE EVALUATION - GIT"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "Model: $MODEL_PATH"
echo "Task: $TASK"
echo "Batch Size: $BATCH_SIZE"
echo "============================================"

# Export PYTHONPATH
export PYTHONPATH=/data1/ososovskyy/babylm_eval:${PYTHONPATH:-}

# GIT is a pure model (not instruction-tuned), so use plain LM mode
export USE_PLAIN_LM=1

# Create results directory
OUTPUT_ROOT="/data1/ososovskyy/babylm_baselines"
RESULTS_DIR="$OUTPUT_ROOT/git_results"
mkdir -p "$RESULTS_DIR"

# Run evaluation
echo "🚀 Starting GIT baseline evaluation on $TASK..."
python -m evaluation_pipeline.sentence_zero_shot.run \
  --model_path_or_name $MODEL_PATH \
  --backend $BACKEND \
  --task $TASK \
  --data_path $DATA_PATH \
  --batch_size $BATCH_SIZE \
  --output_dir $RESULTS_DIR

EXIT_CODE=$?

echo "============================================"
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
