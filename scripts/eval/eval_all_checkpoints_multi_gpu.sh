#!/bin/bash
#SBATCH --job-name=eval_batch
#SBATCH --array=0-18
#SBATCH --partition=gpusALL
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/eval_batch_%A_%a.out

# Actual checkpoint numbers and steps (extracted from directory listing)
CKPT_NUMS=(142 280 420 560 700 840 980 1120 1258 1398 2796 4240 5692 7144 8596 10048 11494 12938 14378)
CKPT_STEPS=(1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100)

MODEL_SUFFIX="tim"  # Will be replaced by launch script: t, ti, or tim

CHECKPOINT_IDX=${SLURM_ARRAY_TASK_ID}
CKPT_NUM=${CKPT_NUMS[$CHECKPOINT_IDX]}
CHECKPOINT_STEP=${CKPT_STEPS[$CHECKPOINT_IDX]}

MODEL_DIR="output/ckpt_mixed_23_${MODEL_SUFFIX}/checkpoint-${CKPT_NUM}-${CHECKPOINT_STEP}Mwords"

echo "============================================"
echo "Evaluating checkpoint: $CHECKPOINT_STEP M words"
echo "Model: $MODEL_DIR"
echo "GPU: $SLURM_JOB_PARTITION on $(hostname)"
echo "Start: $(date)"
echo "============================================"

source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llava-st

export PYTHONPATH=/data1/ososovskyy/babylm_eval:${PYTHONPATH:-}
export USE_PLAIN_LM=1

# Task 1: Entity Tracking
echo ""
echo "=== Evaluating Entity Tracking ==="
python -m evaluation_pipeline.sentence_zero_shot.run \
  --model_path_or_name $MODEL_DIR \
  --backend causal \
  --task entity_tracking \
  --data_path /data1/ososovskyy/babylm_eval/evaluation_data/fast_eval/entity_tracking_fast \
  --batch_size 64 \
  --output_dir output/results_entity_wug
ENTITY_EXIT=$?
echo "ENTITY_TRACKING exit code: $ENTITY_EXIT"

# Task 2: WUG Adjective
echo ""
echo "=== Evaluating WUG Adjective ==="
python -m evaluation_pipeline.sentence_zero_shot.run \
  --model_path_or_name $MODEL_DIR \
  --backend causal \
  --task wug_adj \
  --data_path /data1/ososovskyy/babylm_eval/evaluation_data/fast_eval/wug_adj_nominalization \
  --batch_size 64 \
  --output_dir output/results_entity_wug
WUG_ADJ_EXIT=$?
echo "WUG_ADJ exit code: $WUG_ADJ_EXIT"

# Task 3: WUG Past Tense
echo ""
echo "=== Evaluating WUG Past Tense ==="
python -m evaluation_pipeline.sentence_zero_shot.run \
  --model_path_or_name $MODEL_DIR \
  --backend causal \
  --task wug_past \
  --data_path /data1/ososovskyy/babylm_eval/evaluation_data/fast_eval/wug_past_tense \
  --batch_size 64 \
  --output_dir output/results_entity_wug
WUG_PAST_EXIT=$?
echo "WUG_PAST exit code: $WUG_PAST_EXIT"

echo ""
echo "============================================"
echo "✅ CHECKPOINT ${CHECKPOINT_STEP}M COMPLETE"
echo "End: $(date)"
echo "Exit codes: Entity=$ENTITY_EXIT, WUG_ADJ=$WUG_ADJ_EXIT, WUG_PAST=$WUG_PAST_EXIT"
echo "============================================"
