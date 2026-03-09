#!/bin/bash

# Evaluate all 4 seed TIM models at 100M checkpoint only
# Usage: bash eval_seeds_100M.sh

BACKEND="causal"
EVAL_DIR="/data1/ososovskyy/babylm_eval/evaluation_data/fast_eval"
WORDS_M="100"

export PYTHONPATH=/data1/ososovskyy/babylm_eval:${PYTHONPATH:-}
export USE_PLAIN_LM=1

source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llava-st

echo "============================================"
echo "🧪 EVALUATING SEED MODELS AT 100M CHECKPOINT"
echo "============================================"
echo "Start: $(date)"
echo "Backend: $BACKEND"
echo "Checkpoint: ${WORDS_M}M words"
echo "============================================"

# Loop through all 4 seeds
for SEED in 1 2 3 4; do
    echo ""
    echo "=========================================="
    echo "🌱 SEED ${SEED}"
    echo "=========================================="
    
    MODEL_DIR="/data1/ososovskyy/LLaVA-ST-BabyLM-Configuration/output/ckpt_mixed_23_seed${SEED}_tim"
    
    if [ ! -d "$MODEL_DIR" ]; then
        echo "❌ Model directory not found: $MODEL_DIR"
        continue
    fi
    
    # Find 100M checkpoint
    CHECKPOINT=$(ls -d "$MODEL_DIR"/checkpoint-*"-${WORDS_M}Mwords" 2>/dev/null | sort | tail -n1 || true)
    
    if [ -z "$CHECKPOINT" ]; then
        echo "❌ No 100M checkpoint found in $MODEL_DIR"
        continue
    fi
    
    CKPT_NAME=$(basename "$CHECKPOINT")
    echo "✓ Found checkpoint: $CKPT_NAME"
    echo "✓ Path: $CHECKPOINT"
    
    # Run all evaluations
    echo ""
    echo "--- Running BLIMP FAST ---"
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$CHECKPOINT" \
        --backend $BACKEND \
        --task blimp \
        --data_path "${EVAL_DIR}/blimp_fast" \
        --save_predictions \
        --revision_name "seed${SEED}_100M"
    
    echo ""
    echo "--- Running SUPPLEMENT FAST ---"
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$CHECKPOINT" \
        --backend $BACKEND \
        --task blimp \
        --data_path "${EVAL_DIR}/supplement_fast" \
        --save_predictions \
        --revision_name "seed${SEED}_100M"
    
    echo ""
    echo "--- Running EWOK FAST ---"
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$CHECKPOINT" \
        --backend $BACKEND \
        --task ewok \
        --data_path "${EVAL_DIR}/ewok_fast" \
        --save_predictions \
        --revision_name "seed${SEED}_100M"
    
    echo ""
    echo "--- Running ENTITY TRACKING FAST ---"
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$CHECKPOINT" \
        --backend $BACKEND \
        --task entity_tracking \
        --data_path "${EVAL_DIR}/entity_tracking_fast" \
        --save_predictions \
        --revision_name "seed${SEED}_100M"
    
    echo ""
    echo "--- Running WUG ADJECTIVE ---"
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$CHECKPOINT" \
        --backend $BACKEND \
        --task wug_adj \
        --data_path "${EVAL_DIR}/wug_adj_nominalization" \
        --save_predictions \
        --revision_name "seed${SEED}_100M"
    
    echo ""
    echo "--- Running WUG PAST TENSE ---"
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$CHECKPOINT" \
        --backend $BACKEND \
        --task wug_past \
        --data_path "${EVAL_DIR}/wug_past_tense" \
        --save_predictions \
        --revision_name "seed${SEED}_100M"
    
    echo ""
    echo "✅ SEED ${SEED} COMPLETE"
    echo "=========================================="
done

echo ""
echo "============================================"
echo "✅ ALL SEED EVALUATIONS COMPLETE"
echo "End: $(date)"
echo "============================================"
echo ""
echo "Next step: Run python3 compile_seed_results.py to collect results"
