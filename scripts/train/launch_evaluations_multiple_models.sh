#!/bin/bash

# ======================================================
# 🚀 LAUNCH EVALUATIONS FOR MULTIPLE MODELS
# ======================================================
# Edit this array to specify which model versions to evaluate
MODELS=(5 6 7)  # Change this to the models you want

echo "============================================"
echo "🚀 Launching evaluations for models: ${MODELS[@]}"
echo "============================================"

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "=== Submitting jobs for Model $MODEL ==="
  
  # Check if model directory exists
  MODEL_DIR="output/ckpt_mixed_${MODEL}_tim"
  if [ ! -d "$MODEL_DIR" ]; then
    echo "⚠️  Warning: $MODEL_DIR not found, skipping model $MODEL"
    continue
  fi
  
  # Count available checkpoints
  NUM_CKPTS=$(ls -d $MODEL_DIR/checkpoint-*Mwords 2>/dev/null | wc -l)
  echo "Found $NUM_CKPTS checkpoints for model $MODEL"
  
  # Create temporary script with MODEL_VERSION set
  TEMP_SCRIPT="eval_batch_model_${MODEL}.sh"
  sed "s/MODEL_VERSION=\"23\"/MODEL_VERSION=\"$MODEL\"/" eval_all_checkpoints_batch.sh > $TEMP_SCRIPT
  
  # Submit job array
  JOB_ID=$(sbatch $TEMP_SCRIPT | awk '{print $4}')
  echo "✅ Submitted job array: $JOB_ID for model $MODEL"
  
  # Clean up temp script
  rm $TEMP_SCRIPT
  
  # Small delay between submissions
  sleep 2
done

echo ""
echo "============================================"
echo "✅ All jobs submitted!"
echo "Monitor with: watch -n 5 'squeue -u \$USER'"
echo "============================================"
