#!/bin/bash

# ======================================================
# 🚀 LAUNCH EVALUATIONS FOR ALL 3 MODEL VARIANTS
# ======================================================
# Model 23 variants: t, ti, tim

MODEL_SUFFIXES=(t ti tim)

echo "============================================"
echo "🚀 Launching evaluations for all 3 model variants"
echo "   ckpt_mixed_23_t"
echo "   ckpt_mixed_23_ti"
echo "   ckpt_mixed_23_tim"
echo "============================================"

for SUFFIX in "${MODEL_SUFFIXES[@]}"; do
  echo ""
  echo "=== Submitting jobs for ckpt_mixed_23_${SUFFIX} ==="
  
  # Check if model directory exists
  MODEL_DIR="output/ckpt_mixed_23_${SUFFIX}"
  if [ ! -d "$MODEL_DIR" ]; then
    echo "⚠️  Warning: $MODEL_DIR not found, skipping"
    continue
  fi
  
  # Count available checkpoints
  NUM_CKPTS=$(ls -d $MODEL_DIR/checkpoint-*Mwords 2>/dev/null | wc -l)
  echo "Found $NUM_CKPTS checkpoints"
  
  # Create temporary script with MODEL_SUFFIX set
  TEMP_SCRIPT="eval_batch_23_${SUFFIX}.sh"
  sed "s/MODEL_SUFFIX=\"tim\"/MODEL_SUFFIX=\"$SUFFIX\"/" eval_all_checkpoints_batch.sh > $TEMP_SCRIPT
  
  # Submit job array
  JOB_ID=$(sbatch $TEMP_SCRIPT | awk '{print $4}')
  echo "✅ Submitted job array: $JOB_ID for ckpt_mixed_23_${SUFFIX}"
  
  # Clean up temp script
  rm $TEMP_SCRIPT
  
  # Small delay between submissions
  sleep 2
done

echo ""
echo "============================================"
echo "✅ All 3 model variants submitted!"
echo "This will evaluate Entity Tracking + WUG_ADJ + WUG_PAST"
echo "Monitor with: squeue -u \$USER"
echo "============================================"
