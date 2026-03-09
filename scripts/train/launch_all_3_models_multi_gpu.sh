#!/bin/bash

echo "============================================"
echo "🚀 Launching evaluations for all 3 model variants"
echo "   Using gpusALL partition (H100 + L40 GPUs)"
echo "   ckpt_mixed_23_t"
echo "   ckpt_mixed_23_ti"
echo "   ckpt_mixed_23_tim"
echo "============================================"
echo ""

MODEL_SUFFIXES=(t ti tim)

for suffix in "${MODEL_SUFFIXES[@]}"; do
  echo "=== Submitting jobs for ckpt_mixed_23_${suffix} ==="
  
  # Create temporary script with the correct MODEL_SUFFIX
  TEMP_SCRIPT="eval_batch_${suffix}_temp.sh"
  sed "s/MODEL_SUFFIX=\"tim\"/MODEL_SUFFIX=\"${suffix}\"/" eval_all_checkpoints_multi_gpu.sh > $TEMP_SCRIPT
  chmod +x $TEMP_SCRIPT
  
  # Count checkpoints
  CKPT_COUNT=$(ls output/ckpt_mixed_23_${suffix}/ | grep -c "checkpoint-.*Mwords")
  echo "Found $CKPT_COUNT checkpoints"
  
  # Submit job array
  JOB_ID=$(sbatch $TEMP_SCRIPT | awk '{print $4}')
  echo "✅ Submitted job array: $JOB_ID for ckpt_mixed_23_${suffix}"
  echo ""
  
  # Cleanup temp script
  rm $TEMP_SCRIPT
done

echo "============================================"
echo "✅ All 3 model variants submitted!"
echo "This will evaluate Entity Tracking + WUG_ADJ + WUG_PAST"
echo "Using gpusALL partition for better parallelization"
echo "Monitor with: squeue -u \$USER"
echo "============================================"
