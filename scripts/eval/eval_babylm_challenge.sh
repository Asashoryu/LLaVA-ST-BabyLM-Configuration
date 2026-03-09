#!/bin/bash
#SBATCH --job-name=eval_babylm_loop
#SBATCH --output=logs/eval_loop_%j.out
#SBATCH --error=logs/eval_loop_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpusALL

CHECKPOINT_ROOT="/data1/ososovskyy/LLaVA-ST-BabyLM-Configuration/output/ckpt_mixed_8_tim"

EVAL_DATA_DIR="/data1/ososovskyy/babylm_eval/evaluation_data/fast_eval"

EVAL_CODE_ROOT="/data1/ososovskyy/babylm_eval"

BACKEND="causal"

source ~/.bashrc
# Initialize conda properly for SLURM
eval "$(conda shell.bash hook)"
conda activate llava-st

export PYTHONPATH=$PYTHONPATH:$EVAL_CODE_ROOT

# CHECK: Verify Python and transformers are available
echo "Python: $(which python)"
python -c "import transformers; print('transformers OK')" || { echo "ERROR: transformers not found"; exit 1; }

if [ ! -d "$EVAL_DATA_DIR" ]; then
    echo "ERROR: The data directory '$EVAL_DATA_DIR' does not exist."
    exit 1
fi

# Create results directory
RESULTS_DIR="${CHECKPOINT_ROOT}/results"
mkdir -p "$RESULTS_DIR"

echo "========================================================"
echo "JOB ID: $SLURM_JOB_ID"
echo "Eval Code Root:    $EVAL_CODE_ROOT"
echo "PYTHONPATH:          $PYTHONPATH"
echo "========================================================"



find "$CHECKPOINT_ROOT" -maxdepth 1 -type d -name "*word*" | sort -V | while read ckpt_path; do

    ckpt_name=$(basename "$ckpt_path")
    echo "--------------------------------------------------------"
    echo ">>>> BEGIN CHECKPOINT EVALUATION: $ckpt_name <<<<"

    # 1. BLiMP
    echo "   Evaluating on BLiMP FAST..."
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$ckpt_path" \
        --backend "$BACKEND" \
        --task blimp \
        --data_path "${EVAL_DATA_DIR}/blimp_fast" \
        --output_dir "$RESULTS_DIR" \
        --image_template append_image_token \
        --save_predictions \
        --batch_size 64

    # 2. BLiMP Supplement
    echo "   Evaluating on BLiMP Supplement FAST..."
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$ckpt_path" \
        --backend "$BACKEND" \
        --task blimp \
        --data_path "${EVAL_DATA_DIR}/supplement_fast" \
        --output_dir "$RESULTS_DIR" \
        --image_template append_image_token \
        --save_predictions \
        --batch_size 64

    # 3. EWoK
    echo "   Evaluating on EWoK FAST..."
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$ckpt_path" \
        --backend "$BACKEND" \
        --task ewok \
        --data_path "${EVAL_DATA_DIR}/ewok_fast" \
        --output_dir "$RESULTS_DIR" \
        --image_template append_image_token \
        --save_predictions \
        --batch_size 64

    echo ">>>> CHECKPOINT COMPLETED: $ckpt_name"
    echo "--------------------------------------------------------"

done
