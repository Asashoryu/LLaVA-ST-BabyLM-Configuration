# LLaVA-ST BabyLM Configuration

Local fork focused on training and evaluating lightweight Baby Llama models on Localized Narratives and BabyLM metrics (BLiMP, supplement, EWoK). Includes preprocessing, training, multimodal inference, and linguistic evaluation scripts tailored to this setup.

## Environment
- Python 3.10, CUDA 11.8 recommended. Quick setup:
```bash
conda create -n llava-st python=3.10 -y
conda activate llava-st
pip install --upgrade pip
pip install -e ".[train]"
python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available())"
```

## Base checkpoints
- [create_baby_llama.py](create_baby_llama.py): 12-layer Llama 2 config with `tie_word_embeddings=False`; copies `embed_tokens` into `lm_head`; saves to `baby_llama_baseline` with TinyLlama tokenizer.
- [create_baby_llama_3.py](create_baby_llama_3.py): same pipeline for Llama 3 (vocab 128256, `rope_theta=500000`); saves to `baby_llama3_baseline` with Meta-Llama-3-8B-Instruct tokenizer.
Run the script to materialize the initial weights.

## Localized Narratives data prep
- [preprocess_localized_narratives_json.py](preprocess_localized_narratives_json.py): downloads OpenImages/MSCOCO, segments temporal traces, builds bboxes, optionally saves debug visualizations.
- [generate_ln_for_llava.py](generate_ln_for_llava.py): emits three dataset variants—text only (t), text+image (ti), multimodal interleaved with normalized variables (tim)—as JSON ready for training.

## Training
- [train_ln_t.sh](train_ln_t.sh) launches `llava/train/train_mem.py` with DINOv2-L and fast/slow resampler.
      - `MODEL_PATH` must point to the base checkpoint (e.g., `baby_llama_baseline`).
      - `EXP_ID`: 1 = `_t`, 2 = `_ti`, 3 = `_tim` (also sets `STAGE`).
      - Default `DATA_PATH`: `data/localized_narratives/llava_datasets/all_shards{suffix}_merged.json`.
      - `OUTPUT_DIR`: `output/ckpt_mixed_{version}{suffix}`. Env vars pin BabyLM limits (`BABYLM_WORD_LIMIT`) and tunable parts.

## Multimodal inference and metrics
- Dataset paths live in [inference/config.yaml](inference/config.yaml) (Charades, ST-Align, RefCOCO).
- [inference/multi_task_inference.py](inference/multi_task_inference.py) covers REC, TVG, and STVG/SVG/ELC: loads model (optional LoRA), formats spatial/temporal variables, saves predictions to JSONL.
      - Example: `python inference/multi_task_inference.py --model_path <ckpt> --task rec --dataset refcoco --data_folder <imgs> --ann_path <json> --save_dir eval_outputs --sub_dir run1`
- [inference/multi_task_eval.py](inference/multi_task_eval.py) scans predictions, computes mIoU/recall for REC/TVG and ST-Align metrics, and writes a summary to `total_eval.json`.

## BabyLM linguistic evaluation
- [eval_babylm_challenge.sh](eval_babylm_challenge.sh): loops over checkpoints named `word*` inside `output/ckpt_mixed_*`, running BLiMP, BLiMP supplement, and EWoK via `evaluation_pipeline.sentence_zero_shot` (babylm_eval) with backend `causal`.
- [eval_blimp.sh](eval_blimp.sh) and [eval_blimp_fixed.sh](eval_blimp_fixed.sh): single-checkpoint runs with explicit `--image_template append_image_token`. Set `MODEL_PATH`/`CHECKPOINT_DIR` and `DATA_PATH` before launching.
- [inference_linguistic_1.py](inference_linguistic_1.py): smoke test computing perplexity and simple completions.

## Notes on structure and ignored files
- Raw data, checkpoints, and many local utilities are excluded via [.gitignore](.gitignore); place datasets under `data/` and outputs under `output/` to avoid accidental commits.
- Training and preprocessing scripts use hardcoded paths; confirm them before submitting Slurm jobs.

## Code state
This fork omits the public-facing LLaVA-ST docs. The sections above summarize only the scripts actually used in this local configuration.
