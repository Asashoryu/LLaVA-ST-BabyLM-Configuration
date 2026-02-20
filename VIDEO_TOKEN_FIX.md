# Video Training Issue - Root Cause & Fix

## Problem Identified

The video training was not learning (loss stuck at 10.2, grad_norm ~0.1) while text-only training converged normally (loss 0.2, grad_norm ~6). **Root cause: In plain LM mode, the model was trying to predict video feature tokens as if they were language tokens.**

## Technical Analysis

### What Was Happening

1. **Plain LM Mode** (`USE_PLAIN_LM=1`):
   - Used to train GPT-2 style models without instruction formatting
   - Original implementation: `labels = copy.deepcopy(input_ids)` (no masking)
   - ALL tokens included in loss computation

2. **Video Token Expansion**:
   - `<video>` → `<vid_start>` + 34 × `<vid_patch>` + `<vid_end>` + `<slow_vid_start>` + 322 × `<vid_patch>` + `<slow_vid_end>`
   - Total: **356 video patch tokens** + 2 start/end tokens per path = **360 video-related tokens**

3. **Loss Computation Problem**:
   - Model was asked to predict: `[vid_start, vid_patch, vid_patch, ..., italian_text_tokens]`
   - Video tokens are **placeholders** that get replaced with DINOv2 features in `prepare_inputs_labels_for_multimodal_video()`
   - The model cannot learn to predict DINOv2 feature representations as language tokens!
   - Result: Loss dominated by impossible video token prediction task (~360 tokens) vs actual caption (~10-20 tokens)

### Why Text-Only Worked

- Text-only samples: labels contained ONLY text tokens (no video tokens)
- Model could learn normally: predict Italian text → compute loss → update weights
- Gradients flowed properly because the task was learnable

### Why Video Failed

- Video samples: labels contained ~360 video tokens + ~15 text tokens
- Model tried to learn two tasks:
  1. **Impossible**: Predict DINOv2 features as language tokens (360 tokens, random loss ~10)
  2. **Possible**: Predict Italian caption from video context (15 tokens, learnable)
- Impossible task dominated the loss → gradients mostly reflected failure on video tokens
- Learnable caption task drowned out → minimal learning signal
- Result: grad_norm 70x weaker, loss stuck at ~10

## The Fix

Modified `preprocess_plain_lm()` in [llava/train/train.py](llava/train/train.py) (lines 961-1025):

```python
# CRITICAL FIX: Mask image/video patch tokens so model only predicts text
# Get token IDs for special tokens
im_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
im_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
im_patch_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PATCH_TOKEN)
vid_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_VID_START_TOKEN)
vid_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_VID_END_TOKEN)
vid_patch_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_VIDEO_PATCH_TOKEN)
slow_vid_start_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SLOW_VID_START_TOKEN)
slow_vid_end_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SLOW_VID_END_TOKEN)

# Mask all image/video tokens in labels with IGNORE_INDEX=-100
for target in targets:
    if isinstance(target, torch.Tensor):
        target[(target == im_start_token_id) | (target == im_end_token_id) |
               (target == im_patch_token_id)] = IGNORE_INDEX
        target[(target == vid_start_token_id) | (target == vid_end_token_id) |
               (target == vid_patch_token_id)] = IGNORE_INDEX
        target[(target == slow_vid_start_token_id) | (target == slow_vid_end_token_id)] = IGNORE_INDEX
```

### What This Does

- **Before**: labels = `[30000, 30001, 30001, ..., 1234, 5678, ...]` (video tokens + text tokens)
- **After**: labels = `[-100, -100, -100, ..., 1234, 5678, ...]` (video tokens masked, only text predicted)

PyTorch's CrossEntropyLoss ignores positions with `IGNORE_INDEX=-100` → no loss computed for video tokens.

## Expected Results

With this fix, the model will:

1. ✅ **Not** try to predict video feature tokens (they're masked)
2. ✅ **Only** predict the Italian caption text based on video context
3. ✅ Have a learnable objective (similar to text-only training)
4. ✅ Receive strong gradients from the caption prediction task
5. ✅ Learn video→text association properly

### Comparison

| Aspect               | Before Fix                      | After Fix                 |
| -------------------- | ------------------------------- | ------------------------- |
| **Predicted tokens** | 360 video + 15 text = 375 total | 15 text only              |
| **Loss focus**       | 96% on impossible video tokens  | 100% on learnable caption |
| **Gradient signal**  | Diluted 25:1 (video:text)       | Focused on caption task   |
| **Expected loss**    | ~10 (dominated by video)        | ~0.2-2 (caption only)     |
| **Learning rate**    | Effectively 25x weaker          | Full strength             |

## Next Steps

1. **Restart video training** with the fixed code:
   ```bash
   sbatch train_italian_video.sh  # With EXP_ID=2
   ```

2. **Monitor loss**: Should see loss decrease from epoch 1 (not stuck at 10.2)

3. **Compare with text-only**: Video training should now converge at similar rate

4. **Check grad_norm**: Should be comparable to text-only (~2-6 range)

## Why This Wasn't Caught Earlier

- Original LLaVA-ST was designed for instruction-following formats (Llama, Qwen, Gemma)
- Those formats use different preprocessing functions (e.g., `preprocess_llama3()`)
- They have explicit logic to mask input prompts (including images/videos)
- Plain LM mode was added for GPT-2 compatibility but didn't implement the masking
- The issue only manifested when:
  1. Using `USE_PLAIN_LM=1` (required for GPT-2)
  2. Training with videos (not just images)
  3. Training from random initialization (pretrained models might mask internally)

## Files Modified

- `llava/train/train.py` (lines 961-1025): Added video token masking in `preprocess_plain_lm()`

## Verification

To verify the fix is working, check the training logs for:
- Video samples should have ~15-20 non-IGNORE_INDEX labels (just the caption)
- Loss should start decreasing from epoch 1
- No "Sample has ALL labels masked" warnings for video samples
