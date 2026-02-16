# GPT-2 Wrapper Implementation for LLaVA

## Overview

Successfully created complete GPT-2 support for LLaVA training pipeline by implementing a wrapper that mirrors the existing Llama implementation. This enables the BAMBI Italian GPT-2 model to be used with LLaVA's multimodal infrastructure.

## Files Created/Modified

### 1. New File: `llava/model/language_model/llava_gpt2.py`
Complete GPT-2 wrapper with three main classes:

**LlavaGPT2Config**
- Inherits from `GPT2Config`
- Sets `model_type = "llava_gpt2"`
- Adds generation defaults (temperature, max_new_tokens, etc.)

**LlavaGPT2Model**
- Inherits from both `LlavaMetaModel` and `GPT2Model`
- Provides compatibility properties:
  - `@property model`: Returns `self` (mixin code expects `self.model`)
  - `@property embed_tokens`: Returns `self.wte` (GPT-2's word token embeddings)

**LlavaGPT2ForCausalLM**
- Inherits from both `GPT2LMHeadModel` and `LlavaMetaForCausalLM`
- Main interface for training and inference
- Implements:
  - `get_model()`: Returns `self.transformer` (GPT-2's base model)
  - `forward()`: Handles multimodal inputs (images/videos + text)
  - `generate()`: Inference with multimodal inputs
  - `prepare_inputs_for_generation()`: Prepares inputs for generation
- Registers with HuggingFace `AutoConfig` and `AutoModelForCausalLM`

### 2. Modified: `llava/model/__init__.py`
Added GPT-2 to available models:
```python
"llava_gpt2": "LlavaGPT2ForCausalLM, LlavaGPT2Config"
```

### 3. Modified: `llava/train/train.py`
Added GPT-2/BAMBI detection logic (after line 1715):
```python
elif "gpt2" in model_args.model_name_or_path.lower() or "bambi" in model_args.model_name_or_path.lower():
    model = LlavaGPT2ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        low_cpu_mem_usage=False,
        **customized_kwargs,
    )
```

## Architecture Compatibility

### GPT-2 vs Llama Naming Differences

| Component            | Llama                     | GPT-2              | Solution                 |
| -------------------- | ------------------------- | ------------------ | ------------------------ |
| Base model attribute | `self.model`              | `self.transformer` | `get_model()` method     |
| Embedding layer      | `embed_tokens`            | `wte`              | `@property embed_tokens` |
| Hidden size          | `hidden_size`             | `n_embd`           | Both work (768)          |
| Num layers           | `num_hidden_layers`       | `n_layer`          | Both work (12)           |
| Max positions        | `max_position_embeddings` | `n_positions`      | Both work                |

### Compatibility Strategy

The wrapper uses **property decorators** to provide transparent attribute name translation:

1. **`@property model`** in `LlavaGPT2Model`: Returns `self` so mixin code accessing `self.model.has_init_specific_embeddings` works correctly
2. **`@property embed_tokens`** in `LlavaGPT2Model`: Returns `self.wte` so mixin code accessing `self.get_model().embed_tokens.weight` works correctly
3. **`get_model()` method** in `LlavaGPT2ForCausalLM`: Returns `self.transformer` instead of `self.model`

This approach preserves all multimodal functionality from `LlavaMetaModel` and `LlavaMetaForCausalLM` mixins without modification.

## Integration Testing

Created `scripts_italian/test_gpt2_wrapper.py` with comprehensive tests:

✅ **Direct imports**: All classes import successfully
✅ **Config instantiation**: LlavaGPT2Config creates properly
✅ **Model instantiation**: LlavaGPT2ForCausalLM initializes correctly
✅ **Attribute access**: All compatibility properties work
  - `get_model()` returns transformer
  - `model` property works
  - `embed_tokens` property maps to `wte`
✅ **AutoConfig registration**: HuggingFace recognizes `llava_gpt2` model type

### Test Results
```
======================================================================
Test Summary:
======================================================================
✓ PASS: Imports
✓ PASS: Config
✓ PASS: Model Class
✓ PASS: AutoConfig Registration
======================================================================
✓ All tests passed! GPT-2 wrapper is ready.
```

## Supported Models

The LLaVA training pipeline now supports **7 architectures**:

1. ✅ Llama (llama_llama.py)
2. ✅ Mistral (llava_mistral.py)
3. ✅ Mixtral (llava_mixtral.py)
4. ✅ Qwen (llava_qwen.py)
5. ✅ Gemma (llava_gemma.py)
6. ✅ MPT (llava_mpt.py)
7. ✅ **GPT-2 (llava_gpt2.py)** ← NEW

## Next Steps

1. ✅ GPT-2 wrapper implementation complete
2. ✅ Integration tests passing
3. ⏳ **Start training** with `sbatch train_italian_video.sh`
4. ⏳ Monitor initialization logs
5. ⏳ Verify forward pass and gradient flow
6. ⏳ Check checkpoint saving

## Training Command

```bash
cd /data1/ososovskyy/LLaVA-ST-BabyLM-Configuration
sbatch train_italian_video.sh
```

This will:
- Load BAMBI GPT-2 (babylm_italian/bambi_18_24_sept2025/)
- Load DINOv2-large vision encoder
- Initialize TokenPacker resampler + MLP projector
- Train on 1,398 Italian video-text pairs (WebVid Italian filtered)
- Save checkpoints to output/ckpt_italian/

## Technical Details

### Model Architecture
```
BAMBI GPT-2 (Decoder-only LM)
├── 12 transformer layers
├── 768 hidden size
├── 30,000 Italian vocabulary
└── 58M parameters

+ DINOv2-large (Vision)
  ├── Frozen ViT-L/14
  └── 1024 feature dimension

+ TokenPacker (Resampler)
  ├── Fast track: 64 latents
  ├── Slow track: 32 latents
  └── Output: 96 total tokens per video

+ MLP Projector
  ├── Input: 1024 (DINOv2)
  ├── Hidden: 2560
  └── Output: 768 (GPT-2 hidden)
```

### Multimodal Forward Pass
1. Video frames (8 frames/video) → DINOv2 → patch features
2. TokenPacker resamples patches → 96 latent tokens
3. MLP projects 1024→768 dimensions
4. Spatial/temporal embeddings injected
5. Combined with text token embeddings
6. GPT-2 processes multimodal sequence
7. Language modeling head predicts next tokens

## References

- **Base implementation**: `llava/model/language_model/llava_llama.py`
- **Mixin classes**: `llava/model/llava_arch.py`
- **Training script**: `llava/train/train.py` (lines 1650-1730)
- **Test script**: `scripts_italian/test_gpt2_wrapper.py`

## Debugging Notes

If training fails, check:
1. Model path detection: Should match "gpt2" or "bambi" in path
2. Attribute access: All `self.model` and `self.embed_tokens` should work
3. Config loading: BAMBI config should have correct `vocab_size=30000`
4. Multimodal projector: MLP should initialize with correct dimensions (1024→768)
5. Dataset loading: Video paths should exist and be valid MP4 files
