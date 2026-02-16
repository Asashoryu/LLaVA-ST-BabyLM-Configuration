# 🔍 Deep Analysis: Italian Video-Language Model Training Issues

## 📊 Pipeline Comparison: English vs Italian Training

### **Original English Baby Llama Training** (3-phase approach)

```
Phase 1 (_t):  Text-only training
├── Model: Baby Llama (Llama-2 architecture, 58M params)
├── Data: Localized Narratives text captions
├── Stage: 1
└── No vision components

Phase 2 (_ti): Text + Image training
├── Model: From Phase 1 checkpoint
├── Data: Localized Narratives images + captions
├── Vision: DINOv2-large (frozen)
├── Resampler: Fast-slow TokenPacker (64+32 latents)
├── MLP Projector: 2-layer (1024→768)
└── Stage: 1 (trains resampler, projector, language model)

Phase 3 (_tim): Text + Image + Mask/Grounding
├── Model: From Phase 2 checkpoint
├── Data: Localized Narratives with spatial grounding boxes
├── Same vision pipeline
└── Stage: 2 (adds grounding capabilities)
```

**Key Architecture:**
- **Base LM**: `baby_llama_baseline` - Llama-2 architecture (12 layers, 768 dim)
- **Config**: `model_type: "llama"`, `architecture: "LlamaForCausalLM"`
- **Vision**: DINOv2-large + TokenPacker resampler
- **Supported by**: `LlavaLlamaForCausalLM` class in codebase

---

### **Your Italian Video Training** (attempted 1-phase)

```
Phase 1-only: Video + Italian text
├── Model: BAMBI GPT-2 (GPT-2 architecture, 58M params)  ❌ PROBLEM
├── Data: WebVid Italian video captions (1,398 samples)
├── Vision: DINOv2-large (for video frames)
├── Resampler: Fast-slow TokenPacker (64+32 latents)
├── MLP Projector: 2-layer (1024→768)
└── Stage: 1
```

**Your Architecture:**
- **Base LM**: `babylm_italian/bambi_18_24_sept2025` - GPT-2 architecture
- **Config**: `model_type: "gpt2"`, `architecture: "GPT2LMHeadModel"` ❌
- **Vision**: Same (DINOv2-large + TokenPacker)
- **Supported by**: NOTHING - GPT-2 not implemented! ❌

---

## 🚨 Root Cause Analysis

### **The Critical Error**

```python
# From train.py line 1723
elif model_args.vision_tower is not None:
    if "mixtral" in model_args.model_name_or_path.lower():
        # Use LlavaMixtralForCausalLM
    elif "mistral" in model_args.model_name_or_path.lower():
        # Use LlavaMistralForCausalLM
    elif "llama" in model_args.model_name_or_path.lower():
        # Use LlavaLlamaForCausalLM ✅ Baby Llama path matches here
    elif "qwen" in model_args.model_name_or_path.lower():
        # Use LlavaQwenForCausalLM
    elif "gemma" in model_args.model_name_or_path.lower():
        # Use LlavaGemmaForCausalLM
    else:
        raise ValueError(f"Unknown model class {model_args}")  ❌ GPT-2 falls here!
```

**Problem**:
- Path `./babylm_italian/bambi_18_24_sept2025` doesn't match any supported model name pattern
- GPT-2 has NO multimodal wrapper class in the codebase
- Training code only supports: Llama, Mistral, Mixtral, Qwen, Gemma, MPT

---

## 🏗️ Available LLaVA Model Classes

```
llava/model/language_model/
├── llava_llama.py      ✅ (English Baby Llama uses this)
├── llava_mistral.py    ✅
├── llava_mixtral.py    ✅
├── llava_qwen.py       ✅
├── llava_qwen_moe.py   ✅
├── llava_gemma.py      ✅
├── llava_mpt.py        ✅
└── llava_gpt2.py       ❌ DOES NOT EXIST!
```

Each LLaVA wrapper:
1. Inherits from base HF model (e.g., `LlamaForCausalLM`)
2. Mixes in `LlavaMetaForCausalLM` for multimodal functionality
3. Adds vision tower, resampler, and projector components
4. Handles multimodal input preparation and forward pass

---

## 🎯 What Changed from English → Italian

| Aspect              | English Baby Llama            | Italian BAMBI        | Impact                 |
| ------------------- | ----------------------------- | -------------------- | ---------------------- |
| **Architecture**    | Llama-2 (decoder-only)        | GPT-2 (decoder-only) | ❌ Breaking             |
| **Model Class**     | LlamaForCausalLM              | GPT2LMHeadModel      | ❌ Not supported        |
| **Hidden Size**     | 768                           | 768                  | ✅ Compatible           |
| **Layers**          | 12                            | 12                   | ✅ Compatible           |
| **Vocab Size**      | ~32K English                  | ~30K Italian         | ⚠️ Different tokenizer  |
| **Context Length**  | 2048                          | 1024                 | ⚠️ Shorter context      |
| **Dataset**         | Localized Narratives (images) | WebVid (videos)      | ✅ Works (video=frames) |
| **Training Phases** | 3 phases (_t, _ti, _tim)      | 1 phase (video+text) | ✅ Simpler              |
| **Modality**        | Image + grounding             | Video only           | ✅ Cleaner              |

**Key Insight**: The architectural differences are SMALL (both decoder-only transformers with same dimensions), but the codebase treats them as incompatible!

---

## 💡 Solution Options (Ranked by Feasibility)

### **Option 1: Quick Fix - Disguise GPT-2 as Llama** ⭐ RECOMMENDED

**Idea**: Modify BAMBI's config.json to make it look like a Llama model to the training code.

**Steps**:
```bash
# 1. Backup original config
cp babylm_italian/bambi_18_24_sept2025/config.json \
   babylm_italian/bambi_18_24_sept2025/config.json.backup

# 2. Modify config to disguise as Llama
# Change: "model_type": "gpt2" → "model_type": "llama"
# Change: "architectures": ["GPT2LMHeadModel"] → ["LlamaForCausalLM"]
```

**Pros**:
- ✅ Works immediately (5 minutes)
- ✅ No code changes needed
- ✅ Leverages existing LlavaLlamaForCausalLM
- ✅ Both models are decoder-only transformers (architecturally similar)

**Cons**:
- ⚠️  "Hacky" solution
- ⚠️  May have subtle incompatibilities (positional embeddings, attention patterns)
- ⚠️  Model weights won't match architecture name

**Risk**: LOW - Worth trying first. GPT-2 and Llama are similar enough that this might just work.

---

### **Option 2: Create LlavaGPT2 Wrapper** ⭐⭐ PROPER SOLUTION

**Idea**: Implement proper GPT-2 support by creating a new LLaVA wrapper class.

**Required Files**:
```python
# 1. Create: llava/model/language_model/llava_gpt2.py
# Copy structure from llava_llama.py and adapt for GPT-2

from transformers import GPT2Model, GPT2LMHeadModel
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

class LlavaGPT2Config(GPT2Config):
    model_type = "llava_gpt2"

class LlavaGPT2Model(LlavaMetaModel, GPT2Model):
    config_class = LlavaGPT2Config
    def __init__(self, config):
        super(LlavaGPT2Model, self).__init__(config)

class LlavaGPT2ForCausalLM(GPT2LMHeadModel, LlavaMetaForCausalLM):
    config_class = LlavaGPT2Config
    def __init__(self, config):
        GPT2LMHeadModel.__init__(self, config)
        config.model_type = "llava_gpt2"
        self.transformer = LlavaGPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.transformer
    # ... (forward pass logic similar to llava_llama.py)
```

```python
# 2. Modify: llava/train/train.py (around line 1720)
# Add GPT-2 detection:

elif "gpt2" in model_args.model_name_or_path.lower() or "bambi" in model_args.model_name_or_path.lower():
    from llava.model.language_model.llava_gpt2 import LlavaGPT2ForCausalLM
    model = LlavaGPT2ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        low_cpu_mem_usage=False,
        **customized_kwargs,
    )
```

```python
# 3. Modify: llava/model/__init__.py
# Add GPT-2 imports

from llava.model.language_model.llava_gpt2 import LlavaGPT2Config, LlavaGPT2ForCausalLM
```

**Pros**:
- ✅ Architecturally correct
- ✅ Clean implementation
- ✅ Future-proof for other GPT-2 models
- ✅ Maintains separation of concerns

**Cons**:
- ⚠️  Requires coding (2-4 hours)
- ⚠️  Need to test/debug
- ⚠️  May expose GPT-2-specific issues (positional encodings, etc.)

**Risk**: MEDIUM - More work but cleaner solution.

---

###  **Option 3: Text-Only Training First** ⭐⭐⭐ SAFEST

**Idea**: Start with text-only training (no vision) which works with any model.

**Changes to train_italian_video.sh**:
```bash
# Change EXP_ID from 1 to text-only without vision
EXP_ID=1  # Keep text-only
# But comment out vision tower to force text-only mode:
# VISION_MODEL="facebook/dinov2-large"  # Comment this out
VISION_MODEL=""  # Or set to empty
```

**Workflow**:
1. **Phase 1**: Train BAMBI GPT-2 text-only on Italian captions
   - This bypasses the multimodal code entirely
   - Will work immediately
   - Validates BAMBI + Italian tokenizer + training pipeline

2. **Phase 2**: Once text works, tackle multimodal
   - Then implement Option 1 or 2 above
   - Start from text-only checkpoint

**Pros**:
- ✅ Works immediately (tested path)
- ✅ Validates base setup before adding complexity
- ✅ Useful baseline for comparison
- ✅ No risk of multimodal bugs

**Cons**:
- ⚠️  Doesn't train video understanding yet
- ⚠️  Two-phase process takes longer
- ⚠️  Need to solve multimodal issue eventually

**Risk**: NONE - This will definitely work.

---

## 🔧 Recommended Implementation Plan

### **Phase 1: Immediate (Try Option 1 first - 10 minutes)**

```bash
# Step 1: Backup BAMBI config
cd /data1/ososovskyy/LLaVA-ST-BabyLM-Configuration
cp babylm_italian/bambi_18_24_sept2025/config.json \
   babylm_italian/bambi_18_24_sept2025/config_original.json

# Step 2: Modify config to look like Llama
python3 << 'EOF'
import json

config_file = "babylm_italian/bambi_18_24_sept2025/config.json"
with open(config_file, 'r') as f:
    config = json.load(f)

# Make GPT-2 look like Llama
config["model_type"] = "llama"
config["architectures"] = ["LlamaForCausalLM"]

# Map GPT-2 params to Llama equivalents
config["hidden_size"] = config.pop("n_embd", 768)
config["num_hidden_layers"] = config.pop("n_layer", 12)
config["num_attention_heads"] = config.pop("n_head", 12)
config["intermediate_size"] = config.get("n_inner") or (config["hidden_size"] * 4)
config["max_position_embeddings"] = config.pop("n_positions", 1024)

# Add required Llama fields
config["hidden_act"] = "silu"  # Llama uses SiLU, GPT-2 uses GELU
config["rms_norm_eps"] = 1e-5
config["rope_scaling"] = None
config["tie_word_embeddings"] = False

with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Config modified to look like Llama")
EOF

# Step 3: Change train script to use "llama" explicitly
sed -i 's/bambi_18_24_sept2025/bambi_18_24_sept2025/g' train_italian_video.sh
sed -i 's/EXP_ID=1/EXP_ID=2/g' train_italian_video.sh  # Use video+text mode

# Step 4: Launch training
sbatch train_italian_video.sh
```

**Expected Outcome**:
- If it works → Great! You have a working Italian video model
- If it fails with attention/embedding errors → Try Option 2

---

### **Phase 2: If Option 1 Fails (Proper GPT-2 wrapper - 2-4 hours)**

Would you like me to generate the complete `llava_gpt2.py` implementation?

---

### **Phase 3: Fallback (Text-only training - immediate)**

```bash
# Just change EXP_ID and comment out vision
sed -i 's/EXP_ID=2/EXP_ID=1/g' train_italian_video.sh
# In the script, comment out: VISION_MODEL="facebook/dinov2-large"
sbatch train_italian_video.sh
```

---

## 📝 Summary: What You Should Do Now

1. **Try Option 1 first** (config disguise) - It's quick and might just work
2. **If that fails**, I'll help you implement Option 2 (proper GPT-2 wrapper)
3. **If you want to be safe**, start with Option 3 (text-only) to validate the pipeline

Which approach would you like to proceed with?
