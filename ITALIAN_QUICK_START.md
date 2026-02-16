# 🇮🇹 Italian Video-Language Model - Quick Start Guide

## ✅ Phase 1 Complete: Infrastructure Setup

### What We've Built

1. **✅ Video Processing** - Fixed video loading code in `llava/train/train.py`
2. **✅ Data Inspection** - `scripts_italian/inspect_webvid_data.py`
3. **✅ Dataset Generator** - `scripts_italian/generate_webvid_for_llava_italian.py`
4. **✅ Training Script** - `scripts_italian/train_italian_video.sh`
5. **✅ Directory Structure** - All folders created

---

## 📊 Dataset Status

**Source**: WebVid-10M Italian translations
- **Total samples**: 10,000 ✅
- **Italian captions**: 100% complete ✅
- **Avg caption length**: 16 words, 99 chars
- **Video durations**: 1-238s (avg 17.9s)
- **Format**: JSONL with Italian translations

**Generated Datasets** (100 samples for testing):
- ✅ `data/webvid_italian/processed/webvid_t_it.json` - Text-only
- ✅ `data/webvid_italian/processed/webvid_vi_it.json` - Video+text

---

## 🚀 Quick Start: Run Your First Training

### Step 1: Generate Full Datasets (or use test set)

**Option A: Use test set (100 samples) - Already done! ✅**
```bash
# Already generated with --max-samples 100
ls data/webvid_italian/processed/
```

**Option B: Generate full dataset (10,000 samples)**
```bash
cd /data1/ososovskyy/LLaVA-ST-BabyLM-Configuration
python scripts_italian/generate_webvid_for_llava_italian.py
```

### Step 2: Text-Only Training (Warm-up)

```bash
# Edit train_italian_video.sh
# Set: EXP_ID=1 (text-only Italian)

# Run training
sbatch train_italian_video.sh

# Or run directly (without SLURM):
bash train_italian_video.sh
```

**Expected output**: `output/ckpt_italian_1_t_it/`

**Optional - Resume from pretrained weights:**
To continue training from a previous checkpoint, edit `train_italian_video.sh`:
```bash
PRETRAIN_MM_MLP_ADAPTER="./output/ckpt_italian_1_vi_it/mm_projector.bin"
```

---

### Step 3: Video+Text Training (Multimodal)

**First, you need videos!** Choose one:

**Option A: Download a few videos manually for testing**
```bash
# Example: Download first video
VIDEO_DIR="data/webvid_italian/videos"
mkdir -p $VIDEO_DIR
curl -L "https://ak.picdn.net/shutterstock/videos/21179416/preview/stock-footage-aerial-shot-winter-forest.mp4" \
     -o "$VIDEO_DIR/21179416.mp4"
```

**Option B: Generate download script**
```bash
python scripts_italian/generate_webvid_for_llava_italian.py \
    --max-samples 100 \
    --generate-download-script

# Then run:
bash scripts_italian/download_videos.sh
```

**Then train** (with backbone configuration):
```bash
# Edit train_italian_video.sh
# Set: EXP_ID=2 (video+text Italian)
# Set: TRAIN_VISION_BACKBONE=false (or true for full fine-tuning)

sbatch train_italian_video.sh
```

**Expected output**: `output/ckpt_italian_1_vi_it/`


---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Italian Video-LM                     │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴──────────────────┐
        │                                      │
    ┌───▼────┐                          ┌─────▼──────┐
    │ Video  │                          │   GPT-2    │
    │ Frames │                          │  (BAMBI)   │
    └───┬────┘                          │  Italian   │
        │                               └─────▲──────┘
        │                                     │
  ┌─────▼──────┐                             │
  │  DINOv2-L  │                    ┌────────┴────────┐
  │   Encoder  │                    │  Tokenizer      │
  └─────┬──────┘                    │  (30K Italian)  │
        │                           └─────────────────┘
  ┌─────▼──────┐
  │ Fast-Slow  │
  │ Resampler  │
  │ (64+32)    │
  └─────┬──────┘
        │
  ┌─────▼──────┐
  │ MLP Proj.  │
  │ 1024→768   │
  └─────┬──────┘
        │
        └──────────────────────────► (Combined with text)
```

**Components**:
- **Decoder**: BAMBI GPT-2 (58M params, 12 layers, 768 dim)
- **Vision**: DINOv2-large (frozen)
- **Resampler**: Fast-slow TokenPacker (64+32 latents)
- **Projector**: 2-layer MLP (1024→768)

---

## 📁 Project Structure

```
LLaVA-ST-BabyLM-Configuration/
├── babylm_italian/              # Italian base model
│   ├── bambi_18_24_sept2025/   # GPT-2 checkpoint
│   ├── tokenizer_decoder/       # Italian tokenizer (30K)
│   └── data/                    # Original 750 samples
├── data/webvid_italian/
│   ├── videos/                  # Downloaded videos (*.mp4)
│   ├── annotations/             # Empty (for future use)
│   └── processed/               # Generated datasets
│       ├── webvid_t_it.json    # Text-only (100 samples) ✅
│       └── webvid_vi_it.json   # Video+text (100 samples) ✅
├── scripts_italian/             # Italian-specific scripts
│   ├── inspect_webvid_data.py  # Data analysis ✅
│   ├── generate_webvid_for_llava_italian.py ✅
│   ├── train_italian_video.sh  # Training script ✅
│   └── webvid_italian_stats.txt # Generated stats ✅
├── output/ckpt_italian/         # Model checkpoints
│   ├── ckpt_italian_1_t_it/    # Text-only checkpoint
│   └── ckpt_italian_1_vi_it/   # Video+text checkpoint
└── ITALIAN_PROJECT_PLAN.md      # Complete project plan ✅
```

---

## 🧪 Testing & Validation

### Test 1: Verify Video Loading
```python
# Create: scripts_italian/test_video_load.py
from llava.utils import process_video_with_decord
import argparse

class Args:
    video_fps = 1
    frames_upbound = 100
    num_frames = 8
    force_sample = False

args = Args()
video_file = "data/webvid_italian/videos/21179416.mp4"
video, video_time, frame_time, num_frames = process_video_with_decord(
    video_file, args, split=None
)
print(f"✅ Loaded {len(video)} frames, video time: {video_time:.2f}s")
```

### Test 2: Check Italian Tokenization
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "./babylm_italian/bambi_18_24_sept2025"
)
text = "Un gatto gioca con una palla"
tokens = tokenizer.encode(text)
print(f"Text: {text}")
print(f"Tokens: {len(tokens)} -> {tokens}")
print(f"Decoded: {tokenizer.decode(tokens)}")
```

---

## 📊 Training Monitoring

### WandB Setup
```bash
# Login to WandB
wandb login

# Monitor training
# Project: baby_llama_italian
# Check: https://wandb.ai/your-username/baby_llama_italian
```

### Check Logs
```bash
# Training logs
tail -f logs/train_italian_*.out

# Error logs
tail -f logs/train_italian_*.err

# TensorBoard (if enabled)
tensorboard --logdir output/ckpt_italian_1_t_it
```

---

## 🎯 Next Steps (Phase 2)

### Immediate (Do Now):
1. ✅ **Generate datasets** - Already done for 100 samples
2. ⏳ **Download test videos** - Get 10-20 videos for smoke test
3. ⏳ **Run text-only training** - Test BAMBI integration
4. ⏳ **Test video loading** - Verify decord works

### Short-term (This Week):
5. ⏳ **Smoke test video+text** - Train on 10 samples, 5 steps
6. ⏳ **Full text training** - 100 samples, 1 epoch
7. ⏳ **Debug any issues** - Fix tokenization, video loading
8. ⏳ **Small-scale video training** - 50 samples with videos

### Medium-term (Next 2 Weeks):
9. ⏳ **Scale to 1000 samples** - Full training run
10. ⏳ **Evaluation pipeline** - Italian captioning metrics
11. ⏳ **Hyperparameter tuning** - LR, batch size, etc.
12. ⏳ **Compare with BAMBI baseline** - Text-only performance

---

## 🐛 Troubleshooting

### Issue: BAMBI model not loading
**Solution**: Check path to `babylm_italian/bambi_18_24_sept2025/`
```bash
ls -la babylm_italian/bambi_18_24_sept2025/
# Should see: config.json, pytorch_model.bin, tokenizer files
```

### Issue: Video file not found
**Solution**: Videos need to be downloaded
```bash
# Check if video exists
VIDEO_ID=21179416
ls data/webvid_italian/videos/${VIDEO_ID}.mp4

# Download if missing
python scripts_italian/generate_webvid_for_llava_italian.py \
    --generate-download-script
```

### Issue: CUDA out of memory
**Solution**: Reduce batch size
```bash
# Edit train_italian_video.sh
PER_DEVICE_TRAIN_BATCH_SIZE=8  # Reduce from 16
GRADIENT_ACCUMULATION_STEPS=8  # Increase from 4
```

### Issue: Decord not installed
**Solution**: Install video processing libraries
```bash
conda activate llava-st
pip install decord av
```

---

## 📞 Support

**Documentation**:
- Full plan: `ITALIAN_PROJECT_PLAN.md`
- English training: `train_ln_t.sh` (reference)
- Video processing: `llava/utils.py`

**Scripts**:
- Data inspection: `scripts_italian/inspect_webvid_data.py`
- Dataset generation: `scripts_italian/generate_webvid_for_llava_italian.py`
- Training: `scripts_italian/train_italian_video.sh`

---

## ✅ Checklist for First Run

- [x] Phase 1 complete (infrastructure)
- [x] Datasets generated (100 samples)
- [ ] Download 10 test videos
- [ ] Run text-only training (EXP_ID=1)
- [ ] Verify checkpoint saves
- [ ] Test video loading
- [ ] Run video+text training (EXP_ID=2)
- [ ] Generate test captions
- [ ] Celebrate! 🎉

---

**Status**: Ready for testing! Start with text-only training (EXP_ID=1) on 100 samples.
