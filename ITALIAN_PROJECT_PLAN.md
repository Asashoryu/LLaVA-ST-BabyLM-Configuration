# Italian Video-Language Model Project Plan

## Project Goal
Train a multimodal Italian video-language model using:
- **Decoder**: BAMBI GPT-2 (58M params)
- **Encoder**: BERT-base Italian (similar size)
- **Dataset**: WebVid-10M Italian translations (~1500 samples for testing)
- **Vision**: Existing LLaVA-ST video processing pipeline

---

## Phase 1: Code Audit & Infrastructure ✅ COMPLETED

### 1.1 Video Processing Infrastructure - FOUND ✅
- **Location**: `llava/utils.py`
- **Functions**:
  - `process_video_with_decord(video_file, data_args, split=None)` - Main video loader
  - `process_video_with_pyav(video_file, data_args)` - Alternative loader
- **Capabilities**:
  - Uniform frame sampling (linspace)
  - FPS-based sampling (video_fps parameter)
  - Frame upbound limiting (frames_upbound)
  - Supports video segments (split=[start, end])

### 1.2 Video Token System - FOUND ✅
- **Constants** (`llava/constants.py`):
  ```python
  DEFAULT_VIDEO_TOKEN = "<video>"
  DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
  DEFAULT_VID_START_TOKEN = "<vid_start>"
  DEFAULT_VID_END_TOKEN = "<vid_end>"
  DEFAULT_SLOW_VID_START_TOKEN = "<slow_vid_start>"
  DEFAULT_SLOW_VID_END_TOKEN = "<slow_vid_end>"
  ```

### 1.3 Current Implementation Status - NEEDS FIX ⚠️
- **Problem**: Line 1374 in `llava/train/train.py` has:
  ```python
  raise NotImplementedError("Copia qui il blocco video originale dal backup")
  ```
- **Solution**: Copy complete implementation from backup (lines 1213-1273)

### 1.4 Dataset Location - FOUND ✅
- **Path**: `/data1/ososovskyy/web-vid-10M-Italian/`
- **Files**:
  - `0000_translated_first_1000.jsonl` (4.5MB, ~1000 samples)
  - Translation script, inspection tools
- **Structure**: JSONL with Italian translations

### 1.5 Existing BAMBI Model - FOUND ✅
- **Path**: `babylm_italian/bambi_18_24_sept2025/`
- **Config**: GPT-2, 12 layers, 768 dim, 30K vocab
- **Tokenizer**: Custom Italian BPE (30K tokens)

---

## Phase 2: Dataset Preparation

### 2.1 Inspect WebVid Italian Data Structure
**File**: `scripts_italian/inspect_webvid_data.py`
- Load `0000_translated_first_1000.jsonl`
- Analyze fields: videoid, contentUrl, duration, name_it, etc.
- Check video file availability
- Count valid samples

### 2.2 Create Dataset Converter
**File**: `scripts_italian/generate_webvid_for_llava_italian.py`

**Input**: `0000_translated_first_1000.jsonl`

**Output 1: Text-only** (`webvid_t_it.json`)
```json
{
  "id": "webvid_123",
  "conversations": [
    {"from": "human", "value": ""},
    {"from": "gpt", "value": "Un gatto gioca con una palla"}
  ]
}
```

**Output 2: Video + Text** (`webvid_vi_it.json`)
```json
{
  "id": "webvid_123",
  "video": "data/webvid_italian/videos/123.mp4",
  "conversations": [
    {"from": "human", "value": "<video>"},
    {"from": "gpt", "value": "Un gatto gioca con una palla"}
  ]
}
```

### 2.3 Video Download/Preparation
- Check if videos need downloading from contentUrl
- Organize videos in `data/webvid_italian/videos/`
- Validate video files (format, duration, readability)

---

## Phase 3: Model Architecture

### 3.1 Base Models
**Decoder**: BAMBI GPT-2
- Path: `babylm_italian/bambi_18_24_sept2025/`
- Vocab: 30,000 tokens
- Already trained on Italian

**Encoder**: BERT-base Italian (To Be Decided)
- Options:
  - `dbmdz/bert-base-italian-xxl-cased` (similar size to DINOv2)
  - Keep DINOv2-large for video frames (proven to work)
- **Decision needed**: Use BERT for text encoding or DINOv2 for video?

### 3.2 Vision Processing
- **Keep existing**: DINOv2-large for video frame encoding
- **Token Packer**: Existing fast-slow resampler
- **Frame sampling**: As per existing code (8-100 frames)

---

## Phase 4: Training Pipeline

### 4.1 Fix Video Loading in train.py
**Task**: Replace NotImplementedError with working video loading code from backup

### 4.2 Create Italian Training Script
**File**: `scripts_italian/train_italian_video.sh`

**Key Parameters**:
```bash
EXP_ID=1  # 1=text-only, 2=video+text
MODEL_PATH="./babylm_italian/bambi_18_24_sept2025"
DATA_PATH="data/webvid_italian/webvid_${SUFFIX}_it.json"
VIDEO_FOLDER="data/webvid_italian/videos"
OUTPUT_DIR="output/ckpt_italian_${VERSION}_${SUFFIX}"

# Video processing
VIDEO_FPS=1  # 1 frame per second
FRAMES_UPBOUND=100  # Max 100 frames
NUM_FRAMES=8  # Actual frames used
```

### 4.3 Training Stages
**Stage 1**: Text-only Italian (EXP_ID=1)
- Warm up BAMBI tokenizer embeddings
- Verify Italian text processing

**Stage 2**: Video + Italian text (EXP_ID=2)
- Add video encoder
- Train multimodal adapter
- Use existing LLaVA-ST video sampling

---

## Phase 5: Testing & Validation

### 5.1 Unit Tests
**File**: `scripts_italian/test_italian_video.py`
- Test video loading from WebVid
- Test Italian tokenization
- Test `<video>` token expansion
- Test forward pass with video+text

### 5.2 Smoke Test
- Train on 10 samples for 5 steps
- Verify gradients flow
- Check loss convergence

### 5.3 Small-Scale Training
- Train on 100-200 samples
- 1 epoch, monitor loss
- Verify checkpoint saving

---

## Phase 6: Evaluation

### 6.1 Italian Linguistic Benchmarks
- Adapt BabyLM eval for Italian
- Test on Italian text-only data

### 6.2 Video Captioning
- Generate Italian captions for test videos
- Compute BLEU/METEOR/CIDEr

---

## Immediate Next Steps

✅ **Step 1**: Fix video loading in `llava/train/train.py` (copy from backup)
⏳ **Step 2**: Create `scripts_italian/inspect_webvid_data.py`
⏳ **Step 3**: Create `scripts_italian/generate_webvid_for_llava_italian.py`
⏳ **Step 4**: Create `scripts_italian/train_italian_video.sh`
⏳ **Step 5**: Run first smoke test

---

## Critical Decisions Made

✅ **Models**: GPT-2 (BAMBI) + BERT encoder
✅ **Video sampling**: Use existing LLaVA-ST code (no changes)
✅ **Grounding**: Skip for now (focus on captioning)
✅ **Dataset**: Start with 1500 existing samples

---

## Directory Structure to Create

```
LLaVA-ST-BabyLM-Configuration/
├── data/
│   └── webvid_italian/
│       ├── videos/              # Downloaded video files
│       ├── annotations/         # Original JSONL
│       ├── webvid_t_it.json    # Text-only dataset
│       └── webvid_vi_it.json   # Video+text dataset
├── scripts_italian/
│   ├── inspect_webvid_data.py
│   ├── generate_webvid_for_llava_italian.py
│   ├── train_italian_video.sh
│   └── test_italian_video.py
└── output/
    └── ckpt_italian/            # Italian model checkpoints
```

---

**Status**: Phase 1 Complete ✅ | Ready for Phase 2 Implementation
