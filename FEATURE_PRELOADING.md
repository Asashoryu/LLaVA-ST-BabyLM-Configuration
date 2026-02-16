# Feature Preloading for Italian Video Training

## Overview

To eliminate video loading recursion issues, you can pre-extract DINOv2 features offline and load them during training instead of processing videos at runtime.

## Usage

### 1. Extract Features (One-time setup)

```bash
# Submit feature extraction job (takes ~1-2 hours for 1,406 videos)
sbatch extract_features_cv2.sh

# Monitor progress
squeue -u $USER
tail -f extract_features_cv2_JOBID.out

# Check extracted count
ls data/webvid_italian/features/*.npy | wc -l  # Should be 1406 when complete
```

### 2. Enable Feature Preloading in Training

Edit `train_italian_video.sh` and set:

```bash
USE_PRELOADED_FEATURES=True
PRELOADED_FEATURES_FOLDER="./data/webvid_italian/features"
```

Then train normally:
```bash
sbatch train_italian_video.sh
```

### 3. Disable Feature Preloading (Use Videos Directly)

Edit `train_italian_video.sh` and set:

```bash
USE_PRELOADED_FEATURES=False
```

## What Gets Extracted

- **Input**: Videos in `data/webvid_italian/videos/*.mp4`
- **Processing**:
  - Videos loaded with OpenCV (cv2)
  - Frames sampled at 1 FPS
  - Each frame → DINOv2-large → 1024-dim embedding
- **Output**: `data/webvid_italian/features/VIDEO_ID.npy`
  - Shape: `[num_frames, 1024]`
  - Dtype: `float32`
  - Size: ~50-200KB per video (vs 2-5MB original)

## Benefits

✅ **No recursion errors**: Videos never loaded during training
✅ **Faster I/O**: 10-50x smaller files
✅ **Identical features**: Same embeddings DINOv2 would compute
✅ **Easy toggle**: Switch between preloaded/direct with one variable

## Example Feature Structure

```python
import numpy as np

# Load a feature file
features = np.load("data/webvid_italian/features/1008593641.npy")
print(features.shape)  # (24, 1024) - 24 frames, 1024-dim embeddings
print(features.dtype)  # float32
```

## Files

- **extract_video_features_cv2.py**: Feature extraction script
- **extract_features_cv2.sh**: SLURM job script
- **train_italian_video.sh**: Training script with USE_PRELOADED_FEATURES flag
- **llava/train/train.py**: Modified to support preloaded features (DataArguments)

## Troubleshooting

**Q: Some features failed to extract**
A: Check `data/webvid_italian/features/failed_videos.json` for error details

**Q: Training still tries to load videos**
A: Verify `USE_PRELOADED_FEATURES=True` (not false, False, True - case matters)

**Q: FileNotFoundError for .npy file**
A: Ensure feature extraction completed for all 1,406 videos
