"""
Pre-extract DINOv2 features for all Italian videos offline.
This eliminates video loading during training, avoiding all recursion issues.
"""
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from llava.utils import process_video_with_pyav
from dataclasses import dataclass
from typing import Optional

@dataclass
class DataArguments:
    image_aspect_ratio: str = "pad"
    video_fps: int = 1
    frames_upbound: int = 0
    unfreeze_mm_vision_tower: bool = False

def main():
    # Load Italian dataset
    json_path = "./data/webvid_italian/webvid_italian_merged.json"
    video_folder = "./data/webvid_italian/videos"
    output_folder = "./data/webvid_italian/features"
    os.makedirs(output_folder, exist_ok=True)

    with open(json_path, 'r') as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} samples")

    # Load DINOv2 model
    print("Loading DINOv2-large model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vision_tower = AutoModel.from_pretrained(
        "facebook/dinov2-large",
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

    print(f"Model loaded on {device}")

    # Data args for video processing
    data_args = DataArguments()

    # Track stats
    success_count = 0
    fail_count = 0
    failed_videos = []

    # Process each video
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(dataset, desc="Extracting features")):
            video_id = sample["videoid"]
            video_file = f"{video_id}.mp4"
            video_path = os.path.join(video_folder, video_file)

            # Feature output path
            feature_name = f"{video_id}.npy"
            feature_path = os.path.join(output_folder, feature_name)

            # Skip if already extracted
            if os.path.exists(feature_path):
                success_count += 1
                continue

            try:
                # Load video frames
                video_frames = process_video_with_pyav(video_path, data_args)

                if video_frames is None or len(video_frames) == 0:
                    raise ValueError(f"No frames extracted from {video_file}")

                # Process frames through DINOv2
                # video_frames is a list of PIL Images
                inputs = image_processor(images=video_frames, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Get features
                outputs = vision_tower(**inputs)
                # Use CLS token features: [num_frames, 1024] for dinov2-large
                features = outputs.last_hidden_state[:, 0].cpu().numpy()

                # Save features
                np.save(feature_path, features)
                success_count += 1

            except Exception as e:
                fail_count += 1
                failed_videos.append({
                    "index": idx,
                    "videoid": video_id,
                    "video_file": video_file,
                    "error": str(e)[:200]
                })
                print(f"\nFailed {video_file}: {str(e)[:100]}")
                continue

    # Save failed videos list
    if failed_videos:
        with open(os.path.join(output_folder, "failed_videos.json"), 'w') as f:
            json.dump(failed_videos, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Feature extraction complete!")
    print(f"Success: {success_count}/{len(dataset)}")
    print(f"Failed: {fail_count}/{len(dataset)}")
    if failed_videos:
        print(f"Failed videos saved to {output_folder}/failed_videos.json")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
