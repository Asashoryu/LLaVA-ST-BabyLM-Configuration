"""
Pre-extract DINOv2 features using OpenCV (cv2) instead of pyav/decord.
OpenCV is more stable and avoids recursion issues.
"""
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import cv2
from dataclasses import dataclass

@dataclass
class DataArguments:
    video_fps: int = 1

def load_video_frames_cv2(video_path, fps=1, max_frames=0):
    """Load video frames using OpenCV."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame sampling
    frame_interval = max(1, int(video_fps / fps))

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)

            if max_frames > 0 and len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames extracted from {video_path}")

    return frames

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
        torch_dtype=torch.float32  # Use float32 for feature extraction
    ).to(device).eval()
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

    print(f"Model loaded on {device}")
    print("Starting extraction...\n")

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
                # Load video frames with OpenCV
                video_frames = load_video_frames_cv2(video_path, fps=1, max_frames=0)

                # Process frames through DINOv2 in batches to avoid OOM
                batch_size = 32
                all_features = []

                for i in range(0, len(video_frames), batch_size):
                    batch_frames = video_frames[i:i+batch_size]
                    inputs = image_processor(images=batch_frames, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # Get features
                    outputs = vision_tower(**inputs)
                    # Use CLS token features: [num_frames, 1024] for dinov2-large
                    features = outputs.last_hidden_state[:, 0].cpu().numpy()
                    all_features.append(features)

                # Concatenate all batches
                all_features = np.concatenate(all_features, axis=0)

                # Save features
                np.save(feature_path, all_features)
                success_count += 1

            except Exception as e:
                fail_count += 1
                failed_videos.append({
                    "index": idx,
                    "videoid": video_id,
                    "video_file": video_file,
                    "error": str(e)[:200]
                })
                # Print but continue
                tqdm.write(f"Failed {video_file}: {str(e)[:80]}")
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
