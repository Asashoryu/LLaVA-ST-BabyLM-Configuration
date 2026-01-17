"""
Quick token count diagnostic for LazySupervisedDataset
Usage:
    python token_count_diagnostic.py your_data_path.json
"""
import sys
from llava.train.train import LazySupervisedDataset, DataArguments
import torch

class DummyVisionConfig:
    im_start_token = 32000
    im_end_token = 32001
    im_patch_token = 32002
    image_token_num = 257  # Example: 1 start + 255 patches + 1 end

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python token_count_diagnostic.py <data_path.json>")
        sys.exit(1)
    data_path = sys.argv[1]
    args = DataArguments(data_path=data_path, is_multimodal=True)
    vision_config = DummyVisionConfig()
    ds = LazySupervisedDataset(args, None, vision_config)
    for idx in range(min(5, len(ds))):
        sample = ds[idx]
        input_ids = sample["input_ids"]
        n_im_start = (input_ids == vision_config.im_start_token).sum().item()
        n_im_end = (input_ids == vision_config.im_end_token).sum().item()
        n_im_patch = (input_ids == vision_config.im_patch_token).sum().item()
        print(f"Sample {idx}:")
        print(f"  im_start: {n_im_start}, im_end: {n_im_end}, im_patch: {n_im_patch}")
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  Has image: {'image' in sample}")
        print(f"  First 20 input_ids: {input_ids[:20]}")
        print()
