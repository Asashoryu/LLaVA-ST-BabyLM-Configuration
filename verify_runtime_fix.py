from transformers import AutoImageProcessor
from PIL import Image
import torch


def verify_runtime_override():
    model_name = "facebook/dinov2-large"
    print(f"1. Loading DEFAULT processor for: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)

    print(f"   Default crop_size: {processor.crop_size}")

    # --- SIMULATE THE FIX IN YOUR TRAIN.PY ---
    print("\n2. Applying Runtime Override (Simulating train.py)...")
    if "dinov2" in model_name:
        processor.size = {"height": 518, "width": 518}
        processor.crop_size = {"height": 518, "width": 518}
    # -----------------------------------------

    print(f"   New crop_size: {processor.crop_size}")

    print("\n3. Testing Preprocessing...")
    # Create black image using the NEW dimensions
    h = processor.crop_size['height']
    w = processor.crop_size['width']
    black_image = Image.new('RGB', (w, h), (0, 0, 0))

    # Some processors use `preprocess`, others use `__call__`/`__call__` aliases
    if hasattr(processor, 'preprocess'):
        inputs = processor.preprocess(black_image, return_tensors="pt")
    else:
        inputs = processor(images=black_image, return_tensors="pt")
    pixel_values = inputs["pixel_values"][0]

    print(f"   Final Tensor Shape: {pixel_values.shape}")

    if pixel_values.shape[1] == 518 and pixel_values.shape[2] == 518:
        print("\n✅ SUCCESS: The override works! Training will use 518x518.")
    else:
        print("\n❌ FAILURE: Override didn't take effect.")

if __name__ == "__main__":
    verify_runtime_override()
