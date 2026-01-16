from transformers import AutoImageProcessor
from PIL import Image
import torch


def verify_dinov2_dimensions():
    model_name = "facebook/dinov2-large"
    print(f"🔍 Loading processor for: {model_name}...")

    try:
        # Load the exact processor used in your script
        processor = AutoImageProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"❌ Error loading processor: {e}")
        return

    print("\n--- 1. Inspecting Processor Config ---")
    # This mimics exactly the logic in your modified __get_item
    if hasattr(processor, 'crop_size'):
        print(f"✅ Found 'crop_size' attribute: {processor.crop_size}")
        h = processor.crop_size.get('height', 384)
        w = processor.crop_size.get('width', 384)
    else:
        print("⚠️ 'crop_size' NOT found. Fallback logic would use 384.")
        h = w = 384

    print(f"👉 Resolved Dimensions for Training: {w}x{h}")

    print("\n--- 2. Simulating Black Image Creation ---")
    # Simulate the black image generation
    black_image = Image.new('RGB', (w, h), (0, 0, 0))
    print(f"Generated Black PIL Image size: {black_image.size}")

    print("\n--- 3. Verifying Tensor Output ---")
    try:
        # Simulate preprocessing
        # Some AutoImageProcessor implementations use `__call__` rather than `preprocess`.
        if hasattr(processor, 'preprocess'):
            inputs = processor.preprocess(black_image, return_tensors="pt")
        else:
            inputs = processor(images=black_image, return_tensors="pt")
        pixel_values = inputs.get("pixel_values", inputs.get("pixel_values"))
        if isinstance(pixel_values, torch.Tensor):
            tensor = pixel_values[0]
            print(f"Final Tensor Shape: {tensor.shape}")
        else:
            # sometimes dict values are nested in list
            tensor = pixel_values[0]
            print(f"Final Tensor Shape: {tensor.shape}")

        # Check against DINOv2 native resolution (518)
        if tensor.shape[1] == 518 and tensor.shape[2] == 518:
            print(f"\n✅ SUCCESS: The logic correctly identified 518x518.")
        elif tensor.shape[1] == 384:
            print(f"\n⚠️ WARNING: It defaulted to 384. This might be okay but isn't native DINOv2 resolution.")
        else:
            print(f"\n❌ ERROR: Unexpected dimension {tensor.shape}.")

    except Exception as e:
        print(f"Processing failed: {e}")


if __name__ == "__main__":
    verify_dinov2_dimensions()
