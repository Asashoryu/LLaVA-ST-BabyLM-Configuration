import torch
from types import SimpleNamespace

# Import TokenPacker from the codebase
from llava.model.multimodal_resampler.token_packer import TokenPacker

def run_smoke():
    # Create minimal model_args
    model_args = SimpleNamespace()
    model_args.mm_perceiver_latents = 64
    model_args.mm_perceiver_latents_fast = 32
    model_args.mm_perceiver_depth = 2
    model_args.mm_perceiver_ff_mult = 4
    model_args.mm_perceiver_pretrained = None
    model_args.hidden_size = 768
    model_args.vision_tower = 'facebook/dinov2-large'

    # vision_tower can be None for this test
    vision_tower = None

    packer = TokenPacker(model_args, vision_tower)
    packer.eval()

    # Synthetic DINOv2 output: t=1, hw=37*37, c=hidden_size
    t = 1
    hw = 37 * 37
    c = model_args.hidden_size
    x = torch.randn(t, hw, c)

    with torch.no_grad():
        out = packer(x)

    print("Input shape:", x.shape)
    if isinstance(out, tuple):
        print("Output (slow, fast) shapes:", out[0].shape, out[1].shape)
    else:
        print("Output shape:", out.shape)

if __name__ == '__main__':
    run_smoke()
