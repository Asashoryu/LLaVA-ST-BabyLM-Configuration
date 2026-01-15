import torch
import torch.nn as nn
import re

from .pooler_projector import PoolerProjector

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}

class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    # Visual feature dimension
    vision_dim = config.mm_hidden_size
    # Language model (LLM) dimension
    llm_dim = config.hidden_size

    if projector_type == "linear":
        # Project from vision output dim → LLM dim
        projector = nn.Linear(vision_dim, llm_dim)
        print(f"\n🔧 Building mm_projector (linear):")
        print(f"   vision_dim (input): {vision_dim}")
        print(f"   llm_dim (output): {llm_dim}")
        print(f"   Projector weight shape: {projector.weight.shape}\n")
        return projector

    if projector_type == "pooler":
        return PoolerProjector(config, kwargs["vision_cfg"])

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(vision_dim, llm_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(llm_dim, llm_dim))
        projector = nn.Sequential(*modules)
        print(f"\n🔧 Building mm_projector (mlp{mlp_depth}x_gelu):")
        print(f"   vision_dim (input): {vision_dim}")
        print(f"   llm_dim (output): {llm_dim}")
        print(f"   Depth: {mlp_depth}")
        print(f"   Total layers: {len(modules)}\n")
        return projector

    mlp_gelu_resnet_match = re.match(r"^mlp(\d+)x_res(\d+)x_gelu$", projector_type)
    if mlp_gelu_resnet_match:
        mlp_depth = int(mlp_gelu_resnet_match.group(1))
        res_depth = int(mlp_gelu_resnet_match.group(2))
        modules = [nn.Linear(vision_dim, llm_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(llm_dim, llm_dim))
        for _ in range(res_depth):
            modules.append(SimpleResBlock(llm_dim))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")
