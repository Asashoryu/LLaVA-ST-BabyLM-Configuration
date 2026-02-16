import torch
import torch.nn as nn
import math
from torch.nn.init import trunc_normal_
from torch.nn import functional as F
from llava.utils import rank0_print

# ============================================================================
# 1. TokenPackerAttention
# ============================================================================
class TokenPackerAttention(nn.Module):
    def __init__(self, *, dim, dim_head=128, heads=8, patch_devide_pattern='spatial_temporal',
                 spatial_scale_factor=3, temporal_scale_factor=10):
        super().__init__()
        embed_dim = dim_head * heads
        self.to_q = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.to_k = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.to_v = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_k = nn.LayerNorm(embed_dim)
        self.ln_v = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, heads)

        self.to_out = nn.Linear(embed_dim, dim, bias=False)

        self.patch_devide_pattern = patch_devide_pattern
        self.spatial_scale_factor = spatial_scale_factor
        self.temporal_scale_factor = temporal_scale_factor

    def divide_feature(self, x, kernel_size_t, kernel_size_hw, temporal_token_num, spatial_token_num, c):
        """
        Divide features into patches for attention.
        x: (spatial_tokens, temporal_tokens, c)
        """
        h = w = int(spatial_token_num ** 0.5)
        # Fallback per griglie non quadrate
        if h * w != spatial_token_num:
            h = int(round(spatial_token_num ** 0.5))
            w = spatial_token_num // h

        t = temporal_token_num

        if t == 1:
            kernel_size_t = 1

        # Controllo di sicurezza sui kernel (evita divisioni impossibili)
        if h % kernel_size_hw != 0:
            new_kernel_hw = max(1, h // (h // kernel_size_hw + (1 if h % kernel_size_hw else 0)))
            kernel_size_hw = new_kernel_hw

        if w % kernel_size_hw != 0:
            new_kernel_hw = max(1, w // (w // kernel_size_hw + (1 if w % kernel_size_hw else 0)))
            kernel_size_hw = new_kernel_hw

        if t % kernel_size_t != 0:
            new_kernel_t = max(1, t // (t // kernel_size_t + (1 if t % kernel_size_t else 0)))
            kernel_size_t = new_kernel_t

        reshape_x = x.reshape(h // kernel_size_hw, kernel_size_hw, w // kernel_size_hw, kernel_size_hw,
                              t // kernel_size_t, kernel_size_t, c)
        reshape_x = reshape_x.permute(1, 3, 5, 0, 2, 4, 6)
        reshape_x = reshape_x.reshape(kernel_size_hw ** 2 * kernel_size_t, -1, c)

        return reshape_x

    def forward(self, x, x_multi, attn_mask=None):
        q = self.ln_q(self.to_q(x)).permute(1, 0, 2)
        k = self.ln_k(self.to_k(x_multi)).permute(1, 0, 2)
        v = self.ln_v(self.to_v(x_multi)).permute(1, 0, 2)

        key_spatial_num, key_temporal_num, c = k.shape
        query_spatial_num, query_temporal_num, c = q.shape

        if self.patch_devide_pattern == 'spatial':
            q = self.divide_feature(q, 1, 1, query_temporal_num, query_spatial_num, c)
            k = self.divide_feature(k, 1, self.spatial_scale_factor, key_temporal_num, key_spatial_num, c)
            v = self.divide_feature(v, 1, self.spatial_scale_factor, key_temporal_num, key_spatial_num, c)
        elif self.patch_devide_pattern == 'temporal':
            q = self.divide_feature(q, 1, 1, query_temporal_num, query_spatial_num, c)
            k = self.divide_feature(k, self.temporal_scale_factor, 1, key_temporal_num, key_spatial_num, c)
            v = self.divide_feature(v, self.temporal_scale_factor, 1, key_temporal_num, key_spatial_num, c)
        elif self.patch_devide_pattern == 'spatial_temporal':
            q = self.divide_feature(q, 1, 1, query_temporal_num, query_spatial_num, c)
            k = self.divide_feature(k, self.temporal_scale_factor, self.spatial_scale_factor, key_temporal_num, key_spatial_num, c)
            v = self.divide_feature(v, self.temporal_scale_factor, self.spatial_scale_factor, key_temporal_num, key_spatial_num, c)
        else:
            raise ValueError("Unexpected patch devide pattern")

        # Flatten q/k/v so they share a common batch dimension for MHA.
        # Current shapes are (a, b, c) where a=patch_elems, b=num_windows.
        # MultiheadAttention expects (L, N, E) where N is batch. We set N=1
        # by flattening windows into the sequence dimension: (a*b, 1, c).
        q_shape = q.shape
        k_shape = k.shape
        v_shape = v.shape

        q = q.reshape(q_shape[0] * q_shape[1], 1, c)
        k = k.reshape(k_shape[0] * k_shape[1], 1, c)
        v = v.reshape(v_shape[0] * v_shape[1], 1, c)

        out = self.attn(q, k, v, attn_mask=attn_mask)[0]

        # Restore patch grouping: (a*b, 1, embed) -> (a, b, embed)
        out = out.reshape(q_shape[0], q_shape[1], -1)

        # Map back to original query spatial/temporal grid
        out = out.reshape(query_spatial_num, query_temporal_num, -1)
        out = out.permute(1, 0, 2)
        out = self.to_out(out)

        return out

# ============================================================================
# 2. TokenPackerModule
# ============================================================================
class TokenPackerModule(nn.Module):
    def __init__(
            self,
            *,
            raw_grid=27,
            num_latents=81,
            raw_frames=100,
            num_temporal_latents=20,
            embed_dim=1024,
            num_heads=8,
            visual_dim=3456,
            hidden_size=3584,
            patch_devide_pattern='spatial_temporal',
            spatial_scale_factor=3,
            temporal_scale_factor=5
    ):
        super().__init__()
        self.raw_grid = raw_grid
        self.grid_size = int(num_latents ** 0.5)

        self.raw_frames = raw_frames
        self.num_temporal_latents = num_temporal_latents
        self.num_queries = self.grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.patch_devide_pattern = patch_devide_pattern

        self.devide_attention = TokenPackerAttention(
            dim=visual_dim,
            dim_head=embed_dim // num_heads,
            heads=num_heads,
            patch_devide_pattern=patch_devide_pattern,
            spatial_scale_factor=spatial_scale_factor,
            temporal_scale_factor=temporal_scale_factor
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):
        x_multi = x.clone()
        dtype = x.dtype
        t, hw, c = x.shape

        if self.patch_devide_pattern == 'spatial':
            # --- 1. Query (Latents) ---
            actual_grid = int(round(hw ** 0.5))
            if actual_grid ** 2 != hw:
                actual_grid = int(hw ** 0.5)

            # Reshape e Interpolazione Query alla griglia latente
            x = x.reshape(t, actual_grid, actual_grid, c).permute(0, 3, 1, 2)
            x = F.interpolate(x, size=(self.grid_size, self.grid_size), mode='bilinear')
            x = x.permute(0, 2, 3, 1).reshape(t, -1, c).to(dtype)

            # --- 2. Key/Value (Image Features) - FIX UNIVERSALE (SPAZIALE) ---
            # Le Key DEVONO generare lo stesso numero di finestre delle Query.
            # Query Windows = (grid_size / 1)^2
            # Key Windows = (key_grid / scale)^2
            # Quindi key_grid deve essere esattamente grid_size * scale.

            scale = getattr(self.devide_attention, 'spatial_scale_factor', 1)
            target_key_grid = self.grid_size * scale

            if actual_grid != target_key_grid:
                # Questo blocco previene il crash se usi parametri incompatibili (es. 81 latents con DINOv2)
                # Con 64 latents e DINOv2, questo blocco NON viene eseguito (actual_grid=16 == target=16)
                x_multi = x_multi.reshape(t, actual_grid, actual_grid, c).permute(0, 3, 1, 2)
                x_multi = F.interpolate(x_multi, size=(target_key_grid, target_key_grid), mode='bilinear')
                x_multi = x_multi.permute(0, 2, 3, 1).reshape(t, -1, c).to(dtype)

        elif self.patch_devide_pattern == 'temporal' and t != 1:
            # Resize queries temporalmente
            x = x.reshape(t, -1, c).permute(1, 2, 0)
            x = F.interpolate(x, size=(self.num_temporal_latents,), mode='linear')
            x = x.permute(2, 0, 1).to(dtype)

            # NOTE: For temporal-only path we intentionally DO NOT resize x_multi here.
            # The temporal path compresses the query (x) into `num_temporal_latents`,
            # while x_multi (the full-resolution temporal features) should preserve
            # the original temporal resolution so that the devide_attention can
            # attend across the full temporal axis if needed.

        x = x + self.devide_attention(x, x_multi, attn_mask)
        return x

# ============================================================================
# 3. TokenPacker (Wrapper)
# ============================================================================
class TokenPacker(nn.Module):
    def __init__(self, model_args, vision_tower):
        super().__init__()

        # ✅ CRITICAL FIX: Allow environment variables to override config
        # This bypasses HfArgumentParser issues with config caching
        import os
        slow_latents = int(os.environ.get('MM_PERCEIVER_LATENTS', model_args.mm_perceiver_latents))
        fast_latents = int(os.environ.get('MM_PERCEIVER_LATENTS_FAST', model_args.mm_perceiver_latents_fast))

        # rank0_print(f"\n{'='*70}")
        # rank0_print(f"TokenPacker.__init__():")
        # rank0_print(f"  model_args.mm_perceiver_latents={model_args.mm_perceiver_latents}")
        # rank0_print(f"  model_args.mm_perceiver_latents_fast={model_args.mm_perceiver_latents_fast}")
        # rank0_print(f"  (Env override) slow_latents={slow_latents}")
        # rank0_print(f"  (Env override) fast_latents={fast_latents}")
        # rank0_print(f"{'='*70}\n")
        pass

        self.depth = model_args.mm_perceiver_depth
        self.slow_num_latents = slow_latents
        self.fast_num_latents = fast_latents
        self.ff_mult = model_args.mm_perceiver_ff_mult
        self.pretrained = model_args.mm_perceiver_pretrained

        # ✅ visual_dim: features come AFTER mm_projector, so always = LLM hidden_size
        # For Llama2-7b: hidden_size=768 (NOT vision_tower.hidden_size!)
        visual_dim = getattr(model_args, 'hidden_size', 768)

        vt_name = getattr(model_args, 'vision_tower', '') or ''
        if isinstance(vt_name, str) and 'siglip' in vt_name.lower():
            # SigLIP is special: uses 3584-dim features
            visual_dim = 3584

        # raw grid side: use information from the vision_tower if present
        raw_grid_spatial = getattr(vision_tower, 'num_patches_per_side', None) if vision_tower is not None else None
        if raw_grid_spatial is None:
            if 'dinov2' in (getattr(model_args, 'vision_tower', '') or '').lower():
                raw_grid_spatial = 37  # DINOv2-large: 37×37 patches
            elif 'siglip' in (getattr(model_args, 'vision_tower', '') or '').lower():
                raw_grid_spatial = 27  # SigLIP: 27×27 patches
            else:
                raw_grid_spatial = 24  # Fallback

        # Calcolo Scala Spaziale (Dinamico)
        # FIX: Use round() instead of int() for grid_size to match Gen 6 behavior
        # Gen 6 had FastScale=7, which requires grid_size_fast=6 (from round(sqrt(32))=6)
        # Using int() gives grid_size_fast=5 and FastScale=8, which hurts performance
        grid_size_slow = int(round(self.slow_num_latents ** 0.5))
        spatial_scale_slow = max(1, int(math.ceil(raw_grid_spatial / grid_size_slow)))

        grid_size_fast = int(round(self.fast_num_latents ** 0.5))
        spatial_scale_fast = max(1, int(math.ceil(raw_grid_spatial / grid_size_fast)))

        # rank0_print(f"TokenPacker Init: InputGrid={raw_grid_spatial}, SlowScale={spatial_scale_slow}, FastScale={spatial_scale_fast}")
        # rank0_print(f"TokenPacker: visual_dim={visual_dim} (features from mm_projector)")
        pass

        self.token_packer = TokenPackerModule(
            raw_grid=raw_grid_spatial,
            num_latents=self.slow_num_latents,
            embed_dim=1024,
            num_heads=8,
            visual_dim=visual_dim,
            hidden_size=visual_dim,
            patch_devide_pattern='spatial',
            spatial_scale_factor=spatial_scale_slow,
            temporal_scale_factor=5
        )

        self.token_packer_slow = TokenPackerModule(
            raw_grid=raw_grid_spatial,
            num_latents=self.slow_num_latents,
            embed_dim=1024,
            num_heads=8,
            visual_dim=visual_dim,
            hidden_size=visual_dim,
            patch_devide_pattern='temporal',
            spatial_scale_factor=spatial_scale_slow,
            temporal_scale_factor=5,
            num_temporal_latents=4  ##CRITICAL: Match video frame count (FRAMES_UPBOUND=4)
        )

        self.token_packer_fast = TokenPackerModule(
            raw_grid=raw_grid_spatial,
            num_latents=self.fast_num_latents,
            embed_dim=1024,
            num_heads=8,
            visual_dim=visual_dim,
            hidden_size=visual_dim,
            patch_devide_pattern='spatial',
            spatial_scale_factor=spatial_scale_fast,
            temporal_scale_factor=5
        )

        if self.pretrained is not None:
            rank0_print(f"Loading pretrained TokenPacker from {self.pretrained}")
            self.load_state_dict(torch.load(self.pretrained), strict=False)

    @property
    def config(self):
        return {
            "mm_resampler_type": "fast_slow_resampler",
            "mm_perceiver_depth": self.depth,
            "mm_perceiver_latents": self.slow_num_latents,
            "mm_perceiver_latents_fast": self.fast_num_latents,
            "mm_perceiver_ff_mult": self.ff_mult,
            "mm_perceiver_pretrained": self.pretrained,
        }

    def forward(self, image_features, slow=True, *args, **kwargs):
        # rank0_print(f"\n{'='*60}")
        # rank0_print(f"🔍 TokenPacker.forward() - INPUT")
        # rank0_print(f"  Input shape: {image_features.shape}")
        # rank0_print(f"  Input dtype: {image_features.dtype}")
        # rank0_print(f"  Input device: {image_features.device}")
        # rank0_print(f"  Expected visual_dim: {self.token_packer.devide_attention.to_q[0].in_features}")
        # rank0_print(f"  slow={slow}")

        image_features = self.token_packer(image_features)
        # rank0_print(f"\n🔍 After token_packer (main): {image_features.shape}")

        if slow:
            slow_features = self.token_packer_slow(image_features)
            # rank0_print(f"🔍 After token_packer_slow: {slow_features.shape}")
            fast_features = self.token_packer_fast(image_features)
            # rank0_print(f"🔍 After token_packer_fast: {fast_features.shape}")
            # Dummy for deepspeed
            slow_features = slow_features + fast_features.mean() * 0
            # rank0_print(f"🔍 TokenPacker OUTPUT (slow mode): {slow_features.shape}")
            # rank0_print(f"{'='*60}\n")
            return slow_features
        else:
            slow_features = self.token_packer_slow(image_features)
            # rank0_print(f"🔍 After token_packer_slow: {slow_features.shape}")
            fast_features = self.token_packer_fast(image_features)
            # rank0_print(f"🔍 After token_packer_fast: {fast_features.shape}")
            # Dummy for deepspeed
            slow_features = slow_features + fast_features.mean() * 0
            # rank0_print(f"🔍 TokenPacker OUTPUT (fast+slow mode): slow={slow_features.shape}, fast={fast_features.shape}")
            # rank0_print(f"{'='*60}\n")
            return slow_features, fast_features
