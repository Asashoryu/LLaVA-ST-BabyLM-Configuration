import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from llava.utils import rank0_print

class DinoV2VisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = vision_tower_cfg.mm_vision_select_layer
        self.select_feature = getattr(vision_tower_cfg, "mm_vision_select_feature", "patch")

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            self.load_model()
        else:
            self.cfg_only = AutoConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print(f"{self.vision_tower_name} is already loaded, skipping.")
            return

        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        # Load model; when running on CPU it's safer to use float32 dtypes
        self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        # If there is no CUDA available (running on CPU), convert vision tower to float32
        if not torch.cuda.is_available():
            try:
                self.vision_tower = self.vision_tower.to(dtype=torch.float32)
            except Exception:
                # Best-effort: some models may not support `to(dtype=...)` in-place
                pass
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.last_hidden_state
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            pass
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward(self, images):
        image_forward_outs = None

        # rank0_print(f"\n{'*'*60}")
        # rank0_print(f"🔭 DinoV2VisionTower.forward() START")
        # rank0_print(f"  Input type: {type(images)}")
        # if type(images) is list:
        #     rank0_print(f"  Input list length: {len(images)}")
        #     rank0_print(f"  First image shape: {images[0].shape if images else 'N/A'}")
        # else:
        #     rank0_print(f"  Input tensor shape: {images.shape}")
        # rank0_print(f"  Vision tower loaded: {self.is_loaded}")
        # rank0_print(f"  Vision tower name: {self.vision_tower_name}")

        # If the vision tower has not been loaded (delay_load mode), return dummy outputs
        if not self.is_loaded:
            # Prepare device and dtype defaults
            device = torch.device("cpu")
            dtype = torch.float32

            if type(images) is list:
                image_features = []
                for image in images:
                    # last_hidden_state shape: (batch=1, 1 + num_patches, hidden_size)
                    last_hidden_state = torch.zeros(1, 1 + self.num_patches, self.hidden_size, device=device, dtype=dtype)
                    image_forward_out = type("X", (), {"last_hidden_state": last_hidden_state})()
                    image_feature = self.feature_select(image_forward_out).to(image.dtype)
                    image_features.append(image_feature)
                    image_forward_outs = image_forward_out
            else:
                last_hidden_state = torch.zeros(images.shape[0], 1 + self.num_patches, self.hidden_size, device=device, dtype=dtype)
                image_forward_outs = type("X", (), {"last_hidden_state": last_hidden_state})()
                image_features = self.feature_select(image_forward_outs).to(images.dtype)

            return image_features, image_forward_outs

        # Normal operation when vision tower is loaded
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
                image_forward_outs = image_forward_out
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        # rank0_print(f"  ✅ DinoV2 output shape: {image_features.shape if not isinstance(image_features, list) else [f.shape for f in image_features]}")
        # rank0_print(f"  Output dtype: {image_features.dtype if not isinstance(image_features, list) else image_features[0].dtype}")
        # rank0_print(f"  Expected: [batch, num_patches={self.num_patches}, hidden_size={self.hidden_size}]")
        # rank0_print(f"{'*'*60}\n")

        return image_features, image_forward_outs

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size
