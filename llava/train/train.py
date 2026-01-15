# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import ast
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from PIL import Image, ImageFile
from packaging import version
import numpy as np

import sys
sys.path.append("../../")
import wandb
wandb.init(mode="disabled")

import time
import random
import yaml
import math
import re
import torch

import transformers
import tokenizers
import deepspeed

from transformers import AutoConfig
from torch.utils.data import Dataset
import torch.distributed as dist
from llava.constants import *
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token
from llava.utils import rank0_print, process_video_with_pyav, process_video_with_decord, get_rank, get_world_size

torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


@dataclass
class ModelArguments:
    """
    Defines model parameters for Multimodal training.
    """
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    # Freeze LLM backbone
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=81)
    mm_perceiver_latents_fast: Optional[int] = field(default=9)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)


    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)
    use_downsample_image: Optional[bool] = field(default=False)



@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)

    num_frames: Optional[int] = field(default=100)



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    mm_vision_resampler_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})

    stage: str = field(default="1")
    load_lora_path: str = field(default="")

# @dataclass
# class EvaluationArguments:
#     eval_num_processes: int = field(default=1)
#     task_names: str = field(default=None)
#     model: str = field(default="llava")
#     model_args: Optional[str] = field(default=None)
#     num_fewshot: Optional[int] = field(default=None)
#     batch_size: int = field(default=1)
#     device: Optional[str] = field(default=None)
#     limit: Optional[int] = field(default=None)
#     check_integrity: Optional[bool] = field(default=False)
#     show_task_to_terminal: Optional[bool] = field(default=False)
#     log_samples: Optional[bool] = field(default=True)
#     gen_kwargs: Optional[str] = field(default="")
#     log_samples_suffix: Optional[str] = field(default="")
#     output_path: Optional[str] = field(default="./logs/")


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler", "output_embeddings", "input_embeddings"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    state_dict = trainer.model.state_dict()
    cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
    trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments, vision_config) -> Dict:
    """
    Expands multimodal placeholder tokens (<image>, <video>) into their actual token sequences.
    Transforms symbolic tokens into the detailed token representations that the model expects.

    For images: <image> -> <im_start> + (image_token_num-2) patch tokens + <im_end>
    For videos: <video> -> video_start + (frame_num*token_num-2) patch tokens + video_end
    """
    # Check if this dataset contains multimodal data (images/videos)
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        rank0_print("WARNING: Data is not multimodal, skipping preprocess_multimodal.")
        return sources

    # Counter to track if we found any multimodal elements
    multimodal_count = 0

    # Iterate through all conversations in the batch
    for source in sources:
        # Each source is a list of conversational turns (human-assistant pairs)
        for sentence in source:
            # Handle image tokens
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:  # Check if sentence contains <image>
                # Count how many <image> tokens appear in this sentence (typically should be 1)
                num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))

                # If exactly one image token and it's NOT at the beginning, move it to the start
                if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                    # Remove the <image> token from its current position
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                    # Place <image> at the beginning with a newline separator
                    sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                    sentence["value"] = sentence["value"].strip()

                    # Optional: wrap with <Image> tags for certain model configurations
                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")

                # EXPAND THE PLACEHOLDER TOKEN INTO ACTUAL PATCH TOKENS
                # Create the replacement token sequence:
                # - Start with <im_start> marker
                # - Add (image_token_num - 2) repeated <im_patch> tokens
                #   (e.g., DINOv2 has 256 tokens total: 1 start + 254 patches + 1 end)
                # - End with <im_end> marker
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * (vision_config.image_token_num - 2)
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN

                # Replace the symbolic <image> token with the expanded token sequence
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

                # Clean up any noise from the dataset
                sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

                # Increment counter to indicate we processed multimodal data
                multimodal_count += 1

            # Handle video tokens
            elif DEFAULT_VIDEO_TOKEN in sentence["value"]:  # Check if sentence contains <video>
                # Count how many <video> tokens appear in this sentence
                num_vid = len(re.findall(DEFAULT_VIDEO_TOKEN, sentence["value"]))

                # If exactly one video token and it's NOT at the beginning, move it to the start
                if num_vid == 1 and DEFAULT_VIDEO_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_VIDEO_TOKEN):
                    # Remove the <video> token from its current position
                    sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, "").strip()
                    # Place <video> at the beginning with a newline separator
                    sentence["value"] = DEFAULT_VIDEO_TOKEN + "\n" + sentence["value"]
                    sentence["value"] = sentence["value"].strip()

                # EXPAND VIDEO TOKENS WITH TEMPORAL INFORMATION
                replace_token = ""

                # Check if model uses both slow and fast temporal tokens (some architectures do)
                if vision_config.slow_token:
                    # Fast tokens: <vid_start> + (fast_token_num * fast_frame_num - 2) patches + <vid_end>
                    replace_token += DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * (vision_config.fast_token_num * vision_config.fast_frame_num - 2) + DEFAULT_VID_END_TOKEN
                    # Slow tokens: <slow_vid_start> + (slow_token_num * slow_frame_num - 2) patches + <slow_vid_end>
                    replace_token += DEFAULT_SLOW_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * (vision_config.slow_token_num * vision_config.slow_frame_num - 2) + DEFAULT_SLOW_VID_END_TOKEN
                else:
                    # Only slow tokens (temporal encoding): <vid_start> + patches + <vid_end>
                    replace_token += DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * (vision_config.slow_token_num * vision_config.slow_frame_num - 2) + DEFAULT_VID_END_TOKEN

                # Replace the symbolic <video> token with the expanded temporal token sequence
                sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, replace_token)

                # Increment counter to indicate we processed multimodal data
                multimodal_count += 1

    # ============ RETURN RESULT ============
    # If we found any multimodal elements, return the modified sources
    # If only text (no images/videos), return None to signal pure language modeling
    if multimodal_count > 0:
        return sources
    else:
        return None

def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama_2_pure_lm(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """
    Preprocessing for pure language modeling (no conversational format).
    - Takes only the GPT/assistant text with <image> token
    - All tokens are trainable (no masking)
    - ALL examples are processed (no skipping)
    """
    input_ids_list = []
    labels_list = []

    for i, source in enumerate(sources):
        # Get text from assistant message
        text = None

        if len(source) >= 2 and source[1]["from"] == "gpt":
            text = source[1].get("value", "").strip()

        # If no text, use placeholder to maintain batch consistency
        if not text:
            text = "[EMPTY]"

        # Add <image> token
        prompt = f"<image>\n{text}"

        # Tokenize using tokenizer_image_token for consistency
        token_ids = tokenizer_image_token(prompt, tokenizer)
        # Convert to tensor (tokenizer_image_token returns a list)
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        input_ids_list.append(token_ids)
        # For pure LM, all tokens are trainable (no masking of instructions)
        labels_list.append(token_ids.clone())

    # Stack all tensors into a batch tensor
    input_ids = torch.stack(input_ids_list, dim=0) if len(input_ids_list) > 1 else input_ids_list[0].unsqueeze(0)
    labels = torch.stack(labels_list, dim=0) if len(labels_list) > 1 else labels_list[0].unsqueeze(0)

    return dict(input_ids=input_ids, labels=labels)


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    for conversation, target in zip(conversations, targets):
        total_len: int = int(target.ne(tokenizer.pad_token_id).sum())

        rounds: List[str] = conversation.split(conv.sep)
        re_rounds = []
        for conv_idx in range(0, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))

        cur_len = 1  # Ignore <bos>
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Re-append sep because split on this
            # Now "".join(parts)==rou

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  # Ignore <bos>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # Ignore <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # Ignore <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # Ignore <bos>

            round_len += 2  # sep: <end_of_turn>\n takes 2 tokens
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"warning: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        # Skip this sample if no messages remain after filtering
        if len(source) == 0:
            continue

        # Filter out empty messages
        source = [msg for msg in source if msg.get("value", "").strip()]
        if len(source) == 0:
            continue

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)

            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id



        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        # Skip this sample if no messages remain after filtering
        if len(source) == 0:
            continue

        # Filter out empty messages
        source = [msg for msg in source if msg.get("value", "").strip()]
        if len(source) == 0:
            continue

        input_id, target = [], []

        # System message
        input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)

            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id



        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX

        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(input_ids=input_ids, labels=targets)


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        if len(source) == 0:
            continue

        # Replace empty human messages with default prompt
        if source[0]["from"] == "human" and not source[0].get("value", "").strip():
            source[0]["value"] = "Continue the following text:"

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        if len(source) == 0:
            continue

        # Replace empty human messages with default prompt
        if source[0]["from"] == "human" and not source[0].get("value", "").strip():
            source[0]["value"] = "Continue the following text:"

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f"(#turns={len(re_rounds)} ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        # Use conversational llama_2 preprocessing with roles and instruction masking
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama_v3":
        # rank0_print(f">>> FORCING preprocess_llama3 (conversation version: {conversation_lib.default_conversation.version})")
        return preprocess_llama3(sources, tokenizer, has_image=has_image)

    rank0_print(f"FAILED TO FIND CONVERSATION TYPE: fallback to generic preprocess (conversation version: {conversation_lib.default_conversation.version})")
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

import concurrent.futures
import time

def run_with_timeout(func, timeout, i):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交任务并传递参数
        future = executor.submit(func, i)
        try:
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            print(f'get_item({i}) time out')
            raise NotImplementedError()

ignore_samples = []

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments, vision_config):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.list_data_dict = []
        self.vision_config = vision_config
        self.data_folders = {}

        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)

                    source = dataset.get("source", None)
                    for sample in cur_data_dict:
                        sample["source"] = source
                    data_folder = dataset["data_folder"]
                    if source and data_folder:
                        self.data_folders[source] = data_folder
        # Load single JSON file
        else:
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading Dataset from data_path: {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from data_path: {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

        # Precalculate black image embedding for text-only samples (optimization)
        self.black_image_cached = None
        try:
            processor = data_args.image_processor
            if hasattr(processor, 'crop_size'):
                height = processor.crop_size.get('height', 224)
                width = processor.crop_size.get('width', 224)
            else:
                height = width = getattr(vision_config, 'frame_size', 224)

            # Create and preprocess black image once
            black_image = Image.new('RGB', (width, height), (0, 0, 0))
            self.black_image_cached = processor.preprocess(black_image, return_tensors="pt")["pixel_values"]
            self.black_image_size = black_image.size
            rank0_print(f"Precalculated black image: shape={self.black_image_cached.shape}, size={self.black_image_size}")
        except Exception as e:
            rank0_print(f"Warning: Could not precalculate black image: {e}")
            self.black_image_cached = None

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len)
            else:
                length_list.append(-cur_len)
        return length_list

    def process_image(self, image_file, source=None,overwrite_image_aspect_ratio=None):
        image_folder = self.data_folders.get(source,self.data_args.image_folder)
        processor = self.data_args.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn

        image_size = image.size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            print(f'error sample {i}')
            raise e

    def run_with_timeout(self, func, timeout, i):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the task and pass parameters
            future = executor.submit(func, i)
            try:
                result = future.result(timeout=timeout)
                return result
            except concurrent.futures.TimeoutError:
                print(f'get_item({i}) time out {self.list_data_dict[i]}')
                raise NotImplementedError()

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        # Gets the i-th item. If the sample is in the ignore list, skip to the next one.
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]

        if "image" in sources[0]:
            source_file = self.list_data_dict[i]["image"]
        elif "video" in sources[0]:
            source_file = self.list_data_dict[i]["video"]
        else:
            # print('not mukltimodel')
            source_file = ''

        ignore_flag = False
        for s in ignore_samples:
            if s in source_file:
                ignore_flag = True

        if ignore_flag:
            i = i + 1
            return self._get_item(i)
        # Wait 4 minutes maximum to get the item, then cancel the task
        output = self.run_with_timeout(self.__get_item, 240, i)
        return output

    def __get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]

        # START: modification (Llama 3 role conversion)
        for source_item in sources:
            if "conversations" in source_item:
                for conv in source_item["conversations"]:
                    if "role" in conv and "content" in conv:
                        if conv["role"] == "user":
                            conv["from"] = "human"
                        elif conv["role"] == "assistant":
                            conv["from"] = "gpt"
                        else:
                            conv["from"] = conv["role"]
                        conv["value"] = conv["content"]
                        del conv["role"]
                        del conv["content"]
        # END: modification

        assert len(sources) == 1, "Don't know why it is wrapped to a list"

        processor = self.data_args.image_processor

        # 1. IMAGE CASE
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image_source = self.list_data_dict[i].get("source",None)
            if type(image_file) is list:
                image = [self.process_image(f,image_source) for f in image_file]
                # Handling multi images
                # overwrite to process with simple pad
                if len(image_file) > 1:
                    image = [self.process_image(f,image_source, "pad") for f in image_file]
                    image = [[im[0], im[1], "image"] for im in image]
            else:
                image = [self.process_image(image_file,image_source)]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args, self.vision_config)

        elif "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_source = self.list_data_dict[i].get("source", None)
            video_folder = self.data_folders.get(video_source, self.data_args.video_folder)
            video_file = os.path.join(video_folder, video_file)
            suffix = video_file.split(".")[-1]
            if not os.path.exists(video_file):
                print("File {} not exist!".format(video_file))

            try:
                if "shareVideoGPTV" in video_file:
                    frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
                    frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

                    # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
                    if self.data_args.force_sample:
                        num_frames_to_sample = self.data_args.frames_upbound
                    else:
                        num_frames_to_sample = 10

                    avg_fps = 2

                    total_frames = len(frame_files)
                    sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)


                    frame_time = [i/2 for i in sampled_indices]
                    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

                    video_time = total_frames / avg_fps

                    # Read and store the sampled frames
                    video = []
                    for idx in sampled_indices:
                        frame_path = frame_files[idx]
                        try:
                            with Image.open(frame_path) as img:
                                frame = img.convert("RGB")
                                video.append(frame)
                        except IOError:
                            print(f"Failed to read frame at path: {frame_path}")
                else:
                    meta = self.list_data_dict[i].get("meta", None)
                    if meta is not None:
                        split = meta.get("split", None)
                    else:
                        split = None
                    video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file, self.data_args, split=split)

                processor = self.data_args.image_processor
                image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                if self.data_args.add_time_instruction:
                    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                    sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
                image = [(image, video[0].size, "video")]
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args, self.vision_config)
                # print(sources)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Failed to read video file: {video_file}")
                return self._get_item(i + 1)

        # Handle text-only samples: reuse precalculated black image
        else:
            # Reuse precalculated black image if available
            if self.black_image_cached is not None:
                image = [(self.black_image_cached, self.black_image_size, "image")]
            else:
                # Fallback: create black image on-the-fly if precalculation failed
                processor = self.data_args.image_processor
                if hasattr(processor, 'crop_size'):
                    height = processor.crop_size.get('height', 224)
                    width = processor.crop_size.get('width', 224)
                else:
                    height = width = getattr(self.vision_config, 'frame_size', 224)

                black_image = Image.new('RGB', (width, height), (0, 0, 0))
                image = processor.preprocess(black_image, return_tensors="pt")["pixel_values"]
                image = [(image, black_image.size, "image")]

            # Insert DEFAULT_IMAGE_TOKEN at the beginning of the first conversation
            if sources and len(sources[0].get("conversations", [])) > 0:
                first_msg = sources[0]["conversations"][0]
                if "value" in first_msg and DEFAULT_IMAGE_TOKEN not in first_msg["value"]:
                    first_msg["value"] = DEFAULT_IMAGE_TOKEN + "\n" + first_msg["value"]

            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args, self.vision_config)

        if sources is None:
            num_base_retries = 3
            for attempt_idx in range(num_base_retries):
                try:
                    next_index = min(i + 1, len(self.list_data_dict) - 1)
                    sample = self._get_item(next_index)
                    return sample
                except Exception as e:
                    pass

        # --- FINAL PROCESSING ---
        # has_image is True when we have an image (real or black placeholder)
        has_image = (image is not None)
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        if isinstance(i, int):
            input_ids = data_dict["input_ids"][0]
            labels = data_dict["labels"][0]
            # Ensure tensors (not lists) are returned
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            data_dict = dict(input_ids=input_ids, labels=labels)


        data_dict["image"] = image

        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = self.list_data_dict[i].get("id", i)
        data_dict["variables"] = self.list_data_dict[i].get("variables", {
            "temporal_input_locations": [], "temporal_output_locations": [],
            "spatial_height_input_locations": [], "spatial_height_output_locations": [],
            "spatial_width_input_locations": [], "spatial_width_output_locations": []
        })

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

        # Initialize image-related batch components to maintain batch size consistency
        images = []
        image_sizes = []
        modalities = []

        for instance in instances:
            if "image" in instance and instance["image"] is not None:
                images.extend([im[0] for im in instance["image"]])
                image_sizes.extend([im[1] for im in instance["image"]])
                modalities.extend([im[2] for im in instance["image"]])

        # Assign to batch - use None if no images (will be handled by model)
        # Only include actual image data, not None placeholders
        batch["images"] = images if images else None  # None if all text-only samples
        batch["image_sizes"] = image_sizes if image_sizes else None
        batch["modalities"] = modalities if modalities else None

        variables = []
        for instance in instances:
            if "variables" in instance:
                variables.append(instance['variables'])
            else:
                variables.append({
                    "temporal_input_locations": [],
                    "temporal_output_locations": [],
                    "spatial_height_input_locations": [],
                    "spatial_height_output_locations": [],
                    "spatial_width_input_locations": [],
                    "spatial_width_output_locations": []
                })
        batch["variables"] = variables

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, vision_config) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args, vision_config=vision_config)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def get_model(model_args, training_args, bnb_model_from_pretrained_args):
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    cfg_pretrained = None

    overwrite_config = {}

    # Load config when we need to overwrite any parameter
    needs_config_overwrite = any(
        [
            model_args.rope_scaling_factor is not None,
            model_args.rope_scaling_type is not None,
            model_args.mm_spatial_pool_stride is not None,
            model_args.mm_spatial_pool_out_channels is not None,
            model_args.mm_spatial_pool_mode is not None,
            model_args.mm_resampler_type is not None,
        ]
    )

    # Always load config for mm_perceiver_latents (they have defaults, so check if specified)
    # We need to overwrite when user explicitly passes them via command line
    if needs_config_overwrite or True:  # Always load config to support mm_perceiver_latents
        cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if model_args.use_downsample_image is not None:
        overwrite_config["use_downsample_image"] = model_args.use_downsample_image

    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )
        # overwrite_config["max_sequence_length"] = model_args.max_sequence_length
        # overwrite_config["tokenizer_model_max_length"] = model_args.tokenizer_model_max_length

    if model_args.mm_spatial_pool_stride is not None and model_args.mm_spatial_pool_out_channels is not None and model_args.mm_spatial_pool_mode is not None and model_args.mm_resampler_type is not None:
        overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = model_args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.mm_spatial_pool_mode is not None:
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    # ✅ CRITICAL FIX: ALWAYS override mm_perceiver_latents from model_args
    # These parameters have defaults (81, 9) but users can override via CLI (--mm_perceiver_latents 256)
    # We must ALWAYS set them in overwrite_config to ensure CLI args take precedence
    overwrite_config["mm_perceiver_latents"] = model_args.mm_perceiver_latents
    overwrite_config["mm_perceiver_latents_fast"] = model_args.mm_perceiver_latents_fast
    overwrite_config["mm_perceiver_depth"] = model_args.mm_perceiver_depth
    overwrite_config["mm_perceiver_ff_mult"] = model_args.mm_perceiver_ff_mult

    if overwrite_config:
        assert cfg_pretrained is not None, "cfg_pretrained is None"

        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)

        customized_kwargs["config"] = cfg_pretrained

    if model_args.model_class_name is not None:
        actual_model_class_name = f"{model_args.model_class_name}ForCausalLM"
        model_class = getattr(transformers, actual_model_class_name)
        rank0_print(f"Using model class {model_class} from {model_args.model_class_name}")
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    elif model_args.vision_tower is not None:
        if "mixtral" in model_args.model_name_or_path.lower():
            model = LlavaMixtralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
            from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
        elif "mistral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
            model = LlavaMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif (
            "wizardlm-2" in model_args.model_name_or_path.lower()
            or "vicuna" in model_args.model_name_or_path.lower()
            or "llama" in model_args.model_name_or_path.lower()
            or "yi" in model_args.model_name_or_path.lower()
            or "nous-hermes" in model_args.model_name_or_path.lower()
            and "wizard-2" in model_args.model_name_or_path.lower()
        ):
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif "qwen" in model_args.model_name_or_path.lower():
            if "moe" in model_args.model_name_or_path.lower() or "A14B" in model_args.model_name_or_path:
                model = LlavaQwenMoeForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
                from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

                deepspeed.utils.set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])
            else:
                model = LlavaQwenForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
        elif "gemma" in model_args.model_name_or_path.lower():
            model = LlavaGemmaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        else:
            raise ValueError(f"Unknown model class {model_args}")
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    return model


def train(attn_implementation=None):
    """training entry point.

    training pipeline:
    1. Parse command line arguments and configuration
    2. Initialize model, tokenizer, and vision components
    3. Set up data loaders and preprocessing
    4. Execute training loop with optional checkpointing
    5. Save final model and state
    """
    # Use local rank that is globally defined
    global local_rank

    # Parse command-line arguments into dataclass objects.
    # Each dataclass (ModelArguments, DataArguments, TrainingArguments) groups related hyperparameters.
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Extract and apply CLI overrides on model arguments.
    import sys
    cli_mm_perceiver_latents = None
    cli_mm_perceiver_latents_fast = None
    for i, arg in enumerate(sys.argv):
        if arg == "--mm_perceiver_latents" and i + 1 < len(sys.argv):
            cli_mm_perceiver_latents = int(sys.argv[i + 1])
        elif arg == "--mm_perceiver_latents_fast" and i + 1 < len(sys.argv):
            cli_mm_perceiver_latents_fast = int(sys.argv[i + 1])

    if cli_mm_perceiver_latents is not None:
        model_args.mm_perceiver_latents = cli_mm_perceiver_latents
        rank0_print(f"CLI override: mm_perceiver_latents = {cli_mm_perceiver_latents}")
    if cli_mm_perceiver_latents_fast is not None:
        model_args.mm_perceiver_latents_fast = cli_mm_perceiver_latents_fast
        rank0_print(f"CLI override: mm_perceiver_latents_fast = {cli_mm_perceiver_latents_fast}")

    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")

    # DEBUG: Show mm_perceiver_latents from CLI before get_model
    rank0_print(f"\n{'='*60}")
    rank0_print(f"BEFORE get_model() SHOW mm_perceiver_latents from CLI:")
    rank0_print(f">> model_args.mm_perceiver_latents = {model_args.mm_perceiver_latents}")
    rank0_print(f">> model_args.mm_perceiver_latents_fast = {model_args.mm_perceiver_latents_fast}")
    rank0_print(f"{'='*60}\n")

    local_rank = training_args.local_rank

    # Determine the compute dtype for mixed precision training.
    # Priority: fp16 > bf16 > fp32. This could affect numerical stability and memory usage.
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    # Configure quantization settings if 4-bit or 8-bit quantization is requested.
    # This reduces model size and memory footprint at the cost of precision.
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    # Load the model with optional quantization and device mapping.
    # This creates the full multimodal architecture (LLM + vision tower + projector + resampler)
    model = get_model(model_args, training_args, bnb_model_from_pretrained_args)

    # Disable KV cache during training to ensure gradients flow properly.
    model.config.use_cache = False

    # Configure RoPE (Rotary Position Embeddings) scaling if specified.
    # This allows the model to handle sequences longer than training sequences.
    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    # Enable gradient checkpointing to reduce memory usage during training.
    # This technique trades compute for memory by recomputing intermediate activations during backprop.
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            # Fallback: manually register a forward hook to enable gradients on embeddings.
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, PeftModel

        if training_args.stage == "3" and training_args.load_lora_path != "":

            model.init_special_embeddings()
            model.initialize_embedings(6, model.config.vocab_size+6)

            def load_lora(model, lora_path):
                non_lora_trainables_path = os.path.join(lora_path, 'non_lora_trainables.bin')
                if os.path.exists(non_lora_trainables_path):
                    non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cpu')
                    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
                    if any(k.startswith('model.model.') for k in non_lora_trainables):
                        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                    info = model.load_state_dict(non_lora_trainables, strict=False)
                    rank0_print(f"Load non_lora_trainables unexpected_keys: {info[1]}")
                rank0_print('Loading pretrained LoRA weights...')
                model = PeftModel.from_pretrained(model, lora_path)
                model = model.merge_and_unload()
                rank0_print('Have merged pretrained LoRA weights...')
                return model

            model = load_lora(model, training_args.load_lora_path)

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    # Load the tokenizer with model-specific configuration.
    # Different model families use different padding sides and tokenization strategies.
    # Mistral and derivatives use left padding for better batching in generation.
    if "mistral" in model_args.model_name_or_path.lower() or "mixtral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="left")
    elif "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")
    elif (
        "wizardlm-2" in model_args.model_name_or_path.lower()
        or "vicuna" in model_args.model_name_or_path.lower()
        or "llama" in model_args.model_name_or_path.lower()
        or "yi" in model_args.model_name_or_path.lower()
        or "nous-hermes" in model_args.model_name_or_path.lower()
        and "wizard-2" in model_args.model_name_or_path.lower()
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        # Sono tokens have pad token undefined, we define pad token as eos token
        if tokenizer.pad_token is None:
            rank0_print("ADDED: setting pad_token to eos_token")
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))

            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        rank0_print(f"Tokenizer Pad Token ID: {tokenizer.pad_token_id}")

    rank0_print(f"Prompt version: {model_args.version}")
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # Initialize multimodal components (vision encoder, projector, resampler) if vision tower is specified.
    if model_args.vision_tower is not None:
        # Initialize vision modules and retrieve the vision configuration.
        # This also adds special tokens to the tokenizer for image/video delimiters.
        vision_config = model.get_model().initialize_vision_modules(model_args=model_args, tokenizer=tokenizer, fsdp=training_args.fsdp)
        vision_config.fast_token_num = model_args.mm_perceiver_latents_fast
        vision_config.slow_token_num = model_args.mm_perceiver_latents

        # Move vision tower to the correct device and dtype.
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        # Precompute and cache the projected features for a black (zero) image.
        # This accelerates training when placeholder images are used in the model.

        try:
            from PIL import Image
            if hasattr(data_args, 'image_processor') and data_args.image_processor is not None:
                proc = data_args.image_processor
            else:
                proc = vision_tower.image_processor
        except Exception:
            pass

        if training_args.gradient_checkpointing:
            if not model_args.unfreeze_mm_vision_tower:
                vision_tower._no_grad = True
                # Disable gradient checkpointing for frozen vision tower to avoid warnings
                if hasattr(vision_tower, 'vision_tower') and hasattr(vision_tower.vision_tower, 'gradient_checkpointing_disable'):
                    vision_tower.vision_tower.gradient_checkpointing_disable()
                rank0_print("ADDED: Vision tower is frozen, so it is marked as no_grad for checkpointing")

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_grid_pinpoints is not None:
            if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
                try:
                    patch_size = data_args.image_processor.size[0]
                except Exception as e:
                    patch_size = data_args.image_processor.size["shortest_edge"]

                assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
                # Use regex to extract the range from the input string
                matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
                range_start = tuple(map(int, matches[0]))
                range_end = tuple(map(int, matches[-1]))
                # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
                grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
                # Multiply all elements by patch_size
                data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
            elif isinstance(data_args.image_grid_pinpoints, str):
                data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.image_crop_resolution = data_args.image_crop_resolution
        model.config.image_split_resolution = data_args.image_split_resolution
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_newline_position = model_args.mm_newline_position
        model.config.add_faster_video = model_args.add_faster_video
        model.config.faster_token_stride = model_args.faster_token_stride
        model.config.add_time_instruction = data_args.add_time_instruction
        model.config.force_sample = data_args.force_sample
        model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride

        if model_args.mm_tunable_parts is None:
            model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
            model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
            if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler:
                model.requires_grad_(False)
            if model_args.tune_mm_mlp_adapter:
                mm_projector = model.get_model().mm_projector
                for p in mm_projector.parameters():
                    p.requires_grad = True
            if model_args.tune_mm_vision_resampler:
                vision_resampler = model.get_model().vision_resampler
                for p in vision_resampler.parameters():
                    p.requires_grad = True

            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            if training_args.freeze_mm_mlp_adapter:
                mm_projector = model.get_model().mm_projector
                for p in mm_projector.parameters():
                    p.requires_grad = False

            model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
            if training_args.freeze_mm_vision_resampler:
                vision_resampler = model.get_model().vision_resampler
                for p in vision_resampler.parameters():
                    p.requires_grad = False

            model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
            if model_args.unfreeze_mm_vision_tower:
                vision_tower.requires_grad_(True)
            else:
                vision_tower.requires_grad_(False)

        else:
            # Apply selective gradient enabling based on mm_tunable_parts specification.
            # This allows fine-grained control over which components are trainable.
            rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
            model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts

            # Debug: check what attributes exist (commented out for cleaner output)
            # model_obj = model.get_model()
            # rank0_print(f"DEBUG: model.get_model() attributes with 'resampler' or 'projector':")
            # for name in dir(model_obj):
            #     if 'resampler' in name.lower() or 'projector' in name.lower():
            #         attr = getattr(model_obj, name, None)
            #         if hasattr(attr, 'parameters'):
            #             param_count = sum(1 for _ in attr.parameters())
            #             rank0_print(f"  - {name}: {type(attr).__name__} with {param_count} params")
            #         else:
            #             rank0_print(f"  - {name}: {type(attr).__name__}")

            # Debug: check what's in model.named_parameters() (commented out for cleaner output)
            # rank0_print(f"DEBUG: Checking model.named_parameters() for vision_resampler/mm_projector:")
            # found_resampler = False
            # found_projector = False
            # for param_name, param in model.named_parameters():
            #     if 'vision_resampler' in param_name or 'resampler' in param_name:
            #         found_resampler = True
            #         rank0_print(f"  Found resampler param: {param_name}, requires_grad={param.requires_grad}")
            #         break
            # for param_name, param in model.named_parameters():
            #     if 'mm_projector' in param_name or ('projector' in param_name and 'vision' not in param_name):
            #         found_projector = True
            #         rank0_print(f"  Found projector param: {param_name}, requires_grad={param.requires_grad}")
            #         break
            # rank0_print(f"  Resampler found: {found_resampler}, Projector found: {found_projector}")

            # Freeze all parameters by default, then selectively unfreeze specified components.
            model.requires_grad_(False)
            vision_tower.requires_grad_(False)
            mm_projector = model.get_model().mm_projector
            if mm_projector is not None:
                mm_projector.requires_grad_(False)
            vision_resampler = model.get_model().vision_resampler
            if vision_resampler is not None:
                vision_resampler.requires_grad_(False)

            tunable_parts = model_args.mm_tunable_parts.split(",")
            if "mm_mlp_adapter" in tunable_parts:
                mm_projector = model.get_model().mm_projector
                if mm_projector is not None:
                    for p in mm_projector.parameters():
                        p.requires_grad = True
            if "mm_vision_resampler" in tunable_parts:
                vision_resampler = model.get_model().vision_resampler
                if vision_resampler is not None:
                    for p in vision_resampler.parameters():
                        p.requires_grad = True
            if "mm_vision_tower" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" in name:
                        param.requires_grad_(True)
            if "mm_language_model" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name:
                        param.requires_grad_(True)
            if training_args.lora_enable:
                for name,param in model.named_parameters():
                    if "lora" in name.lower():
                        param.requires_grad_(True)
            if get_rank() == 0:
                rank0_print("\n" + "="*60)
                rank0_print("🔍 GRADIENT CHECK")
                rank0_print("="*60)

                trainable_params = []
                frozen_params = []

                for name, param in model.named_parameters():
                    if param.requires_grad:
                        trainable_params.append((name, param.numel()))
                    else:
                        frozen_params.append((name, param.numel()))

                total_trainable = sum(p[1] for p in trainable_params)
                total_frozen = sum(p[1] for p in frozen_params)

                rank0_print(f"✅ Trainable: {len(trainable_params)} params ({total_trainable/1e6:.2f}M)")
                rank0_print(f"❄️  Frozen: {len(frozen_params)} params ({total_frozen/1e6:.2f}M)")

                rank0_print("\nFirst 5 trainable:")
                for name, _ in trainable_params[:5]:
                    rank0_print(f"  - {name}")

                # Check if vision_resampler or mm_projector are in trainable
                vision_resampler_trainable = [p for p in trainable_params if 'vision_resampler' in p[0]]
                mm_projector_trainable = [p for p in trainable_params if 'mm_projector' in p[0]]
                rank0_print(f"\n🔍 Vision Resampler in Trainable: {len(vision_resampler_trainable)} params")
                rank0_print(f"🔍 MM Projector in Trainable: {len(mm_projector_trainable)} params")

                rank0_print("\nFirst 5 frozen:")
                for name, _ in frozen_params[:5]:
                    rank0_print(f"  - {name}")

                rank0_print("="*60 + "\n")

        total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
        trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
        rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")
        if training_args.bits in [4, 8]:
            mm_projector = model.get_model().mm_projector
            if mm_projector is not None:
                mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

    model.model.config.max_frame = 100
    if getattr(model.model.config, "use_downsample_image", False):
        vision_config.image_token_num = 9 * 9

    # Configure the number of image placeholder tokens in the input text.
    # This determines how many im_patch tokens will replace each <image> placeholder during preprocessing.
    # We need to determine the ACTUAL number of tokens produced by vision_tower + resampler pipeline.
    try:
        # For fast_slow_resampler with DINOv2, the output is always 256 tokens (16x16 grid)
        if model_args.mm_resampler_type == "fast_slow_resampler":
            rank0_print("🔍 Detected fast_slow_resampler - using known output size")
            # Fast_slow_resampler with DINOv2 produces 256 tokens (16x16)
            vision_config.image_token_num = 256
            rank0_print(f"✅ image_token_num set for fast_slow_resampler: {vision_config.image_token_num}")
        elif hasattr(model_args, "mm_perceiver_latents") and model_args.mm_perceiver_latents and int(model_args.mm_perceiver_latents) > 0:
            # For other resamplers that actually compress, use mm_perceiver_latents
            vision_config.image_token_num = int(model_args.mm_perceiver_latents)
            rank0_print(f"image_token_num set from mm_perceiver_latents: {vision_config.image_token_num}")
        else:
            # Fallback: use vision tower output size
            if hasattr(vision_tower, "num_patches_per_side"):
                patches_side = int(vision_tower.num_patches_per_side)
                if patches_side > 0:
                    vision_config.image_token_num = patches_side * patches_side
                    rank0_print(f"✅ image_token_num from vision_tower.num_patches_per_side: {patches_side}x{patches_side} -> {vision_config.image_token_num}")
            elif hasattr(vision_tower, "config") and hasattr(vision_tower.config, "image_size") and hasattr(vision_tower.config, "patch_size"):
                patch_size = getattr(vision_tower.config, "patch_size", 14)
                image_size = getattr(vision_tower.config, "image_size", 518)
                patches_side = image_size // patch_size
                vision_config.image_token_num = patches_side * patches_side
                rank0_print(f"✅ image_token_num from config (image_size={image_size}, patch_size={patch_size}): {patches_side}x{patches_side} -> {vision_config.image_token_num}")
    except Exception as e:
        rank0_print(f"⚠️  Exception while determining image_token_num: {e}")
        import traceback
        traceback.print_exc()

    rank0_print(f"Final vision_config.image_token_num: {vision_config.image_token_num}")

    print("tokenizer.vocab_size", tokenizer.vocab_size)

    # Execute stage-specific initialization and token embedding setup.
    # Stage 1: Add image and video special tokens; resize embeddings.
    # Stage 2: Add temporal and spatial tokens for advanced multimodal understanding.
    # Stage 3: Full setup with all token types and maximum trainable parameters.

    if training_args.stage == "1":
        # Stage 1: Basic multimodal setup with image and video tokens.
        origin_token_num = len(tokenizer)

        num_image_tokens = model.initialize_image_tokenizer(tokenizer=tokenizer)
        num_video_tokens = model.initialize_video_tokenizer(tokenizer=tokenizer)

        new_token_num = num_image_tokens + num_video_tokens
        cur_token_num = origin_token_num + new_token_num

        assert len(tokenizer) == cur_token_num, \
            f"Tokenizer size mismatch: {len(tokenizer)} vs {cur_token_num}"

        model.initialize_embedings(new_token_num, cur_token_num)

        rank0_print("\n" + "="*60)
        rank0_print("EMBEDDINGS INITIALIZATION CHECK")
        rank0_print("="*60)

        embed_tokens = model.get_input_embeddings()
        lm_head = model.get_output_embeddings()

        rank0_print(f"Input embeddings shape: {embed_tokens.weight.shape}")
        rank0_print(f"Output embeddings shape: {lm_head.weight.shape}")
        rank0_print(f"Vocab size: {model.config.vocab_size}")
        rank0_print(f"Tokenizer size: {len(tokenizer)}")

        if embed_tokens.weight.data_ptr() == lm_head.weight.data_ptr():
            rank0_print("⚠️  WARNING: Input and output embeddings share memory!")
            rank0_print("   This may cause gradient issues.")
        else:
            rank0_print("✅ Input and output embeddings are separate.")

        if torch.isnan(embed_tokens.weight).any():
            raise ValueError("NaN detected in input embeddings!")
        if torch.isnan(lm_head.weight).any():
            raise ValueError("NaN detected in output embeddings!")

        rank0_print("="*60 + "\n")

    elif training_args.stage == "2":
        origin_token_num = len(tokenizer)

        num_image_tokens = model.initialize_image_tokenizer(tokenizer=tokenizer)
        num_video_tokens = model.initialize_video_tokenizer(tokenizer=tokenizer)

        new_token_num = model.initialize_spatial_temporal_tokens(
            tokenizer=tokenizer,
            num_spatial_tokens=vision_config.spatial_token_num,
            num_temporal_tokens=vision_config.fast_frame_num
        )

        model.model.init_special_embeddings()

        print(f"DEBUG: Tokenizer len: {len(tokenizer)}")
        print(f"DEBUG: Model embeddings shape before: {model.get_input_embeddings().weight.shape}")

        model.resize_token_embeddings(len(tokenizer))

        print(f"DEBUG: Model embeddings shape after: {model.get_input_embeddings().weight.shape}")
        print(f"DEBUG: Model output head shape: {model.get_output_embeddings().weight.shape}")

        model.model.spatial_height_input_embeddings.weight.requires_grad = True
        model.model.spatial_height_output_embeddings.weight.requires_grad = True
        model.model.spatial_width_input_embeddings.weight.requires_grad = True
        model.model.spatial_width_output_embeddings.weight.requires_grad = True
        model.model.temporal_input_embeddings.weight.requires_grad = True
        model.model.temporal_output_embeddings.weight.requires_grad = True

    elif training_args.stage == "3":
        origin_token_num = len(tokenizer)

        num_image_tokens = model.initialize_image_tokenizer(tokenizer=tokenizer)
        num_video_tokens = model.initialize_video_tokenizer(tokenizer=tokenizer)

        model.initialize_spatial_temporal_tokens(tokenizer=tokenizer, num_spatial_tokens=vision_config.spatial_token_num, num_temporal_tokens=vision_config.fast_frame_num)

        # CRITICAL FIX: Also initialize special embeddings in stage 3!
        model.model.init_special_embeddings()

        for p in model.get_input_embeddings().parameters():
            p.requires_grad = True
        for p in model.get_output_embeddings().parameters():
            p.requires_grad = False
        model.model.spatial_height_input_embeddings.weight.requires_grad = True
        model.model.spatial_height_output_embeddings.weight.requires_grad = True
        model.model.spatial_width_input_embeddings.weight.requires_grad = True
        model.model.spatial_width_output_embeddings.weight.requires_grad = True
        model.model.temporal_input_embeddings.weight.requires_grad = True
        model.model.temporal_output_embeddings.weight.requires_grad = True

    if get_rank() == 0:
        os.makedirs(training_args.output_dir, exist_ok=True)
        trainable_params = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
        output_file = os.path.join(training_args.output_dir, 'trainable_params.txt')
        output_string = "Trainable Parameters:\n"
        for name, shape in trainable_params:
            output_string += f"{name}: {shape}\n"
        with open(output_file, 'w') as f:
            f.write(output_string)
        print(output_string)

    # Create data loaders for training and evaluation.
    # This applies all preprocessing: tokenization, image loading, label masking, etc.
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, vision_config=vision_config)

    # Setup BabyLM callback for word limit tracking if enabled.
    train_dataset = data_module['train_dataset']

    babylm_enabled = os.environ.get("BABYLM_ENABLED", "true").lower() == "true"
    callbacks_list = []

    if babylm_enabled:
        from llava.train.babylm_callback import BabyLMCheckpointCallback
        babylm_cb = BabyLMCheckpointCallback(
            tokenizer=tokenizer,
            dataset=train_dataset,
            target_word_limit=1_000_000_000
        )
        callbacks_list.append(babylm_cb)
        rank0_print("✅ BabyLM callback attivato")

        # ✅ DEBUG: Stampa il batch size reale
        # rank0_print(f"\n{'='*60}")
        # rank0_print("DEBUG BATCH SIZE:")
        # rank0_print(f"  per_device_train_batch_size: {training_args.per_device_train_batch_size}")
        # rank0_print(f"  gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
        # rank0_print(f"  world_size: {training_args.world_size}")
        # rank0_print(f"  Batch size effettivo calcolato: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size}")
        # rank0_print(f"{'='*60}\n")

    # Create the trainer with custom callbacks for BabyLM word limit tracking.
    trainer = LLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks_list,
        **data_module
    )

    # Execute the main training loop with optional checkpoint resumption.
    # The trainer handles:
    # - Forward pass through the multimodal model
    # - Gradient computation and optimization
    # - Checkpoint saving and evaluation
    # - Logging and callback invocation
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        # Resume training from the latest checkpoint if checkpoints exist.
        trainer.train(resume_from_checkpoint=False)
    else:
        # Start fresh training if no checkpoints are found.
        trainer.train()
    # Save trainer state (optimizer, scheduler, etc.) for potential resumption.
    trainer.save_state()

    # Re-enable KV caching for inference after training completes.
    model.config.use_cache = True

    # Save the final trained model with appropriate format based on training configuration.
    # If LoRA was used, save LoRA weights separately from base model.
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        # Save full model state dict in a format compatible with HuggingFace transformers.
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=360))

    train()
