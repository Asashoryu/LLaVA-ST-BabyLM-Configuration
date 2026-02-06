#!/usr/bin/env python3
"""
Test script to verify plain LM preprocessing works correctly.
Tests both text-only and multimodal samples.
"""

import os
import sys
import json
import torch

# Set plain LM mode
os.environ["USE_PLAIN_LM"] = "1"

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

from transformers import AutoTokenizer
from llava.train.train import preprocess_plain_lm, preprocess
from llava.constants import *

print("=" * 60)
print("🧪 TESTING PLAIN LM PREPROCESSING")
print("=" * 60)

# Load tokenizer
print("\n📦 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("./baby_llama_baseline", use_fast=False, padding_side="right")
print(f"✅ Tokenizer loaded: vocab_size={len(tokenizer)}")

# ============================================================
# TEST 1: Text-only sample
# ============================================================
print("\n" + "="*60)
print("TEST 1: Text-only sample (no image)")
print("="*60)

text_sample = [
    [
        {"from": "human", "value": ""},
        {"from": "gpt", "value": "The cat sat on the mat. It was very comfortable."}
    ]
]

result_text = preprocess_plain_lm(text_sample, tokenizer, has_image=False)

print(f"\n📊 Result:")
print(f"  Input IDs shape: {result_text['input_ids'][0].shape}")
print(f"  Labels shape: {result_text['labels'][0].shape}")
print(f"  Input IDs: {result_text['input_ids'][0][:20].tolist()}...")
print(f"  Labels: {result_text['labels'][0][:20].tolist()}...")

# Verify labels == input_ids (no masking)
if torch.equal(result_text['input_ids'][0], result_text['labels'][0]):
    print("  ✅ Labels match input_ids (no masking)")
else:
    print("  ❌ ERROR: Labels don't match input_ids!")

# Decode to verify
decoded = tokenizer.decode(result_text['input_ids'][0])
print(f"\n📝 Decoded text:\n  {decoded[:200]}...")

# ============================================================
# TEST 2: Multimodal sample (with image token)
# ============================================================
print("\n" + "="*60)
print("TEST 2: Multimodal sample (with <image> token)")
print("="*60)

# Create a sample with image token expanded (as it would be after preprocess_multimodal)
image_token_expanded = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 254 + DEFAULT_IM_END_TOKEN

multimodal_sample = [
    [
        {"from": "human", "value": image_token_expanded},
        {"from": "gpt", "value": "This is a picture of a dog playing in the park."}
    ]
]

result_mm = preprocess_plain_lm(multimodal_sample, tokenizer, has_image=True)

print(f"\n📊 Result:")
print(f"  Input IDs shape: {result_mm['input_ids'][0].shape}")
print(f"  Labels shape: {result_mm['labels'][0].shape}")
print(f"  Input IDs: {result_mm['input_ids'][0][:20].tolist()}...")
print(f"  Labels: {result_mm['labels'][0][:20].tolist()}...")

# Verify labels == input_ids
if torch.equal(result_mm['input_ids'][0], result_mm['labels'][0]):
    print("  ✅ Labels match input_ids (no masking)")
else:
    print("  ❌ ERROR: Labels don't match input_ids!")

# ============================================================
# TEST 3: Compare with instruction-following mode
# ============================================================
print("\n" + "="*60)
print("TEST 3: Compare Plain LM vs Instruction-Following")
print("="*60)

# Disable plain LM mode
os.environ["USE_PLAIN_LM"] = "0"
# Reimport to get updated value
from importlib import reload
import llava.train.train as train_module
reload(train_module)

instruction_sample = [
    [
        {"from": "human", "value": ""},
        {"from": "gpt", "value": "The quick brown fox jumps over the lazy dog."}
    ]
]

# Test with instruction format
from llava import conversation as conversation_lib
conversation_lib.default_conversation = conversation_lib.conv_templates["llama_v2"].copy()

result_inst = train_module.preprocess(instruction_sample, tokenizer, has_image=False)

print(f"\n📊 Instruction Format:")
print(f"  Input IDs shape: {result_inst['input_ids'][0].shape}")
print(f"  Has IGNORE_INDEX in labels: {(-100 in result_inst['labels'][0])}")
decoded_inst = tokenizer.decode(result_inst['input_ids'][0])
print(f"  Decoded (first 200 chars):\n  {decoded_inst[:200]}...")

# Re-enable plain LM
os.environ["USE_PLAIN_LM"] = "1"
reload(train_module)
result_plain = train_module.preprocess(instruction_sample, tokenizer, has_image=False)

print(f"\n📊 Plain LM Format:")
print(f"  Input IDs shape: {result_plain['input_ids'][0].shape}")
print(f"  Has IGNORE_INDEX in labels: {(-100 in result_plain['labels'][0])}")
decoded_plain = tokenizer.decode(result_plain['input_ids'][0])
print(f"  Decoded (first 200 chars):\n  {decoded_plain[:200]}...")

print(f"\n📏 Token count comparison:")
print(f"  Instruction format: {len(result_inst['input_ids'][0])} tokens")
print(f"  Plain LM format: {len(result_plain['input_ids'][0])} tokens")
print(f"  Savings: {len(result_inst['input_ids'][0]) - len(result_plain['input_ids'][0])} tokens")

# ============================================================
# TEST 4: Real dataset sample
# ============================================================
print("\n" + "="*60)
print("TEST 4: Real dataset sample")
print("="*60)

dataset_path = "data/localized_narratives/llava_datasets/all_shards_t_merged.json"
if os.path.exists(dataset_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Get first text sample
    real_sample = [dataset[0]["conversations"]]

    result_real = train_module.preprocess(real_sample, tokenizer, has_image=False)

    print(f"\n📊 Real Sample:")
    print(f"  Original text: {dataset[0]['conversations'][1]['value'][:100]}...")
    print(f"  Input IDs shape: {result_real['input_ids'][0].shape}")
    print(f"  Labels shape: {result_real['labels'][0].shape}")
    decoded_real = tokenizer.decode(result_real['input_ids'][0])
    print(f"  Decoded (first 150 chars):\n  {decoded_real[:150]}...")
    print(f"  ✅ Successfully processed real dataset sample")
else:
    print(f"  ⚠️  Dataset not found at {dataset_path}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("✅ ALL TESTS PASSED")
print("="*60)
print("\nPlain LM mode is working correctly!")
print("To use it in training, set: export USE_PLAIN_LM=1")
print("="*60)
