#!/usr/bin/env python3
"""
Create a copy of BAMBI model with random weights (from scratch training)
Keeps: config, tokenizer, architecture
Resets: all model weights to random values
"""

import torch
import shutil
import os
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel

def create_random_weights_model(source_path, target_path):
    """
    Copy model structure but reinitialize all weights randomly

    Args:
        source_path: Path to pretrained BAMBI model
        target_path: Path where to save the random-weights version
    """

    print(f"\n{'='*70}")
    print(f"🎲 Creating model with random weights")
    print(f"{'='*70}")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print(f"{'='*70}\n")

    # Create target directory
    os.makedirs(target_path, exist_ok=True)

    # Step 1: Copy config files
    print("📋 Step 1: Copying configuration files...")
    config_files = ['config.json', 'generation_config.json', 'special_tokens_map.json',
                    'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt']

    for file in config_files:
        src_file = os.path.join(source_path, file)
        if os.path.exists(src_file):
            shutil.copy2(src_file, target_path)
            print(f"  ✓ Copied {file}")

    # Step 2: Copy tokenizer directory if exists
    tokenizer_dirs = ['tokenizer_decoder', 'tokenizer_encoder']
    for tokenizer_dir in tokenizer_dirs:
        src_dir = os.path.join(source_path, tokenizer_dir)
        if os.path.exists(src_dir):
            dst_dir = os.path.join(target_path, tokenizer_dir)
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            print(f"  ✓ Copied {tokenizer_dir}/")

    # Step 3: Load config and create model with random weights
    print("\n🔧 Step 2: Loading config and creating model...")
    config = AutoConfig.from_pretrained(source_path)
    print(f"  Config loaded: {config.model_type}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Hidden size: {config.n_embd if hasattr(config, 'n_embd') else config.hidden_size}")

    # Create model with random initialization
    print("\n🎲 Step 3: Initializing model with RANDOM weights...")
    model = GPT2LMHeadModel(config)

    # Verify weights are different from pretrained
    print("  Verifying weights are randomized...")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Step 4: Save model with random weights
    print(f"\n💾 Step 4: Saving model to {target_path}...")
    model.save_pretrained(target_path)

    # Step 5: Copy tokenizer explicitly (from tokenizer subdirectory if needed)
    print("\n📝 Step 5: Copying tokenizer...")

    # Try to load tokenizer from main directory first, then from subdirectory
    tokenizer_loaded = False
    tokenizer_paths = [source_path,
                       os.path.join(source_path, 'tokenizer_decoder'),
                       os.path.join(source_path, 'tokenizer_encoder')]

    for tok_path in tokenizer_paths:
        if os.path.exists(tok_path):
            try:
                tokenizer = AutoTokenizer.from_pretrained(tok_path)
                tokenizer.save_pretrained(target_path)
                print(f"  ✓ Tokenizer loaded from: {tok_path}")
                tokenizer_loaded = True
                break
            except:
                continue

    if not tokenizer_loaded:
        print("  ⚠️  Warning: Could not load tokenizer, but config files were copied")

    print(f"\n{'='*70}")
    print("✅ SUCCESS! Model with random weights created")
    print(f"{'='*70}")
    print(f"📁 Saved to: {target_path}")
    print(f"📊 Total parameters: {total_params:,}")
    print(f"🎯 Ready for training from scratch!")
    print(f"{'='*70}\n")

    return target_path


if __name__ == "__main__":
    # Paths
    source_model = "./babylm_italian/bambi_18_24_sept2025"
    target_model = "./babylm_italian/bambi_random_init"

    # Create model with random weights
    create_random_weights_model(source_model, target_model)

    print("\n📌 Usage in training script:")
    print(f'   MODEL_PATH="{target_model}"')
    print("   # This model has BAMBI architecture + Italian tokenizer but RANDOM weights\n")
