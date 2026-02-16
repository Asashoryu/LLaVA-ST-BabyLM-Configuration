#!/usr/bin/env python3
"""
Quick inference test for Italian video-language model
Tests the trained model with Italian prompts to evaluate training quality
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

# ============================================================================
# CONFIGURATION - Change these paths as needed
# ============================================================================
MODEL_PATH = "./babylm_italian/bambi_18_24_sept2025"
CHECKPOINT_PATH = "./output/ckpt_italian_1_vi_it"  # Final model location
TOKENIZER_PATH = "./babylm_italian/tokenizer_decoder"

# ============================================================================
# ITALIAN TEST PROMPTS
# ============================================================================
ITALIAN_PROMPTS = [
    "Il gatto nero",
    "Ci sono un signore",
    "Sopra la panca la capra campa, sotto la",
    "Devi toccare il fondo per",
    "Andiamo insieme",
    "L'usignolo sta",
    "Osservando il cielo",
    "Studiamo insieme la",
    "Dove",
    "Il sole splende nel",
]

# ============================================================================
# LOAD MODEL AND TOKENIZER
# ============================================================================
print(f"🔧 Loading Italian tokenizer from: {TOKENIZER_PATH}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

print(f"🔧 Loading base model from: {MODEL_PATH}")
print(f"🔧 Loading trained model from: {CHECKPOINT_PATH}")

# Load model - the checkpoint folder contains the full trained model
try:
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_PATH)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Model loaded on {'cuda' if torch.cuda.is_available() else 'cpu'}")

except Exception as e:
    print(f"⚠️  Error loading model: {e}")
    sys.exit(1)
# ============================================================================
# GENERATION SETTINGS
# ============================================================================
GEN_CONFIG = {
    "max_new_tokens": 20,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

# ============================================================================
# RUN INFERENCE
# ============================================================================
print("\n" + "="*70)
print("🎬 TESTING ITALIAN VIDEO-LANGUAGE MODEL")
print("="*70 + "\n")

device = "cuda" if torch.cuda.is_available() else "cpu"

for i, prompt in enumerate(ITALIAN_PROMPTS, 1):
    print(f"[{i}/{len(ITALIAN_PROMPTS)}] Prompt: {prompt}")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            **GEN_CONFIG
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated part (after prompt)
    completion = generated_text[len(prompt):].strip()

    print(f"   → {completion}")
    print()

print("="*70)
print("✓ Inference test completed!")
print("="*70)
