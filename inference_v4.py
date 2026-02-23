#!/usr/bin/env python3
"""
inference_phase1.py - Chat & Test Script for Karla (Phase 1)
============================================================
Lädt das trainierte CTM-Modell und generiert Text autoregressiv.
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Importiere unsere Karla-Klassen
from karla.models.karla import create_karla
from karla.utils.config import KarlaConfig

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_p=0.9, device="cuda"):
    model.eval()
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"\n--- Generiere Antwort (CTM-Scale: {model.ctm_scale().item():.4f}) ---")
    print(prompt, end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Da unser CTM parallel über die ganze Sequenz läuft, 
            # füttern wir einfach die bisherige Sequenz ein (bis max 512)
            seq_len = input_ids.size(1)
            if seq_len > 512:
                input_ids = input_ids[:, -512:]
                
            # Forward pass (schnell dank Flash Attention)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids)
                
            # Wir nehmen nur die Logits des allerletzten Tokens
            next_token_logits = outputs.logits[0, -1, :]
            
            # Temperature
            next_token_logits = next_token_logits / temperature
            
            # Top-P (Nucleus Sampling)
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Decode and print live
            token_str = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(token_str, end="", flush=True)
            
            # Append to input
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Stop if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    print("\n\n--- Ende ---")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Pfad zur .pt Datei")
    parser.add_argument("--prompt", type=str, default="give code for a simple calculator. please use a tool, to be able to code in python. to use the tool write: <tool>. and if you are done with the python code write:</tool>. a script then detects this tool call of yours and adds the code into my IDE for testing.", help="Task")
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--temp", type=float, default=0.7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Lade Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
    
    print("Initialisiere Architektur...")
    config = KarlaConfig()
    model = create_karla(config).to(device)
    
    print(f"Lade Gewichte von {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    generate_text(model, tokenizer, args.prompt, args.max_tokens, args.temp, device=device)

if __name__ == "__main__":
    main()