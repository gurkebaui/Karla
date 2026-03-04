#!/usr/bin/env python3
"""
inference_v4.py - The Perfect Chat Interface
============================================
Nutzt das Qwen Chat-Template, Repetition Penalty und sauberes Sampling.
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from karla.models.karla import create_karla
from karla.utils.config import KarlaConfig

def generate_chat(model, tokenizer, user_input: str, max_new_tokens=300, temperature=0.7, top_p=0.9, repetition_penalty=1.1, device="cuda"):
    model.eval()
    
    # 1. Wir nutzen das offfizielle Chat-Template!
    messages = [{"role": "user", "content": user_input}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"\n--- Generiere Antwort (CTM-Scale: {model.ctm_scale().item():.4f}, L1-Scale: {model.l1_scale().item():.4f}) ---")
    
    generated_tokens =[]
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq_len = input_ids.size(1)
            # Begrenze auf die maximal trainierten 512 Tokens (sliding window)
            if seq_len > 512:
                model_input = input_ids[:, -512:]
            else:
                model_input = input_ids
                
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(model_input)
                
            next_token_logits = outputs.logits[0, -1, :].clone()
            
            # --- REPETITION PENALTY ---
            # Bestraft Tokens, die das Modell gerade erst generiert hat
            if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                for token in set(generated_tokens):
                    if next_token_logits[token] > 0:
                        next_token_logits[token] /= repetition_penalty
                    else:
                        next_token_logits[token] *= repetition_penalty
            
            # Temperature
            next_token_logits = next_token_logits / temperature
            
            # Top-P (Nucleus Sampling)
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample das nächste Token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Speichere für Repetition Penalty und Input
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # --- STOP BEDINGUNGEN ---
            # Qwens EOS Token oder das ChatML End-Token <|im_end|>
            if next_token.item() == tokenizer.eos_token_id or \
               next_token.item() == tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]:
                break
                
            # Live Print (übersetzt das einzelne Token in Text)
            token_str = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(token_str, end="", flush=True)
            
    print("\n\n--- Ende ---")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True, help="Deine Frage an Karla")
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--temp", type=float, default=0.6)
    parser.add_argument("--rep-penalty", type=float, default=1.15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Cosmos-Reason2-2B", trust_remote_code=True)
    
    config = KarlaConfig()
    model = create_karla(config).to(device)
    print(f"Lade Gewichte von {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    
    state_dict = checkpoint["model_state_dict"]
    fixed_state_dict = {}
    
    for k, v in state_dict.items():
        # 1. "model." Präfix vom Wrapper entfernen
        new_key = k[6:] if k.startswith("model.") else k
        
        # 2. Wir ignorieren das eingefrorene Qwen-Backbone (L0)!
        # HuggingFace hat das bereits in perfektem 4-Bit geladen.
        # Wir wollen nur unser hart trainiertes L1, L2 und die Scales.
        if new_key.startswith("l0."):
            continue
            
        fixed_state_dict[new_key] = v
            
    # Wir nutzen strict=False, weil PyTorch meckern wird, dass L0 fehlt. Das ist aber gewollt!
    missing, unexpected = model.load_state_dict(fixed_state_dict, strict=False)
    
    print(f"Erfolgreich geladen! CTM-Scale: {model.ctm_scale().item():.4f}, L1-Scale: {model.l1_scale().item():.4f}")
    # ----------------------------------------------------------------
            
    # Jetzt laden wir die echten, 35 Stunden lang trainierten Gewichte!
    #model.load_state_dict(fixed_state_dict, strict=True)
    # ----------------------------------------------------------------
    
    generate_chat(model, tokenizer, args.prompt, args.max_tokens, args.temp, args.rep_penalty, device=device)

if __name__ == "__main__":
    main()