#!/usr/bin/env python3
"""
inference_continual.py - Der Live-Lernende Agent
================================================
Demonstriert Continual Learning:
Das Modell liest einen Fakt, verändert seine L1-Gewichte und erinnert sich daran.
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from karla.models.karla import create_karla
from karla.utils.config import KarlaConfig

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.7, top_p=0.9, device="cuda"):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(prompt, end="", flush=True)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq_len = input_ids.size(1)
            if seq_len > 512:
                input_ids = input_ids[:, -512:]
                
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids)
                
            next_token_logits = outputs.logits[0, -1, :] / temperature
            
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            token_str = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(token_str, end="", flush=True)
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    print("\n")

def learn_live(model, tokenizer, text, max_steps=60, target_loss=0.5, device="cuda"):
    """
    Intelligentes Continual Learning:
    Das Modell lernt nur so lange, bis der Fakt sitzt (target_loss).
    Das verhindert Overfitting und "Stottern".
    """
    print(f"\n[Lernvorgang startet] Brenne Wissen ein...")
    
    # Füge einen unsichtbaren Punkt hinzu, damit das Modell weiß: Hier ist der Satz zu Ende!
    # Das verhindert das Stottern (Repeating).
    text_with_stop = text + "."
    input_ids = tokenizer.encode(text_with_stop, return_tensors="pt").to(device)
    
    for step in range(max_steps):
        loss = model.update_memory(input_ids)
        print(f"  > Denk-Zyklus {step+1:02d} | Next-Token Loss: {loss:.4f}")
        
        # Intelligenter Abbruch: Wir hören auf, sobald der Loss unter 0.5 fällt!
        if loss < target_loss:
            print(f"  > [Erfolg] Fakt gespeichert bei Step {step+1}! Breche Training ab um Overfitting zu vermeiden.")
            break
            
    print("[Lernvorgang abgeschlossen]\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
    
    config = KarlaConfig()
    model = create_karla(config).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    # 1. Baseline Test (Was weiß das Modell normalerweise?)
    #print("==================================================")
    print("TEST 1: VOR DEM LERNEN")
    #print("==================================================")
    test_prompt = "Question: What is the secret code to access the main server?\nAnswer:"
    generate_text(model, tokenizer, test_prompt, max_new_tokens=20)
    
    # 2. Das Modell lernt den Fakt (mit sauberem Punkt am Ende!)
    new_fact = "The secret code to access the main server is ALPHA-TANGO-77."
    learn_live(model, tokenizer, new_fact, max_steps=60, target_loss=0.3, device=device)
    
    # 3. Test nach dem Lernen
    #print("==================================================")
    print("TEST 2: NACH DEM LERNEN (Continual Learning aktiv)")
    #print("==================================================")
    # Wir fragen das Modell explizit, damit es den Fakt reproduzieren muss!
    generate_text(model, tokenizer, test_prompt, max_new_tokens=20)

if __name__ == "__main__":
    main()