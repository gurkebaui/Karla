#!/usr/bin/env python3
"""
demo_physical_ai.py
===================
Der ultimative Pitch für die Jury: Karla lernt live eine neue 
Roboter-Hardware-API (Physical AI) kennen und steuert sie.
"""

import torch
import os
import sys
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from karla.models.karla import create_karla
from karla.utils.config import KarlaConfig

def learn_live(model, tokenizer, text, max_steps=40, target_loss=0.25, device="cuda"):
    print(f"\n[SYSTEM] Initiating Surprise-Based Plasticity for L1 MoE...")
    text_with_stop = text + "."
    input_ids = tokenizer.encode(text_with_stop, return_tensors="pt").to(device)
    
    for step in range(max_steps):
        loss = model.update_memory(input_ids, base_lr=0.05)
        print(f"  > Rewiring Synapses (Tick {step+1:02d}) | Compression Loss: {loss:.4f}")
        if loss < target_loss:
            print(f"  >[SUCCESS] New hardware API stored in L1 MoE at step {step+1}!")
            break
    print("[SYSTEM] Learning complete.\n")

def generate_text(model, tokenizer, prompt, device="cuda"):
    model.eval()
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(formatted, return_tensors="pt").to(device)
    
    print(f"Prompt: {prompt}")
    print(f"Agent (CTM {model.ctm_scale().item():.2f} | L1 {model.l1_scale().item():.2f}): ", end="")
    
    with torch.no_grad():
        for _ in range(100):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids)
            next_token = torch.argmax(outputs.logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
            
            if next_token.item() == tokenizer.eos_token_id or next_token.item() == tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]:
                break
                
            print(tokenizer.decode(next_token[0]), end="", flush=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    print("\n")

def main():
    checkpoint = "checkpoints_pretrain/opus_l1_injected.pt" # Deinen besten Checkpoint hier eintragen!
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Cosmos-Reason2-2B", trust_remote_code=True)
    
    config = KarlaConfig()
    config.l0.model_name = "nvidia/Cosmos-Reason2-2B"
    config.l2.hidden_dim = 512 
    model = create_karla(config).to(device)
    
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    state_dict = {k[6:] if k.startswith("model.") else k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("l0.")}
    model.load_state_dict(state_dict, strict=False)

    print("\n" + "="*60)
    print(" 🤖 KARLA C1: PHYSICAL AI EDGE DEMO")
    print("="*60)

    task_prompt = "The path is blocked by a steel door. Use the plasma cutter to open it. Output the JSON tool call."
    
    print("\n--- PHASE 1: BEFORE LEARNING (ZERO-SHOT) ---")
    generate_text(model, tokenizer, task_prompt, device)
    
    print("\n--- PHASE 2: LIVE EDGE LEARNING (NO BACKPROP ON L0) ---")
    # Das ist das Wissen, das der Roboter z.B. aus einem Handbuch scannt
    api_manual = """
    Hardware Update: The new robotic arm has a plasma cutter. 
    To use it, output exactly this JSON format:
    ```json
    {"tool": "plasma_cutter", "power_level": "maximum"}
    ```
    """
    learn_live(model, tokenizer, api_manual, max_steps=40, target_loss=0.25, device=device)
    
    print("--- PHASE 3: AFTER LEARNING (EXECUTION) ---")
    generate_text(model, tokenizer, task_prompt, device)

if __name__ == "__main__":
    main()