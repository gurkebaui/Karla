#!/usr/bin/env python3
"""
train_l1_opus.py - The Opus Injection (L1-Only Training)
========================================================
Wir trainieren AUSSCHLIESSLICH das L1-MoE (und den l1_scale) auf das 
brillante Reasoning-Verhalten von Claude 3 Opus.
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from karla.models.karla import create_karla
from karla.utils.config import KarlaConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("Opus-L1")

def main():
    checkpoint_path = "checkpoints_pretrain/INTERRUPTED_step_4383.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Lade Tokenizer und Architektur auf {device}...")
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Cosmos-Reason2-2B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    config = KarlaConfig()
    config.l0.model_name = "nvidia/Cosmos-Reason2-2B"
    config.l2.hidden_dim = 512
    
    model = create_karla(config).to(device)
    
    logger.info(f"Lade Checkpoint {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = {k[6:] if k.startswith("model.") else k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("l0.")}
    model.load_state_dict(state_dict, strict=False)

    # =========================================================================
    # OPTIMIZER SETUP: NUR L1 LERNT!
    # L0 ist eingefroren. L2 (CTM) ist im Eval-Modus und lernt NICHTS.
    # =========================================================================
    model.l0.eval()
    model.l2.eval()
    model.l1.train()
    
    # Aggressiver Optimizer für das L1-MoE
    l1_optimizer = torch.optim.AdamW([
        {'params': model.l1.parameters(), 'lr': 1e-4},
        {'params':[model.l1_scale_raw], 'lr': 5e-4} # L1 soll lernen, lauter zu werden!
    ], weight_decay=0.01)

    logger.info("Lade Opus-4.6-Reasoning Dataset...")
    # Streaming = False, da 2100 Beispiele problemlos in den RAM passen
    ds = load_dataset("json", data_files="data/opus.jsonl", split="train")
    
    epochs = 2 # Wir gehen zweimal über die 2100 Beispiele
    total_steps = len(ds) * epochs
    current_step = 0
    
    logger.info(f"=== STARTE L1-INJEKTION ({total_steps} Steps) ===")
    
    for epoch in range(epochs):
        ds = ds.shuffle() # Gut durchmischen
        
        for item in ds:
            l1_optimizer.zero_grad()
            
            # ChatML Formatierung
            messages = item["messages"]
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except:
                continue
                
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            # Truncate falls zu lang für VRAM (Opus denkt oft sehr lang)
            if input_ids.size(1) > 1024: 
                input_ids = input_ids[:, -1024:]
                
            labels = input_ids.clone()
            
            # Smart Masking: Wir trainieren nur auf Assistant-Outputs!
            in_assistant_turn = False
            im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
            for i in range(input_ids.size(1)):
                if input_ids[0, i] == im_start:
                    if i + 3 < input_ids.size(1):
                        lookahead = tokenizer.decode(input_ids[0, i+1:i+4]).strip().lower()
                        in_assistant_turn = lookahead.startswith("assistant")
                if in_assistant_turn:
                    labels[0, i] = input_ids[0, i]
                else:
                    labels[0, i] = -100
                if input_ids[0, i] == im_start:
                    labels[0, i] = -100
                    
            if not (labels != -100).any():
                continue

            # Forward Pass (Nur L1 wird upgedatet)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
                
            if torch.isnan(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.l1.parameters(), 1.0)
            l1_optimizer.step()
            
            current_step += 1
            
            if current_step % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} | Step {current_step}/{total_steps} | Loss: {loss.item():.4f} | L1-Scale: {model.l1_scale().item():.4f}")
                
    # Speichern des Opus-injizierten Modells
    save_path = "checkpoints_pretrain/opus_l1_injected.pt"
    torch.save({"model_state_dict": model.state_dict()}, save_path)
    logger.info(f"=== INJEKTION ABGESCHLOSSEN. Gespeichert unter {save_path} ===")

if __name__ == "__main__":
    main()