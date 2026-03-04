#!/usr/bin/env python3
"""
continual_eval.py - The Ultimate Continual Learning Test
========================================================
Testet Karlas Fähigkeit, über das L1-MoE nacheinander zwei völlig 
verschiedene Programmiersprachen (JS und C++) zu lernen, ohne 
die erste wieder zu vergessen.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import sys
import random

# Pfade setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from karla.models.karla import create_karla
from karla.utils.config import KarlaConfig

class KarlaContinualAgent:
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Agent] Lade auf {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained("nvidia/Cosmos-Reason2-2B", trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        config = KarlaConfig()
        config.l0.model_name = "nvidia/Cosmos-Reason2-2B"
        config.l2.hidden_dim = 512 # CTM Dimension unseres Checkpoints
        
        self.model = create_karla(config).to(self.device)
        
        if os.path.exists(checkpoint_path):
            print(f"[Agent] Lade SFT-Checkpoint {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            
            state_dict = {}
            for k, v in ckpt["model_state_dict"].items():
                key = k[6:] if k.startswith("model.") else k
                if not key.startswith("l0."): # L0 wird frisch von HF geladen
                    state_dict[key] = v

            self.model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} nicht gefunden!")
            
        self.model.eval()
        print("[Agent] Bereit.")

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=200, temperature=0.6, top_p=0.9):
        self.model.eval() 
        
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        prompt_len = input_ids.size(1)

        for _ in range(max_new_tokens):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(input_ids)
                
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
            
            if next_token.item() == self.tokenizer.eos_token_id or \
               next_token.item() == self.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]:
                break
                
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        generated_ids = input_ids[0, prompt_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def update_memory(self, messages, base_lr=0.05):
        """
        Nimmt ein ChatML Array entgegen und trainiert NUR das L1 MoE.
        """
        self.model.l1.train() 
        self.model.l1_optimizer.zero_grad()
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Max length protection
        if input_ids.size(1) > 512:
            input_ids = input_ids[:, -512:]
            
        labels = input_ids.clone()
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss
            
        if loss is None or torch.isnan(loss): return 0.0

        loss_val = loss.item()
        
        # Surprise-Based Plasticity
        surprise_factor = min(max((loss_val - 0.5) / 4.0, 0.0), 1.0)
        dynamic_lr = base_lr * surprise_factor
        
        if surprise_factor <= 0.01:
            self.model.eval()
            return loss_val
            
        for param_group in self.model.l1_optimizer.param_groups:
            if len(param_group['params']) == 1: 
                param_group['lr'] = dynamic_lr * 5.0 
            else:
                param_group['lr'] = dynamic_lr

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.l1.parameters(), max_norm=1.0)
        self.model.l1_optimizer.step()
        
        self.model.eval()
        return loss_val

# ==============================================================================
# DATA PREPARATION (Läd JS und C++ direkt via HF Streaming)
# ==============================================================================
def get_language_samples(lang_name: str, count: int):
    print(f"Sammle {count} {lang_name} Beispiele...")
    ds = load_dataset("ronantakizawa/github-top-code", split="train", streaming=True)
    samples =[]
    
    for item in ds:
        if item.get("file_language") == lang_name:
            # Baue ChatML Format
            messages =[
                {"role": "user", "content": f"Write the {lang_name} code for {item['file_path']}."},
                {"role": "assistant", "content": str(item['content'])[:1500]} # Limitiere auf 1500 Chars
            ]
            samples.append(messages)
            if len(samples) >= count:
                break
    return samples

# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================
if __name__ == "__main__":
    # Nutze deinen SFT Checkpoint (vor dem fehlerhaften RL)
    CHECKPOINT = "checkpoints_pretrain/INTERRUPTED_step_4383.pt" 
    
    agent = KarlaContinualAgent(CHECKPOINT)
    
    print("\n--- [PHASE 0] DATEN LADEN ---")
    js_samples = get_language_samples("JavaScript", 200)
    cpp_samples = get_language_samples("C++", 200)
    
    print("\n--- [PHASE 1] LERNE JAVASCRIPT ---")
    # Wir iterieren 2x über die Daten (Epochen), um sie gut einzuprägen
    for epoch in range(2):
        random.shuffle(js_samples)
        total_loss = 0
        for i, msg in enumerate(js_samples):
            loss = agent.update_memory(msg)
            total_loss += loss
            if i % 50 == 0:
                print(f"  JS Epoch {epoch+1} | Sample {i}/200 | Avg Loss: {total_loss/(i+1):.4f}")
    
    print("\n--- [PHASE 2] LERNE C++ (Mit 10% JavaScript Replay-Buffer) ---")
    # Wir mischen 20 JS Samples in die 200 C++ Samples, um Forgetting zu verhindern
    mixed_samples = cpp_samples + random.sample(js_samples, 20)
    
    for epoch in range(2):
        random.shuffle(mixed_samples)
        total_loss = 0
        for i, msg in enumerate(mixed_samples):
            loss = agent.update_memory(msg)
            total_loss += loss
            if i % 50 == 0:
                print(f"  C++ Epoch {epoch+1} | Sample {i}/{len(mixed_samples)} | Avg Loss: {total_loss/(i+1):.4f}")
                
    print("\n==================================================")
    print("             FINALE TESTS (THE MOMENT OF TRUTH)       ")
    print("==================================================")
    
    print("\n[TEST 1: JAVASCRIPT]")
    js_prompt = "Write a simple Express.js server that listens on port 3000 and returns 'Hello Karla' on the root route. Only output the code."
    print(f"Prompt: {js_prompt}")
    js_res = agent.generate(js_prompt, max_new_tokens=150)
    print(f"\nModel Output:\n{js_res}\n")
    
    print("-" * 50)
    
    print("\n[TEST 2: C++]")
    cpp_prompt = "Write a C++ class for a simple BankAccount with a deposit and withdraw method. Only output the code."
    print(f"Prompt: {cpp_prompt}")
    cpp_res = agent.generate(cpp_prompt, max_new_tokens=200)
    print(f"\nModel Output:\n{cpp_res}\n")