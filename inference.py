#!/usr/bin/env python3
"""
inference_hacker.py — Force Karla to think harder
=================================================
Allows boosting the L1/CTM scales during inference to uncover
what the reasoning head is actually thinking.
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from karla.utils.config import KarlaConfig
from karla.models.karla import create_karla

# Suppress noisy logs
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Hacker")

class KarlaHacker:
    def __init__(self, checkpoint_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"⚡ Loading Karla from {checkpoint_path}...")
        
        self.config = KarlaConfig()
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.l0.model_name, trust_remote_code=True
        )
        
        self.model = create_karla(self.config)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state, strict=False)
        
        # Move to GPU
        for name, module in self.model.named_children():
            if name != 'l0': module.to(self.device)
        self.model.l1_scale.data = self.model.l1_scale.data.to(self.device)
        self.model.ctm_scale.data = self.model.ctm_scale.data.to(self.device)
        self.model.eval()
        
        print(f"   Original L1 Scale:  {self.model.l1_scale.item():.4f}")
        print(f"   Original CTM Scale: {self.model.ctm_scale.item():.4f}")

    def generate(self, prompt, boost_factor=1.0, temp=0.7):
        # 1. Apply Boost
        original_l1 = self.model.l1_scale.item()
        original_ctm = self.model.ctm_scale.item()
        
        self.model.l1_scale.data *= boost_factor
        self.model.ctm_scale.data *= boost_factor
        
        print(f"\n🧪 Generating with Boost x{boost_factor}")
        print(f"   Effective L1:  {self.model.l1_scale.item():.4f}")
        print(f"   Effective CTM: {self.model.ctm_scale.item():.4f}")
        print(f"   Prompt: {prompt}")
        print(f"{'-'*60}")
        
        # 2. Format & Tokenize
        msgs = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        # 3. Generate
        generated = input_ids[0].tolist()
        
        print("🤖 ", end="", flush=True)
        for _ in range(200):
            current = torch.tensor([generated], device=self.device)
            with torch.no_grad():
                out = self.model(current)
                logits = out.logits[0, -1, :]
            
            # Sampling
            if temp == 0:
                token = logits.argmax().item()
            else:
                probs = F.softmax(logits / temp, dim=-1)
                token = torch.multinomial(probs, 1).item()
            
            if token == self.tokenizer.eos_token_id:
                break
                
            generated.append(token)
            print(self.tokenizer.decode([token]), end="", flush=True)
            
        print("\n")
        
        # 4. Restore Original Scales
        self.model.l1_scale.data.fill_(original_l1)
        self.model.ctm_scale.data.fill_(original_ctm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str, default="What are the symptoms of diabetes?")
    parser.add_argument("-b", "--boost", type=float, default=10.0, help="Multiply Karla's influence by this factor")
    parser.add_argument("-t", "--temp", type=float, default=0.7)
    args = parser.parse_args()
    
    # Use RL checkpoint preferably
    ckpt = "checkpoints_rl/best_rl_model.pt"
    if not os.path.exists(ckpt):
        ckpt = "checkpoints/best_model.pt"
        
    hacker = KarlaHacker(ckpt)
    hacker.generate(args.prompt, boost_factor=1.0, temp=args.temp)       # Normal
    hacker.generate(args.prompt, boost_factor=args.boost, temp=args.temp) # Boosted