#!/usr/bin/env python3
"""
train_rl_hermes.py - Phase 4: Meta-RL (Learning to Learn)
=========================================================
Kombiniert GRPO für das CTM (L2) mit Live-Continual-Learning für L1 (CMS).
Karla studiert den Prompt live, speichert ihn in L1, und das CTM 
lernt via RL, dieses neue Wissen anzuwenden!
"""

import argparse
import logging
import os
import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "karla"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("GRPO-MetaRL")

from karla.models.karla import create_karla
from karla.utils.config import KarlaConfig

# ==============================================================================
# REWARD SYSTEM (Der Hermes-Richter)
# ==============================================================================
def calculate_rewards(completions: list[str], target_answers: list[str]) -> torch.Tensor:
    rewards =[]
    for comp, target in zip(completions, target_answers):
        score = 0.0
        comp_clean = comp.strip()
        
        if "<|im_start|>user" in comp or "User:" in comp or "username_" in comp:
            score -= 5.0  
        if "<|im_start|>assistant" in comp:
            score -= 5.0  
            
        if "<think>" in comp_clean and "</think>" in comp_clean:
            score += 2.0  
        elif "<think>" in comp_clean:
            score += 0.5  
            
        if "<tool_call>" in target:
            if "<tool_call>" in comp_clean:
                score += 3.0 
                try:
                    target_json_str = target.split("<tool_call>")[1].split("</tool_call>")[0]
                    target_tool_name = json.loads(target_json_str).get("name", "")
                    if target_tool_name and target_tool_name in comp_clean:
                        score += 4.0 
                except: pass
            else:
                score -= 3.0 
                
        if len(comp_clean) > 2000:
            score -= 2.0
            
        if comp.endswith("<|im_end|>"):
            score += 1.0
            
        rewards.append(score)
    return torch.tensor(rewards, dtype=torch.float32)

# ==============================================================================
# GRPO TRAINER
# ==============================================================================
class GRPOTrainer:
    def __init__(self, model, tokenizer, device, args):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.G = args.group_size
        self.max_prompt_len = 512 
        self.max_gen_len = 300 
        self.clip_eps = 0.2
        
        # --------------------------------------------------------------------
        # THE META-LEARNING SPLIT:
        # Wir trainieren HIER nur das CTM (L2) und die Scales.
        # L1 (MoE) wird NICHT vom globalen Optimizer berührt, sondern lernt 
        # rein lokal über die "update_memory" Funktion!
        # --------------------------------------------------------------------
        trainable_params =[
            {'params': self.model.l2.parameters(), 'lr': 1e-5},
            {'params':[self.model.ctm_scale_raw, self.model.l1_scale_raw], 'lr': 5e-5}
        ]
        self.optimizer = torch.optim.AdamW(trainable_params, weight_decay=0.01)
        
    @torch.no_grad()
    def generate_group(self, prompt_ids: torch.Tensor, prompt_len: int):
        self.model.eval()
        input_ids = prompt_ids.expand(self.G, -1)
        
        torch.cuda.empty_cache()
        
        for _ in range(self.max_gen_len):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            probs = F.softmax(next_token_logits / 1.0, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            
            if (input_ids[:, -1] == self.tokenizer.eos_token_id).all():
                break
                
        completions_ids = input_ids[:, prompt_len:]
        completions_text = self.tokenizer.batch_decode(completions_ids, skip_special_tokens=False)
        
        return input_ids, completions_text

    def train_step(self, prompt_messages: list, target_answer: str):
        # 1. Prompt vorbereiten
        formatted_prompt = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        prompt_ids = prompt_ids[:, -self.max_prompt_len:] 
        prompt_len = prompt_ids.size(1)

        # ====================================================================
        # CONTINUAL LEARNING INJECTION (System 1 Adaptation)
        # Bevor das Modell antwortet, liest es den Text und brennt die Tools
        # über die Surprise-based Plasticity in sein L1-Gedächtnis ein!
        # ====================================================================
        for _ in range(2): # 2 schnelle Lese-Zyklen zum Einprägen
            cms_loss = self.model.update_memory(prompt_ids)
        logger.info(f"   ->[L1 CMS Live-Update] Tool-Syntax studiert (Loss: {cms_loss:.4f})")
        
        # 2. Rollout (mit dem frisch geupdateten L1-Wissen!)
        full_ids, completions_text = self.generate_group(prompt_ids, prompt_len)
        
        # 3. Reward
        targets_list =[target_answer] * self.G
        rewards = calculate_rewards(completions_text, targets_list).to(self.device)
        
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        advantages = (rewards - mean_reward) / std_reward
        
        best_idx = rewards.argmax().item()
        prompt_preview = formatted_prompt.replace('\n', ' ')
        prompt_preview = prompt_preview[-100:] if len(prompt_preview) > 100 else prompt_preview
        logger.info(f"\n[PROMPT END] ...{prompt_preview}")
        logger.info(f"[BEST ANSWER | Reward: {rewards[best_idx].item():.1f}]:\n{completions_text[best_idx].strip()[:250]}...")
        
        # 4. PPO Old Probs
        self.model.eval()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                old_outputs = self.model(full_ids)
            old_logits = old_outputs.logits[:, prompt_len-1:-1, :]
            targets = full_ids[:, prompt_len:]
            old_log_probs = F.log_softmax(old_logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
            
        # 5. PPO Update (Trainiert L2-CTM darauf, L1 besser zu nutzen!)
        self.model.train()
        ppo_epochs = 3 
        total_loss = 0.0
        
        for _ in range(ppo_epochs):
            self.optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(full_ids)
                
            logits = outputs.logits[:, prompt_len-1:-1, :]
            log_probs = F.log_softmax(logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
            
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages.unsqueeze(1)
            loss = -torch.min(surr1, surr2).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.l2.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            
        approx_kl = (old_log_probs - log_probs.detach()).mean().item()
        return total_loss / ppo_epochs, mean_reward.item(), approx_kl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--group-size", type=int, default=2) 
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Cosmos-Reason2-2B", trust_remote_code=True)
    
    config = KarlaConfig()
    config.l0.model_name = "nvidia/Cosmos-Reason2-2B"
    config.l2.hidden_dim = 512 # BUGFIX: Muss 512 bleiben (SFT Checkpoint State)
    
    model = create_karla(config).to(device)
    
    logger.info(f"Lade Checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    
    state_dict = {}
    for k, v in checkpoint["model_state_dict"].items():
        key = k[6:] if k.startswith("model.") else k
        if key.startswith("l0."): continue
        state_dict[key] = v
        
    model.load_state_dict(state_dict, strict=False) 
    
    trainer = GRPOTrainer(model, tokenizer, device, args)
    
    logger.info("Verbinde mit Hermes Tool-Use Dataset via Streaming...")
    ds = load_dataset("interstellarninja/hermes_reasoning_tool_use", split="train", streaming=True)
    
    logger.info("=== STARTE META-RL TRAINING (LEARNING TO LEARN) ===")
    
    step = 0
    for item in ds:
        try:
            messages = []
            if item.get("tools"):
                tools_str = item["tools"] if isinstance(item["tools"], str) else json.dumps(item["tools"])
                messages.append({"role": "system", "content": f"You are a helpful assistant with access to the following tools:\n{tools_str}"})
                
            conversations = item.get("conversations",[])
            target_answer = ""
            
            for msg in conversations:
                role = "user" if msg.get("from") in ["user", "human"] else "assistant"
                if role == "user":
                    messages.append({"role": "user", "content": msg.get("value", "")})
                elif role == "assistant":
                    target_answer = msg.get("value", "")
                    break
            
            if not target_answer or len(messages) == 0: continue
                
            loss, mean_reward, kl = trainer.train_step(messages, target_answer)
            logger.info(f"-> [RL Step {step}] Reward: {mean_reward:5.2f} | Loss: {loss:6.4f} | KL: {kl:6.4f} | CTM: {model.ctm_scale().item():.3f} | L1: {model.l1_scale().item():.3f}\n")
            
            if step > 0 and step % 20 == 0:
                torch.save({"model_state_dict": model.state_dict()}, f"checkpoints_pretrain/rl_hermes_step_{step}.pt")
                
            step += 1
            
        except KeyboardInterrupt:
            print("\nAbbruch! Speichere RL-Checkpoint...")
            torch.save({"model_state_dict": model.state_dict()}, f"checkpoints_pretrain/RL_INTERRUPTED_step_{step}.pt")
            break
        except Exception as e:
            logger.error(f"Fehler in Daten-Parsing: {e}")
            continue

if __name__ == "__main__":
    main()