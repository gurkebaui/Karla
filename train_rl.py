#!/usr/bin/env python3
"""
train_rl.py - Karla Phase 4: Local JSONL Reinforcement Learning
===============================================================
Trainiert Karla (CTM + MoE) auf logisches Denken anhand lokaler
Claude Opus Reasoning-Daten.
"""

import argparse
import logging
import os
import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "karla"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("GRPO-Opus")

from karla.models.karla import create_karla
from karla.utils.config import KarlaConfig

# ==============================================================================
# REWARD SYSTEM (Der universelle Opus-Richter)
# ==============================================================================
def calculate_rewards(prompts: list[str], completions: list[str], target_answers: list[str]) -> torch.Tensor:
    rewards =[]
    for prompt, comp, target in zip(prompts, completions, target_answers):
        score = 0.0
        comp_clean = comp.strip()
        target_clean = target.strip()
        
        # 1. ALIGNMENT PENALTIES (Keine Rollen-Halluzinationen!)
        if "<|im_start|>" in comp_clean:
            score -= 5.0  
            
        # 2. THINKING REWARD (DeepSeek-R1 Style)
        if "<think>" in comp_clean and "</think>" in comp_clean:
            score += 2.0  # Volle Punkte für sauberes Denken
        elif "<think>" in comp_clean:
            score += 0.5  # Zumindest angefangen zu denken...
            
        # 3. ZIEL-ERKENNUNG (Stimmt das Ergebnis mit Opus überein?)
        # Da Opus seine Final Answer meist ganz am Ende gibt, vergleichen wir 
        # die letzten ~40 Zeichen (ohne Leerzeichen für höhere Robustheit).
        target_tail = "".join(target_clean[-40:].split())
        comp_tail = "".join(comp_clean[-100:].split()) # Wir suchen in den letzten 100 Zeichen von Karla
        
        if len(target_tail) > 5 and target_tail in comp_tail:
            score += 4.0 # Genial! Richtiges Ergebnis gefunden.
            
        # 4. LÄNGEN-STRAFE
        if len(comp_clean) > 2000:
            score -= 1.0
            
        # 5. SAUBERES ENDE
        if comp.endswith("<|im_end|>"):
            score += 1.0
            
        rewards.append(score)
        
    return torch.tensor(rewards, dtype=torch.float32)

# ==============================================================================
# LOKALER JSONL STREAMER
# ==============================================================================
def stream_local_jsonl(filepath):
    """Liest die JSONL Zeile für Zeile, braucht fast 0 RAM."""
    if not os.path.exists(filepath):
        logger.error(f"Dataset {filepath} nicht gefunden!")
        sys.exit(1)
        
    while True: # Endlosschleife (Epochen) über die Datei
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    yield json.loads(line)
                except Exception:
                    pass

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
        self.max_gen_len = 256 # Genug Platz für Gedanken und Antwort
        self.clip_eps = 0.2
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=1e-5, weight_decay=0.01)
        
    @torch.no_grad()
    def generate_group(self, prompt_text: str):
        self.model.eval()
        messages =[{"role": "user", "content": prompt_text}]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        prompt_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        prompt_ids = prompt_ids[:, -self.max_prompt_len:] 
        prompt_len = prompt_ids.size(1)
        
        input_ids = prompt_ids.expand(self.G, -1)
        
        for _ in range(self.max_gen_len):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Temperatur = 1.0 (Hohe Kreativität für GRPO-Varianz)
            probs = F.softmax(next_token_logits / 1.0, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            
            if (input_ids[:, -1] == self.tokenizer.eos_token_id).all():
                break
                
        completions_ids = input_ids[:, prompt_len:]
        completions_text = self.tokenizer.batch_decode(completions_ids, skip_special_tokens=False)
        
        return input_ids, completions_text, prompt_len

    def train_step(self, prompt_text: str, target_answer: str):
        torch.cuda.empty_cache() 
        full_ids, completions_text, prompt_len = self.generate_group(prompt_text)
        
        prompts_list =[prompt_text] * self.G
        targets_list = [target_answer] * self.G
        rewards = calculate_rewards(prompts_list, completions_text, targets_list).to(self.device)
        
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        advantages = (rewards - mean_reward) / std_reward
        
        best_idx = rewards.argmax().item()
        logger.info(f"\n[PROMPT] {prompt_text[:80].replace(chr(10), ' ')}...")
        logger.info(f"[BEST ANSWER | Reward: {rewards[best_idx].item():.1f}]:\n{completions_text[best_idx].strip()[:250]}...")
        
        self.model.eval()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                old_outputs = self.model(full_ids)
            old_logits = old_outputs.logits[:, prompt_len-1:-1, :]
            targets = full_ids[:, prompt_len:]
            old_log_probs = F.log_softmax(old_logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
            
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            
        approx_kl = (old_log_probs - log_probs.detach()).mean().item()
        return total_loss / ppo_epochs, mean_reward.item(), approx_kl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, help="Pfad zur Opus JSONL Datei")
    parser.add_argument("--group-size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Cosmos-Reason2-2B", trust_remote_code=True)
    
    config = KarlaConfig()
    config.l0.model_name = "nvidia/Cosmos-Reason2-2B"
    config.l2.hidden_dim = 512 # BUGFIX: Muss 512 bleiben (SFT Checkpoint State)
    
    model = create_karla(config).to(device)
    
    logger.info(f"Lade Checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    
    # Intelligentes Filtern: Wir wollen NUR unsere trainierten CTM/L1 Gewichte
    state_dict = {}
    for k, v in checkpoint["model_state_dict"].items():
        # Entferne Wrapper-Präfix
        key = k[6:] if k.startswith("model.") else k
        
        # Ignoriere ALLES, was zum gefrorenen Backbone (l0) gehört
        # Egal ob l0.model.model.visual oder l0.model.embed_tokens
        if key.startswith("l0."):
            continue
            
        state_dict[key] = v
    
    # Jetzt laden wir nur die relevanten Teile!
    # Wir nutzen strict=False, weil das Modell (l0) natürlich mehr Parameter hat 
    # als wir hier laden (wir laden ja nur l1 und l2).
    # Aber wir prüfen manuell, ob L1 und L2 geladen wurden.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    # Sicherheits-Check: Haben wir L1 und L2 geladen?
    l2_loaded = any("l2.synapse" in k for k in state_dict.keys())
    if not l2_loaded:
        logger.warning("WARNUNG: CTM (L2) Gewichte scheinen nicht im Checkpoint zu sein!")
    else:
        logger.info("CTM und L1 Gewichte erfolgreich extrahiert und geladen.")
    # Checkpoint Strict=True um Fehler wie vorhin absolut auszuschließen!
    #model.load_state_dict(state_dict, strict=True) 
    
    trainer = GRPOTrainer(model, tokenizer, device, args)
    
    logger.info(f"Verbinde mit lokalem Dataset: {args.dataset}")
    data_stream = stream_local_jsonl(args.dataset)
    
    logger.info("=== STARTE OPUS RL TRAINING ===")
    
    for step, item in enumerate(data_stream):
        try:
            # Extrahiere User Prompt und Assistant (Target) Antwort
            user_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), None)
            target_msg = next((m["content"] for m in item["messages"] if m["role"] == "assistant"), None)
            
            if not user_msg or not target_msg:
                continue
                
            loss, mean_reward, kl = trainer.train_step(user_msg, target_msg)
            logger.info(f"->[RL Step {step}] Reward: {mean_reward:5.2f} | Loss: {loss:6.4f} | KL: {kl:6.4f} | CTM: {model.ctm_scale().item():.3f}\n")
            
            if step > 0 and step % 20 == 0:
                save_path = f"checkpoints_pretrain/rl_step_{step}.pt"
                torch.save({"model_state_dict": model.state_dict()}, save_path)
                
        except KeyboardInterrupt:
            print("\nAbbruch! Speichere RL-Checkpoint...")
            torch.save({"model_state_dict": model.state_dict()}, f"checkpoints_pretrain/RL_INTERRUPTED_step_{step}.pt")
            break
        except Exception as e:
            logger.error(f"Fehler in Step {step}: {e}")
            continue

if __name__ == "__main__":
    main()