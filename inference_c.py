#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import os
import sys

# Pfade setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from karla.models.karla import create_karla
from karla.utils.config import KarlaConfig

class KarlaAgent:
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Agent] Lade auf {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        config = KarlaConfig()
        self.model = create_karla(config).to(self.device)
        
        if os.path.exists(checkpoint_path):
            print(f"[Agent] Lade Checkpoint {checkpoint_path}...")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            # Filter 'model.' prefix if present (compatibility with training wrapper)
            state_dict = {}
            for k, v in ckpt["model_state_dict"].items():
                if k.startswith("model."):
                    state_dict[k[6:]] = v
                else:
                    state_dict[k] = v

            # strict=False ist wichtig, falls L1 neu ist oder L0 gefroren war
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[Agent] Warnung: Keys nicht gefunden (neue Module?): {len(missing)}")
        else:
            print("[Agent] Starte ohne Checkpoint (Frisch).")
            
        # WICHTIG: Modell in Eval-Modus setzen (Dropout deaktivieren)
        self.model.eval()
        print("[Agent] Bereit.")

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
        """
        Generierungsfunktion - EXAKT wie in deinem funktionierenden inference_v4.py
        """
        self.model.eval() # Safety Check
        
        # Apply Chat Template (IMPORTANT for Qwen/ChatML models)
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Optional: Context kürzen
        if input_ids.size(1) > 512:
            input_ids = input_ids[:, -512:]

        for _ in range(max_new_tokens):
            # 1. Autocast ist PFLICHT für 4-bit Modelle!
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self.model(input_ids)
                
            next_token_logits = outputs.logits[0, -1, :]
            
            # Temperature
            next_token_logits = next_token_logits / temperature
            
            # Top-P Sampling (Logik aus inference_v4.py)
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop Condition
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        # Decode only the NEW tokens
        generated_ids = input_ids[0, len(self.tokenizer.encode(formatted_prompt, add_special_tokens=False)):]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def update_memory(self, text):
        """
        Lernt einen neuen Fakt. 
        Korrektur: L0 bleibt IMMER im Eval-Modus (Dropout aus!).
        """
        # 1. Nur L1 und L2 in Train-Modus setzen (L0 BLEIBT in Eval!)
        # Das verhindert, dass Dropout im Frozen Backbone Daten zerstört.
        self.model.l1.train() 
        self.model.l2.train() # CTM trainiert auch mit!
        
        # L0 sicherheitshalber in eval lassen
        self.model.l0.eval()
        
        # 2. Optimizer prep
        self.model.l1_optimizer.zero_grad()
        
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        labels = input_ids.clone()
        
        # 3. Forward mit Autocast
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss
            
        if loss is None or torch.isnan(loss):
            print("[Agent] Fehler: Kein Loss berechnet.")
            return 0.0

        loss_val = loss.item()
        
        # 4. Nested Learning Gatekeeper (Surprise-Based Plasticity)
        # NICHT lernen, wenn Loss zu niedrig (Wissen vorhanden)
        if loss_val < 0.3:
             # Alles zurück auf Eval
            self.model.eval()
            return loss_val
            
        # NICHT lernen, wenn Loss zu hoch (Datensalat)
        if loss_val > 8.0:
            print(f"[Agent] Loss zu hoch (Garbage? {loss_val:.2f}). Skip Update.")
            self.model.eval()
            return loss_val

        # 5. Backward
        loss.backward()
        
        # Gradient Clipping (Lebenswichtig für Stabilität)
        torch.nn.utils.clip_grad_norm_(self.model.l1.parameters(), max_norm=1.0)
        
        # 6. Optimizer Step
        self.model.l1_optimizer.step()
        
        # 7. Zurück in den sicheren Eval-Modus
        self.model.eval()
        
        return loss_val

def learn_live(agent, text, max_steps=40, target_loss=0.3):
    print(f"\n[Lernvorgang startet] Brenne Wissen ein...")
    # Punkt am Ende hilft dem Modell, den Satz abzuschließen
    text_to_learn = text if text.endswith(".") else text + "."
    
    for step in range(max_steps):
        loss = agent.update_memory(text_to_learn)
        print(f"  > Denk-Zyklus {step+1:02d}/{max_steps} | Loss: {loss:.4f}")
        
        if loss < target_loss:
            print(f"  > [Erfolg] Fakt gespeichert bei Step {step+1}!")
            break
    print("[Lernvorgang abgeschlossen]")


# -------------------
# MAIN TEST (C++ Edition)
# -------------------
if __name__ == "__main__":
    # Nimm den BESTEN SFT Checkpoint (nicht den unterbrochenen RL Checkpoint)
    checkpoint = "checkpoints_pretrain/step_4383.pt" 
    # Falls du den RL checkpoint nehmen willst: "checkpoints_pretrain/RL_INTERRUPTED_step_87.pt"
    
    if not os.path.exists(checkpoint):
        print(f"Checkpoint nicht gefunden: {checkpoint}")
        # Fallback auf den letzten bekannten guten
        checkpoint = "checkpoints_pretrain/INTERRUPTED_step_20706.pt" 

    agent = KarlaAgent(checkpoint)
    
    # TEST 1: Baseline (Kann sie C++?)
    print("\n==================================================")
    print("TEST 1: VOR DEM LERNEN (C++ Task)")
    print("==================================================")
    prompt = "Write a C++ function that calculates the factorial of a number. Only code, no explanation."
    response = agent.generate(prompt, max_new_tokens=150)
    print(f"Response:\n{response}")
    
    # TEST 2: LIVE LERNEN (C++ Syntax Injektion)
    print("\n==================================================")
    print("LIVE LERNEN: C++ SYNTAX INJEKTION")
    print("==================================================")
    
    # Ein kleines Stück C++ Wissen
    cpp_knowledge = """In C++, a function to calculate the square of a number looks like this:
int square(int x) {
    return x * x;
}"""
    
    learn_live(agent, cpp_knowledge, max_steps=40, target_loss=0.3)
    
    # TEST 3: Resultat (Hat das Wissen geholfen?)
    print("\n==================================================")
    print("TEST 2: NACH DEM LERNEN (C++ Task)")
    print("==================================================")
    # Wir stellen genau dieselbe Frage nochmal!
    response_after = agent.generate(prompt, max_new_tokens=150)
    print(f"Response:\n{response_after}")