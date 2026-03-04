#!/usr/bin/env python3
"""
interactive_chat.py - Karla Command Line Interface
==================================================
Kommandos:
  !clear           - Löscht den aktuellen Chat-Verlauf.
  !learn <text>    - Karla lernt den Text live (Surprise-based Plasticity).
  !save <name.pt>  - Speichert das L1/L2 Gehirn als neuen Checkpoint ab.
  quit             - Beendet den Chat.
"""

import torch
import os
import sys
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from karla.models.karla import create_karla
from karla.utils.config import KarlaConfig

def main():
    checkpoint = "checkpoints_pretrain/opus_l1_injected.pt" # Deinen besten nehmen!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Lade Karla Agent...")
    tokenizer = AutoTokenizer.from_pretrained("nvidia/Cosmos-Reason2-2B", trust_remote_code=True)
    
    config = KarlaConfig()
    config.l0.model_name = "nvidia/Cosmos-Reason2-2B"
    config.l2.hidden_dim = 512
    model = create_karla(config).to(device)
    
    if os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        state_dict = {k[6:] if k.startswith("model.") else k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("l0.")}
        model.load_state_dict(state_dict, strict=False)
        print("Gehirn (CTM + MoE) geladen!")
    
    model.eval()
    
    chat_history =[]
    
    print("\n" + "="*50)
    print("Willkommen bei Karla C1 (NVIDIA Cosmos Edition)")
    print("Tippe '!learn [text]' um ihr live Fakten beizubringen.")
    print("Tippe '!clear' für einen neuen Kontext.")
    print("Tippe '!save name.pt' um das Gehirn zu speichern.")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\nDu: ")
            if not user_input.strip(): continue
            if user_input.lower() in ["quit", "exit"]: break
            
            # --- COMMANDS ---
            if user_input.startswith("!clear"):
                chat_history = []
                print("[System] Chat-Verlauf gelöscht.")
                continue
                
            if user_input.startswith("!save"):
                parts = user_input.split(" ", 1)
                if len(parts) > 1:
                    filename = parts[1]
                    torch.save({"model_state_dict": model.state_dict()}, filename)
                    print(f"[System] Karla's Gehirn gespeichert als {filename}!")
                else:
                    print("[System] Bitte Dateinamen angeben: !save karla_v2.pt")
                continue
                
            if user_input.startswith("!learn"):
                fact = user_input.split(" ", 1)[1]
                print(f"[System] Karla studiert den Fakt...")
                input_ids = tokenizer.encode(fact + ".", return_tensors="pt").to(device)
                
                for step in range(50):
                    loss = model.update_memory(input_ids)
                    if loss < 0.25: # Strenge Loss-Grenze für perfektes Auswendiglernen
                        print(f"  -> Verstanden nach {step+1} Zyklen! (Loss: {loss:.4f})")
                        break
                continue

            # --- NORMALER CHAT ---
            chat_history.append({"role": "user", "content": user_input})
            
            prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            print(f"Karla: ", end="")
            
            generated_text = ""
            with torch.no_grad():
                for _ in range(30000): # Max Tokens
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(input_ids)
                    
                    next_token_logits = outputs.logits[0, -1, :]
                    next_token = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
                    
                    if next_token.item() == tokenizer.eos_token_id or next_token.item() == tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]:
                        break
                        
                    token_str = tokenizer.decode(next_token[0])
                    print(token_str, end="", flush=True)
                    generated_text += token_str
                    
                    input_ids = torch.cat([input_ids, next_token], dim=1)
            
            chat_history.append({"role": "assistant", "content": generated_text})
            print() # Newline nach Antwort

        except KeyboardInterrupt:
            print("\n[System] Chat beendet.")
            break
        except Exception as e:
            print(f"\n[Fehler] {e}")

if __name__ == "__main__":
    main()