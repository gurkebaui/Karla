#!/usr/bin/env python3
"""
Unified Dataset Loader for Karla C1 Training
============================================
Fix: Behandelt 'content': null in Tool-Use Datensätzen korrekt.
"""

import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional
import random
import os
import logging

logger = logging.getLogger("UnifiedDataset")

class MessageFormatter:
    SYSTEM_INSTRUCTION_REASONING = "You are a helpful AI assistant. Think step by step and show your reasoning."

    @classmethod
    def format_legacy(cls, prompt: str, response: str, reasoning: str = "") -> List[Dict]:
        messages = [{"role": "system", "content": cls.SYSTEM_INSTRUCTION_REASONING}]
        messages.append({"role": "user", "content": prompt})
        
        if reasoning:
            ans = f"Here's my analysis:\n\n{reasoning}\n\n**Answer:**\n{response}"
        else:
            ans = response
            
        messages.append({"role": "assistant", "content": ans})
        return messages

class UnifiedDataset(Dataset):
    def __init__(
        self,
        data_paths: List[str],
        tokenizer: Any,
        max_length: int = 512,
        seed: int = 42,
        use_pope: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_pope = use_pope
        
        self.im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
        self.examples: List[List[Dict]] = []
        
        random.seed(seed)
        for path in data_paths:
            if not os.path.exists(path): 
                continue
            
            logger.info(f"Lade Datei: {path}")
            
            # 1. PARQUET (Tool-Use): Wir laden ALLES!
            if path.endswith('.parquet'):
                try:
                    df = pd.read_parquet(path)
                    col_name = 'messages' if 'messages' in df.columns else ('conversations' if 'conversations' in df.columns else df.columns[0])
                    tools_data = df[col_name].tolist()
                    self.examples.extend(tools_data)
                    logger.info(f"-> {len(tools_data)} Tool-Use Samples aus Parquet geladen.")
                except Exception as e:
                    logger.error(f"Fehler beim Laden von {path}: {e}")
                
            # 2. JSON/JSONL (Altes Wissen): RAM-SCHUTZ! Maximal 20.000 Samples!
            elif path.endswith('.json') or path.endswith('.jsonl'):
                loaded_count = 0
                max_json_samples = 20000 
                
                with open(path, "r", encoding="utf-8") as f:
                    # Smart Peek
                    try:
                        first_char = f.read(1)
                        f.seek(0)
                        
                        if first_char == '[':
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data[:max_json_samples]:
                                    if "messages" in item:
                                        self.examples.append(item["messages"])
                                        loaded_count += 1
                                    elif isinstance(item, list):
                                        self.examples.append(item)
                                        loaded_count += 1
                                    elif "instruction" in item and "output" in item:
                                        self.examples.append(MessageFormatter.format_legacy(item["instruction"], item["output"]))
                                        loaded_count += 1
                        else:
                            for line in f:
                                if loaded_count >= max_json_samples: break
                                line = line.strip()
                                if not line: continue
                                try:
                                    item = json.loads(line)
                                    if "messages" in item:
                                        self.examples.append(item["messages"])
                                        loaded_count += 1
                                    elif isinstance(item, list):
                                        self.examples.append(item)
                                        loaded_count += 1
                                    elif "instruction" in item and "output" in item:
                                        self.examples.append(MessageFormatter.format_legacy(item["instruction"], item["output"]))
                                        loaded_count += 1
                                except: pass
                    except Exception as e:
                        logger.warning(f"Fehler beim Lesen von {path}: {e}")
                        
                logger.info(f"-> {loaded_count} Legacy Samples geladen.")

        random.shuffle(self.examples)
        logger.info(f"=== Gesamt: {len(self.examples)} Konversationen im RAM ===")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw_messages = self.examples[idx]
        
        # --- SANITIZE MESSAGES ---
        # Der wichtigste Fix: Wir stellen sicher, dass 'content' niemals None ist.
        messages = []
        for msg in raw_messages:
            clean_msg = msg.copy()
            # Wenn content None ist (bei Tool Calls oft der Fall), mache es zum leeren String
            if clean_msg.get("content") is None:
                clean_msg["content"] = ""
            
            # Sicherheitscheck: Alles muss String sein
            if clean_msg.get("content") is not None:
                clean_msg["content"] = str(clean_msg["content"])
                
            messages.append(clean_msg)
        # -------------------------
        
        try:
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception as e:
            # Fallback falls Template immer noch crasht (sollte nicht passieren)
            # Wir bauen einen simplen String
            full_text = ""
            for m in messages:
                full_text += f"<|im_start|>{m.get('role','user')}\n{m.get('content','')}<|im_end|>\n"
        
        enc = self.tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        
        if input_ids.numel() > self.max_length:
            input_ids = input_ids[-self.max_length:]
            
        seq_len = input_ids.numel()
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        
        in_assistant_turn = False
        for i in range(seq_len):
            if input_ids[i] == self.im_start:
                # Lookahead
                if i + 3 < seq_len:
                    lookahead = self.tokenizer.decode(input_ids[i+1:i+4]).strip().lower()
                    if lookahead.startswith("assistant") or lookahead.startswith("tool"):
                        in_assistant_turn = True
                    else:
                        in_assistant_turn = False
                else:
                    in_assistant_turn = False
                    
            if in_assistant_turn:
                labels[i] = input_ids[i]
                
            if input_ids[i] == self.im_start:
                labels[i] = -100

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        if seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def create_dataloaders(data_paths: List[str], tokenizer: Any, batch_size: int = 4, max_length: int = 512, train_ratio: float = 0.95, **kwargs):
    val_paths = [p for p in data_paths if 'valid' in p.lower() or 'test' in p.lower()]
    train_paths = [p for p in data_paths if p not in val_paths]
    
    if not train_paths: train_paths = val_paths

    logger.info("=== Erstelle Train Dataset ===")
    train_dataset = UnifiedDataset(train_paths, tokenizer, max_length, use_pope=True)
    
    if val_paths:
        logger.info("=== Erstelle Validation Dataset ===")
        val_dataset = UnifiedDataset(val_paths, tokenizer, max_length, use_pope=False)
    else:
        total_len = len(train_dataset.examples)
        split_idx = int(total_len * train_ratio)
        val_dataset = UnifiedDataset([], tokenizer, max_length)
        val_dataset.examples = train_dataset.examples[split_idx:]
        train_dataset.examples = train_dataset.examples[:split_idx]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader