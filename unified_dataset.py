#!/usr/bin/env python3
"""
Unified Dataset Loader
======================
Extrem schnelle Version für bereits bereinigte SFT-Parquet-Dateien.
"""

import os
import random
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

logger = logging.getLogger("UnifiedDataset")

class CleanMappedDataset(Dataset):
    def __init__(self, data_paths, tokenizer, max_length=512, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]
        
        self.datasets =[]
        for path in data_paths:
            if not os.path.exists(path): continue
            logger.info(f"Mappe saubere Datei: {path} ...")
            self.datasets.append(load_dataset("parquet", data_files=path, split="train"))

        self.index_map =[]
        for ds_idx, ds in enumerate(self.datasets):
            self.index_map.extend([(ds_idx, i) for i in range(len(ds))])
            
        random.seed(seed)
        random.shuffle(self.index_map)
        logger.info(f"=== READY: {len(self.index_map)} Beispiele im schnellen Zugriff! ===")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ds_idx, item_idx = self.index_map[idx]
        item = self.datasets[ds_idx][item_idx]
        messages = item["messages"]
        
        # CPU-Schutz für sehr lange Code-Dateien
        if len(messages) > 10: messages = messages[-10:]
            
        try:
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            if len(full_text) > self.max_length * 5:
                full_text = full_text[-(self.max_length * 5):]
        except Exception:
            return self.__getitem__((idx + 1) % len(self.index_map)) 
            
        enc = self.tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        
        if input_ids.numel() > self.max_length:
            input_ids = input_ids[-self.max_length:]
            
        seq_len = input_ids.numel()
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        
        # Smart Masking (Lernt nur Assistant-Tokens)
        in_assistant_turn = False
        for i in range(seq_len):
            if input_ids[i] == self.im_start:
                if i + 3 < seq_len:
                    lookahead = self.tokenizer.decode(input_ids[i+1:i+4]).strip().lower()
                    in_assistant_turn = lookahead.startswith("assistant") or lookahead.startswith("tool")
                else:
                    in_assistant_turn = False
                    
            if in_assistant_turn: labels[i] = input_ids[i]
            if input_ids[i] == self.im_start: labels[i] = -100

        pad_id = self.tokenizer.pad_token_id or 0
        if seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

        labels[attention_mask == 0] = -100
        
        if not (labels != -100).any():
            return self.__getitem__((idx + 1) % len(self.index_map))

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def create_dataloaders(tokenizer, batch_size=4, max_length=512, **kwargs):
    data_paths =[os.path.join("data", f) for f in os.listdir("data") if f.endswith('.parquet')]
    if not data_paths: raise ValueError("Keine .parquet Dateien in data/ gefunden!")
        
    train_dataset = CleanMappedDataset(data_paths, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    
    return train_loader, None