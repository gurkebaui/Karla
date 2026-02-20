#!/usr/bin/env python3
"""
finetune_scales.py — Fix scales on existing pre-trained model
==============================================================

This script takes an existing pre-trained model and fine-tunes it
with proper scale values, forcing CTM to contribute meaningfully.

Use this if you already have a pre-trained checkpoint but the scales
are too small (near zero).
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from karla.utils.config import KarlaConfig
from karla.models.karla import create_karla

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Finetune-Scales")


class SimpleDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        text = data.get("prompt") or data.get("question") or ""
                        if data.get("reasoning"):
                            text += " " + data["reasoning"]
                        if data.get("answer"):
                            text += " Answer: " + data["answer"]
                        
                        tokens = tokenizer.encode(
                            text, truncation=True, max_length=max_length
                        )
                        if len(tokens) > 10:
                            self.samples.append(tokens)
                    except:
                        continue
        
        logger.info(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {"input_ids": input_ids, "labels": labels}


def init_scales(model, l1_scale: float, ctm_scale: float):
    """Initialize scales to target values."""
    if l1_scale > 0:
        l1_raw = math.log(math.exp(l1_scale) - 1 + 1e-8)
        model.l1_scale_raw.data = torch.tensor(l1_raw)
    if ctm_scale > 0:
        ctm_raw = math.log(math.exp(ctm_scale) - 1 + 1e-8)
        model.ctm_scale_raw.data = torch.tensor(ctm_raw)


def finetune_scales(args):
    logger.info("=" * 60)
    logger.info("FINE-TUNING SCALES")
    logger.info("=" * 60)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    config = KarlaConfig()
    model = create_karla(config)
    
    # Load existing checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif "ctm_state_dict" in ckpt:
            model.l2.load_state_dict(ckpt["ctm_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        
        logger.info("Checkpoint loaded")
    
    # Initialize scales to target values
    init_scales(model, args.target_l1_scale, args.target_ctm_scale)
    logger.info(f"Scales initialized: L1={model.l1_scale().item():.4f}, CTM={model.ctm_scale().item():.4f}")
    
    # Move to device
    for name, module in model.named_children():
        if name != "l0":
            module.to(device)
    model.l1_scale_raw.data = model.l1_scale_raw.data.to(device)
    model.ctm_scale_raw.data = model.ctm_scale_raw.data.to(device)
    
    # Dataset
    dataset = SimpleDataset(args.data_path, tokenizer, max_length=args.max_length)
    
    if len(dataset) == 0:
        logger.warning("No data, using dummy data for scale tuning")
        # Create dummy data
        dummy_tokens = tokenizer.encode("The quick brown fox jumps over the lazy dog.")
        dummy_tokens = dummy_tokens + [tokenizer.pad_token_id] * (args.max_length - len(dummy_tokens))
        dataset = [{"input_ids": torch.tensor(dummy_tokens), "labels": torch.tensor(dummy_tokens)}]
    
    dataloader = DataLoader(
        dataset if isinstance(dataset, list) else dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        } if isinstance(batch[0], dict) else batch
    )
    
    # Optimizer - only for scales and CTM output projection
    # This freezes most weights and only trains what's needed for scale adaptation
    scale_params = [model.l1_scale_raw, model.ctm_scale_raw]
    
    if args.train_ctm_output:
        # Also train CTM output projection
        scale_params.extend(model.l2.out_proj.parameters())
        logger.info("Training CTM output projection too")
    
    optimizer = torch.optim.Adam(scale_params, lr=args.lr)
    
    # Fine-tuning loop
    model.train()
    
    for step in range(args.num_steps):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            
            # Forward
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            
            # Add scale regularization
            l1_scale = model.l1_scale()
            ctm_scale = model.ctm_scale()
            
            # Encourage scales to stay in range
            scale_reg = torch.zeros((), device=device)
            if l1_scale < args.min_scale:
                scale_reg = scale_reg + (args.min_scale - l1_scale) ** 2
            if ctm_scale < args.min_scale:
                scale_reg = scale_reg + (args.min_scale - ctm_scale) ** 2
            
            total_loss = loss + args.scale_weight * scale_reg
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if step % args.log_interval == 0:
                logger.info(
                    f"Step {step} | loss {loss.item():.4f} | "
                    f"scale_reg {scale_reg.item():.4f} | "
                    f"l1 {l1_scale.item():.4f} | ctm {ctm_scale.item():.4f}"
                )
            
            break  # One batch per step
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "finetuned_model.pt")
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "scales": {
            "l1_scale": model.l1_scale().item(),
            "ctm_scale": model.ctm_scale().item(),
        }
    }, output_path)
    
    logger.info(f"Saved to: {output_path}")
    logger.info(f"Final scales: L1={model.l1_scale().item():.4f}, CTM={model.ctm_scale().item():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune scales")
    
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-path", type=str, default="data/micro_pope_data.jsonl")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--output-dir", type=str, default="checkpoints_finetuned")
    
    parser.add_argument("--target-l1-scale", type=float, default=0.1)
    parser.add_argument("--target-ctm-scale", type=float, default=0.3)
    parser.add_argument("--min-scale", type=float, default=0.05)
    
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--scale-weight", type=float, default=0.1)
    
    parser.add_argument("--train-ctm-output", action="store_true",
                        help="Also train CTM output projection")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-interval", type=int, default=10)
    
    args = parser.parse_args()
    finetune_scales(args)


if __name__ == "__main__":
    main()
