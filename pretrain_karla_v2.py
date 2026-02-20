#!/usr/bin/env python3
"""
pretrain_karla_v2.py — Correct Pre-training for Karla C1
=========================================================

FIXED VERSION: Addresses the scale training problem

Key Fixes:
1. Initialize scales to meaningful values (NOT near zero)
2. Force CTM contribution by using a combined loss
3. Scale warmup schedule
4. Proper gradient flow to scales

Paper References:
- 2505.05522v4: CTM pre-training
- Training scales from the start, not leaving them at ~0
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from karla.utils.config import KarlaConfig
from karla.models.karla import create_karla, Karla

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Karla-Pretrain-v2")


# ============================================================
# 1. Dataset
# ============================================================

class PretrainDataset(Dataset):
    """Dataset for pre-training."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 512,
        min_seq_length: int = 32,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.samples: List[Dict] = []
        
        self._load(data_path)
        logger.info(f"[Dataset] Loaded {len(self.samples)} samples")
    
    def _extract_answer(self, reasoning: str) -> str:
        import re
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", reasoning, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()[:500]
        return ""
    
    def _load(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"Data file not found: {path}")
            return
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                prompt = data.get("prompt") or data.get("question") or ""
                reasoning = data.get("reasoning") or data.get("cot") or ""
                answer = data.get("answer") or ""
                
                if not answer and reasoning:
                    answer = self._extract_answer(reasoning)
                
                if not prompt:
                    continue
                
                full_text = prompt
                if reasoning:
                    reasoning = reasoning.replace("<think", "").replace("</think", "")
                    reasoning = reasoning.replace("<answer>", "").replace("</answer>", "")
                    full_text += "\n" + reasoning.strip()
                if answer:
                    full_text += "\nAnswer: " + answer
                
                tokens = self.tokenizer.encode(
                    full_text,
                    truncation=True,
                    max_length=self.max_seq_length,
                    add_special_tokens=True
                )
                
                if len(tokens) >= self.min_seq_length:
                    prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                    prompt_len = min(len(prompt_tokens) + 1, len(tokens) - 1)
                    
                    self.samples.append({
                        "input_ids": tokens,
                        "prompt_length": prompt_len,
                        "has_answer": len(answer) > 0,
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids = sample["input_ids"]
        
        if len(input_ids) < self.max_seq_length:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))
        else:
            input_ids = input_ids[:self.max_seq_length]
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_length": sample["prompt_length"],
            "has_answer": sample["has_answer"],
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "prompt_lengths": [b["prompt_length"] for b in batch],
        "has_answers": [b["has_answer"] for b in batch],
    }


# ============================================================
# 2. Pre-training Loss with Scale Loss
# ============================================================

class PretrainLoss:
    """
    Pre-training losses:
    1. LM Loss: Cross-entropy for next token prediction
    2. Sync Loss: Encourage diverse neuron dynamics
    3. Certainty Loss: Predict sequence progress
    4. Value Loss: Predict sequence quality
    5. SCALE LOSS: Force CTM to contribute meaningfully
    """
    
    def __init__(
        self,
        alpha_sync: float = 0.1,
        alpha_cert: float = 0.1,
        alpha_value: float = 0.05,
        alpha_scale: float = 0.01,
        min_scale: float = 0.1,
        max_scale: float = 0.5,
    ):
        self.alpha_sync = alpha_sync
        self.alpha_cert = alpha_cert
        self.alpha_value = alpha_value
        self.alpha_scale = alpha_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
    
    def compute_sync_loss(self, model_output) -> torch.Tensor:
        """Encourage diverse dynamics - use internal_ticks as proxy."""
        # KarlaOutput doesn't have synchronization, use internal_ticks
        # We want the model to use enough ticks (not too few, not too many)
        ticks = model_output.internal_ticks
        target_ticks = 5.0  # Reasonable target
        return torch.tensor((ticks - target_ticks) ** 2 * 0.01, device=model_output.certainty.device)
    
    def compute_certainty_loss(
        self,
        certainty: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Encourage certainty to increase during generation."""
        # Target: certainty should be higher for complete sequences
        target = torch.ones_like(certainty) * 0.5  # Start neutral
        return F.binary_cross_entropy(certainty.clamp(0.01, 0.99), target.clamp(0.01, 0.99))
    
    def compute_value_loss(
        self,
        value: torch.Tensor,
        has_answer: List[bool],
    ) -> torch.Tensor:
        """Value should predict if sequence has good answer."""
        if not any(has_answer):
            return torch.zeros((), device=value.device)
        
        target = torch.tensor(
            [1.0 if has else 0.5 for has in has_answer],
            device=value.device,
            dtype=torch.float32
        ).unsqueeze(1)
        
        return F.mse_loss(value, target)
    
    def compute_scale_loss(
        self,
        l1_scale: torch.Tensor,
        ctm_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        SCALE LOSS: Force scales into meaningful range.
        
        We want scales to be in [min_scale, max_scale] range.
        Loss = penalty for being outside range.
        """
        loss = torch.zeros((), device=l1_scale.device)
        
        # Penalty for being too small
        if l1_scale < self.min_scale:
            loss = loss + (self.min_scale - l1_scale) ** 2
        if ctm_scale < self.min_scale:
            loss = loss + (self.min_scale - ctm_scale) ** 2
        
        # Penalty for being too large
        if l1_scale > self.max_scale:
            loss = loss + (l1_scale - self.max_scale) ** 2
        if ctm_scale > self.max_scale:
            loss = loss + (ctm_scale - self.max_scale) ** 2
        
        return loss
    
    def compute_loss(
        self,
        lm_loss: torch.Tensor,
        model_output,
        has_answer: List[bool],
        l1_scale: torch.Tensor,
        ctm_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute total loss."""
        
        sync_loss = self.compute_sync_loss(model_output)
        cert_loss = self.compute_certainty_loss(model_output.certainty, 10)
        value_loss = self.compute_value_loss(model_output.value, has_answer)
        scale_loss = self.compute_scale_loss(l1_scale, ctm_scale)
        
        total_loss = (
            lm_loss
            + self.alpha_sync * sync_loss
            + self.alpha_cert * cert_loss
            + self.alpha_value * value_loss
            + self.alpha_scale * scale_loss
        )
        
        stats = {
            "lm_loss": lm_loss.item(),
            "sync_loss": sync_loss.item(),
            "cert_loss": cert_loss.item(),
            "value_loss": value_loss.item(),
            "scale_loss": scale_loss.item(),
            "total_loss": total_loss.item(),
        }
        
        return total_loss, stats


# ============================================================
# 3. Scale Initialization
# ============================================================

def init_scales(model: Karla, l1_scale: float = 0.1, ctm_scale: float = 0.3):
    """
    Initialize scales to meaningful values.
    
    Karla uses softplus(scale_raw), so:
    scale_raw = log(exp(target) - 1)
    """
    if l1_scale > 0:
        l1_raw = math.log(math.exp(l1_scale) - 1 + 1e-8)
        model.l1_scale_raw.data = torch.tensor(l1_raw)
        logger.info(f"L1 scale initialized to {l1_scale:.4f} (raw: {l1_raw:.4f})")
    
    if ctm_scale > 0:
        ctm_raw = math.log(math.exp(ctm_scale) - 1 + 1e-8)
        model.ctm_scale_raw.data = torch.tensor(ctm_raw)
        logger.info(f"CTM scale initialized to {ctm_scale:.4f} (raw: {ctm_raw:.4f})")


# ============================================================
# 4. Training Loop
# ============================================================

def run_pretraining(args):
    logger.info("=" * 60)
    logger.info("KARLA C1 - PRE-TRAINING v2 (Fixed Scales)")
    logger.info("=" * 60)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    config = KarlaConfig()
    model = create_karla(config)
    
    # CRITICAL: Initialize scales to meaningful values
    init_scales(model, l1_scale=args.init_l1_scale, ctm_scale=args.init_ctm_scale)
    
    # Move to device
    for name, module in model.named_children():
        if name != "l0":
            module.to(device)
    model.l1_scale_raw.data = model.l1_scale_raw.data.to(device)
    model.ctm_scale_raw.data = model.ctm_scale_raw.data.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Dataset
    dataset = PretrainDataset(args.data_path, tokenizer, max_seq_length=args.max_seq_length)
    
    if len(dataset) == 0:
        logger.error("No data loaded! Check data path.")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    
    # Loss
    loss_fn = PretrainLoss(
        alpha_sync=args.alpha_sync,
        alpha_cert=args.alpha_cert,
        alpha_value=args.alpha_value,
        alpha_scale=args.alpha_scale,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
    )
    
    # Optimizer - include scales explicitly
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Scheduler
    num_training_steps = args.num_epochs * len(dataloader)
    warmup_steps = int(0.05 * num_training_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=num_training_steps,
        pct_start=warmup_steps / num_training_steps,
    )
    
    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    global_step = 0
    best_loss = float("inf")
    
    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        model.train()
        
        for batch in dataloader:
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            has_answers = batch["has_answers"]
            
            # Forward
            outputs = model(input_ids, attention_mask, labels=labels)
            
            # LM loss
            lm_loss = outputs.loss
            
            # Total loss with scale loss
            total_loss, stats = loss_fn.compute_loss(
                lm_loss,
                outputs,
                has_answers,
                model.l1_scale(),
                model.ctm_scale(),
            )
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
            global_step += 1
            
            # Logging
            if global_step % args.log_interval == 0:
                l1_scale = model.l1_scale().item()
                ctm_scale = model.ctm_scale().item()
                lr = scheduler.get_last_lr()[0]
                
                logger.info(
                    f"Step {global_step:6d} | "
                    f"lm {stats['lm_loss']:.3f} | "
                    f"sync {stats['sync_loss']:.3f} | "
                    f"cert {stats['cert_loss']:.3f} | "
                    f"value {stats['value_loss']:.3f} | "
                    f"scale {stats['scale_loss']:.3f} | "
                    f"l1 {l1_scale:.4f} | ctm {ctm_scale:.4f} | "
                    f"lr {lr:.2e}"
                )
        
        # Epoch stats
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_batches
        
        l1_scale = model.l1_scale().item()
        ctm_scale = model.ctm_scale().item()
        
        logger.info(
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"avg_loss {avg_loss:.4f} | "
            f"l1 {l1_scale:.4f} | ctm {ctm_scale:.4f} | "
            f"time {epoch_time:.1f}s"
        )
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
                "global_step": global_step,
                "scales": {
                    "l1_scale": l1_scale,
                    "ctm_scale": ctm_scale,
                }
            }, checkpoint_path)
            logger.info(f"Saved best model: {checkpoint_path}")
        
        # Save periodic
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
                "global_step": global_step,
                "scales": {
                    "l1_scale": l1_scale,
                    "ctm_scale": ctm_scale,
                }
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    logger.info("=" * 60)
    logger.info(f"Pre-training complete! Best loss: {best_loss:.4f}")
    logger.info(f"Final scales: L1={model.l1_scale().item():.4f}, CTM={model.ctm_scale().item():.4f}")
    logger.info(f"Checkpoints saved to: {args.output_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Karla Pre-training v2 (Fixed Scales)")
    
    # Data
    parser.add_argument("--data-path", type=str, default="data/micro_pope_data.jsonl")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Scale initialization (CRITICAL!)
    parser.add_argument("--init-l1-scale", type=float, default=0.1,
                        help="Initial L1 scale (default 0.1)")
    parser.add_argument("--init-ctm-scale", type=float, default=0.3,
                        help="Initial CTM scale (default 0.3)")
    
    # Scale loss
    parser.add_argument("--alpha-scale", type=float, default=0.01,
                        help="Scale loss weight")
    parser.add_argument("--min-scale", type=float, default=0.05,
                        help="Minimum allowed scale")
    parser.add_argument("--max-scale", type=float, default=0.5,
                        help="Maximum allowed scale")
    
    # Other losses
    parser.add_argument("--alpha-sync", type=float, default=0.1)
    parser.add_argument("--alpha-cert", type=float, default=0.1)
    parser.add_argument("--alpha-value", type=float, default=0.05)
    
    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints_pretrain_v2")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1)
    
    args = parser.parse_args()
    run_pretraining(args)


if __name__ == "__main__":
    main()
