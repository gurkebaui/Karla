#!/usr/bin/env python3
"""
pretrain_ctm.py — CTM Pre-training nach Paper 2505.05522v4
===========================================================

Vollständiges Pre-training für das CTM (Continuous Thought Machine).

Paper-basierte Training-Komponenten:
1. Supervised Pre-training mit Cross-Entropy Loss
2. Synchronization Loss: Encourages diverse neuron dynamics
3. Certainty Regularization: Predict target certainty
4. Value Pre-training: Learn to estimate returns
5. Curriculum: Start with few ticks, increase over training

Paper Quote (Section 4):
"The CTM is pre-trained on the downstream task with additional 
regularization losses before RL fine-tuning."
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
from karla.models.l2_ctm import CTMHead, L2Output

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CTM-Pretrain")


# ============================================================
# 1. Pre-training Dataset
# ============================================================

class PretrainDataset(Dataset):
    """
    Dataset für CTM Pre-training.
    
    Verwendet die gleichen Daten wie RL, aber formatiert für:
    - Next-token prediction (supervised)
    - Certainty targets (based on sequence position)
    - Value targets (based on answer quality proxy)
    """
    
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
        logger.info(f"[Pretrain] Loaded {len(self.samples)} samples")
    
    def _extract_answer(self, reasoning: str) -> str:
        import re
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", reasoning, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()[:500]
        return ""
    
    def _load(self, path: str):
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
                
                # Format: prompt + reasoning + answer
                full_text = prompt
                if reasoning:
                    # Clean reasoning
                    reasoning = reasoning.replace("<think", "").replace("</think", "")
                    reasoning = reasoning.replace("<answer>", "").replace("</answer>", "")
                    full_text += "\n" + reasoning.strip()
                if answer:
                    full_text += "\nAnswer: " + answer
                
                # Tokenize
                tokens = self.tokenizer.encode(
                    full_text, 
                    truncation=True, 
                    max_length=self.max_seq_length,
                    add_special_tokens=True
                )
                
                if len(tokens) >= self.min_seq_length:
                    # Compute targets
                    prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                    prompt_len = min(len(prompt_tokens) + 1, len(tokens) - 1)  # +1 for BOS
                    
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
        
        # Pad/truncate
        if len(input_ids) < self.max_seq_length:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))
        else:
            input_ids = input_ids[:self.max_seq_length]
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Mask for CTM (we want CTM to process the whole sequence)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # Certainty target: high at end of sequence (answer), low at start
        seq_len = attention_mask.sum().item()
        certainty_target = torch.zeros(self.max_seq_length)
        if seq_len > 0:
            # Certainty increases towards the end
            for i in range(int(seq_len)):
                certainty_target[i] = min(1.0, i / max(seq_len - 1, 1))
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_length": sample["prompt_length"],
            "certainty_target": certainty_target,
            "has_answer": sample["has_answer"],
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "prompt_lengths": [b["prompt_length"] for b in batch],
        "certainty_targets": torch.stack([b["certainty_target"] for b in batch]),
        "has_answers": [b["has_answer"] for b in batch],
    }


# ============================================================
# 2. CTM Pre-training Losses (Paper-based)
# ============================================================

class CTMPretrainLoss:
    """
    Combined pre-training losses for CTM.
    
    Loss Components:
    1. L_lm: Language modeling loss (cross-entropy)
    2. L_sync: Synchronization loss (encourage diverse dynamics)
    3. L_cert: Certainty loss (predict sequence progress)
    4. L_value: Value loss (predict sequence quality)
    
    Total: L = L_lm + α_sync * L_sync + α_cert * L_cert + α_value * L_value
    """
    
    def __init__(
        self,
        alpha_sync: float = 0.1,
        alpha_cert: float = 0.1,
        alpha_value: float = 0.05,
        sync_target: float = 0.5,  # Target sync variance
    ):
        self.alpha_sync = alpha_sync
        self.alpha_cert = alpha_cert
        self.alpha_value = alpha_value
        self.sync_target = sync_target
    
    def compute_sync_loss(self, sync_stat: torch.Tensor) -> torch.Tensor:
        """
        Synchronization Loss: Encourage diverse neuron dynamics.
        
        Paper (Section 3.4):
        "We encourage the neurons to have diverse dynamics by 
        penalizing low synchronization variance."
        
        Target: High variance in synchronization values across ticks.
        """
        # sync_stat should have variance (diverse dynamics)
        # Loss = (target_variance - actual_variance)^2
        # But we want to maximize variance, so minimize negative variance
        
        if sync_stat.dim() == 1:
            sync_var = sync_stat.var()
        else:
            sync_var = sync_stat.var(dim=-1).mean()
        
        # Encourage variance to be close to target (not too high, not too low)
        loss = F.mse_loss(sync_var, torch.tensor(self.sync_target, device=sync_stat.device))
        
        return loss
    
    def compute_certainty_loss(
        self,
        certainty: torch.Tensor,
        certainty_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Certainty Loss: Predict sequence progress.
        
        The CTM should learn to be uncertain at the start (prompt)
        and certain at the end (answer).
        """
        # certainty: (B, 1) - final certainty from CTM
        # certainty_target: (B, S) - per-position targets
        
        # Use mean of last few positions as target
        if certainty_target.dim() == 2:
            # Take mean of last 10 positions (or all if shorter)
            B, S = certainty_target.shape
            target = certainty_target[:, -10:].mean(dim=1, keepdim=True)
        else:
            target = certainty_target
        
        loss = F.binary_cross_entropy(certainty.clamp(0.01, 0.99), target.clamp(0.01, 0.99))
        
        return loss
    
    def compute_value_loss(
        self,
        value: torch.Tensor,
        has_answer: List[bool],
    ) -> torch.Tensor:
        """
        Value Loss: Predict sequence quality.
        
        Sequences with answers should have higher value.
        """
        if not any(has_answer):
            return torch.zeros((), device=value.device)
        
        # Target: 1.0 for sequences with answer, 0.0 for others
        target = torch.tensor(
            [1.0 if has else 0.0 for has in has_answer],
            device=value.device,
            dtype=torch.float32
        ).unsqueeze(1)
        
        loss = F.mse_loss(value, target)
        
        return loss
    
    def compute_loss(
        self,
        lm_loss: torch.Tensor,
        ctm_output: L2Output,
        certainty_target: torch.Tensor,
        has_answer: List[bool],
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute total pre-training loss."""
        
        # Synchronization loss
        sync_loss = self.compute_sync_loss(ctm_output.synchronization)
        
        # Certainty loss
        cert_loss = self.compute_certainty_loss(ctm_output.certainty, certainty_target)
        
        # Value loss
        value_loss = self.compute_value_loss(ctm_output.value, has_answer)
        
        # Total
        total_loss = (
            lm_loss
            + self.alpha_sync * sync_loss
            + self.alpha_cert * cert_loss
            + self.alpha_value * value_loss
        )
        
        stats = {
            "lm_loss": lm_loss.item(),
            "sync_loss": sync_loss.item(),
            "cert_loss": cert_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }
        
        return total_loss, stats


# ============================================================
# 3. Curriculum Learning
# ============================================================

class TickCurriculum:
    """
    Curriculum for internal ticks during pre-training.
    
    Paper (Section 4.2):
    "We start with fewer internal ticks and gradually increase
    to allow the CTM to learn incrementally."
    """
    
    def __init__(
        self,
        start_ticks: int = 2,
        end_ticks: int = 10,
        warmup_steps: int = 1000,
    ):
        self.start_ticks = start_ticks
        self.end_ticks = end_ticks
        self.warmup_steps = warmup_steps
    
    def get_ticks(self, step: int) -> int:
        """Get number of ticks for current step."""
        if step >= self.warmup_steps:
            return self.end_ticks
        
        # Linear interpolation
        progress = step / self.warmup_steps
        ticks = self.start_ticks + int(progress * (self.end_ticks - self.start_ticks))
        
        return ticks


# ============================================================
# 4. CTM Wrapper for Pre-training
# ============================================================

class CTMPretrainer(nn.Module):
    """
    CTM with LM head for pre-training.
    
    Takes hidden states from L0 (Qwen) and trains CTM to:
    1. Produce useful representations for language modeling
    2. Learn synchronization dynamics
    3. Predict certainty and value
    """
    
    def __init__(
        self,
        ctm: CTMHead,
        hidden_dim: int = 1536,
        vocab_size: int = 151936,
    ):
        super().__init__()
        self.ctm = ctm
        self.hidden_dim = hidden_dim
        
        # LM head for next-token prediction
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Scale for CTM contribution (trainable)
        self.ctm_scale = nn.Parameter(torch.zeros(1))  # Start at 0
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_ticks: Optional[int] = None,
    ) -> Tuple[torch.Tensor, L2Output]:
        """
        Forward pass.
        
        Args:
            hidden_states: (B, S, hidden_dim) from L0
            attention_mask: (B, S)
            max_ticks: Override CTM ticks
            
        Returns:
            logits: (B, S, vocab_size)
            ctm_output: L2Output from CTM
        """
        B, S, D = hidden_states.shape
        device = hidden_states.device
        
        # Create dummy L1 memory (zeros for pre-training)
        l1_mem = torch.zeros(B, S, 256, device=device)
        
        # CTM forward
        ctm_output = self.ctm(
            hidden_states, 
            l1_mem, 
            attention_mask=attention_mask,
            max_ticks=max_ticks
        )
        
        # Add CTM features to hidden states
        scale = torch.sigmoid(self.ctm_scale)  # 0 to 1
        enhanced = hidden_states + scale * ctm_output.features.unsqueeze(1)
        
        # LM head
        logits = self.lm_head(enhanced)
        
        return logits, ctm_output


# ============================================================
# 5. Training Loop
# ============================================================

def run_pretraining(args):
    logger.info("=" * 60)
    logger.info("CTM PRE-TRAINING (Paper 2505.05522v4)")
    logger.info("=" * 60)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = PretrainDataset(
        args.data_path,
        tokenizer,
        max_seq_length=args.max_seq_length,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    
    # Create CTM
    config = KarlaConfig()
    ctm = CTMHead(
        hidden_dim=config.l2.hidden_dim,
        num_neurons=config.l2.num_neurons,
        num_internal_ticks=config.l2.num_internal_ticks,
        use_bitnet=config.l2.use_bitnet,
        l0_hidden_dim=1536,
        nlm_history_length=config.l2.nlm_history_length,
        nlm_hidden_dim=config.l2.nlm_hidden_dim,
        num_action_pairs=config.l2.num_action_pairs,
        num_output_pairs=config.l2.num_output_pairs,
        attn_heads=config.l2.attn_heads,
        truncation_period=args.truncation_period,
    )
    
    # Create pre-trainer
    pretrainer = CTMPretrainer(
        ctm=ctm,
        hidden_dim=1536,
        vocab_size=tokenizer.vocab_size,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in pretrainer.parameters())
    trainable_params = sum(p.numel() for p in pretrainer.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss
    loss_fn = CTMPretrainLoss(
        alpha_sync=args.alpha_sync,
        alpha_cert=args.alpha_cert,
        alpha_value=args.alpha_value,
    )
    
    # Curriculum
    curriculum = TickCurriculum(
        start_ticks=args.start_ticks,
        end_ticks=args.end_ticks,
        warmup_steps=args.curriculum_steps,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        pretrainer.parameters(),
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
    
    # Load L0 for hidden states
    logger.info("Loading L0 (Qwen) for hidden states...")
    from karla.models.l0_perception import L0Perception
    l0 = L0Perception(model_name=args.model_name, bits=4)
    l0.eval()
    for param in l0.parameters():
        param.requires_grad = False
    logger.info("L0 loaded (frozen)")
    
    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    global_step = 0
    best_loss = float("inf")
    
    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_lm_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Get ticks from curriculum
            current_ticks = curriculum.get_ticks(global_step)
            
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            certainty_targets = batch["certainty_targets"].to(device)
            has_answers = batch["has_answers"]
            
            # Get L0 hidden states (no grad)
            with torch.no_grad():
                l0_out = l0(input_ids, attention_mask)
                hidden_states = l0_out.hidden_states.to(device).float()
            
            # Forward through CTM pretrainer
            logits, ctm_output = pretrainer(
                hidden_states,
                attention_mask=attention_mask,
                max_ticks=current_ticks,
            )
            
            # Compute LM loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            
            # Compute total loss
            total_loss, stats = loss_fn.compute_loss(
                lm_loss,
                ctm_output,
                certainty_targets,
                has_answers,
            )
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), args.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            # Update stats
            epoch_loss += total_loss.item()
            epoch_lm_loss += lm_loss.item()
            num_batches += 1
            global_step += 1
            
            # Logging
            if global_step % args.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {global_step:6d} | "
                    f"ticks {current_ticks:2d} | "
                    f"lm {stats['lm_loss']:.4f} | "
                    f"sync {stats['sync_loss']:.4f} | "
                    f"cert {stats['cert_loss']:.4f} | "
                    f"value {stats['value_loss']:.4f} | "
                    f"total {stats['total_loss']:.4f} | "
                    f"lr {lr:.2e} | "
                    f"scale {torch.sigmoid(pretrainer.ctm_scale).item():.4f}"
                )
        
        # Epoch stats
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_batches
        avg_lm_loss = epoch_lm_loss / num_batches
        
        logger.info(
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"avg_loss {avg_loss:.4f} | "
            f"avg_lm {avg_lm_loss:.4f} | "
            f"time {epoch_time:.1f}s"
        )
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(args.output_dir, "best_ctm.pt")
            torch.save({
                "ctm_state_dict": ctm.state_dict(),
                "lm_head_state_dict": pretrainer.lm_head.state_dict(),
                "ctm_scale": pretrainer.ctm_scale.item(),
                "epoch": epoch,
                "loss": avg_loss,
                "global_step": global_step,
            }, checkpoint_path)
            logger.info(f"Saved best checkpoint: {checkpoint_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, f"ctm_epoch_{epoch+1}.pt")
            torch.save({
                "ctm_state_dict": ctm.state_dict(),
                "lm_head_state_dict": pretrainer.lm_head.state_dict(),
                "ctm_scale": pretrainer.ctm_scale.item(),
                "epoch": epoch,
                "loss": avg_loss,
                "global_step": global_step,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    logger.info("=" * 60)
    logger.info(f"Pre-training complete! Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoint saved to: {args.output_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="CTM Pre-training")
    
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
    
    # CTM
    parser.add_argument("--truncation-period", type=int, default=4)
    
    # Curriculum
    parser.add_argument("--start-ticks", type=int, default=2, help="Starting internal ticks")
    parser.add_argument("--end-ticks", type=int, default=10, help="Final internal ticks")
    parser.add_argument("--curriculum-steps", type=int, default=1000, help="Steps to reach end_ticks")
    
    # Loss weights
    parser.add_argument("--alpha-sync", type=float, default=0.1, help="Synchronization loss weight")
    parser.add_argument("--alpha-cert", type=float, default=0.1, help="Certainty loss weight")
    parser.add_argument("--alpha-value", type=float, default=0.05, help="Value loss weight")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints_ctm_pretrain")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1)
    
    args = parser.parse_args()
    run_pretraining(args)


if __name__ == "__main__":
    main()
