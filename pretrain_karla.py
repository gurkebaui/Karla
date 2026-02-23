#!/usr/bin/env python3
"""
pretrain_karla.py - Phase 3: Tool-Use & Continual Pretraining
=============================================================
Features:
- Resume Capability (Load previous CTM state)
- Parquet Support (For Tool-Use Datasets)
- Parallel Sequence-Level CTM (Fast & Stable)
- Graceful Exit on Ctrl+C
"""

import argparse
import logging
import os
import sys
import time
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "karla"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PretrainKarla")

from karla.models.karla import create_karla
from karla.utils.config import KarlaConfig

@dataclass
class PretrainConfig:
    data_paths: List[str] = field(default_factory=list)
    max_length: int = 512       
    train_ratio: float = 0.95
    max_samples_per_file: Optional[int] = None
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    l2_lr: float = 1e-4  
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0   
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.1
    use_pope: bool = True
    use_amp: bool = True
    nan_patience: int = 5
    output_dir: str = "checkpoints_pretrain"
    save_interval: int = 500
    eval_interval: int = 200
    log_interval: int = 10
    device: str = "cuda"
    use_mock_l0: bool = False
    use_multi_loss: bool = True  
    multi_loss_start_step: int = 50  

class KarlaPretrainWrapper(nn.Module):
    def __init__(self, config: KarlaConfig = None, use_mock_l0: bool = False):
        super().__init__()
        self.config = config or KarlaConfig()
        self.model = create_karla(self.config, use_mock=use_mock_l0)
        
        # Check CTM Scale sanity
        with torch.no_grad():
            if self.model.ctm_scale().item() < 0.1 or self.model.ctm_scale().item() > 2.0:
                self.model.ctm_scale_raw.data.fill_(-0.6)
        logger.info(f"CTM scale: {self.model.ctm_scale().item():.4f}")
        
        # Check L1 Scale sanity (if L1 exists)
        if hasattr(self.model, 'l1_scale'):
            logger.info(f"L1 scale:  {self.model.l1_scale().item():.4f}")

    def forward(self, input_ids, attention_mask=None, labels=None, collect_tick_outputs=False):
        return self.model(input_ids, attention_mask, labels, collect_tick_outputs=collect_tick_outputs)

    def ctm_scale(self): return self.model.ctm_scale()
    def count_parameters(self): return self.model.count_parameters()

    def clip_scales(self, max_val: float = 1.0):
        with torch.no_grad():
            if self.model.ctm_scale().item() > max_val:
                self.model.ctm_scale_raw.data.clamp_(max=math.log(math.exp(max_val) - 1))
            if hasattr(self.model, 'l1_scale') and self.model.l1_scale().item() > max_val:
                 self.model.l1_scale_raw.data.clamp_(max=math.log(math.exp(max_val) - 1))

class StableTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader, self.val_loader, self.config = train_loader, val_loader, config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Move params to device
        for name, module in self.model.model.named_children():
            if name != "l0": module.to(self.device)
        self.model.model.ctm_scale_raw.data = self.model.model.ctm_scale_raw.data.to(self.device)
        if hasattr(self.model.model, 'l1_scale_raw'):
             self.model.model.l1_scale_raw.data = self.model.model.l1_scale_raw.data.to(self.device)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.l2_lr, weight_decay=config.weight_decay, betas=(0.9, 0.95), eps=1e-8)

        total_steps = len(train_loader) * config.num_epochs // max(config.gradient_accumulation_steps, 1)
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        def lr_lambda(step):
            if step < warmup_steps: return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return self.config.min_lr_ratio + (1 - self.config.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler('cuda', init_scale=2**12) if self.use_amp else None 
        self.global_step, self.epoch, self.best_loss, self.consecutive_nans = 0, 0, float("inf"), 0
        
        os.makedirs(os.path.abspath(config.output_dir), exist_ok=True)

    def train(self):
        logger.info("=" * 60)
        logger.info("STARTING TRAINING (Phase 2/3)")
        logger.info("Press Ctrl+C to save checkpoint and exit safely.")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            for epoch in range(self.config.num_epochs):
                self.epoch = epoch
                self.consecutive_nans = 0
                
                epoch_loss = self._train_epoch()
                if math.isnan(epoch_loss): break
                
                val_loss = self._evaluate()
                if not math.isnan(val_loss):
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self._save_checkpoint("best_model.pt")
            
            logger.info(f"Training completed in {(time.time() - start_time)/3600:.2f} hours")

        except KeyboardInterrupt:
            logger.info("\n" + "="*60)
            logger.info("INTERRUPT DETECTED (Ctrl+C)")
            logger.info(f"Saving emergency checkpoint at step {self.global_step}...")
            self._save_checkpoint(f"INTERRUPTED_step_{self.global_step}.pt")
            logger.info("Exiting gracefully. Goodbye!")
            sys.exit(0)

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss, num_batches, accum_step = 0.0, 0, 0
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask").to(self.device) if batch.get("attention_mask") is not None else None
            labels = batch.get("labels", input_ids.clone()).to(self.device)

            with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu', enabled=self.use_amp):
                collect_multi = self.config.use_multi_loss and self.global_step >= self.config.multi_loss_start_step
                outputs = self.model(input_ids, attention_mask, labels, collect_tick_outputs=collect_multi)
                loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=self.device)
                scaled_loss = loss / self.config.gradient_accumulation_steps

            if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                self.consecutive_nans += 1
                if self.consecutive_nans >= self.config.nan_patience: break
                self.optimizer.zero_grad()
                continue

            self.consecutive_nans = 0
            if self.use_amp: self.scaler.scale(scaled_loss).backward()
            else: scaled_loss.backward()

            total_loss += loss.item()
            num_batches += 1
            accum_step += 1

            if accum_step >= self.config.gradient_accumulation_steps:
                grad_norm = sum(p.grad.norm().item() ** 2 for p in self.model.parameters() if p.requires_grad and p.grad is not None)
                
                if math.isnan(grad_norm) or math.isinf(grad_norm):
                    self.optimizer.zero_grad()
                    accum_step = 0
                    continue

                if self.use_amp: self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.config.max_grad_norm)
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else: self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                accum_step = 0
                self.model.clip_scales(max_val=1.0)
                self.global_step += 1

                if self.global_step % self.config.log_interval == 0:
                    l1_info = f" | L1: {self.model.model.l1_scale().item():.4f}" if hasattr(self.model.model, 'l1_scale') else ""
                    logger.info(f"[Step {self.global_step}] Loss: {total_loss/num_batches:.4f} | LR: {self.scheduler.get_last_lr()[0]:.2e} | CTM: {self.model.ctm_scale().item():.4f}{l1_info} | GradNorm: {math.sqrt(grad_norm):.2f}")

                if self.global_step % self.config.save_interval == 0: self._save_checkpoint(f"step_{self.global_step}.pt")

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        total_loss, num_batches = 0.0, 0
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask").to(self.device) if batch.get("attention_mask") is not None else None
            labels = batch.get("labels", input_ids.clone()).to(self.device)
            outputs = self.model(input_ids, attention_mask, labels)
            if outputs.loss is not None and not math.isnan(outputs.loss.item()):
                total_loss += outputs.loss.item()
                num_batches += 1
            if num_batches >= 20: break
        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, filename: str):
        save_dir = os.path.abspath(self.config.output_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_state": {
                "global_step": self.global_step,
                "epoch": self.epoch,
            },
            "scales": {
                "ctm_scale": self.model.ctm_scale().item(),
            },
        }
        if hasattr(self.model.model, 'l1_scale'):
            checkpoint["scales"]["l1_scale"] = self.model.model.l1_scale().item()
        
        try:
            torch.save(checkpoint, save_path)
            logger.info(f"SUCCESS: Saved checkpoint to: {save_path}")
        except Exception as e:
            logger.error(f"CRITICAL SAVE ERROR at {save_path}: {e}")
            try:
                home_path = os.path.join(os.path.expanduser("~"), f"EMERGENCY_{filename}")
                torch.save(checkpoint, home_path)
                logger.warning(f"Saved to FALLBACK location: {home_path}")
            except Exception as e2:
                logger.error(f"Even fallback failed: {e2}")

    def load_checkpoint(self, path: str):
        logger.info(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location="cpu")
        
        # Strict=False ist wichtig, weil L1 neu dazu kommt!
        missing, unexpected = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        if missing:
            logger.info(f"New modules initialized (not in checkpoint): {len(missing)} keys (e.g. L1)")
        
        # Nur laden, wenn die Shapes passen (da L1 neu ist, könnte Optimizer crashen)
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.global_step = checkpoint["training_state"]["global_step"]
            self.epoch = checkpoint["training_state"]["epoch"]
            logger.info(f"Resumed successfully from step {self.global_step}")
        except Exception as e:
            logger.warning(f"Could not load optimizer state (expected if architecture changed). Starting fresh optimizer. Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Karla Phase 3 Pretraining")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=512)
    
    # HIER IST DAS FEHLENDE ARGUMENT:
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pt)")

    args = parser.parse_args()

    # Jetzt inkl. Parquet!
    data_paths = []
    data_dir = Path(args.data_dir)
    if data_dir.exists():
        for ext in ["*.parquet", "*.jsonl", "*.json"]:
            data_paths.extend(str(p) for p in data_dir.glob(ext))

    if not data_paths: 
        logger.error("No data files found!")
        sys.exit(1)

    config = PretrainConfig(data_paths=data_paths, num_epochs=args.epochs, batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum, l2_lr=args.lr, max_length=args.max_length)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = KarlaPretrainWrapper(KarlaConfig())
    from unified_dataset import create_dataloaders
    
    # Unified Dataset ist jetzt schlau genug für Parquet & JSON
    train_loader, val_loader = create_dataloaders(data_paths=data_paths, tokenizer=tokenizer, batch_size=config.batch_size, max_length=config.max_length)

    trainer = StableTrainer(model, train_loader, val_loader, config)
    
    # HIER WIRD GELADEN:
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    trainer.train()

if __name__ == "__main__": main()