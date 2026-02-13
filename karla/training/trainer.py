"""
Karla Training Pipeline - Nested Learning
=========================================
Training implementation with:
1. Nested Learning (multi-timescale updates)
2. POPE (Privileged On-Policy Exploration)
3. Grokking Detection
4. Gradient checkpointing and memory optimization

Key Training Phases:
1. Phase 1: Next-token prediction warmup
2. Phase 2: POPE-guided reasoning training
3. Phase 3: Fine-tuning with adaptive compute
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import time
import json
import os
from datetime import datetime
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("KarlaTrainer")


@dataclass
class TrainingState:
    """Mutable training state"""
    global_step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    best_val_loss: float = float('inf')
    grokking_detected: bool = False
    grokking_step: int = -1
    patience_counter: int = 0
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)


class GrokkingDetector:
    """
    Detects grokking phenomenon during training.
    
    From OpenAI paper: "Generalization beyond overfitting on small algorithmic datasets"
    Grokking is when validation accuracy suddenly improves long after overfitting.
    
    Signs of grokking:
    1. Training loss near zero but validation loss high
    2. Sudden drop in validation loss after plateau
    3. Weight decay helps induce grokking
    """
    
    def __init__(
        self,
        patience: int = 1000,
        threshold: float = 0.1,
        min_train_loss: float = 0.01,
    ):
        self.patience = patience
        self.threshold = threshold
        self.min_train_loss = min_train_loss
        
        self.best_val_loss = float('inf')
        self.steps_since_improvement = 0
        self.grokking_detected = False
    
    def update(
        self,
        train_loss: float,
        val_loss: float,
        step: int,
    ) -> Tuple[bool, str]:
        """
        Update detector with new losses.
        
        Returns: (grokking_detected, status_message)
        """
        message = ""
        
        # Check for improvement
        if val_loss < self.best_val_loss - self.threshold:
            self.best_val_loss = val_loss
            self.steps_since_improvement = 0
            message = f"Validation improved to {val_loss:.4f}"
        else:
            self.steps_since_improvement += 1
        
        # Check for grokking conditions
        if (
            train_loss < self.min_train_loss
            and val_loss > train_loss * 10
            and self.steps_since_improvement > self.patience // 2
        ):
            message = "Potential grokking: training converged, waiting for generalization..."
        
        # Detect grokking: sudden improvement after long plateau
        if (
            self.steps_since_improvement > self.patience
            and not self.grokking_detected
            and val_loss < self.best_val_loss * 0.5
        ):
            self.grokking_detected = True
            message = f"GROKKING DETECTED at step {step}! Val loss: {val_loss:.4f}"
            return True, message
        
        return False, message


class KarlaTrainer:
    """
    Trainer for Karla with Nested Learning.
    
    Implements the 3-level update strategy:
    - L0 (Perception): Never updated (frozen)
    - L1 (Memory): Updated via Delta Rule every N batches
    - L2 (Reasoning): Updated via backprop every batch
    
    POPE Integration:
    - Hard problems receive oracle prefixes
    - Mixed training for transfer
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        
        # Learning rates
        l2_lr: float = 3e-4,
        l1_lr: float = 0.01,
        
        # Nested learning
        l1_update_frequency: int = 10,
        
        # Training
        num_epochs: int = 100,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        weight_decay: float = 0.01,
        
        # Mixed precision
        use_amp: bool = True,
        
        # Logging
        log_interval: int = 10,
        eval_interval: int = 100,
        save_interval: int = 500,
        
        # Output
        output_dir: str = "checkpoints",
        run_name: str = "karla_v1",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Learning rates
        self.l2_lr = l2_lr
        self.l1_lr = l1_lr
        
        # Nested learning
        self.l1_update_frequency = l1_update_frequency
        
        # Training params
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        
        # Mixed precision (nur wenn CUDA verfügbar)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Logging
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        
        # Output
        self.output_dir = output_dir
        self.run_name = run_name
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize optimizer (only for trainable params)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=l2_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Learning rate scheduler (cosine with warmup)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,  # First restart after 1000 steps
            T_mult=2,  # Double period after each restart
        )
        
        # Training state
        self.state = TrainingState()
        
        # Grokking detector
        self.grokking_detector = GrokkingDetector()
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device (wichtig für CUDA!)
        self.model = self.model.to(self.device)
        
        logger.info(f"[Trainer] Initialized on {self.device}")
        logger.info(f"[Trainer] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    def train(self):
        """Main training loop"""
        logger.info(f"[Trainer] Starting training for {self.num_epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.state.epoch, self.num_epochs):
            self.state.epoch = epoch
            epoch_loss = self._train_epoch(epoch)
            
            # Validation
            if self.eval_dataloader is not None:
                val_loss = self._evaluate()
                self.state.val_losses.append(val_loss)
                
                # Check for grokking
                grokked, message = self.grokking_detector.update(
                    epoch_loss, val_loss, self.state.global_step
                )
                if grokked:
                    self.state.grokking_detected = True
                    self.state.grokking_step = self.state.global_step
                    logger.info(f"[Trainer] {message}")
                
                # Save best model
                if val_loss < self.state.best_val_loss:
                    self.state.best_val_loss = val_loss
                    self._save_checkpoint("best_model.pt")
            
            logger.info(f"[Trainer] Epoch {epoch} completed. Loss: {epoch_loss:.4f}")
            
            # Save checkpoint
            self._save_checkpoint(f"epoch_{epoch}.pt")
        
        total_time = time.time() - start_time
        logger.info(f"[Trainer] Training completed in {total_time/3600:.2f} hours")
    
    def _train_epoch(self, epoch: int) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch.get('labels', input_ids.clone())
            labels = labels.to(self.device)
            
            # Forward pass
            device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
            with autocast(device_type=device_type, enabled=self.use_amp):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.max_grad_norm
                )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                # Update L1 memory (delta rule)
                if self.state.global_step % self.l1_update_frequency == 0:
                    self.model.update_memory()
                
                self.state.global_step += 1
                
                # Logging
                if self.state.global_step % self.log_interval == 0:
                    avg_loss = total_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    self.state.train_losses.append(avg_loss)
                    self.state.learning_rates.append(lr)
                    
                    logger.info(
                        f"[Step {self.state.global_step}] "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Ticks: {outputs.internal_ticks}"
                    )
                
                # Evaluation
                if self.eval_dataloader and self.state.global_step % self.eval_interval == 0:
                    val_loss = self._evaluate()
                    logger.info(f"[Step {self.state.global_step}] Val Loss: {val_loss:.4f}")
                
                # Save checkpoint
                if self.state.global_step % self.save_interval == 0:
                    self._save_checkpoint(f"step_{self.state.global_step}.pt")
        
        return total_loss / num_batches
    
    def _evaluate(self) -> float:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch.get('labels', input_ids.clone())
                labels = labels.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches
    
    def _save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_state': {
                'global_step': self.state.global_step,
                'epoch': self.state.epoch,
                'best_loss': self.state.best_loss,
                'best_val_loss': self.state.best_val_loss,
                'grokking_detected': self.state.grokking_detected,
                'grokking_step': self.state.grokking_step,
            },
            'train_losses': self.state.train_losses,
            'val_losses': self.state.val_losses,
        }
        
        path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, path)
        logger.info(f"[Trainer] Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        state = checkpoint['training_state']
        self.state.global_step = state['global_step']
        self.state.epoch = state['epoch']
        self.state.best_loss = state['best_loss']
        self.state.best_val_loss = state['best_val_loss']
        self.state.grokking_detected = state['grokking_detected']
        self.state.grokking_step = state['grokking_step']
        self.state.train_losses = checkpoint['train_losses']
        self.state.val_losses = checkpoint['val_losses']
        
        logger.info(f"[Trainer] Loaded checkpoint from {path}")
        logger.info(f"[Trainer] Resuming from step {self.state.global_step}")


def train_karla(
    config: Any,
    use_mock: bool = False,
    use_lite: bool = False,
):
    """
    Convenience function to train Karla from config.
    """
    from karla.models.karla import create_karla
    from karla.data.dataset import ReasoningDataset, create_dataloader
    
    # Create model
    model = create_karla(config, use_mock=use_mock, use_lite=use_lite)
    
    # Create datasets
    train_dataset = ReasoningDataset(
        config.training.data_path,
        pope_prefix_ratio=config.training.pope_prefix_ratio,
        use_pope=config.training.use_pope,
    )
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
    )
    
    # Create trainer
    trainer = KarlaTrainer(
        model=model,
        train_dataloader=train_loader,
        l2_lr=config.training.l2_lr,
        l1_lr=config.training.l1_lr,
        l1_update_frequency=config.training.l1_update_frequency,
        num_epochs=config.training.num_epochs,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        weight_decay=config.l2.weight_decay,
        use_amp=config.training.mixed_precision,
        output_dir=config.training.output_dir,
        run_name=config.training.run_name,
    )
    
    # Train
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    print("Karla Training Pipeline")
    print("Run: python train_karla.py --config config.yaml")
