#!/usr/bin/env python3
"""
C1 Architecture Training Script
================================
Usage:
    python train_karla.py --mode train
    python train_karla.py --mode train --epochs 5 --batch-size 2
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from karla.utils.config import KarlaConfig
from karla.models.karla import create_karla
from karla.data.dataset import ReasoningDataset, create_dataloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("C1")


def run_train_mode(args):
    logger.info("=" * 60)
    logger.info("C1 ARCHITECTURE - TRAINING")
    logger.info("=" * 60)

    config = KarlaConfig()

    # CLI overrides
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.l2_lr:
        config.training.l2_lr = args.l2_lr
    if args.data_path:
        config.training.data_path = args.data_path

    device = torch.device(config.training.device)

    # GPU info
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu} ({mem:.1f} GB)")

    # === 1. Tokenizer ===
    logger.info("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.l0.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Vocab size: {len(tokenizer)}")

    # === 2. Model ===
    logger.info("Creating model...")
    model = create_karla(config)

    counts = model.count_parameters()
    for k, v in counts.items():
        logger.info(f"  {k}: {v:,}")

    model.l1.to(device)
    model.l2.to(device)
    model.l0_proj.to(device)
    model.l1_proj.to(device)

    # === 3. Dataset ===
    logger.info(f"Loading dataset: {config.training.data_path}")
    dataset = ReasoningDataset(
        data_path=config.training.data_path,
        tokenizer=tokenizer,
        max_length=config.training.max_seq_length,
    )
    dataloader = create_dataloader(
        dataset, batch_size=config.training.batch_size
    )

    # === 4. Optimizer ===
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training.l2_lr,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95),
    )

    total_steps = len(dataloader) * config.training.num_epochs
    warmup_steps = min(100, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-2)  # min 1% LR during warmup
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP scaler
    use_amp = config.training.mixed_precision and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # === 5. Training Loop ===
    accum_steps = config.training.gradient_accumulation_steps
    logger.info(f"Epochs: {config.training.num_epochs}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Gradient accumulation: {accum_steps}")
    logger.info(f"Effective batch: {config.training.batch_size * accum_steps}")
    logger.info(f"AMP: {use_amp}")

    os.makedirs(config.training.output_dir, exist_ok=True)
    global_step = 0
    best_loss = float('inf')
    nan_count = 0

    model.train()

    for epoch in range(config.training.num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch.get('labels', input_ids.clone()).to(device)

            # Forward
            try:
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(input_ids, attention_mask, labels)
                        loss = outputs.loss / accum_steps
                else:
                    outputs = model(input_ids, attention_mask, labels)
                    loss = outputs.loss / accum_steps
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM at step {global_step}! Clearing cache.")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise

            # NaN check — skip backward but do NOT zero_grad
            # (keep valid accumulated gradients from other sub-steps)
            if not torch.isfinite(loss):
                nan_count += 1
                if nan_count <= 10:
                    logger.warning(
                        f"Step {global_step}, sub-step {batch_idx % accum_steps}: "
                        f"NaN/Inf loss! (total: {nan_count})"
                    )
                if nan_count > 200:
                    logger.error("Too many NaN losses. Stopping.")
                    return
                # Fall through to accumulation check below (don't backward)
            else:
                # Backward — only if loss is finite
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss.item() * accum_steps
                epoch_steps += 1

            # === Optimizer step at accumulation boundary ===
            if (batch_idx + 1) % accum_steps == 0:
                if use_amp:
                    # 1. Unscale gradients
                    scaler.unscale_(optimizer)

                    # 2. Clip gradients
                    nn.utils.clip_grad_norm_(
                        trainable_params, config.training.max_grad_norm
                    )

                    # 3. Step (internally skips if grads are inf/nan)
                    scaler.step(optimizer)

                    # 4. ALWAYS update scaler — this is critical!
                    scaler.update()
                else:
                    grad_norm = nn.utils.clip_grad_norm_(
                        trainable_params, config.training.max_grad_norm
                    )
                    if torch.isfinite(grad_norm):
                        optimizer.step()
                    else:
                        logger.warning(f"Step {global_step}: Non-finite grad norm!")

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                # L1 memory updates
                if global_step % config.training.l1_update_frequency == 0:
                    model.update_memory()

                # Logging
                if global_step % config.training.log_interval == 0:
                    avg_loss = epoch_loss / max(epoch_steps, 1)
                    lr = optimizer.param_groups[0]['lr']
                    logger.info(
                        f"Epoch {epoch} | Step {global_step} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                        f"Ticks: {outputs.internal_ticks} | "
                        f"NaN: {nan_count}"
                    )

        # Epoch summary
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch} complete | "
            f"Loss: {avg_epoch_loss:.4f} | "
            f"Time: {elapsed:.1f}s | "
            f"Steps: {epoch_steps}"
        )

        # Save checkpoint
        if avg_epoch_loss < best_loss and epoch_steps > 0:
            best_loss = avg_epoch_loss
            path = os.path.join(config.training.output_dir, "best_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_epoch_loss,
                'global_step': global_step,
            }, path)
            logger.info(f"Saved best model: {path}")

        path = os.path.join(config.training.output_dir, f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), path)

    logger.info("Training complete!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Total NaN losses: {nan_count}")

def main():
    parser = argparse.ArgumentParser(description="C1 Architecture Training")
    parser.add_argument("--mode", default="train", choices=["train", "test", "eval"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--l2-lr", type=float, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--quick", action="store_true")

    args = parser.parse_args()

    if args.mode == "train":
        run_train_mode(args)
    else:
        logger.info(f"Mode '{args.mode}' not yet implemented")


if __name__ == "__main__":
    main()