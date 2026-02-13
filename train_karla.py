#!/usr/bin/env python3
"""
Karla Training Script
=====================
Main entry point for training Karla models.

Usage:
    # Quick test (mock model, lite components)
    python train_karla.py --mode test
    
    # Full training
    python train_karla.py --mode train
    
    # Custom configuration
    python train_karla.py --mode train --l2-lr 1e-4 --epochs 50
    
    # Evaluation only
    python train_karla.py --mode eval --checkpoint checkpoints/best_model.pt

The Self-Evolving Reasoner (Frankenstein 2.0)
Target: DeepSeek-Level Reasoning on RTX 4060 Ti

Architecture:
┌─────────────────────────────────────────┐
│  L0: Qwen 2.5-1.5B 4-bit (Frozen)      │  ~2GB VRAM
│  L1: CMS Engram Memory (CPU)           │  Scalable RAM
│  L2: CTM + BitNet Reasoning (Trainable)│  ~50MB VRAM
└─────────────────────────────────────────┘
"""

import argparse
import os
import sys
import torch
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from karla.utils.config import KarlaConfig
from karla.models.karla import create_karla, Karla
from karla.models.l0_perception import L0Perception, L0PerceptionMock
from karla.training.trainer import KarlaTrainer
from karla.data.dataset import ReasoningDataset, create_dataloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-8s │ %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Karla")


def print_banner():
    """Print startup banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██╗  ██╗ █████╗ ██████╗ ██╗     ███████╗██████╗           ║
║   ██║ ██╔╝██╔══██╗██╔══██╗██║     ██╔════╝██╔══██╗          ║
║   █████╔╝ ███████║██████╔╝██║     █████╗  ██████╔╝          ║
║   ██╔═██╗ ██╔══██║██╔══██╗██║     ██╔══╝  ██╔══██╗          ║
║   ██║  ██╗██║  ██║██║  ██║███████╗███████╗██║  ██║          ║
║   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝          ║
║                                                               ║
║         The Self-Evolving Reasoner (Frankenstein 2.0)        ║
║                                                               ║
║   L0: Perception (Frozen)  │  L1: Memory (CPU)  │  L2: CTM   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def create_test_config() -> KarlaConfig:
    """Create configuration for quick testing"""
    config = KarlaConfig()
    
    # L0: Use mock for testing
    config.l0.hidden_size = 256
    
    # L1: Lightweight memory
    config.l1.embedding_dim = 128
    config.l1.num_heads = 4
    
    # L2: Small CTM
    config.l2.hidden_dim = 128
    config.l2.num_neurons = 64
    config.l2.num_internal_ticks = 5
    
    # Training: Quick iterations
    config.training.num_epochs = 3
    config.training.batch_size = 2
    config.training.gradient_accumulation_steps = 2
    config.training.log_interval = 1
    config.training.eval_interval = 5
    
    return config


def create_production_config() -> KarlaConfig:
    """Create configuration for production training"""
    config = KarlaConfig()
    
    # L0: Full Qwen 2.5-1.5B
    config.l0.model_name = "Qwen/Qwen2.5-1.5B"
    config.l0.bits = 4
    config.l0.frozen = True
    
    # L1: Larger memory
    config.l1.embedding_dim = 512
    config.l1.num_heads = 8
    config.l1.memory_size = 500_000
    
    # L2: Full CTM
    config.l2.hidden_dim = 512
    config.l2.num_neurons = 256
    config.l2.num_internal_ticks = 10
    config.l2.use_bitnet = True
    
    # Training
    config.training.num_epochs = 100
    config.training.batch_size = 4
    config.training.gradient_accumulation_steps = 4
    config.training.l2_lr = 3e-4
    config.training.weight_decay = 0.01  # Important for grokking!
    
    return config


def run_test_mode():
    """Run quick test with mock components"""
    logger.info("=" * 60)
    logger.info("RUNNING TEST MODE")
    logger.info("=" * 60)
    
    config = create_test_config()
    
    # Create model with mock components
    logger.info("Creating model (mock mode)...")
    model = create_karla(config, use_mock=True, use_lite=True)
    
    # Print model info
    param_counts = model.count_parameters()
    logger.info(f"Model parameters: {param_counts}")
    
    # Create dataset WITHOUT tokenizer (mock mode)
    logger.info("Creating dataset...")
    dataset = ReasoningDataset(
        data_path="data/micro_pope_data.jsonl",  # Will create synthetic if missing
        tokenizer=None,  # No tokenizer in mock mode
        max_length=128,
    )
    
    dataloader = create_dataloader(dataset, batch_size=config.training.batch_size)
    
    # Create trainer
    trainer = KarlaTrainer(
        model=model,
        train_dataloader=dataloader,
        l2_lr=config.training.l2_lr,
        num_epochs=config.training.num_epochs,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_interval=config.training.log_interval,
        output_dir="checkpoints_test",
        run_name="karla_test",
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Test completed successfully!")
    return trainer


def run_train_mode(args):
    """Run full training"""
    logger.info("=" * 60)
    logger.info("RUNNING TRAIN MODE")
    logger.info("=" * 60)
    
    # Create config
    if args.quick:
        config = create_test_config()
        use_mock = True
        use_lite = True
    else:
        config = create_production_config()
        use_mock = False
        use_lite = False
    
    # Override with command line args
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.l2_lr:
        config.training.l2_lr = args.l2_lr
    if args.data_path:
        config.training.data_path = args.data_path
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.warning("No GPU detected! Training on CPU will be slow.")
    
    # === WICHTIG: Tokenizer laden BEVOR Dataset erstellt wird ===
    tokenizer = None
    
    if not use_mock:
        logger.info("Loading tokenizer from Qwen...")
        try:
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(
                config.l0.model_name,
                trust_remote_code=True,
            )
            
            # Ensure pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
            
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")
            logger.info("Will proceed without tokenizer (will cause issues!)")
    
    # Create model
    logger.info("Creating model...")
    model = create_karla(config, use_mock=use_mock, use_lite=use_lite)
    
    # If tokenizer wasn't loaded separately, try to get it from model
    if tokenizer is None and hasattr(model.l0, 'tokenizer'):
        # Trigger model loading to get tokenizer
        _ = model.l0.tokenize("test")
        tokenizer = model.l0.tokenizer
        logger.info(f"Got tokenizer from model. Vocab size: {len(tokenizer)}")
    
    # Print model info
    param_counts = model.count_parameters()
    memory_estimate = model.estimate_memory()
    
    logger.info(f"Model parameters:")
    for k, v in param_counts.items():
        logger.info(f"  {k}: {v:,}")
    
    logger.info(f"Memory estimate:")
    for k, v in memory_estimate.items():
        logger.info(f"  {k}: {v:.2f} GB")
    
    # === Dataset mit Tokenizer erstellen ===
    logger.info(f"Loading dataset from: {config.training.data_path}")
    dataset = ReasoningDataset(
        data_path=config.training.data_path,
        tokenizer=tokenizer,  # <-- TOKENIZER HIER ÜBERGEBEN!
        max_length=config.training.max_seq_length,
        pope_prefix_ratio=config.training.pope_prefix_ratio,
        use_pope=config.training.use_pope,
    )
    
    dataloader = create_dataloader(
        dataset,
        batch_size=config.training.batch_size,
    )
    
    # Create trainer
    trainer = KarlaTrainer(
        model=model,
        train_dataloader=dataloader,
        l2_lr=config.training.l2_lr,
        l1_lr=config.training.l1_lr,
        l1_update_frequency=config.training.l1_update_frequency,
        num_epochs=config.training.num_epochs,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        weight_decay=config.l2.weight_decay,
        use_amp=config.training.mixed_precision,
        log_interval=config.training.log_interval,
        eval_interval=config.training.eval_interval,
        save_interval=config.training.save_interval,
        output_dir=config.training.output_dir,
        run_name=config.training.run_name,
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    logger.info("Starting training...")
    logger.info(f"Configuration:")
    logger.info(f"  Epochs: {config.training.num_epochs}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  L2 LR: {config.training.l2_lr}")
    logger.info(f"  L1 LR: {config.training.l1_lr}")
    logger.info(f"  Weight decay: {config.l2.weight_decay}")
    
    start_time = datetime.now()
    trainer.train()
    elapsed = datetime.now() - start_time
    
    logger.info(f"Training completed in {elapsed}")
    logger.info(f"Best validation loss: {trainer.state.best_val_loss:.4f}")
    
    if trainer.state.grokking_detected:
        logger.info(f"⚡ GROKKING DETECTED at step {trainer.state.grokking_step}!")
    
    return trainer


def run_eval_mode(args):
    """Run evaluation only"""
    logger.info("=" * 60)
    logger.info("RUNNING EVAL MODE")
    logger.info("=" * 60)
    
    if not args.checkpoint:
        logger.error("Checkpoint required for eval mode. Use --checkpoint path")
        return None
    
    # Create model
    config = create_production_config()
    model = create_karla(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded checkpoint: {args.checkpoint}")
    logger.info(f"Training step: {checkpoint['training_state']['global_step']}")
    
    # TODO: Add evaluation logic
    
    logger.info("Evaluation complete")
    return model


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Karla - The Self-Evolving Reasoner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "train", "eval"],
        help="Running mode (default: test)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use lightweight model for quick testing"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size"
    )
    
    parser.add_argument(
        "--l2-lr",
        type=float,
        default=None,
        help="Learning rate for L2 (CTM)"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to training data (JSONL)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to load"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Run appropriate mode
    if args.mode == "test":
        return run_test_mode()
    elif args.mode == "train":
        return run_train_mode(args)
    elif args.mode == "eval":
        return run_eval_mode(args)


if __name__ == "__main__":
    main()
