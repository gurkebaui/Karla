"""
Karla - The Self-Evolving Reasoner
==================================
A 3-level nested learning architecture for efficient reasoning.

Quick Start:
    python train_karla.py --mode train --config default

Architecture:
    L0 (Perception): Qwen 2.5-1.5B 4-bit (Frozen)
    L1 (Memory): CMS Engram with Delta Rule Updates
    L2 (Reasoning): CTM Head with BitNet 1.58

Target Hardware: RTX 4060 Ti (16GB VRAM)
"""

import torch
from karla.utils.config import KarlaConfig, L0Config, L1Config, L2Config, TrainingConfig
from karla.models.karla import create_karla, Karla
from karla.training.trainer import KarlaTrainer, train_karla
from karla.data.dataset import ReasoningDataset, create_dataloader

__version__ = "0.1.0"
__author__ = "Karla Project"

__all__ = [
    # Config
    "KarlaConfig",
    "L0Config", 
    "L1Config",
    "L2Config",
    "TrainingConfig",
    # Model
    "Karla",
    "create_karla",
    # Training
    "KarlaTrainer",
    "train_karla",
    # Data
    "ReasoningDataset",
    "create_dataloader",
]
