"""
Configuration for Karla / C1 Architecture
"""

from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class L0Config:
    model_name: str = "Qwen/Qwen2.5-1.5B"
    bits: int = 4
    hidden_size: int = 1536


@dataclass
class L1Config:
    embedding_dim: int = 512
    num_heads: int = 4
    ngram_orders: List[int] = field(default_factory=lambda: [2, 3])
    table_size: int = 50000
    learning_rate: float = 0.001


@dataclass
class L2Config:
    hidden_dim: int = 512
    num_neurons: int = 256
    num_internal_ticks: int = 10
    use_bitnet: bool = True
    weight_decay: float = 0.01


@dataclass
class TrainingConfig:
    data_path: str = "data/micro_pope_data.jsonl"
    max_seq_length: int = 512
    num_epochs: int = 10
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    l2_lr: float = 1e-4
    l1_lr: float = 0.001
    l1_update_frequency: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    mixed_precision: bool = True
    use_pope: bool = True
    pope_prefix_ratio: float = 0.3
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    output_dir: str = "checkpoints"
    run_name: str = "c1_v1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class KarlaConfig:
    l0: L0Config = field(default_factory=L0Config)
    l1: L1Config = field(default_factory=L1Config)
    l2: L2Config = field(default_factory=L2Config)
    training: TrainingConfig = field(default_factory=TrainingConfig)