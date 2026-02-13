"""
Karla Configuration - The Self-Evolving Reasoner
================================================
Configuration file for the 3-Level Nested Learning Architecture.

Architecture Overview:
- L0 (Perception): Qwen 2.5-1.5B 4-bit (Frozen) - Feature Extractor
- L1 (Memory): CMS Engram - Hash-based Memory on CPU/RAM
- L2 (Reasoning): CTM Head with BitNet 1.58 - Trainable Reasoning Module
"""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class L0Config:
    """L0 Perception Layer Configuration - Qwen 2.5-1.5B (Frozen)"""
    model_name: str = "Qwen/Qwen2.5-1.5B"
    bits: int = 4  # 4-bit quantization (NF4)
    device_map: str = "auto"
    frozen: bool = True  # Never update these weights
    hidden_size: int = 1536  # Qwen 2.5-1.5B hidden dimension
    num_layers: int = 28
    vocab_size: int = 151936


@dataclass
class L1Config:
    """L1 Memory System Configuration - CMS Engram"""
    # Memory dimensions
    embedding_dim: int = 512  # Memory vector dimension
    num_heads: int = 8  # Multi-head hashing
    memory_size: int = 100_000  # Number of memory slots (initial)
    
    # N-gram configuration
    ngram_orders: List[int] = field(default_factory=lambda: [2, 3, 4])
    
    # Hash configuration
    hash_prime_sizes: List[int] = field(default_factory=lambda: [100003, 100019, 100043])
    
    # Update configuration
    learning_rate: float = 0.01  # Delta rule learning rate
    update_frequency: int = 10  # Update every N batches
    
    # Memory management
    device: str = "cpu"  # Run on CPU for large memory
    offload_to_disk: bool = False  # Future: SSD offloading


@dataclass
class L2Config:
    """L2 Reasoning Head Configuration - CTM with BitNet"""
    # CTM Architecture
    hidden_dim: int = 512  # CTM internal dimension
    num_neurons: int = 256  # Number of neurons in CTM
    num_internal_ticks: int = 10  # Internal reasoning steps
    
    # BitNet configuration
    use_bitnet: bool = True  # Use ternary weights (-1, 0, 1)
    
    # Synapse model (U-Net style MLP)
    synapse_depth: int = 3
    synapse_hidden_dim: int = 512
    
    # Neuron-level models
    nlm_hidden_dim: int = 64
    nlm_history_length: int = 20  # M in paper
    
    # Output
    vocab_size: int = 151936  # Match Qwen vocab
    output_dim: int = 1536  # Match Qwen hidden for residual
    
    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.01  # Important for grokking!
    
    # Adaptive compute
    use_adaptive_compute: bool = True
    certainty_threshold: float = 0.95


@dataclass
class TrainingConfig:
    """Training Configuration"""
    # Dataset
    data_path: str = "data/micro_pope_data.jsonl"
    max_seq_length: int = 2048
    
    # Training loop
    num_epochs: int = 100
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # Learning rates
    l2_lr: float = 3e-4  # CTM learning rate
    l1_lr: float = 0.01  # Engram learning rate (delta rule)
    
    # Nested learning
    l1_update_frequency: int = 10  # Update L1 every N batches
    gradient_checkpointing: bool = True
    
    # POPE configuration
    use_pope: bool = True
    pope_prefix_ratio: float = 0.3  # Use 30% of oracle as prefix
    pope_mixed_ratio: float = 0.5  # 50% guided, 50% unguided
    
    # Grokking detection
    grokking_patience: int = 1000  # Steps without improvement
    grokking_threshold: float = 0.1  # Validation loss drop
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    
    # Output
    output_dir: str = "checkpoints"
    run_name: str = "karla_v1"


@dataclass
class KarlaConfig:
    """Main Configuration combining all levels"""
    l0: L0Config = field(default_factory=L0Config)
    l1: L1Config = field(default_factory=L1Config)
    l2: L2Config = field(default_factory=L2Config)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        # Ensure vocab sizes match
        self.l2.vocab_size = self.l0.vocab_size
        self.l2.output_dim = self.l0.hidden_size
