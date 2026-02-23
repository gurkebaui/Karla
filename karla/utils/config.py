#!/usr/bin/env python3
"""
Karla C1 Configuration
======================

Centralized configuration for all training components.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class L0Config:
    """L0 - Perception Layer (Frozen Qwen)"""
    model_name: str = "Qwen/Qwen2.5-1.5B"
    bits: int = 4  # 4-bit quantization
    frozen: bool = True


@dataclass
class L1Config:
    """L1 - Engram Memory Layer"""
    embedding_dim: int = 512
    num_heads: int = 4
    ngram_orders: List[int] = field(default_factory=lambda: [2, 3])
    table_size: int = 50000
    learning_rate: float = 0.001
    conv_kernel_size: int = 4
    conv_zero_init: bool = True

    # Scale initialization
    # softplus(-2) ≈ 0.127 (good for L1)
    initial_scale_raw: float = -2.0


@dataclass
class L2Config:
    """L2 - CTM Head (Continuous Thought Machine)"""
    hidden_dim: int = 512
    num_neurons: int = 256
    num_internal_ticks: int = 10
    use_bitnet: bool = True
    nlm_history_length: int = 25
    nlm_hidden_dim: int = 32
    num_action_pairs: int = 512
    num_output_pairs: int = 512
    attn_heads: int = 8
    truncation_period: int = 4
    dropout: float = 0.1

    # Adaptive compute
    use_adaptive_compute: bool = True
    certainty_threshold: float = 0.95

    # Scale initialization
    # softplus(-0.4) ≈ 0.344 (good for CTM)
    initial_scale_raw: float = -0.4


@dataclass
class DataConfig:
    """Data configuration"""
    # Data paths
    train_data_paths: List[str] = field(default_factory=list)
    val_data_path: Optional[str] = None

    # Format detection
    supported_formats: List[str] = field(
        default_factory=lambda: ["messages", "instruction", "personality"]
    )

    # Processing
    max_length: int = 512
    train_ratio: float = 0.95
    max_samples_per_file: Optional[int] = None

    # POPE
    use_pope: bool = True
    pope_prefix_ratio: float = 0.3
    pope_hard_ratio: float = 0.5


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0

    # Learning rates
    l2_lr: float = 1e-4
    l1_lr: float = 0.001
    weight_decay: float = 0.01

    # Schedule
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1

    # L1 updates (delta rule)
    l1_update_frequency: int = 10

    # Mixed precision
    use_amp: bool = True

    # Logging
    log_interval: int = 10
    eval_interval: int = 200
    save_interval: int = 10

    # Output
    output_dir: str = "checkpoints"
    run_name: str = "karla_v1"

    # Device
    device: str = "cuda"


@dataclass
class RLConfig:
    """RL training configuration"""
    # POPE
    guidance_ratio: float = 0.5

    # GRPO
    clip_low: float = 0.2
    clip_high: float = 5.0
    lr: float = 1e-6

    # Rollout
    num_steps: int = 500
    problems_per_step: int = 4
    rollouts_per_problem: int = 4
    max_response_tokens: int = 256
    temperature: float = 0.8

    # Logging
    log_interval: int = 1
    save_interval: int = 50

    # Output
    output_dir: str = "checkpoints_rl"


@dataclass
class KarlaConfig:
    """Complete Karla configuration"""
    l0: L0Config = field(default_factory=L0Config)
    l1: L1Config = field(default_factory=L1Config)
    l2: L2Config = field(default_factory=L2Config)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    rl: RLConfig = field(default_factory=RLConfig)

    def __post_init__(self):
        """Validate configuration."""
        # Ensure output directories exist
        os.makedirs(self.training.output_dir, exist_ok=True)
        os.makedirs(self.rl.output_dir, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str) -> "KarlaConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls(
            l0=L0Config(**data.get("l0", {})),
            l1=L1Config(**data.get("l1", {})),
            l2=L2Config(**data.get("l2", {})),
            data=DataConfig(**data.get("data", {})),
            training=TrainingConfig(**data.get("training", {})),
            rl=RLConfig(**data.get("rl", {})),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "l0": self.l0.__dict__,
            "l1": self.l1.__dict__,
            "l2": self.l2.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "rl": self.rl.__dict__,
        }


# Default configuration
DEFAULT_CONFIG = KarlaConfig()


# ============================================================
# Hardware-specific configurations
# ============================================================

def get_config_for_hardware(vram_gb: float = 16, ram_gb: float = 32) -> KarlaConfig:
    """
    Get configuration optimized for specific hardware.

    Args:
        vram_gb: GPU VRAM in GB
        ram_gb: System RAM in GB
    """
    config = KarlaConfig()

    if vram_gb >= 24:
        # High-end GPU (RTX 4090, A100)
        config.training.batch_size = 4
        config.training.gradient_accumulation_steps = 4
        config.l2.num_neurons = 512
        config.l1.table_size = 100000

    elif vram_gb >= 16:
        # Mid-range GPU (RTX 4060 Ti 16GB, RTX 4080)
        config.training.batch_size = 2
        config.training.gradient_accumulation_steps = 8
        config.l2.num_neurons = 256
        config.l1.table_size = 50000

    elif vram_gb >= 12:
        # Entry-level (RTX 3060, 4070)
        config.training.batch_size = 1
        config.training.gradient_accumulation_steps = 16
        config.l2.num_neurons = 128
        config.l2.hidden_dim = 256
        config.l1.embedding_dim = 256
        config.l1.table_size = 25000
        config.data.max_length = 1024

    else:
        # Low VRAM - use more aggressive optimizations
        config.training.batch_size = 1
        config.training.gradient_accumulation_steps = 32
        config.l2.num_neurons = 64
        config.l2.hidden_dim = 128
        config.l1.embedding_dim = 128
        config.l1.table_size = 10000
        config.data.max_length = 512

    # RAM-dependent settings
    if ram_gb >= 64:
        config.l1.table_size = min(config.l1.table_size * 2, 100000)
    elif ram_gb < 16:
        config.l1.table_size = max(config.l1.table_size // 2, 10000)

    return config


# ============================================================
# Preset configurations
# ============================================================

PRESET_CONFIGS = {
    "default": KarlaConfig,
    "fast": lambda: KarlaConfig(
        training=TrainingConfig(
            num_epochs=1,
            batch_size=4,
            gradient_accumulation_steps=4,
            save_interval=1000,
        ),
        data=DataConfig(max_length=1024),
    ),
    "full": lambda: KarlaConfig(
        training=TrainingConfig(
            num_epochs=5,
            batch_size=2,
            gradient_accumulation_steps=8,
        ),
        l2=L2Config(num_neurons=512, hidden_dim=1024),
        l1=L1Config(table_size=100000),
    ),
    "debug": lambda: KarlaConfig(
        training=TrainingConfig(
            num_epochs=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            log_interval=1,
            eval_interval=10,
            save_interval=100,
        ),
        data=DataConfig(max_samples_per_file=100, max_length=512),
    ),
}


def get_preset_config(preset: str) -> KarlaConfig:
    """Get a preset configuration."""
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESET_CONFIGS.keys())}")
    return PRESET_CONFIGS[preset]()


if __name__ == "__main__":
    # Print default configuration
    import json
    config = KarlaConfig()
    print(json.dumps(config.to_dict(), indent=2))
