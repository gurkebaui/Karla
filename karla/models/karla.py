"""
karla.py — Main Karla C1 Model
==============================
Hierarchical architecture with:
- L0: Qwen 2.5-1.5B (frozen perception)
- L1: Engram Memory (hash-based N-gram lookup)
- L2: CTM Head (reasoning with certainty + value heads)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
import logging

from .l0_perception import L0Perception, L0PerceptionMock
from .l1_engram import EngramMemory, EngramMemoryLite
from .l2_ctm import CTMHead, CTMHeadLite

logger = logging.getLogger("Karla")


@dataclass
class KarlaOutput:
    """Output from Karla model."""
    logits: torch.Tensor              # (B, S, vocab_size)
    internal_ticks: int               # CTM internal ticks used
    certainty: torch.Tensor           # (B, 1) - certainty from CTM
    value: torch.Tensor               # (B, 1) - V(s) from CTM value head
    loss: Optional[torch.Tensor] = None


class Karla(nn.Module):
    """
    Karla C1: Hierarchical Reasoning Model.
    """
    def __init__(
        self,
        l0_model_name: str = "Qwen/Qwen2.5-1.5B",
        l0_bits: int = 4,
        l1_embedding_dim: int = 512,
        l1_num_heads: int = 4,
        l1_ngram_orders=None,
        l1_table_size: int = 50000,
        l1_learning_rate: float = 0.001,
        l1_conv_kernel_size: int = 4,
        l1_conv_zero_init: bool = True,
        l2_hidden_dim: int = 512,
        l2_num_neurons: int = 256,
        l2_num_internal_ticks: int = 10,
        l2_use_bitnet: bool = True,
        l2_nlm_history_length: int = 25,
        l2_nlm_hidden_dim: int = 32,
        l2_num_action_pairs: int = 512,
        l2_num_output_pairs: int = 512,
        l2_attn_heads: int = 8,
        l2_truncation_period: int = 4,
        vocab_size: int = 151936,
        use_mock_l0: bool = False,
        use_lite_l1: bool = False,
        use_lite_l2: bool = False,
    ):
        super().__init__()

        if l1_ngram_orders is None:
            l1_ngram_orders = [2, 3]

        self.vocab_size = vocab_size
        l0_hidden_dim = 1536

        # L0: Perception (Frozen)
        if use_mock_l0:
            self.l0 = L0PerceptionMock(hidden_size=l0_hidden_dim, vocab_size=vocab_size)
        else:
            self.l0 = L0Perception(model_name=l0_model_name, bits=l0_bits)

        # L1: Engram Memory
        if use_lite_l1:
            self.l1 = EngramMemoryLite(embedding_dim=l1_embedding_dim, hidden_dim=l0_hidden_dim)
        else:
            self.l1 = EngramMemory(
                embedding_dim=l1_embedding_dim,
                num_heads=l1_num_heads,
                ngram_orders=l1_ngram_orders,
                table_size=l1_table_size,
                hidden_dim=l0_hidden_dim,
                learning_rate=l1_learning_rate,
                conv_kernel_size=l1_conv_kernel_size,
                conv_zero_init=l1_conv_zero_init,
            )

        # L2: CTM Head
        if use_lite_l2:
            self.l2 = CTMHeadLite(hidden_dim=l2_hidden_dim, l0_hidden_dim=l0_hidden_dim)
        else:
            self.l2 = CTMHead(
                hidden_dim=l2_hidden_dim,
                num_neurons=l2_num_neurons,
                num_internal_ticks=l2_num_internal_ticks,
                use_bitnet=l2_use_bitnet,
                l0_hidden_dim=l0_hidden_dim,
                nlm_history_length=l2_nlm_history_length,
                nlm_hidden_dim=l2_nlm_hidden_dim,
                num_action_pairs=l2_num_action_pairs,
                num_output_pairs=l2_num_output_pairs,
                attn_heads=l2_attn_heads,
                truncation_period=l2_truncation_period,
            )

        # L1 -> Qwen space projection
        self.l1_seq_proj = nn.Sequential(
            nn.Linear(l1_embedding_dim, l0_hidden_dim),
            nn.LayerNorm(l0_hidden_dim),
        )

        # Scales: softplus(raw) ensures positive values
        self.l1_scale_raw = nn.Parameter(torch.tensor(-10.0))
        self.ctm_scale_raw = nn.Parameter(torch.tensor(-10.0))

    def l1_scale(self) -> torch.Tensor:
        return F.softplus(self.l1_scale_raw)

    def ctm_scale(self) -> torch.Tensor:
        return F.softplus(self.ctm_scale_raw)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        max_ticks: Optional[int] = None,
    ) -> KarlaOutput:
        device = input_ids.device

        # L0: Perception
        l0_out = self.l0(input_ids, attention_mask)
        l0_hidden = l0_out.hidden_states.to(device)  # (B, S, 1536)

        # L1: Memory
        l1_out = self.l1(input_ids, l0_hidden)
        l1_mem = l1_out.memory_vectors.to(device)    # (B, S, emb_dim)

        # L2: CTM Reasoning
        l2_out = self.l2(l0_hidden, l1_mem, attention_mask=attention_mask, max_ticks=max_ticks)
        ctm_bcast = l2_out.features.unsqueeze(1)      # (B, 1, 1536)

        # Feature Fusion
        l1_contrib = self.l1_seq_proj(l1_mem)         # (B, S, 1536)

        features = (
            l0_hidden
            + self.l1_scale() * l1_contrib
            + self.ctm_scale() * ctm_bcast
        )

        # Output
        logits = self.l0.decode_head(features)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            V = logits.size(-1)
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, V),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return KarlaOutput(
            logits=logits,
            internal_ticks=l2_out.internal_ticks,
            certainty=l2_out.certainty,
            value=l2_out.value,
            loss=loss,
        )

    def update_memory(self):
        """Apply queued L1 memory updates (delta rule)."""
        if hasattr(self.l1, "apply_updates"):
            self.l1.apply_updates()

    def count_parameters(self) -> dict:
        """Count parameters by component."""
        counts = {
            "l0": sum(p.numel() for p in self.l0.parameters()),
            "l1": sum(p.numel() for p in self.l1.parameters()),
            "l2": sum(p.numel() for p in self.l2.parameters()),
            "projections": sum(p.numel() for p in self.l1_seq_proj.parameters()) + 2,
        }
        counts["total"] = sum(counts.values())
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts


def create_karla(config=None, use_mock=False, use_lite=False, **kwargs):
    """Factory function to create Karla from config."""
    cfg = {}
    if config is not None:
        cfg = {
            "l0_model_name": config.l0.model_name,
            "l0_bits": config.l0.bits,
            "l1_embedding_dim": config.l1.embedding_dim,
            "l1_num_heads": config.l1.num_heads,
            "l1_ngram_orders": config.l1.ngram_orders,
            "l1_table_size": config.l1.table_size,
            "l1_learning_rate": config.l1.learning_rate,
            "l2_hidden_dim": config.l2.hidden_dim,
            "l2_num_neurons": config.l2.num_neurons,
            "l2_num_internal_ticks": config.l2.num_internal_ticks,
            "l2_use_bitnet": config.l2.use_bitnet,
        }
    cfg.update(kwargs)
    return Karla(use_mock_l0=use_mock, use_lite_l1=use_lite, use_lite_l2=use_lite, **cfg)
