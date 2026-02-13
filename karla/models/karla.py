"""
Karla - C1 Architecture
"""

import torch
import torch.nn as nn
from typing import Optional, List
from dataclasses import dataclass
import logging

from .l0_perception import L0Perception, L0PerceptionMock, L0Output
from .l1_engram import EngramMemory, EngramMemoryLite
from .l2_ctm import CTMHead, CTMHeadLite

logger = logging.getLogger("Karla")


@dataclass
class KarlaOutput:
    logits: torch.Tensor
    internal_ticks: int
    certainty: torch.Tensor
    loss: Optional[torch.Tensor] = None


class Karla(nn.Module):
    def __init__(
        self,
        l0_model_name="Qwen/Qwen2.5-1.5B",
        l0_bits=4,
        l1_embedding_dim=512,
        l1_num_heads=4,
        l1_ngram_orders=None,
        l1_table_size=50000,
        l2_hidden_dim=512,
        l2_num_neurons=256,
        l2_num_internal_ticks=10,
        l2_use_bitnet=True,
        vocab_size=151936,
        use_mock_l0=False,
        use_lite_l1=False,
        use_lite_l2=False,
    ):
        super().__init__()

        if l1_ngram_orders is None:
            l1_ngram_orders = [2, 3]

        self.vocab_size = vocab_size
        self.l2_hidden_dim = l2_hidden_dim
        l0_hidden_dim = 1536

        logger.info("[Karla] Initializing components...")

        if use_mock_l0:
            self.l0 = L0PerceptionMock(hidden_size=l0_hidden_dim)
        else:
            self.l0 = L0Perception(model_name=l0_model_name, bits=l0_bits)

        if use_lite_l1:
            self.l1 = EngramMemoryLite(embedding_dim=l1_embedding_dim, hidden_dim=l0_hidden_dim)
        else:
            self.l1 = EngramMemory(
                embedding_dim=l1_embedding_dim,
                num_heads=l1_num_heads,
                ngram_orders=l1_ngram_orders,
                table_size=l1_table_size,
                hidden_dim=l0_hidden_dim,
            )

        if use_lite_l2:
            self.l2 = CTMHeadLite(hidden_dim=l2_hidden_dim, vocab_size=vocab_size)
        else:
            self.l2 = CTMHead(
                hidden_dim=l2_hidden_dim,
                num_neurons=l2_num_neurons,
                num_internal_ticks=l2_num_internal_ticks,
                use_bitnet=l2_use_bitnet,
                vocab_size=vocab_size,
            )

        self.l0_proj = nn.Sequential(
            nn.Linear(l0_hidden_dim, l2_hidden_dim),
            nn.LayerNorm(l2_hidden_dim),
        )

        if l1_embedding_dim != l2_hidden_dim:
            self.l1_proj = nn.Sequential(
                nn.Linear(l1_embedding_dim, l2_hidden_dim),
                nn.LayerNorm(l2_hidden_dim),
            )
        else:
            self.l1_proj = nn.LayerNorm(l2_hidden_dim)

        logger.info("[Karla] Model initialized successfully")

    def _get_last_valid_positions(self, attention_mask, batch_size, device):
        """Get the index of the last non-padding token per batch element."""
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1  # 0-indexed
            # Clamp to valid range
            seq_lengths = seq_lengths.clamp(min=0)
        else:
            # No mask — assume all tokens valid, use last position
            seq_len = 0  # will be set by caller
            seq_lengths = None
        return seq_lengths

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.size(0)
        device = input_ids.device

        # === L0: Perception ===
        l0_out = self.l0(input_ids, attention_mask)

        # === L1: Memory ===
        l1_out = self.l1(input_ids, l0_out.hidden_states)

        # Use last VALID position for L1 (not absolute last = padding)
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1).long() - 1
            seq_lengths = seq_lengths.clamp(min=0)
            batch_idx = torch.arange(batch_size, device=device)
            l1_vec = l1_out.memory_vectors[batch_idx, seq_lengths]
        else:
            l1_vec = l1_out.memory_vectors[:, -1, :]

        l1_vec = self.l1_proj(l1_vec)

        # l0_out.last_hidden already uses last valid position (from L0)
        l0_vec = self.l0_proj(l0_out.last_hidden)

        # === L2: Reasoning ===
        l2_out = self.l2(l0_vec, l1_vec)

        # === Loss ===
        loss = None
        if labels is not None:
            loss = self._compute_loss(l2_out.logits, labels, device)

        return KarlaOutput(
            logits=l2_out.logits,
            internal_ticks=l2_out.internal_ticks,
            certainty=l2_out.certainty,
            loss=loss,
        )

    def _compute_loss(self, logits, labels, device):
        """
        Find the last valid (non -100) label per batch element.
        
        This prevents NaN from cross_entropy when labels[:, -1] is -100
        (which happens for all padded sequences).
        """
        batch_size, seq_len = labels.shape
        valid_mask = labels != -100  # (batch, seq_len)

        # Check if ANY valid targets exist
        if not valid_mask.any():
            # No valid targets at all — return zero loss with gradient
            return logits.sum() * 0.0

        # Find last valid position per batch element
        # Create position indices: [[0,1,2,...],[0,1,2,...]]
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Set invalid positions to -1 so they don't win the max()
        valid_positions = positions.masked_fill(~valid_mask, -1)
        last_valid_pos = valid_positions.max(dim=1).values  # (batch,)

        # Identify batch elements that have at least one valid target
        has_valid = last_valid_pos >= 0  # (batch,)

        if not has_valid.any():
            return logits.sum() * 0.0

        # Gather targets at last valid position
        batch_idx = torch.arange(batch_size, device=device)
        target = labels[batch_idx[has_valid], last_valid_pos[has_valid]]

        # Safety: clamp target to valid vocab range
        target = target.clamp(0, self.vocab_size - 1)

        # Compute loss only on valid batch elements
        loss = nn.functional.cross_entropy(
            logits[has_valid].view(-1, self.vocab_size),
            target.view(-1),
        )

        return loss

    def update_memory(self):
        if hasattr(self.l1, "apply_updates"):
            self.l1.apply_updates()

    def count_parameters(self):
        counts = {
            "l0": sum(p.numel() for p in self.l0.parameters()),
            "l1": sum(p.numel() for p in self.l1.parameters()),
            "l2": sum(p.numel() for p in self.l2.parameters()),
            "projections": sum(p.numel() for p in self.l0_proj.parameters())
            + sum(p.numel() for p in self.l1_proj.parameters()),
        }
        counts["total"] = sum(counts.values())
        counts["trainable"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return counts

    def estimate_memory(self):
        c = self.count_parameters()
        return {
            "l0_vram_gb": c["l0"] * 0.5 / 1e9,
            "l1_ram_gb": c["l1"] * 4 / 1e9,
            "l2_vram_gb": c["l2"] * 4 / 1e9,
            "total_vram_gb": (c["l0"] * 0.5 + c["l2"] * 4) / 1e9,
        }


def create_karla(config=None, use_mock=False, use_lite=False, **kwargs):
    cfg = {}
    if config is not None:
        cfg = {
            "l0_model_name": config.l0.model_name,
            "l0_bits": config.l0.bits,
            "l1_embedding_dim": config.l1.embedding_dim,
            "l1_num_heads": config.l1.num_heads,
            "l1_ngram_orders": config.l1.ngram_orders,
            "l2_hidden_dim": config.l2.hidden_dim,
        }
    cfg.update(kwargs)
    return Karla(use_mock_l0=use_mock, use_lite_l1=use_lite, use_lite_l2=use_lite, **cfg)