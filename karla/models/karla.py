"""
Karla - C1 Architecture (Residual Design)
==========================================

Data flow:
  L0(Qwen) → hidden_states (post-norm, frozen)
  L1(Engram) → per-position memory vectors
  L2(CTM) → pooled reasoning vector (broadcast to all positions)
  
  features = L0_hidden + l1_scale * L1_contribution + ctm_scale * CTM_broadcast
  logits = Qwen.lm_head(features)  — for ALL positions
  
Key insight: At initialization, ctm_scale=0 and l1_scale=0.
So features = L0_hidden → logits = Qwen's own predictions.
The model STARTS as Qwen and learns refinements.
"""

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
import logging

from .l0_perception import L0Perception, L0PerceptionMock, L0Output
from .l1_engram import EngramMemory, EngramMemoryLite
from .l2_ctm import CTMHead, CTMHeadLite

logger = logging.getLogger("Karla")


@dataclass
class KarlaOutput:
    logits: torch.Tensor            # (batch, seq, vocab_size)
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

        # === L0: Perception + Decoder ===
        if use_mock_l0:
            self.l0 = L0PerceptionMock(hidden_size=l0_hidden_dim, vocab_size=vocab_size)
        else:
            self.l0 = L0Perception(model_name=l0_model_name, bits=l0_bits)

        # === L1: Engram Memory ===
        if use_lite_l1:
            self.l1 = EngramMemoryLite(
                embedding_dim=l1_embedding_dim, hidden_dim=l0_hidden_dim
            )
        else:
            self.l1 = EngramMemory(
                embedding_dim=l1_embedding_dim,
                num_heads=l1_num_heads,
                ngram_orders=l1_ngram_orders,
                table_size=l1_table_size,
                hidden_dim=l0_hidden_dim,
            )

        # === L2: CTM Reasoning ===
        if use_lite_l2:
            self.l2 = CTMHeadLite(
                hidden_dim=l2_hidden_dim, l0_hidden_dim=l0_hidden_dim
            )
        else:
            self.l2 = CTMHead(
                hidden_dim=l2_hidden_dim,
                num_neurons=l2_num_neurons,
                num_internal_ticks=l2_num_internal_ticks,
                use_bitnet=l2_use_bitnet,
                l0_hidden_dim=l0_hidden_dim,
            )

        # === Projections for CTM input (pooled vectors) ===
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

        # === NEW: Per-position L1 → Qwen space (for residual) ===
        self.l1_seq_proj = nn.Sequential(
            nn.Linear(l1_embedding_dim, l0_hidden_dim),
            nn.LayerNorm(l0_hidden_dim),
        )

        # === NEW: Learnable mixing scales ===
        # Zero-init: at start, output = pure Qwen (no L1/CTM contribution)
        # Model learns to increase these as it discovers useful signals
        self.ctm_scale = nn.Parameter(torch.zeros(1))
        self.l1_scale = nn.Parameter(torch.zeros(1))

        logger.info("[Karla] Model initialized (Residual Architecture)")
        logger.info("[Karla] At init: output = pure Qwen predictions")

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.size(0)
        device = input_ids.device

        # === L0: Full sequence perception (frozen) ===
        l0_out = self.l0(input_ids, attention_mask)
        l0_hidden = l0_out.hidden_states.to(device)  # (B, S, 1536)

        # === L1: Full sequence memory ===
        l1_out = self.l1(input_ids, l0_hidden)
        l1_memory = l1_out.memory_vectors.to(device)  # Force GPU

        # === Pool for CTM ===
        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1).long() - 1
            seq_lens = seq_lens.clamp(min=0)
            bidx = torch.arange(batch_size, device=device)
            l1_pooled = l1_memory[bidx, seq_lens]
        else:
            l1_pooled = l1_memory[:, -1, :]

        l0_last = l0_out.last_hidden.to(device)
        l0_for_ctm = self.l0_proj(l0_last)              # (B, l2_dim)
        l1_for_ctm = self.l1_proj(l1_pooled)            # (B, l2_dim)

        # === L2: CTM thinks ===
        l2_out = self.l2(l0_for_ctm, l1_for_ctm)        # features: (B, 1536)

        # === Combine: Residual Architecture ===
        l1_contrib = self.l1_seq_proj(l1_memory)        # (B, S, 1536)
        ctm_bcast = l2_out.features.unsqueeze(1)        # (B, 1, 1536)

        features = (
            l0_hidden
            + self.l1_scale * l1_contrib
            + self.ctm_scale * ctm_bcast
        )

        # === Decode ALL positions ===
        logits = self.l0.decode_head(features)          # (B, S, V)

        # === Causal LM Loss ===
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            valid_mask = shift_labels != -100
            if valid_mask.any():
                V = logits.size(-1)
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, V),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                loss = logits.sum() * 0.0

        return KarlaOutput(
            logits=logits,
            internal_ticks=l2_out.internal_ticks,
            certainty=l2_out.certainty,
            loss=loss,
        )

    def update_memory(self):
        if hasattr(self.l1, "apply_updates"):
            self.l1.apply_updates()

    def count_parameters(self):
        counts = {
            "l0": sum(p.numel() for p in self.l0.parameters()),
            "l1": sum(p.numel() for p in self.l1.parameters()),
            "l2": sum(p.numel() for p in self.l2.parameters()),
            "projections": (
                sum(p.numel() for p in self.l0_proj.parameters())
                + sum(p.numel() for p in self.l1_proj.parameters())
                + sum(p.numel() for p in self.l1_seq_proj.parameters())
                + 2  # ctm_scale, l1_scale
            ),
        }
        counts["total"] = sum(counts.values())
        counts["trainable"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return counts


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