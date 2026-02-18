"""
l1_engram_v2.py — Engram Memory with Reward-Weighted Updates
=============================================================

Enhanced with Phase D features:
- Reward-weighted gradient updates (higher reward = stronger consolidation)
- Advantage-based weighting (normalized rewards)
- Separate handling of positive/negative rewards
- CMS-style memory tiers (episodic buffer → long-term memory)

Key Design (from Design-Review):
- w = clip(advantage, 0, ...) für positive Updates
- Negative Updates mit kleinerer magnitude
- Value-based priority für memory consolidation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np
import threading
import logging
from collections import defaultdict

logger = logging.getLogger("L1EngramV2")


@dataclass
class L1Output:
    memory_vectors: torch.Tensor
    retrieval_weights: torch.Tensor


@dataclass
class WeightedGradient:
    """Stores gradient with its weight for prioritized updates."""
    gradient: torch.Tensor
    weight: float
    timestamp: int
    source: str  # "reward", "advantage", "value"


class MultiHeadHasher:
    """Deterministic N-gram hashing."""

    def __init__(self, ngram_orders=[2, 3], num_heads=4, table_size=50000):
        self.ngram_orders = ngram_orders
        self.num_heads = num_heads
        self.table_size = table_size
        rng = np.random.RandomState(42)
        self.multipliers = [
            rng.randint(2, 2**31, size=num_heads) for _ in ngram_orders
        ]

    def hash_ngram(self, ngram, head_idx, order_idx):
        mult = int(self.multipliers[order_idx][head_idx])
        h = 0
        for t in ngram:
            h = (h * mult + int(t)) % self.table_size
        return h

    def hash_position(self, token_ids, position):
        result = {}
        for n_idx, n in enumerate(self.ngram_orders):
            if position < n - 1:
                continue
            ngram = tuple(token_ids[position - n + 1 : position + 1].tolist())
            for h_idx in range(self.num_heads):
                result[(n_idx, h_idx)] = self.hash_ngram(ngram, h_idx, n_idx)
        return result


class EngramMemory(nn.Module):
    """
    Hash-based O(1) lookup memory with Reward-Weighted Updates.
    
    Key Enhancements:
    1. Gradients are weighted by reward/advantage before applying
    2. Positive rewards strengthen patterns (consolidation)
    3. Negative rewards weaken patterns (with smaller magnitude)
    4. Supports multiple weighting modes: reward, advantage, value
    
    Usage:
        # Set weights before apply_updates
        engram.set_sample_weights(sample_ids, rewards, mode="advantage")
        engram.apply_updates()
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        num_heads: int = 4,
        ngram_orders: List[int] = None,
        table_size: int = 50000,
        hidden_dim: int = 1536,
        learning_rate: float = 0.001,
        negative_lr_scale: float = 0.3,  # Smaller updates for negative rewards
        min_weight: float = 0.1,  # Minimum weight for any update
        max_weight: float = 3.0,  # Maximum weight for strong consolidation
        conv_kernel_size: int = 4,
        conv_zero_init: bool = True,
        # CMS-style tiers
        use_priority_queue: bool = True,
        consolidation_threshold: float = 0.5,
    ):
        super().__init__()
        if ngram_orders is None:
            ngram_orders = [2, 3]

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ngram_orders = ngram_orders
        self.table_size = table_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.negative_lr_scale = negative_lr_scale
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.use_priority_queue = use_priority_queue
        self.consolidation_threshold = consolidation_threshold

        self.num_tables = len(ngram_orders) * num_heads
        self.concat_dim = embedding_dim * self.num_tables

        # === CPU Embedding tables ===
        for n_idx in range(len(ngram_orders)):
            for h_idx in range(num_heads):
                buf = torch.randn(table_size, embedding_dim) * 0.01
                self.register_buffer(f"emb_n{n_idx}_h{h_idx}", buf, persistent=True)

        self.hasher = MultiHeadHasher(ngram_orders, num_heads, table_size)

        # === Context-aware gating ===
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.mem_norm = nn.LayerNorm(self.concat_dim)
        self.query_proj = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.key_proj = nn.Linear(self.concat_dim, embedding_dim, bias=False)
        self.val_proj = nn.Linear(self.concat_dim, embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_norm = nn.LayerNorm(embedding_dim)

        # === Depthwise causal conv ===
        self.conv_kernel_size = int(conv_kernel_size)
        self.depthwise_conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=self.conv_kernel_size,
            groups=embedding_dim,
            bias=True,
        )
        if conv_zero_init:
            nn.init.zeros_(self.depthwise_conv.weight)
            nn.init.zeros_(self.depthwise_conv.bias)

        # === Weighted Gradient Accumulator ===
        self._grad_accumulator: Dict[Tuple[str, int], WeightedGradient] = {}
        self._sample_weights: Dict[int, float] = {}  # sample_id -> weight
        self._current_sample_id: Optional[int] = None
        
        # CMS-style tracking
        self._access_counts: Dict[Tuple[str, int], int] = defaultdict(int)
        self._consolidated: set = set()
        
        self._lock = threading.Lock()
        self._queued_count = 0
        self._total_updates = 0
        self._update_timestamp = 0

        logger.info("[L1 V2] EngramMemoryV2 initialized with Reward-Weighted Updates")
        logger.info(f"[L1 V2] tables: {self.num_tables} × {table_size} (CPU)")
        logger.info(f"[L1 V2] negative_lr_scale={negative_lr_scale}, weight_range=[{min_weight}, {max_weight}]")

    def _apply(self, fn):
        """Prevent embedding tables from moving to GPU."""
        super()._apply(fn)
        for name, buf in self.named_buffers(recurse=False):
            if name.startswith("emb_"):
                setattr(self, name, buf.to("cpu"))
        return self

    def _retrieve_embeddings(self, token_ids, compute_device):
        B, S = token_ids.shape
        out = torch.zeros(B, S, self.concat_dim, device=compute_device, dtype=torch.float32)

        for b in range(B):
            for pos in range(S):
                hashes = self.hasher.hash_position(token_ids[b], pos)
                if not hashes:
                    continue
                parts = []
                for n_idx in range(len(self.ngram_orders)):
                    for h_idx in range(self.num_heads):
                        key = (n_idx, h_idx)
                        if key in hashes:
                            table = getattr(self, f"emb_n{n_idx}_h{h_idx}")
                            vec = table[hashes[key]].to(compute_device, non_blocking=True)
                            parts.append(vec)
                            # Track access for consolidation
                            if self.use_priority_queue:
                                dict_key = (f"emb_n{n_idx}_h{h_idx}", hashes[key])
                                self._access_counts[dict_key] += 1
                        else:
                            parts.append(torch.zeros(self.embedding_dim, device=compute_device))
                out[b, pos] = torch.cat(parts, dim=0)
        return out

    def _queue_gradient_updates(self, token_ids_cpu, grad_cpu, sample_id: Optional[int] = None):
        """
        Queue gradient updates with optional sample weighting.
        
        Args:
            token_ids_cpu: Token IDs on CPU
            grad_cpu: Gradients on CPU
            sample_id: Optional sample ID for looking up weight
        """
        B, S = token_ids_cpu.shape
        
        # Get weight for this sample
        weight = 1.0  # Default
        if sample_id is not None and sample_id in self._sample_weights:
            weight = self._sample_weights[sample_id]
        
        with self._lock:
            self._update_timestamp += 1
            
            for b in range(B):
                for pos in range(S):
                    hashes = self.hasher.hash_position(token_ids_cpu[b], pos)
                    if not hashes:
                        continue
                    offset = 0
                    for n_idx in range(len(self.ngram_orders)):
                        for h_idx in range(self.num_heads):
                            key = (n_idx, h_idx)
                            if key in hashes:
                                name = f"emb_n{n_idx}_h{h_idx}"
                                idx = int(hashes[key])
                                g = grad_cpu[b, pos, offset : offset + self.embedding_dim]
                                
                                dict_key = (name, idx)
                                
                                # Apply weight to gradient
                                weighted_grad = g * weight
                                
                                if dict_key in self._grad_accumulator:
                                    # Accumulate with running average
                                    existing = self._grad_accumulator[dict_key]
                                    # Weighted average based on number of updates
                                    combined_grad = (existing.gradient + weighted_grad) / 2
                                    combined_weight = (existing.weight + weight) / 2
                                    self._grad_accumulator[dict_key] = WeightedGradient(
                                        gradient=combined_grad,
                                        weight=combined_weight,
                                        timestamp=self._update_timestamp,
                                        source="reward" if weight != 1.0 else "default",
                                    )
                                else:
                                    self._grad_accumulator[dict_key] = WeightedGradient(
                                        gradient=weighted_grad.clone(),
                                        weight=weight,
                                        timestamp=self._update_timestamp,
                                        source="reward" if weight != 1.0 else "default",
                                    )
                                self._queued_count += 1
                            offset += self.embedding_dim

    def forward(self, token_ids, hidden_states, sample_id: Optional[int] = None):
        """
        Forward pass with optional sample ID for reward weighting.
        
        Args:
            token_ids: Token IDs
            hidden_states: Hidden states from L0
            sample_id: Optional sample ID (set via set_sample_weights)
        """
        compute_device = self.query_proj.weight.device

        raw = self._retrieve_embeddings(token_ids, compute_device)
        raw = raw.detach().requires_grad_(True)

        if self.training:
            captured_ids = token_ids.detach().cpu()
            current_sample = sample_id if sample_id is not None else self._current_sample_id

            def hook_fn(grad):
                self._queue_gradient_updates(captured_ids, grad.detach().cpu(), current_sample)

            raw.register_hook(hook_fn)

        raw_n = self.mem_norm(raw)
        h_n = self.query_norm(hidden_states)

        q = self.query_proj(h_n)
        k = self.key_proj(raw_n)
        v = self.val_proj(raw_n)

        scale = self.embedding_dim**0.5
        qn = F.normalize(q, dim=-1)
        kn = F.normalize(k, dim=-1)
        gate_logits = (qn * kn).sum(dim=-1, keepdim=True) / scale
        gate = torch.sigmoid(gate_logits)

        x = self.out_norm(self.out_proj(gate * v))

        # Depthwise causal conv
        xt = x.transpose(1, 2)
        pad = self.conv_kernel_size - 1
        xt_pad = F.pad(xt, (pad, 0))
        conv = self.depthwise_conv(xt_pad)[..., : xt.size(-1)].transpose(1, 2)
        y = F.silu(conv) + x

        return L1Output(memory_vectors=y, retrieval_weights=gate)

    def set_sample_weights(
        self,
        sample_weights: Dict[int, float],
        mode: str = "advantage",
        normalize: bool = True,
    ):
        """
        Set weights for samples before apply_updates.
        
        Args:
            sample_weights: Dict mapping sample_id -> reward/advantage/value
            mode: How to interpret weights
                - "reward": Direct reward value (0-1)
                - "advantage": Normalized advantage (can be negative)
                - "value": Value estimate from CTM
            normalize: Whether to normalize weights
        """
        if not sample_weights:
            return
        
        self._sample_weights = {}
        
        for sample_id, raw_weight in sample_weights.items():
            if mode == "reward":
                # Reward: clip to [0, 1], then scale
                w = max(0.0, min(1.0, raw_weight))
                # High reward = strong consolidation
                # Zero reward = minimal update
                w = self.min_weight + w * (self.max_weight - self.min_weight)
                
            elif mode == "advantage":
                # Advantage: can be negative
                if raw_weight >= 0:
                    # Positive advantage: strengthen
                    w = self.min_weight + raw_weight * (self.max_weight - self.min_weight)
                else:
                    # Negative advantage: weaken (smaller magnitude)
                    w = self.negative_lr_scale * raw_weight  # Negative weight
                    
            elif mode == "value":
                # Value estimate: typically in [-1, 1] after normalization
                w = 1.0 + raw_weight  # Centered around 1
                w = max(self.min_weight, min(self.max_weight, w))
            else:
                w = 1.0
            
            self._sample_weights[sample_id] = w
        
        if normalize and len(self._sample_weights) > 1:
            # Normalize to prevent drift
            weights = list(self._sample_weights.values())
            mean_w = sum(abs(w) for w in weights) / len(weights)
            if mean_w > 0:
                for sid in self._sample_weights:
                    self._sample_weights[sid] /= mean_w

    def set_current_sample(self, sample_id: Optional[int]):
        """Set the current sample ID for the next forward pass."""
        self._current_sample_id = sample_id

    def apply_updates(self, max_updates=None, consolidate: bool = True) -> Tuple[int, Dict]:
        """
        Apply accumulated gradient updates with weighting.
        
        Args:
            max_updates: Maximum number of updates to apply
            consolidate: Whether to mark frequently-accessed entries as consolidated
            
        Returns:
            Tuple of (update_count, stats_dict)
        """
        stats = {
            "total_updates": 0,
            "positive_updates": 0,
            "negative_updates": 0,
            "consolidated_count": 0,
            "avg_weight": 0.0,
        }
        
        with self._lock:
            if not self._grad_accumulator:
                return 0, stats

            count = 0
            total_weight = 0.0
            
            for (name, idx), weighted_grad in self._grad_accumulator.items():
                table = getattr(self, name)
                weight = weighted_grad.weight
                
                # Determine effective learning rate based on weight sign
                if weight >= 0:
                    effective_lr = self.learning_rate * abs(weight)
                    stats["positive_updates"] += 1
                else:
                    effective_lr = self.learning_rate * abs(weight) * self.negative_lr_scale
                    stats["negative_updates"] += 1
                
                # Apply update
                with torch.no_grad():
                    update = effective_lr * weighted_grad.gradient.to(table.device)
                    if weight >= 0:
                        table[idx] -= update  # Standard SGD direction
                    else:
                        table[idx] += update  # Reverse for negative (unlearn)
                
                count += 1
                total_weight += abs(weight)
                self._total_updates += 1
                
                # Consolidation check
                if consolidate and self.use_priority_queue:
                    dict_key = (name, idx)
                    if dict_key not in self._consolidated:
                        if self._access_counts[dict_key] >= 5:  # Accessed 5+ times
                            if weighted_grad.weight >= self.consolidation_threshold:
                                self._consolidated.add(dict_key)
                                stats["consolidated_count"] += 1
                
                if max_updates is not None and count >= max_updates:
                    break

            self._grad_accumulator.clear()
            self._queued_count = 0
            
            stats["total_updates"] = count
            stats["avg_weight"] = total_weight / max(count, 1)
            
            return count, stats

    def get_consolidation_stats(self) -> Dict:
        """Get statistics about memory consolidation."""
        return {
            "total_entries": self.table_size * self.num_tables,
            "consolidated": len(self._consolidated),
            "high_access": sum(1 for c in self._access_counts.values() if c >= 5),
            "total_updates": self._total_updates,
        }

    def reset_access_counts(self):
        """Reset access counts (call between epochs)."""
        self._access_counts.clear()


class EngramMemoryLite(nn.Module):
    """Lite version without reward-weighted updates."""
    
    def __init__(self, embedding_dim=256, hidden_dim=1536, **kwargs):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, token_ids, hidden_states, sample_id=None):
        mem = self.norm(self.proj(hidden_states))
        B, S, _ = mem.shape
        return L1Output(mem, torch.ones(B, S, 1, device=mem.device))

    def set_sample_weights(self, *args, **kwargs):
        pass

    def apply_updates(self, *a, **k):
        return 0, {}


def create_engram(use_lite=False, **kwargs):
    """Factory function."""
    if use_lite:
        return EngramMemoryLite(**kwargs)
    return EngramMemory(**kwargs)
