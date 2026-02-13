"""
L1 Memory System - CMS Engram
==============================
Hash-based O(1) lookup memory.
Embedding tables on CPU, gating on GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import threading
import logging

logger = logging.getLogger("L1Engram")


@dataclass
class L1Output:
    memory_vectors: torch.Tensor
    retrieval_weights: torch.Tensor


class MultiHeadHasher:
    """Deterministic N-gram hashing"""

    def __init__(self, ngram_orders=[2, 3], num_heads=4, table_size=50000):
        self.ngram_orders = ngram_orders
        self.num_heads = num_heads
        self.table_size = table_size
        np.random.seed(42)
        self.multipliers = [
            np.random.randint(2, 2**31, size=num_heads) for _ in ngram_orders
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
    Hash-based memory with gated retrieval.
    
    CRITICAL: All retrieved embeddings are normalized before gating
    to prevent NaN from extreme concatenated values.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_heads: int = 4,
        ngram_orders: List[int] = None,
        table_size: int = 50000,
        hidden_dim: int = 1536,
        learning_rate: float = 0.001,
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

        self.num_tables = len(ngram_orders) * num_heads
        self.concat_dim = embedding_dim * self.num_tables

        # Embedding tables as CPU buffers
        for n_idx in range(len(ngram_orders)):
            for h_idx in range(num_heads):
                buf = torch.randn(table_size, embedding_dim) * 0.01
                self.register_buffer(f"emb_n{n_idx}_h{h_idx}", buf)

        self.hasher = MultiHeadHasher(ngram_orders, num_heads, table_size)

        # Gating layers (trainable, on GPU)
        self.gate_query = nn.Linear(hidden_dim, embedding_dim)
        self.gate_key = nn.Linear(self.concat_dim, embedding_dim)
        self.gate_value = nn.Linear(self.concat_dim, embedding_dim)
        
        # Normalization BEFORE gating projections
        self.raw_norm = nn.LayerNorm(self.concat_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        self.output_norm = nn.LayerNorm(embedding_dim)

        # Delta rule
        self.update_queue = []
        self._lock = threading.Lock()
        self._total_updates = 0

        logger.info(f"[L1] Engram Memory initialized")
        logger.info(f"[L1] Embedding tables on CPU, gating layers follow model device")
        logger.info(f"[L1] Embedding dim: {embedding_dim}, Heads: {num_heads}")
        logger.info(f"[L1] N-gram orders: {ngram_orders}")
        logger.info(f"[L1] Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _retrieve_embeddings(self, token_ids, compute_device):
        batch_size, seq_len = token_ids.shape
        output = torch.zeros(
            batch_size, seq_len, self.concat_dim,
            device=compute_device, dtype=torch.float32,
        )

        for b in range(batch_size):
            for pos in range(seq_len):
                hashes = self.hasher.hash_position(token_ids[b], pos)
                if not hashes:
                    continue
                parts = []
                for n_idx in range(len(self.ngram_orders)):
                    for h_idx in range(self.num_heads):
                        key = (n_idx, h_idx)
                        if key in hashes:
                            table = getattr(self, f"emb_n{n_idx}_h{h_idx}")
                            vec = table[hashes[key]].to(compute_device)
                            parts.append(vec)
                        else:
                            parts.append(
                                torch.zeros(self.embedding_dim, device=compute_device)
                            )
                output[b, pos] = torch.cat(parts, dim=0)

        return output

    def forward(self, token_ids, hidden_states):
        compute_device = self.gate_query.weight.device

        # 1. Retrieve raw embeddings
        raw_memory = self._retrieve_embeddings(token_ids, compute_device)
        
        # 2. Normalize BEFORE projecting (prevents NaN from 4096-dim concat)
        raw_memory = self.raw_norm(raw_memory)
        hidden_normed = self.query_norm(hidden_states)

        # 3. Gated retrieval
        query = self.gate_query(hidden_normed)
        key = self.gate_key(raw_memory)
        value = self.gate_value(raw_memory)

        scale = self.embedding_dim ** 0.5
        gate_scores = (query * key).sum(dim=-1, keepdim=True) / scale
        gate_weights = torch.sigmoid(gate_scores)

        gated = gate_weights * value
        output = self.output_norm(self.output_proj(gated))

        return L1Output(memory_vectors=output, retrieval_weights=gate_weights)

    def apply_updates(self, max_updates=10000):
        with self._lock:
            if not self.update_queue:
                return 0
            count = 0
            for name, idx, grad in self.update_queue[:max_updates]:
                table = getattr(self, name)
                with torch.no_grad():
                    table[idx] += self.learning_rate * grad.to(table.device)
                count += 1
                self._total_updates += 1
            self.update_queue = self.update_queue[max_updates:]
            if count > 0:
                logger.debug(f"[L1] Applied {count} updates")
            return count


class EngramMemoryLite(nn.Module):
    def __init__(self, embedding_dim=256, hidden_dim=1536, **kwargs):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, token_ids, hidden_states):
        mem = self.norm(self.proj(hidden_states))
        B, S, _ = mem.shape
        return L1Output(mem, torch.ones(B, S, 1, device=mem.device))

    def apply_updates(self):
        pass


def create_engram_memory(lite=False, **kwargs):
    if lite:
        return EngramMemoryLite(**kwargs)
    return EngramMemory(**kwargs)