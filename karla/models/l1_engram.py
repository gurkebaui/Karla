"""
L1 Memory System - CMS Engram (Hash-based Memory on CPU)
=========================================================
Implementation of the Engram memory system from DeepSeek's paper.

Key Principle:
- Treats memory as learnable parameters with O(1) lookup
- Uses multi-head hashing for robust retrieval
- Runs on CPU/RAM for large memory capacity
- Updates via Delta Rule (not backprop through memory)

From Engram Paper:
- N-gram embeddings with tokenizer compression
- Context-aware gating for retrieved embeddings
- Deterministic addressing enables prefetching

CRITICAL FIX: Autograd Hook for Gradient Flow
==============================================
The embedding tables have requires_grad=False, so gradients normally
don't flow back to them. We use register_hook() on the output tensor
to capture gradients during backward pass and queue them for Delta Rule updates.
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
    """Output from L1 memory layer"""
    memory_vectors: torch.Tensor  # (batch, seq_len, mem_dim)
    retrieval_weights: torch.Tensor  # (batch, seq_len, num_heads)


class MultiHeadHasher:
    """
    Multi-head hashing for N-gram lookup.
    
    From Engram paper: Each head uses a deterministic hash function
    to map N-grams to indices in embedding tables.
    """
    
    def __init__(
        self,
        ngram_orders: List[int] = [2, 3, 4],
        num_heads: int = 8,
        prime_sizes: List[int] = [100003, 100019, 100043],
    ):
        self.ngram_orders = ngram_orders
        self.num_heads = num_heads
        self.prime_sizes = prime_sizes
        
        # Generate random multipliers for each head
        # Using fixed seed for reproducibility
        np.random.seed(42)
        self.multipliers = [
            [np.random.randint(2, prime_size - 1) for _ in range(num_heads)]
            for prime_size in prime_sizes
        ]
    
    def _hash_ngram(self, ngram: Tuple[int, ...], head_idx: int, prime_idx: int) -> int:
        """
        Hash an N-gram to an index using multiplicative-XOR hashing.
        
        From paper: φ_{n,k}(g) = (Σ g[i] * multiplier^i) mod prime_size
        """
        prime_size = self.prime_sizes[prime_idx]
        multiplier = self.multipliers[prime_idx][head_idx]
        
        hash_val = 0
        for i, token in enumerate(ngram):
            hash_val = (hash_val * multiplier + token) % prime_size
        
        return hash_val
    
    def hash_tokens(
        self,
        token_ids: torch.Tensor,
        position: int,
    ) -> Dict[Tuple[int, int], int]:
        """
        Hash all N-grams ending at position.
        
        Returns: Dict mapping (ngram_order, head_idx) -> hash_index
        """
        hashes = {}
        
        for n_idx, n in enumerate(self.ngram_orders):
            if position < n - 1:
                continue
            
            ngram = tuple(token_ids[position - n + 1:position + 1].tolist())
            
            for h_idx in range(self.num_heads):
                hash_idx = self._hash_ngram(ngram, h_idx, n_idx)
                hashes[(n, h_idx)] = hash_idx
        
        return hashes
    
    def hash_all_positions(
        self,
        token_ids: torch.Tensor,
    ) -> Dict[int, Dict[Tuple[int, int], int]]:
        """
        Hash N-grams for all positions in a sequence.
        
        Returns: Dict mapping position -> {(ngram_order, head_idx) -> hash_index}
        """
        seq_len = token_ids.shape[0]
        all_hashes = {}
        
        for pos in range(seq_len):
            all_hashes[pos] = self.hash_tokens(token_ids, pos)
        
        return all_hashes


class EngramMemory(nn.Module):
    """
    Engram Memory Module - Hash-based lookup memory.
    
    This memory:
    1. Stores static embeddings indexed by N-gram hashes
    2. Retrieves embeddings via O(1) lookup
    3. Updates via Delta Rule (not backprop through memory)
    4. Runs on CPU for large memory capacity
    
    Key insight from paper: Memory acts like learnable parameters
    but with different update frequency (slow, steady learning).
    
    CRITICAL: Gradient Flow via Autograd Hook
    ==========================================
    The embedding tables have requires_grad=False. To capture gradients
    for Delta Rule updates, we register a hook on the memory_vectors tensor.
    During backward pass, this hook captures the gradient and queues it
    for asynchronous CPU updates.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        num_heads: int = 8,
        ngram_orders: List[int] = [2, 3, 4],
        prime_sizes: List[int] = [100003, 100019, 100043],
        device: str = "cpu",
        learning_rate: float = 0.01,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ngram_orders = ngram_orders
        self.prime_sizes = prime_sizes
        self.device = device
        self.learning_rate = learning_rate
        
        # Total dimension of concatenated embeddings
        self.total_embed_dim = embedding_dim * num_heads * len(ngram_orders)
        
        # Create embedding tables for each (ngram_order, head) combination
        # Stored on CPU for large memory capacity
        self.embedding_tables = nn.ParameterDict()
        
        for n_idx, (n, prime_size) in enumerate(zip(ngram_orders, prime_sizes)):
            for h_idx in range(num_heads):
                table_name = f"table_n{n}_h{h_idx}"
                # Initialize embeddings
                table = nn.Parameter(
                    torch.randn(prime_size, embedding_dim) * 0.02,
                    requires_grad=False,  # NOT updated via backprop!
                )
                self.embedding_tables[table_name] = table
        
        # Hasher
        self.hasher = MultiHeadHasher(ngram_orders, num_heads, prime_sizes)
        
        # Gating mechanism (context-aware) - these ARE trainable
        self.gate_query = nn.Linear(embedding_dim, embedding_dim)
        self.gate_key = nn.Linear(self.total_embed_dim, embedding_dim)
        self.gate_value = nn.Linear(self.total_embed_dim, embedding_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Delta rule update queue
        self.update_queue = []
        self._update_lock = threading.Lock()
        
        # Storage for forward pass context (needed for gradient hook)
        self._current_token_ids = None
        self._current_hashes = None
        
        # Statistics
        self._total_updates = 0
        
        logger.info(f"[L1] Engram Memory initialized on {device}")
        logger.info(f"[L1] Embedding dim: {embedding_dim}, Heads: {num_heads}")
        logger.info(f"[L1] N-gram orders: {ngram_orders}")
        logger.info(f"[L1] Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def retrieve(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve memory vectors for a sequence.
        
        Args:
            token_ids: (batch, seq_len) token IDs
            hidden_states: (batch, seq_len, hidden_dim) from L0
            
        Returns:
            memory_vectors: (batch, seq_len, total_embed_dim) raw retrieved embeddings
            gated_output: (batch, seq_len, embedding_dim) gated and projected output
        """
        batch_size, seq_len = token_ids.shape
        
        # Pre-compute all hashes for efficiency
        all_hashes_batch = []
        for b in range(batch_size):
            all_hashes_batch.append(self.hasher.hash_all_positions(token_ids[b]))
        
        # Collect all retrieved embeddings
        all_embeddings = []
        
        for b in range(batch_size):
            batch_embeddings = []
            
            for pos in range(seq_len):
                hashes = all_hashes_batch[b][pos]
                
                if not hashes:
                    # No valid N-grams (start of sequence)
                    combined = torch.zeros(self.total_embed_dim, device=self.device)
                else:
                    pos_embeddings = []
                    
                    # Retrieve in fixed order for consistent dimensions
                    for n_idx, n in enumerate(self.ngram_orders):
                        for h_idx in range(self.num_heads):
                            key = (n, h_idx)
                            if key in hashes:
                                table_name = f"table_n{n}_h{h_idx}"
                                table = self.embedding_tables[table_name]
                                embedding = table[hashes[key]]
                                pos_embeddings.append(embedding)
                            else:
                                # Padding for missing N-grams
                                pos_embeddings.append(
                                    torch.zeros(self.embedding_dim, device=self.device)
                                )
                    
                    combined = torch.cat(pos_embeddings, dim=0)
                
                batch_embeddings.append(combined)
            
            all_embeddings.append(torch.stack(batch_embeddings))
        
        # Stack batches: (batch, seq_len, total_embed_dim)
        memory_vectors = torch.stack(all_embeddings)
        
        # Apply context-aware gating
        query = self.gate_query(hidden_states)
        key = self.gate_key(memory_vectors)
        value = self.gate_value(memory_vectors)
        
        # Attention-style gating (simplified, per-position)
        gate_scores = (query * key).sum(dim=-1, keepdim=True) / (self.embedding_dim ** 0.5)
        gate_weights = torch.sigmoid(gate_scores)
        gated_memory = gate_weights * value
        
        # Project to output dimension
        output = self.output_proj(gated_memory)
        
        return memory_vectors, output, gate_weights
    
    def _create_gradient_hook(self, token_ids: torch.Tensor, batch_idx: int):
        """
        Create a gradient hook for a specific batch item.
        
        This hook is called during backward pass and captures gradients
        for Delta Rule updates.
        """
        # Pre-compute hashes for this batch item
        all_hashes = self.hasher.hash_all_positions(token_ids[batch_idx])
        
        def hook_fn(grad):
            """
            Gradient hook called during backward pass.
            
            Args:
                grad: Gradient tensor (seq_len, total_embed_dim)
            """
            if not self.training:
                return
            
            seq_len = grad.shape[0]
            
            with self._update_lock:
                for pos in range(seq_len):
                    if pos not in all_hashes:
                        continue
                    
                    hashes = all_hashes[pos]
                    
                    # Extract gradient for each head
                    grad_pos = grad[pos]
                    head_dim = self.embedding_dim
                    
                    grad_idx = 0
                    for n_idx, n in enumerate(self.ngram_orders):
                        for h_idx in range(self.num_heads):
                            key = (n, h_idx)
                            if key in hashes:
                                # Get gradient for this head
                                head_grad = grad_pos[grad_idx:grad_idx + head_dim]
                                
                                # Queue update
                                table_name = f"table_n{n}_h{h_idx}"
                                hash_idx = hashes[key]
                                self.update_queue.append((
                                    table_name,
                                    hash_idx,
                                    head_grad.detach().cpu().clone()
                                ))
                            
                            grad_idx += head_dim
            
            return None  # Don't modify gradient
        
        return hook_fn
    
    def forward(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> L1Output:
        """
        Forward pass: Retrieve memory vectors.
        
        Args:
            token_ids: (batch, seq_len)
            hidden_states: (batch, seq_len, hidden_dim)
            
        Returns:
            L1Output with memory vectors
        """
        # Retrieve embeddings
        raw_memory_vectors, gated_output, gate_weights = self.retrieve(token_ids, hidden_states)
        
        # === CRITICAL: Register gradient hooks ===
        # This is the "missing link" that connects gradients to Delta Rule updates
        if self.training:
            for b in range(token_ids.shape[0]):
                hook = self._create_gradient_hook(token_ids, b)
                # Register hook on gated output (what flows to L2)
                gated_output.register_hook(hook)
        
        # Store for debugging
        self._current_token_ids = token_ids.detach().clone()
        
        return L1Output(
            memory_vectors=gated_output,  # Return gated output for L2
            retrieval_weights=gate_weights,
        )
    
    def queue_update(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        gradients: torch.Tensor,
    ):
        """
        Queue a memory update via Delta Rule.
        
        This is now called automatically via gradient hooks during backward pass.
        Kept for manual updates if needed.
        
        Delta Rule: Memory_new = Memory_old + lr * gradient
        """
        with self._update_lock:
            for i, (tokens, pos, grad) in enumerate(zip(token_ids, positions, gradients)):
                hashes = self.hasher.hash_tokens(tokens, pos.item())
                
                for (n, h_idx), hash_idx in hashes.items():
                    table_name = f"table_n{n}_h{h_idx}"
                    self.update_queue.append((
                        table_name, 
                        hash_idx, 
                        grad.detach().cpu().clone()
                    ))
    
    def apply_updates(self, max_updates: int = 10000):
        """
        Apply queued updates to memory.
        
        This is called periodically (e.g., every N batches).
        Updates are applied on CPU to avoid GPU memory pressure.
        
        Args:
            max_updates: Maximum number of updates to apply (for memory safety)
        """
        with self._update_lock:
            if not self.update_queue:
                return 0
            
            updates_applied = 0
            
            for table_name, hash_idx, grad in self.update_queue[:max_updates]:
                table = self.embedding_tables[table_name]
                
                # Delta rule update: W_new = W_old + lr * grad
                with torch.no_grad():
                    # Gradient is already on CPU from hook
                    table.data[hash_idx] += self.learning_rate * grad.to(table.device)
                
                updates_applied += 1
                self._total_updates += 1
            
            # Clear processed updates
            self.update_queue = self.update_queue[max_updates:]
            
            if updates_applied > 0:
                logger.info(f"[L1] Applied {updates_applied} memory updates (total: {self._total_updates})")
            
            return updates_applied
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics"""
        with self._update_lock:
            queue_size = len(self.update_queue)
        
        return {
            "queue_size": queue_size,
            "total_updates": self._total_updates,
            "num_tables": len(self.embedding_tables),
            "table_size_mb": sum(
                p.numel() * 4 / 1e6 for p in self.embedding_tables.values()
            ),
        }


class EngramMemoryLite(nn.Module):
    """
    Lightweight Engram for testing/prototyping.
    Uses smaller memory tables for faster iteration.
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 4,
        ngram_orders: List[int] = [2, 3],
        memory_size: int = 10000,
        device: str = "cpu",
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.ngram_orders = ngram_orders
        self.device = device
        
        # Simplified: single embedding table
        total_size = memory_size * num_heads * len(ngram_orders)
        self.memory_table = nn.Parameter(
            torch.randn(total_size, embedding_dim) * 0.02,
            requires_grad=False,
        )
        
        # Simple hashing
        self.hasher = MultiHeadHasher(ngram_orders, num_heads, [total_size])
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
        logger.info(f"[L1-Lite] Engram Lite initialized on {device}")
    
    def forward(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> L1Output:
        batch_size, seq_len = token_ids.shape
        
        # Simple lookup (mock implementation)
        memory_vectors = torch.randn(
            batch_size, seq_len, self.embedding_dim,
            device=self.device
        ) * 0.1
        
        memory_vectors = self.output_proj(memory_vectors)
        
        return L1Output(
            memory_vectors=memory_vectors,
            retrieval_weights=torch.ones(batch_size, seq_len, 1, device=self.device),
        )
    
    def queue_update(self, *args, **kwargs):
        pass  # No updates for lite version
    
    def apply_updates(self):
        pass
    
    def get_memory_stats(self) -> Dict[str, float]:
        return {"type": "lite", "numel": self.memory_table.numel()}


def create_engram_memory(
    lite: bool = False,
    **kwargs
) -> nn.Module:
    """Factory function to create Engram memory"""
    if lite:
        return EngramMemoryLite(**kwargs)
    return EngramMemory(**kwargs)
