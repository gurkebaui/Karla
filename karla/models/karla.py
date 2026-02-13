"""
Karla - The Self-Evolving Reasoner
==================================
Main model that integrates L0 (Perception), L1 (Memory), and L2 (Reasoning).

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                         INPUT                                │
│                        Tokens                                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   L0: Perception (Frozen)                    │
│                  Qwen 2.5-1.5B 4-bit                         │
│                    ↓ Hidden States                           │
└───────────┬─────────────────────────────────────┬───────────┘
            │                                     │
            ▼                                     ▼
┌───────────────────────┐             ┌───────────────────────┐
│    L1: Memory (CPU)   │             │   Direct to L2        │
│    CMS Engram         │             │   (for fusion)        │
│    ↓ Memory Vectors   │             │                       │
└───────────┬───────────┘             └───────────┬───────────┘
            │                                     │
            └─────────────────┬───────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   L2: Reasoning (Trainable)                  │
│                  CTM Head + BitNet                           │
│                   ↓ Logits                                   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
                       OUTPUT

Nested Learning Principle:
- L0: Update frequency = 0 (frozen)
- L1: Update frequency = low (slow learning via delta rule)
- L2: Update frequency = high (fast learning via backprop)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import logging

from .l0_perception import L0Perception, L0PerceptionMock, create_l0_perception
from .l1_engram import EngramMemory, EngramMemoryLite, create_engram_memory
from .l2_ctm import CTMHead, CTMHeadLite, create_ctm_head


@dataclass
class KarlaOutput:
    """Output from the full Karla model"""
    logits: torch.Tensor
    hidden_states: torch.Tensor
    memory_vectors: torch.Tensor
    internal_ticks: int
    certainty: torch.Tensor
    loss: Optional[torch.Tensor] = None


class Karla(nn.Module):
    """
    Karla - The Self-Evolving Reasoner
    
    A 3-level nested learning architecture for efficient reasoning
    on edge devices (target: RTX 4060 Ti).
    
    Key Features:
    1. Frozen perception backbone (L0) for semantic understanding
    2. Hash-based memory system (L1) for knowledge storage
    3. CTM reasoning head (L2) for iterative thinking
    
    Memory Budget:
    - L0 (Qwen 4-bit): ~2 GB VRAM
    - L1 (Engram): CPU RAM (scalable)
    - L2 (CTM + BitNet): ~50 MB VRAM
    
    Total: ~2.5 GB VRAM for inference
    """
    
    def __init__(
        self,
        # L0 Configuration
        l0_model_name: str = "Qwen/Qwen2.5-1.5B",
        l0_bits: int = 4,
        l0_frozen: bool = True,
        
        # L1 Configuration
        l1_embedding_dim: int = 512,
        l1_num_heads: int = 8,
        l1_memory_size: int = 100_000,
        
        # L2 Configuration
        l2_hidden_dim: int = 512,
        l2_num_neurons: int = 256,
        l2_num_internal_ticks: int = 10,
        l2_use_bitnet: bool = True,
        
        # General
        vocab_size: int = 151936,
        use_mock_l0: bool = False,
        use_lite_l1: bool = False,
        use_lite_l2: bool = False,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.use_mock_l0 = use_mock_l0
        
        # Initialize components
        logging.info("[Karla] Initializing components...")
        
        # L0: Perception (Frozen)
        if use_mock_l0:
            self.l0 = L0PerceptionMock(hidden_size=l2_hidden_dim * 2)
        else:
            self.l0 = L0Perception(
                model_name=l0_model_name,
                bits=l0_bits,
                frozen=l0_frozen,
            )
        
        # L1: Memory (Engram)
        if use_lite_l1:
            self.l1 = EngramMemoryLite(
                embedding_dim=l1_embedding_dim,
                num_heads=l1_num_heads,
            )
        else:
            self.l1 = EngramMemory(
                embedding_dim=l1_embedding_dim,
                num_heads=l1_num_heads,
            )
        
        # L2: Reasoning (CTM)
        if use_lite_l2:
            self.l2 = CTMHeadLite(
                hidden_dim=l2_hidden_dim,
                num_internal_ticks=l2_num_internal_ticks,
                vocab_size=vocab_size,
            )
        else:
            self.l2 = CTMHead(
                hidden_dim=l2_hidden_dim,
                num_neurons=l2_num_neurons,
                num_internal_ticks=l2_num_internal_ticks,
                vocab_size=vocab_size,
                use_bitnet=l2_use_bitnet,
            )
        
        # Projection layers to align dimensions
        # L0 hidden -> L2 hidden
        # Mock L0 has hidden_size = l2_hidden_dim * 2
        # Real Qwen has hidden_size = 1536
        l0_hidden_size = l2_hidden_dim * 2 if use_mock_l0 else 1536
        self.l0_proj = nn.Linear(l0_hidden_size, l2_hidden_dim)
        
        # L1 memory (512) -> L2 hidden (512)
        # Already aligned if dimensions match
        
        logging.info("[Karla] Model initialized successfully")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> KarlaOutput:
        """
        Forward pass through all three levels.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            labels: Target labels for training (batch, seq_len)
            
        Returns:
            KarlaOutput with logits and intermediate states
        """
        # ========== L0: Perception ==========
        # Extract hidden states from frozen backbone
        l0_output = self.l0(input_ids, attention_mask)
        l0_hidden = l0_output.last_hidden  # (batch, hidden_dim)
        
        # Project to L2 dimension
        l0_hidden_proj = self.l0_proj(l0_hidden)  # (batch, l2_hidden_dim)
        
        # Stop gradient for L0 (nested learning principle)
        l0_hidden_proj = l0_hidden_proj.detach()
        
        # ========== L1: Memory ==========
        # Retrieve relevant memory vectors
        l1_output = self.l1(input_ids, l0_output.hidden_states)
        l1_memory = l1_output.memory_vectors[:, -1, :]  # Last position
        
        # ========== L2: Reasoning ==========
        # Run CTM reasoning head
        l2_output = self.l2(l0_hidden_proj, l1_memory)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = l2_output.logits
            shift_labels = labels[:, -1]  # Last token
            
            # Cross-entropy loss
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return KarlaOutput(
            logits=l2_output.logits,
            hidden_states=l0_hidden,
            memory_vectors=l1_memory,
            internal_ticks=l2_output.internal_ticks,
            certainty=l2_output.certainty,
            loss=loss,
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            
        Returns:
            Generated token IDs
        """
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(generated)
                logits = outputs.logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                generated = torch.cat([generated, next_token], dim=-1)
        
        return generated
    
    def update_memory(self):
        """Apply queued memory updates (L1 delta rule)"""
        self.l1.apply_updates()
    
    def get_trainable_parameters(self) -> Dict[str, nn.Parameter]:
        """Get only trainable parameters (L2 + projections)"""
        trainable = {}
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable[name] = param
        
        return trainable
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component"""
        counts = {
            'l0_perception': sum(p.numel() for p in self.l0.parameters()),
            'l1_memory': sum(p.numel() for p in self.l1.parameters()),
            'l2_reasoning': sum(p.numel() for p in self.l2.parameters()),
            'projections': sum(p.numel() for p in self.l0_proj.parameters()),
        }
        
        counts['total'] = sum(counts.values())
        counts['trainable'] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        
        return counts
    
    def estimate_memory(self) -> Dict[str, float]:
        """Estimate memory usage in GB"""
        # L0: 4-bit quantized
        l0_mem = self.count_parameters()['l0_perception'] * 0.5 / 1e9  # 4-bit = 0.5 bytes
        
        # L1: CPU memory (float32)
        l1_mem = self.count_parameters()['l1_memory'] * 4 / 1e9
        
        # L2: BitNet (ternary, approx 2 bits) + gradients
        l2_params = self.count_parameters()['l2_reasoning']
        l2_mem = l2_params * 0.25 / 1e9  # Approximate
        l2_mem_grad = l2_params * 4 / 1e9  # Gradients are float32
        
        return {
            'l0_vram_gb': l0_mem,
            'l1_ram_gb': l1_mem,
            'l2_vram_gb': l2_mem + l2_mem_grad,
            'total_vram_gb': l0_mem + l2_mem + l2_mem_grad,
        }


def create_karla(
    config: Optional[Any] = None,
    use_mock: bool = False,
    use_lite: bool = False,
    **kwargs
) -> Karla:
    """
    Factory function to create a Karla model.
    
    Args:
        config: Configuration object
        use_mock: Use mock L0 for testing
        use_lite: Use lightweight components
        **kwargs: Override config parameters
    """
    if config is not None:
        kwargs = {
            'l0_model_name': config.l0.model_name,
            'l0_bits': config.l0.bits,
            'l0_frozen': config.l0.frozen,
            'l1_embedding_dim': config.l1.embedding_dim,
            'l1_num_heads': config.l1.num_heads,
            'l2_hidden_dim': config.l2.hidden_dim,
            'l2_num_neurons': config.l2.num_neurons,
            'l2_num_internal_ticks': config.l2.num_internal_ticks,
            'l2_use_bitnet': config.l2.use_bitnet,
            **kwargs
        }
    
    return Karla(
        use_mock_l0=use_mock,
        use_lite_l1=use_lite,
        use_lite_l2=use_lite,
        **kwargs
    )
