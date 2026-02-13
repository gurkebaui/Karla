"""
L2 Reasoning Head - CTM with BitNet 1.58
========================================
Implementation of the Continuous Thought Machine (CTM) from Sakana AI
combined with BitNet 1.58 ternary quantization.

Key Principles from CTM Paper:
1. Neuron-level temporal processing with unique weights per neuron
2. Neural synchronization as latent representation
3. Internal "thought" ticks decoupled from data sequence
4. Adaptive computation time

Key Principles from BitNet:
1. Ternary weights: -1, 0, +1
2. Extremely memory efficient
3. Fast matrix operations with bitwise ops

Combined: A lightweight but powerful reasoning engine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import math


@dataclass
class L2Output:
    """Output from L2 reasoning layer"""
    logits: torch.Tensor  # (batch, vocab_size)
    internal_ticks: int   # Number of internal steps taken
    synchronization: torch.Tensor  # Neural sync matrix
    certainty: torch.Tensor  # Model's confidence


class BitLinear(nn.Module):
    """
    BitNet 1.58 Linear Layer - Ternary Weights
    
    Weights are quantized to {-1, 0, +1} during forward pass.
    This provides extreme memory efficiency and fast computation.
    
    From BitNet paper:
    - Uses absmean quantization
    - Scales weights by 1/mean(|W|)
    - Maintains gradient flow through STE (Straight-Through Estimator)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard float weights (trained normally)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def quantize_weights(self) -> torch.Tensor:
        """
        Quantize weights to ternary {-1, 0, +1}.
        
        Uses absmean quantization from BitNet paper:
        W_q = Round(W / γ) where γ = mean(|W|)
        Then clamp to {-1, 0, +1}
        """
        # Compute scaling factor (absmean)
        gamma = self.weight.abs().mean() + 1e-8
        
        # Quantize
        w_scaled = self.weight / gamma
        w_quantized = torch.round(w_scaled)
        
        # Clamp to ternary
        w_ternary = torch.clamp(w_quantized, -1, 1)
        
        return w_ternary
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized weights.
        
        Uses Straight-Through Estimator (STE) for gradients:
        - Forward: Uses quantized weights
        - Backward: Gradients flow to original weights
        """
        # Quantize weights for forward pass
        w_quantized = self.quantize_weights()
        
        # STE: Use quantized weights for forward, original for backward
        # PyTorch's autograd handles this through detach()
        w_ste = w_quantized + (self.weight - self.weight.detach())
        
        # Compute output
        output = F.linear(x, w_ste, self.bias)
        
        return output


class NeuronLevelModel(nn.Module):
    """
    Neuron-Level Model (NLM) from CTM paper.
    
    Each neuron has its own unique weights to process
    the history of pre-activations. This creates complex,
    neuron-specific temporal dynamics.
    
    From CTM paper: "Neurons as first-class citizens"
    """
    
    def __init__(
        self,
        num_neurons: int,
        history_length: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        self.num_neurons = num_neurons
        self.history_length = history_length
        self.hidden_dim = hidden_dim
        
        # Each neuron has its own MLP to process history
        # Implemented as grouped convolutions for efficiency
        self.processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(history_length, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_neurons)
        ])
    
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """
        Process history for each neuron.
        
        Args:
            history: (batch, num_neurons, history_length)
            
        Returns:
            post_activations: (batch, num_neurons)
        """
        batch_size = history.size(0)
        outputs = []
        
        for i, processor in enumerate(self.processors):
            # Process history for neuron i
            neuron_history = history[:, i, :]  # (batch, history_length)
            output = processor(neuron_history)  # (batch, 1)
            outputs.append(output)
        
        post_activations = torch.cat(outputs, dim=1)  # (batch, num_neurons)
        
        return post_activations


class SynapseModel(nn.Module):
    """
    Synapse Model from CTM paper.
    
    A U-Net style MLP that interconnects neurons
    in a shared latent space. Produces pre-activations
    for the next internal tick.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        depth: int = 3,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.depth = depth
        
        # Build U-Net style layers
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        current_dim = hidden_dim
        encoder_dims = []
        
        for i in range(depth):
            next_dim = current_dim // 2
            self.encoder.append(BitLinear(current_dim, next_dim))
            encoder_dims.append(current_dim)
            current_dim = next_dim
        
        self.bottleneck = BitLinear(current_dim, current_dim)
        
        for i in range(depth):
            prev_dim = encoder_dims[-(i+1)]
            self.decoder.append(BitLinear(current_dim, prev_dim))
            current_dim = prev_dim
        
        self.activation = nn.GELU()
    
    def forward(self, z: torch.Tensor, attention_out: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through synapse model.
        
        Args:
            z: Current post-activations (batch, hidden_dim)
            attention_out: Attention output (batch, hidden_dim)
            
        Returns:
            pre_activations: (batch, hidden_dim)
        """
        # Concatenate inputs
        x = torch.cat([z, attention_out], dim=-1) if attention_out is not None else z
        
        # Encode
        skip_connections = []
        for layer in self.encoder:
            x = self.activation(layer(x))
            skip_connections.append(x)
        
        # Bottleneck
        x = self.activation(self.bottleneck(x))
        
        # Decode with skip connections
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i < len(skip_connections):
                skip = skip_connections[-(i+1)]
                if x.size(-1) == skip.size(-1):
                    x = x + skip
            x = self.activation(x)
        
        return x


class CTMHead(nn.Module):
    """
    Continuous Thought Machine (CTM) Head.
    
    The core reasoning engine that:
    1. Runs multiple internal "thought" ticks
    2. Builds neural synchronization over time
    3. Produces final output after iterative refinement
    
    From CTM paper: "Thinking before speaking"
    """
    
    def __init__(
        self,
        hidden_dim: int = 512,
        num_neurons: int = 256,
        num_internal_ticks: int = 10,
        nlm_history_length: int = 20,
        vocab_size: int = 151936,
        use_bitnet: bool = True,
        use_adaptive_compute: bool = True,
        certainty_threshold: float = 0.95,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_neurons = num_neurons
        self.num_internal_ticks = num_internal_ticks
        self.vocab_size = vocab_size
        self.use_adaptive_compute = use_adaptive_compute
        self.certainty_threshold = certainty_threshold
        
        # Linear layer type (BitNet or standard)
        Linear = BitLinear if use_bitnet else nn.Linear
        
        # Input projection (from L0 + L1)
        self.input_proj = Linear(hidden_dim * 2, hidden_dim)  # L0 + L1
        
        # Synapse model
        self.synapse = SynapseModel(hidden_dim, depth=3)
        
        # Neuron-level models
        self.nlm = NeuronLevelModel(
            num_neurons=num_neurons,
            history_length=nlm_history_length,
            hidden_dim=64,
        )
        
        # Output heads
        self.output_head = Linear(hidden_dim, vocab_size)
        self.certainty_head = Linear(hidden_dim, 1)
        
        # Initial state (learnable)
        self.initial_z = nn.Parameter(torch.randn(1, num_neurons) * 0.02)
        self.initial_history = nn.Parameter(
            torch.randn(1, num_neurons, nlm_history_length) * 0.02
        )
        
        # Synchronization computation
        self.sync_proj_q = Linear(hidden_dim, hidden_dim // 4)
        self.sync_proj_k = Linear(hidden_dim, hidden_dim // 4)
        
        print(f"[L2] CTM Head initialized with {num_internal_ticks} internal ticks")
        print(f"[L2] Using BitNet: {use_bitnet}")
    
    def compute_synchronization(
        self,
        post_activations_history: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute neural synchronization matrix.
        
        From CTM paper: Synchronization is the correlation
        between neurons' activity over time.
        
        This serves as the latent representation.
        """
        # Stack history: (batch, num_ticks, num_neurons)
        history_stack = torch.stack(post_activations_history, dim=1)
        
        # Compute correlations between neuron pairs
        # Using simplified cosine similarity
        batch_size = history_stack.size(0)
        
        # Project to lower dimension
        q = self.sync_proj_q(history_stack.mean(dim=1))  # (batch, dim)
        k = self.sync_proj_k(history_stack.mean(dim=1))  # (batch, dim)
        
        # Synchronization score
        sync = F.cosine_similarity(q, k, dim=-1)
        
        return sync
    
    def forward(
        self,
        l0_hidden: torch.Tensor,
        l1_memory: torch.Tensor,
        max_ticks: Optional[int] = None,
    ) -> L2Output:
        """
        Forward pass with internal reasoning ticks.
        
        Args:
            l0_hidden: Hidden states from L0 (batch, hidden_dim)
            l1_memory: Memory vectors from L1 (batch, hidden_dim)
            max_ticks: Override default number of ticks
            
        Returns:
            L2Output with logits and internal state
        """
        batch_size = l0_hidden.size(0)
        num_ticks = max_ticks or self.num_internal_ticks
        
        # Initialize state
        z = self.initial_z.expand(batch_size, -1).clone()  # (batch, num_neurons)
        history = self.initial_history.expand(batch_size, -1, -1).clone()
        
        # Fuse L0 and L1 inputs
        fused_input = torch.cat([l0_hidden, l1_memory], dim=-1)
        attention_out = self.input_proj(fused_input)
        
        # Store history for synchronization
        post_activation_history = []
        
        # Internal ticks (thinking loop)
        for tick in range(num_ticks):
            # 1. Synapse model: compute pre-activations
            pre_activations = self.synapse(z, attention_out)
            
            # 2. Update history
            history = torch.roll(history, -1, dims=-1)
            history[..., -1] = pre_activations
            
            # 3. Neuron-level model: compute post-activations
            z = self.nlm(history)
            post_activation_history.append(z)
            
            # 4. Check for early stopping (adaptive compute)
            if self.use_adaptive_compute and tick >= num_ticks // 2:
                certainty = torch.sigmoid(self.certainty_head(z.mean(dim=-1, keepdim=True)))
                if (certainty > self.certainty_threshold).all():
                    num_ticks = tick + 1
                    break
        
        # Compute synchronization
        sync = self.compute_synchronization(post_activation_history)
        
        # Output
        output_z = z.mean(dim=-1, keepdim=True).expand(-1, self.hidden_dim)
        logits = self.output_head(output_z)
        certainty = torch.sigmoid(self.certainty_head(z.mean(dim=-1, keepdim=True)))
        
        return L2Output(
            logits=logits,
            internal_ticks=num_ticks,
            synchronization=sync,
            certainty=certainty,
        )


class CTMHeadLite(nn.Module):
    """
    Lightweight CTM for testing.
    Simpler architecture for faster prototyping.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_internal_ticks: int = 5,
        vocab_size: int = 151936,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_internal_ticks = num_internal_ticks
        self.vocab_size = vocab_size
        
        # Input projection
        self.input_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Simple recurrent core
        self.core = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Output head
        self.output_head = nn.Linear(hidden_dim, vocab_size)
        self.certainty_head = nn.Linear(hidden_dim, 1)
        
        self.certainty_threshold = 0.95
    
    def forward(
        self,
        l0_hidden: torch.Tensor,
        l1_memory: torch.Tensor,
        max_ticks: Optional[int] = None,
    ) -> L2Output:
        batch_size = l0_hidden.size(0)
        num_ticks = max_ticks or self.num_internal_ticks
        
        # Fuse inputs
        fused = torch.cat([l0_hidden, l1_memory], dim=-1)
        h = self.input_proj(fused)
        
        # Recurrent ticks
        for tick in range(num_ticks):
            h = self.core(h, h)
            
            # Adaptive compute
            if tick >= num_ticks // 2:
                certainty = torch.sigmoid(self.certainty_head(h))
                if (certainty > self.certainty_threshold).all():
                    break
        
        logits = self.output_head(h)
        certainty = torch.sigmoid(self.certainty_head(h))
        
        return L2Output(
            logits=logits,
            internal_ticks=num_ticks,
            synchronization=h.mean(dim=-1),
            certainty=certainty,
        )


def create_ctm_head(
    lite: bool = False,
    **kwargs
) -> nn.Module:
    """Factory function to create CTM head"""
    if lite:
        return CTMHeadLite(**kwargs)
    return CTMHead(**kwargs)
