"""
L2 Reasoning Head - CTM with BitNet 1.58
========================================
Stabilized version with robust BitLinear and LayerNorm throughout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class L2Output:
    logits: torch.Tensor
    internal_ticks: int
    synchronization: torch.Tensor
    certainty: torch.Tensor


class BitLinear(nn.Module):
    """
    BitNet 1.58 — Ternary quantization with STE.
    Quantization runs in float32 for numerical stability under AMP.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_normal_(self.weight)
        self.weight.data *= 0.1

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save input dtype for output casting
        input_dtype = x.dtype
        
        # Force float32 for quantization (AMP stability)
        x_f32 = x.float()
        w_f32 = self.weight.float()

        # Clamp weights to prevent extreme values
        w_f32 = w_f32.clamp(-2.0, 2.0)

        # Scale factor with robust epsilon
        gamma = w_f32.abs().mean().clamp(min=1e-4)

        # Quantize to {-1, 0, +1}
        w_scaled = w_f32 / gamma
        w_quant = w_scaled.round().clamp(-1, 1)

        # Dequantize
        w_deq = w_quant * gamma

        # STE: use quantized forward, gradient flows to original
        w_ste = w_f32 + (w_deq - w_f32).detach()

        # Linear operation in float32
        bias_f32 = self.bias.float() if self.bias is not None else None
        out = F.linear(x_f32, w_ste, bias_f32)

        # Cast back to input dtype (float16 under AMP, float32 otherwise)
        return out.to(input_dtype)

class NeuronLevelModel(nn.Module):
    """Shared MLP for per-neuron temporal processing"""

    def __init__(self, num_neurons: int, history_length: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(history_length, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: (batch, num_neurons, history_length)
        return self.mlp(history).squeeze(-1)  # (batch, num_neurons)


class SynapseModel(nn.Module):
    """Synapse connecting neuron state with attention input"""

    def __init__(self, num_neurons: int, hidden_dim: int, use_bitnet: bool = True):
        super().__init__()
        Linear = BitLinear if use_bitnet else nn.Linear
        
        self.z_proj = nn.Linear(num_neurons, hidden_dim)
        self.z_norm = nn.LayerNorm(hidden_dim)
        
        self.net = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_neurons),  # Output layer: standard Linear
        )

    def forward(self, z: torch.Tensor, attention_out: torch.Tensor) -> torch.Tensor:
        z_proj = self.z_norm(self.z_proj(z))
        combined = torch.cat([z_proj, attention_out], dim=-1)
        return self.net(combined)


class CTMHead(nn.Module):
    """
    Continuous Thought Machine reasoning head.
    Stabilized with LayerNorm at every projection boundary.
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

        Linear = BitLinear if use_bitnet else nn.Linear

        # Input: concat(l0_proj, l1_proj) = hidden_dim * 2
        self.input_proj = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Core
        self.synapse = SynapseModel(num_neurons, hidden_dim, use_bitnet)
        self.nlm = NeuronLevelModel(num_neurons, nlm_history_length)

        # Output path - use standard Linear for output_head (not BitNet)
        # BitNet on the final projection to vocab causes instability
        self.neuron_proj = nn.Sequential(
            Linear(num_neurons, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.output_head = nn.Linear(hidden_dim, vocab_size)  # Standard!
        self.certainty_head = nn.Linear(hidden_dim, 1)

        # Learnable initial state (small init)
        self.initial_z = nn.Parameter(torch.randn(1, num_neurons) * 0.01)
        self.initial_history = nn.Parameter(
            torch.randn(1, num_neurons, nlm_history_length) * 0.01
        )

        print(f"[L2] CTM Head: {num_internal_ticks} ticks, BitNet={use_bitnet}")
        total = sum(p.numel() for p in self.parameters())
        print(f"[L2] Parameters: {total:,}")

    def forward(self, l0_hidden, l1_memory, max_ticks=None):
        batch_size = l0_hidden.size(0)
        num_ticks = max_ticks or self.num_internal_ticks

        z = self.initial_z.expand(batch_size, -1).clone()
        history = self.initial_history.expand(batch_size, -1, -1).clone()

        # Fuse inputs — force float32 to prevent AMP issues in loop
        l0_hidden = l0_hidden.float()
        l1_memory = l1_memory.float()
        fused = torch.cat([l0_hidden, l1_memory], dim=-1)
        attention_out = self.input_proj(fused)

        activations = []
        ticks_done = 0

        for tick in range(num_ticks):
            ticks_done += 1

            if tick > 0 and tick % 3 == 0:
                z = z.detach()
                history = history.detach()

            pre_act = self.synapse(z, attention_out)

            history = torch.roll(history, -1, dims=-1)
            history = history.clone()
            history[..., -1] = pre_act

            z = self.nlm(history)
            z = torch.clamp(z, -5.0, 5.0)

            # Extra NaN guard
            if torch.isnan(z).any():
                z = torch.nan_to_num(z, nan=0.0)

            activations.append(z)

            if self.use_adaptive_compute and tick >= 2 and not self.training:
                with torch.no_grad():
                    z_check = self.neuron_proj(z)
                    cert = torch.sigmoid(self.certainty_head(z_check))
                    if cert.min() > self.certainty_threshold:
                        break

        z_agg = torch.stack(activations, dim=1).mean(dim=1)

        z_proj = self.neuron_proj(z_agg)
        logits = self.output_head(z_proj)
        certainty = torch.sigmoid(self.certainty_head(z_proj))
        sync = torch.stack(activations, dim=1).var(dim=1).mean(dim=-1)

        return L2Output(logits, ticks_done, sync, certainty)

class CTMHeadLite(nn.Module):
    """Lightweight CTM for testing"""

    def __init__(self, hidden_dim=256, num_internal_ticks=5, vocab_size=151936, **kw):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.core = nn.GRUCell(hidden_dim, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_size)
        self.certainty_head = nn.Linear(hidden_dim, 1)
        self.num_internal_ticks = num_internal_ticks

    def forward(self, l0_hidden, l1_memory, max_ticks=None):
        fused = torch.cat([l0_hidden, l1_memory], dim=-1)
        h = self.input_proj(fused)
        for _ in range(max_ticks or self.num_internal_ticks):
            h = self.core(h, h)
        return L2Output(
            self.output_head(h), self.num_internal_ticks,
            h.mean(-1), torch.sigmoid(self.certainty_head(h))
        )


def create_ctm_head(lite=False, **kwargs):
    if lite:
        return CTMHeadLite(**kwargs)
    return CTMHead(**kwargs)