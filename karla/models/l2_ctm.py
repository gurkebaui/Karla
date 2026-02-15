"""
L2 Reasoning Head - CTM with BitNet 1.58
==========================================
Processes POOLED vectors (one per sequence).
Outputs feature vectors in Qwen's space (1536-dim).
These get ADDED to L0's output as a residual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class L2Output:
    features: torch.Tensor          # (batch, l0_hidden_dim)
    internal_ticks: int
    synchronization: torch.Tensor
    certainty: torch.Tensor


class BitLinear(nn.Module):
    """BitNet 1.58 — float32 quantization for AMP stability."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_normal_(self.weight)
        self.weight.data *= 0.1
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        input_dtype = x.dtype
        x_f32, w_f32 = x.float(), self.weight.float()
        w_f32 = w_f32.clamp(-2.0, 2.0)
        gamma = w_f32.abs().mean().clamp(min=1e-4)
        w_quant = (w_f32 / gamma).round().clamp(-1, 1)
        w_deq = w_quant * gamma
        w_ste = w_f32 + (w_deq - w_f32).detach()
        bias_f32 = self.bias.float() if self.bias is not None else None
        return F.linear(x_f32, w_ste, bias_f32).to(input_dtype)


class NeuronLevelModel(nn.Module):
    def __init__(self, num_neurons, history_length, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(history_length, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, history):
        return self.mlp(history).squeeze(-1)


class SynapseModel(nn.Module):
    def __init__(self, num_neurons, hidden_dim, use_bitnet=True):
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
            nn.Linear(hidden_dim, num_neurons),
        )

    def forward(self, z, attention_out):
        z_proj = self.z_norm(self.z_proj(z))
        return self.net(torch.cat([z_proj, attention_out], dim=-1))


class CTMHead(nn.Module):
    def __init__(
        self,
        hidden_dim=512,
        num_neurons=256,
        num_internal_ticks=10,
        nlm_history_length=20,
        l0_hidden_dim=1536,
        use_bitnet=True,
        use_adaptive_compute=True,
        certainty_threshold=0.95,
        vocab_size=None,  # ignored
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_neurons = num_neurons
        self.num_internal_ticks = num_internal_ticks
        self.use_adaptive_compute = use_adaptive_compute
        self.certainty_threshold = certainty_threshold

        Linear = BitLinear if use_bitnet else nn.Linear

        self.input_proj = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.synapse = SynapseModel(num_neurons, hidden_dim, use_bitnet)
        self.nlm = NeuronLevelModel(num_neurons, nlm_history_length)

        self.neuron_proj = nn.Sequential(
            Linear(num_neurons, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Project to Qwen space — standard Linear (not BitNet)
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, l0_hidden_dim),
            nn.LayerNorm(l0_hidden_dim),
        )

        self.certainty_head = nn.Linear(hidden_dim, 1)

        self.initial_z = nn.Parameter(torch.randn(1, num_neurons) * 0.01)
        self.initial_history = nn.Parameter(
            torch.randn(1, num_neurons, nlm_history_length) * 0.01
        )

        total = sum(p.numel() for p in self.parameters())
        print(f"[L2] CTM Head: {num_internal_ticks} ticks, BitNet={use_bitnet}")
        print(f"[L2] Output: {hidden_dim} → {l0_hidden_dim} (Qwen space)")
        print(f"[L2] Parameters: {total:,}")

    def forward(self, l0_hidden, l1_memory, max_ticks=None):
        batch_size = l0_hidden.size(0)
        num_ticks = max_ticks or self.num_internal_ticks

        l0_hidden = l0_hidden.float()
        l1_memory = l1_memory.float()

        z = self.initial_z.expand(batch_size, -1).clone()
        history = self.initial_history.expand(batch_size, -1, -1).clone()

        fused = torch.cat([l0_hidden, l1_memory], dim=-1)
        attention_out = self.input_proj(fused)

        activations = []

        for tick in range(num_ticks):
            if tick > 0 and tick % 3 == 0:
                z = z.detach()
                history = history.detach()

            pre_act = self.synapse(z, attention_out)
            history = torch.roll(history, -1, dims=-1)
            history = history.clone()
            history[..., -1] = pre_act
            z = self.nlm(history)
            z = torch.clamp(z, -5.0, 5.0)

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
        z_hidden = self.neuron_proj(z_agg)
        features = self.feature_proj(z_hidden)
        certainty = torch.sigmoid(self.certainty_head(z_hidden))
        sync = torch.stack(activations, dim=1).var(dim=1).mean(dim=-1)

        return L2Output(features, len(activations), sync, certainty)


class CTMHeadLite(nn.Module):
    def __init__(self, hidden_dim=256, num_internal_ticks=5, l0_hidden_dim=1536, **kw):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        self.core = nn.GRUCell(hidden_dim, hidden_dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, l0_hidden_dim), nn.LayerNorm(l0_hidden_dim)
        )
        self.certainty_head = nn.Linear(hidden_dim, 1)
        self.num_internal_ticks = num_internal_ticks

    def forward(self, l0_hidden, l1_memory, max_ticks=None):
        fused = torch.cat([l0_hidden, l1_memory], dim=-1)
        h = self.input_proj(fused)
        for _ in range(max_ticks or self.num_internal_ticks):
            h = self.core(h, h)
        return L2Output(
            self.feature_proj(h), self.num_internal_ticks,
            h.mean(-1), torch.sigmoid(self.certainty_head(h))
        )