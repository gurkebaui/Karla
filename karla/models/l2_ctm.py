"""
l2_ctm.py — Parallel Sequence-Level CTM Head
Optimization: Flash Attention + Tanh Stability + TBPTT
"""

from dataclasses import dataclass, field
from typing import Optional, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TickOutput:
    features: torch.Tensor
    certainty: torch.Tensor
    value: torch.Tensor
    tick: int

@dataclass
class L2Output:
    features: torch.Tensor
    internal_ticks: int
    synchronization: torch.Tensor
    certainty: torch.Tensor
    value: torch.Tensor
    tick_outputs: List[TickOutput] = field(default_factory=list)

class PrivateNLMs(nn.Module):
    def __init__(self, D: int, M: int, dhidden: int):
        super().__init__()
        self.D, self.M, self.dhidden = D, M, dhidden
        self.w1 = nn.Parameter(torch.empty(M, dhidden, D))
        self.b1 = nn.Parameter(torch.zeros(dhidden, D))
        self.norm1 = nn.LayerNorm([dhidden, D]) 
        self.w2 = nn.Parameter(torch.empty(dhidden, D))
        self.b2 = nn.Parameter(torch.zeros(D))
        nn.init.xavier_normal_(self.w1, gain=0.1)
        nn.init.xavier_normal_(self.w2, gain=0.1)

    def forward(self, a_hist: torch.Tensor) -> torch.Tensor:
        a_scaled = a_hist / math.sqrt(self.M)
        B, S, D, M = a_scaled.shape
        a_flat = a_scaled.view(-1, D, M)
        
        out = torch.einsum("edm,mhd->ehd", a_flat, self.w1) 
        out = out + self.b1.unsqueeze(0)
        out = self.norm1(out.view(B, S, self.dhidden, D))
        
        # CRITICAL FIX: Tanh prevents exponential growth in RNNs
        out = torch.tanh(out)
        
        out_flat = out.view(-1, self.dhidden, D)
        z_flat = torch.einsum("ehd,hd->ed", out_flat, self.w2) + self.b2.unsqueeze(0)
        
        # Return strictly bounded states
        return torch.tanh(z_flat.view(B, S, D))

class SynapseMLP(nn.Module):
    def __init__(self, in_dim: int, D: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(), # Tanh statt GELU
            nn.Dropout(dropout),
            nn.Linear(hidden, D)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))

class SyncComputer(nn.Module):
    def __init__(self, D: int, num_pairs: int, seed: int = 42):
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)
        self.register_buffer("idx_left", torch.randint(0, D, (num_pairs,), generator=g))
        self.register_buffer("idx_right", torch.randint(0, D, (num_pairs,), generator=g))
        self.r_raw = nn.Parameter(torch.zeros((num_pairs,))) 

    def init_state(self, B: int, S: int, device: torch.device):
        num_pairs = self.idx_left.numel()
        alpha = torch.zeros(B, S, num_pairs, device=device)
        beta = torch.ones(B, S, num_pairs, device=device) 
        return alpha, beta

    def step(self, z: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        gamma = torch.sigmoid(self.r_raw).view(1, 1, -1)
        
        zi = z[..., self.idx_left]
        zj = z[..., self.idx_right]
        prod = zi * zj # Bounded since z is bounded by Tanh
        
        alpha = gamma * alpha + (1.0 - gamma) * prod
        beta = gamma * beta + (1.0 - gamma) * 1.0
        
        sync = alpha / beta.clamp(min=1e-4)
        return sync, alpha, beta

class CTMHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        num_neurons: int = 256,
        num_internal_ticks: int = 10,
        nlm_history_length: int = 25,
        nlm_hidden_dim: int = 32,
        num_action_pairs: int = 512,
        num_output_pairs: int = 512,
        attn_heads: int = 8,
        l0_hidden_dim: int = 2048,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.D = num_neurons
        self.T = num_internal_ticks
        self.M = nlm_history_length
        self.attn_heads = attn_heads
        
        t1 = max(1, self.T // 3)
        t2 = max(2, 2 * self.T // 3)
        self.multi_loss_ticks = [t1, t2, self.T]
        self.multi_loss_weights = [0.1, 0.3, 1.0]

        self.input_norm = nn.LayerNorm(l0_hidden_dim) 
        self.synapse = SynapseMLP(self.D + hidden_dim, self.D, hidden_dim, dropout)
        self.nlm = PrivateNLMs(self.D, self.M, nlm_hidden_dim)
        
        self.sync_action = SyncComputer(self.D, num_action_pairs, seed=123)
        self.sync_out = SyncComputer(self.D, num_output_pairs, seed=456)

        self.q_proj = nn.Linear(num_action_pairs, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_attn = nn.Linear(hidden_dim, hidden_dim)

        self.l0_kv_proj = nn.Linear(l0_hidden_dim, hidden_dim)
        
        self.out_proj = nn.Sequential(
            nn.Linear(num_output_pairs, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(), # GELU is fine here, it's outside the recurrent loop
            nn.Linear(hidden_dim, l0_hidden_dim)
        )
        self.final_norm = nn.LayerNorm(l0_hidden_dim)

        self.certainty_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.z_init = nn.Parameter(torch.randn(self.D) * 0.01)
        self.a_hist_init = nn.Parameter(torch.randn(self.D, self.M) * 0.01)

    def get_multi_loss_config(self):
        return {'multi_loss_ticks': self.multi_loss_ticks, 'multi_loss_weights': self.multi_loss_weights}

    def forward(self, l0_seq: torch.Tensor, attention_mask=None, collect_tick_outputs=False) -> L2Output:
        B, S, _ = l0_seq.shape
        device = l0_seq.device
        
        l0_seq_norm = self.input_norm(l0_seq.float())
        kv = self.l0_kv_proj(l0_seq_norm)
        
        head_dim = self.hidden_dim // self.attn_heads
        k = self.k_proj(kv).view(B, S, self.attn_heads, head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(B, S, self.attn_heads, head_dim).transpose(1, 2)
        
        z = self.z_init.view(1, 1, self.D).expand(B, S, -1).clone()
        a_hist = self.a_hist_init.view(1, 1, self.D, self.M).expand(B, S, -1, -1).clone()
        alpha_a, beta_a = self.sync_action.init_state(B, S, device)
        alpha_o, beta_o = self.sync_out.init_state(B, S, device)

        tick_outputs = []
        last_features, last_cert, last_value = None, None, None

        for t in range(self.T):
            tick_num = t + 1
            
            # CRITICAL FIX: Truncated Backpropagation Through Time (TBPTT)
            # This stops the gradients from stacking up and exploding!
            if self.training and t > 0 and t % 3 == 0:
                z = z.detach()
                a_hist = a_hist.detach()
                alpha_a, beta_a = alpha_a.detach(), beta_a.detach()
                alpha_o, beta_o = alpha_o.detach(), beta_o.detach()
            
            sync_a, alpha_a, beta_a = self.sync_action.step(z, alpha_a, beta_a)
            q = self.q_proj(sync_a).view(B, S, self.attn_heads, head_dim).transpose(1, 2)
            
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
            o = self.out_attn(attn_out.transpose(1, 2).reshape(B, S, self.hidden_dim))

            a_t = self.synapse(torch.cat([z, o], dim=-1))
            a_hist = torch.roll(a_hist, shifts=-1, dims=-1)
            a_hist[..., -1] = a_t
            
            z = self.nlm(a_hist)
            sync_o, alpha_o, beta_o = self.sync_out.step(z, alpha_o, beta_o)
            
            if collect_tick_outputs and tick_num in self.multi_loss_ticks:
                feat = self.final_norm(self.out_proj(sync_o))
                h_out = self.out_proj[0:3](sync_o) 
                tick_outputs.append(TickOutput(
                    features=feat, 
                    certainty=torch.sigmoid(self.certainty_head(h_out)), 
                    value=self.value_head(h_out), 
                    tick=tick_num
                ))
            
            if t == self.T - 1:
                last_features = self.final_norm(self.out_proj(sync_o))
                h_out = self.out_proj[0:3](sync_o)
                last_cert = torch.sigmoid(self.certainty_head(h_out))
                last_value = self.value_head(h_out)

        return L2Output(
            features=last_features, internal_ticks=self.T,
            synchronization=z.abs().mean(dim=-1),
            certainty=last_cert, value=last_value, tick_outputs=tick_outputs
        )