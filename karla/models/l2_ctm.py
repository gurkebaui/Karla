"""
l2_ctm.py — Continuous Thought Machine (CTM) Head for Karla C1
===============================================================
Enhanced with:
- Value Head V(s) for AlphaZero-style self-improvement
- Plan/Answer mode for guided reasoning
- True adaptive compute with warmup
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class L2Output:
    """Output from CTM Head."""
    features: torch.Tensor           # (B, l0_hidden_dim) - projected to Qwen space
    internal_ticks: int              # Number of internal ticks used
    synchronization: torch.Tensor    # (B,) - sync statistic for interpretability
    certainty: torch.Tensor          # (B, 1) - sigmoid(certainty_head(h_out))
    value: torch.Tensor              # (B, 1) - V(s) estimate for RL


@dataclass 
class PlanAnswerOutput:
    """Combined output from plan + answer phases."""
    plan_features: torch.Tensor
    plan_ticks: int
    plan_certainty: torch.Tensor
    plan_value: torch.Tensor
    
    answer_features: torch.Tensor
    answer_ticks: int
    answer_certainty: torch.Tensor
    answer_value: torch.Tensor
    
    features: torch.Tensor
    internal_ticks: int
    synchronization: torch.Tensor
    certainty: torch.Tensor
    value: torch.Tensor
    
    plan_early_stopped: bool = False
    answer_early_stopped: bool = False


class BitLinear(nn.Module):
    """BitNet 1.58-ish linear layer."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_normal_(self.weight)
        self.weight.data *= 0.1
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_f32 = x.float()
        w_f32 = self.weight.float().clamp(-2.0, 2.0)
        gamma = w_f32.abs().mean().clamp(min=1e-4)
        wq = (w_f32 / gamma).round().clamp(-1, 1)
        wd = wq * gamma
        w_ste = w_f32 + (wd - w_f32).detach()
        b = self.bias.float() if self.bias is not None else None
        return F.linear(x_f32, w_ste, b).to(input_dtype)


class PrivateNLMs(nn.Module):
    """Private Neuron-Level Models."""
    def __init__(self, D: int, M: int, dhidden: int):
        super().__init__()
        self.D, self.M, self.dhidden = D, M, dhidden
        self.w1 = nn.Parameter(torch.empty(M, dhidden, D))
        self.b1 = nn.Parameter(torch.zeros(dhidden, D))
        self.w2 = nn.Parameter(torch.empty(dhidden, D))
        self.b2 = nn.Parameter(torch.zeros(D))
        nn.init.xavier_normal_(self.w1)
        nn.init.xavier_normal_(self.w2)

    def forward(self, a_hist: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("bdm,mhd->bhd", a_hist, self.w1) + self.b1.unsqueeze(0)
        out = F.gelu(out)
        z = torch.einsum("bhd,hd->bd", out, self.w2) + self.b2.unsqueeze(0)
        return z


class SynapseMLP(nn.Module):
    """Synapse model: f_syn(concat(z_t, o_t)) -> a_t"""
    def __init__(
        self,
        in_dim: int,
        D: int,
        hidden: int,
        depth: int = 3,
        dropout: float = 0.1,
        use_bitnet: bool = True
    ):
        super().__init__()
        Linear = BitLinear if use_bitnet else nn.Linear

        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [
                Linear(d, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            d = hidden
        layers += [Linear(d, D)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SyncComputer(nn.Module):
    """Subsampled synchronization with learnable decay."""
    def __init__(self, D: int, num_pairs: int, seed: int = 42):
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)
        left = torch.randint(0, D, (num_pairs,), generator=g)
        right = torch.randint(0, D, (num_pairs,), generator=g)
        self.register_buffer("idx_left", left, persistent=True)
        self.register_buffer("idx_right", right, persistent=True)
        self.r_raw = nn.Parameter(torch.zeros(num_pairs))

    def init_state(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.zeros(
            batch_size, self.idx_left.numel(),
            device=device, dtype=torch.float32
        )
        beta = torch.zeros(
            batch_size, self.idx_left.numel(),
            device=device, dtype=torch.float32
        )
        return alpha, beta

    def step(
        self,
        z: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = F.softplus(self.r_raw)
        gamma = torch.exp(-r).unsqueeze(0)
        zi = z.index_select(1, self.idx_left)
        zj = z.index_select(1, self.idx_right)
        prod = zi * zj
        alpha = gamma * alpha + prod
        beta = gamma * beta + 1.0
        sync = alpha / torch.sqrt(beta.clamp(min=1e-8))
        return sync, alpha, beta
#comment

class CTMHead(nn.Module):
    """
    CTM Head with Value Head, Plan/Answer mode, and Adaptive Compute.
    
    Features:
    - Value head V(s) for RL
    - plan_answer_forward() for guided reasoning
    - Adaptive compute with training warmup
    """
    def __init__(
        self,
        hidden_dim: int = 512,
        num_neurons: int = 256,
        num_internal_ticks: int = 10,
        plan_ticks: int = 3,
        nlm_history_length: int = 25,
        nlm_hidden_dim: int = 32,
        num_action_pairs: int = 512,
        num_output_pairs: int = 512,
        attn_heads: int = 8,
        l0_hidden_dim: int = 1536,
        use_bitnet: bool = True,
        use_adaptive_compute: bool = True,
        certainty_threshold: float = 0.95,
        sync_threshold: float = 0.8,
        min_ticks: int = 2,
        warmup_steps: int = 100,
        warmup_ticks: int = 10,
        truncation_period: int = 4,
        dropout: float = 0.1,
        vocab_size: int = None,
        value_init_range: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.D = num_neurons
        self.T = num_internal_ticks
        self.plan_ticks = plan_ticks
        self.answer_ticks = num_internal_ticks - plan_ticks
        self.M = nlm_history_length
        
        self.use_adaptive_compute = use_adaptive_compute
        self.certainty_threshold = certainty_threshold
        self.sync_threshold = sync_threshold
        self.min_ticks = min_ticks
        
        self.warmup_steps = warmup_steps
        self.warmup_ticks = warmup_ticks
        self._training_step = 0
        self._adaptive_enabled = True
        
        self.truncation_period = truncation_period

        # Core components
        self.synapse = SynapseMLP(
            in_dim=self.D + hidden_dim,
            D=self.D,
            hidden=hidden_dim,
            depth=3,
            dropout=dropout,
            use_bitnet=use_bitnet,
        )
        self.nlm = PrivateNLMs(D=self.D, M=self.M, dhidden=nlm_hidden_dim)
        self.sync_action = SyncComputer(D=self.D, num_pairs=num_action_pairs, seed=123)
        self.sync_out = SyncComputer(D=self.D, num_pairs=num_output_pairs, seed=456)

        # Attention
        self.q_proj = nn.Sequential(
            nn.Linear(num_action_pairs, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.kv_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, attn_heads, batch_first=True, dropout=dropout
        )

        # Input projections
        self.l0_kv_proj = nn.Sequential(
            nn.Linear(l0_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self._l1_kv_proj = None

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(num_output_pairs, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, l0_hidden_dim),
            nn.LayerNorm(l0_hidden_dim),
        )

        # Heads
        self.certainty_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        nn.init.uniform_(self.value_head.weight, -value_init_range, value_init_range)
        nn.init.zeros_(self.value_head.bias)

        # Plan conditioning
        self.plan_cond_proj = nn.Sequential(
            nn.Linear(l0_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Initial states
        self.z_init = nn.Parameter(torch.randn(1, self.D) * 0.01)
        self.a_hist_init = nn.Parameter(torch.randn(1, self.D, self.M) * 0.01)

    def set_training_step(self, step: int):
        """Update training step for warmup logic."""
        self._training_step = step

    def set_adaptive_enabled(self, enabled: bool):
        """Manually enable/disable adaptive compute."""
        self._adaptive_enabled = enabled

    def _should_use_adaptive(self) -> bool:
        if not self._adaptive_enabled:
            return False
        if self._training_step >= self.warmup_steps:
            return True
        return False

    def _get_max_ticks(self) -> int:
        if self._training_step < self.warmup_steps:
            return min(self.warmup_ticks, self.T)
        return self.T

    def _check_early_stop(self, certainty: torch.Tensor, sync: torch.Tensor, tick: int) -> bool:
        if tick < self.min_ticks:
            return False
        cert_condition = (certainty >= self.certainty_threshold).all()
        sync_mean = sync.mean(dim=-1).mean()
        sync_condition = sync_mean >= self.sync_threshold
        return cert_condition.item() or sync_condition.item()

    def _run_ticks(
        self,
        kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        B: int,
        max_ticks: int,
        device: torch.device,
        plan_condition: Optional[torch.Tensor] = None,
        detach_every: Optional[int] = None,
        use_adaptive: bool = True,
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        z = self.z_init.expand(B, -1).clone()
        a_hist = self.a_hist_init.expand(B, -1, -1).clone()
        alpha_a, beta_a = self.sync_action.init_state(B, device)
        alpha_o, beta_o = self.sync_out.init_state(B, device)

        used_ticks = 0
        last_cert = torch.zeros(B, 1, device=device)
        last_value = torch.zeros(B, 1, device=device)
        last_sync = torch.zeros(B, self.sync_out.idx_left.numel(), device=device)
        h_out = None
        early_stopped = False

        for t in range(max_ticks):
            used_ticks += 1

            if detach_every and t > 0 and t % detach_every == 0:
                z = z.detach()
                a_hist = a_hist.detach()
                alpha_a, beta_a = alpha_a.detach(), beta_a.detach()
                alpha_o, beta_o = alpha_o.detach(), beta_o.detach()

            sync_a, alpha_a, beta_a = self.sync_action.step(z, alpha_a, beta_a)
            q = self.q_proj(sync_a).unsqueeze(1)

            if plan_condition is not None:
                q = q + plan_condition.unsqueeze(1)

            attn_out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
            o = attn_out.squeeze(1)

            a_t = self.synapse(torch.cat([z, o], dim=-1))

            a_hist = torch.roll(a_hist, shifts=-1, dims=-1).contiguous()
            a_hist[..., -1] = a_t

            z = self.nlm(a_hist)
            z = torch.clamp(z, -5.0, 5.0)

            sync_o, alpha_o, beta_o = self.sync_out.step(z, alpha_o, beta_o)
            h_out = self.out_proj[0:3](sync_o)

            last_cert = torch.sigmoid(self.certainty_head(h_out))
            last_value = self.value_head(h_out)
            last_sync = sync_o

            if use_adaptive:
                if self._check_early_stop(last_cert, last_sync, t + 1):
                    early_stopped = True
                    break

        return z, used_ticks, h_out, last_cert, last_value, early_stopped

    def forward(
        self,
        l0_seq: torch.Tensor,
        l1_seq: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_ticks: Optional[int] = None,
    ) -> L2Output:
        B, S, _ = l0_seq.shape
        device = l0_seq.device
        
        use_adaptive = self._should_use_adaptive()
        actual_max = max_ticks if max_ticks is not None else self._get_max_ticks()

        kv0 = self.l0_kv_proj(l0_seq.float())
        if l1_seq is None:
            kv = kv0
        else:
            l1_dim = l1_seq.size(-1)
            if self._l1_kv_proj is None or self._l1_kv_proj[0].in_features != l1_dim:
                self._l1_kv_proj = nn.Sequential(
                    nn.Linear(l1_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                ).to(device)
            kv1 = self._l1_kv_proj(l1_seq.float())
            kv = self.kv_norm(kv0 + kv1)

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        truncation = self.truncation_period if self.training else None
        z, used_ticks, h_out, certainty, value, early_stopped = self._run_ticks(
            kv, key_padding_mask, B, actual_max, device,
            detach_every=truncation,
            use_adaptive=use_adaptive,
        )

        sync_o, _, _ = self.sync_out.step(z, *self.sync_out.init_state(B, device))
        features = self.out_proj(sync_o)
        sync_stat = z.abs().mean(dim=-1)

        return L2Output(
            features=features,
            internal_ticks=used_ticks,
            synchronization=sync_stat,
            certainty=certainty,
            value=value,
        )

    def plan_answer_forward(
        self,
        l0_seq: torch.Tensor,
        l1_seq: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> PlanAnswerOutput:
        B, S, _ = l0_seq.shape
        device = l0_seq.device
        
        use_adaptive = self._should_use_adaptive()

        kv0 = self.l0_kv_proj(l0_seq.float())
        if l1_seq is None:
            kv = kv0
        else:
            l1_dim = l1_seq.size(-1)
            if self._l1_kv_proj is None or self._l1_kv_proj[0].in_features != l1_dim:
                self._l1_kv_proj = nn.Sequential(
                    nn.Linear(l1_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                ).to(device)
            kv1 = self._l1_kv_proj(l1_seq.float())
            kv = self.kv_norm(kv0 + kv1)

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        plan_max_ticks = min(self.plan_ticks, self._get_max_ticks())
        z_plan, plan_ticks, h_plan, cert_plan, val_plan, plan_early = self._run_ticks(
            kv, key_padding_mask, B, plan_max_ticks, device,
            use_adaptive=use_adaptive,
        )
        
        sync_o, _, _ = self.sync_out.step(z_plan, *self.sync_out.init_state(B, device))
        plan_features = self.out_proj(sync_o)

        plan_cond = self.plan_cond_proj(plan_features)
        answer_max_ticks = self._get_max_ticks()
        
        z_answer, answer_ticks, h_answer, cert_answer, val_answer, answer_early = self._run_ticks(
            kv, key_padding_mask, B, answer_max_ticks, device,
            plan_condition=plan_cond,
            use_adaptive=use_adaptive,
        )

        sync_o, _, _ = self.sync_out.step(z_answer, *self.sync_out.init_state(B, device))
        answer_features = self.out_proj(sync_o)
        sync_stat = z_answer.abs().mean(dim=-1)

        return PlanAnswerOutput(
            plan_features=plan_features,
            plan_ticks=plan_ticks,
            plan_certainty=cert_plan,
            plan_value=val_plan,
            answer_features=answer_features,
            answer_ticks=answer_ticks,
            answer_certainty=cert_answer,
            answer_value=val_answer,
            features=answer_features,
            internal_ticks=plan_ticks + answer_ticks,
            synchronization=sync_stat,
            certainty=cert_answer,
            value=val_answer,
            plan_early_stopped=plan_early,
            answer_early_stopped=answer_early,
        )

    def get_adaptive_stats(self) -> dict:
        return {
            "training_step": self._training_step,
            "warmup_steps": self.warmup_steps,
            "in_warmup": self._training_step < self.warmup_steps,
            "adaptive_enabled": self._should_use_adaptive(),
            "current_max_ticks": self._get_max_ticks(),
        }


class CTMHeadLite(nn.Module):
    """Lightweight CTM for testing/fallback."""
    def __init__(
        self,
        hidden_dim: int = 256,
        num_internal_ticks: int = 5,
        l0_hidden_dim: int = 1536,
        **kwargs
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.core = nn.GRUCell(hidden_dim, hidden_dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, l0_hidden_dim),
            nn.LayerNorm(l0_hidden_dim)
        )
        self.certainty_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.num_internal_ticks = num_internal_ticks

    def forward(
        self,
        l0_hidden: torch.Tensor,
        l1_memory: torch.Tensor,
        attention_mask=None,
        max_ticks=None,
    ) -> L2Output:
        fused = torch.cat([l0_hidden, l1_memory], dim=-1)
        h = self.input_proj(fused)
        
        for _ in range(max_ticks or self.num_internal_ticks):
            h = self.core(h, h)
            
        return L2Output(
            features=self.feature_proj(h),
            internal_ticks=self.num_internal_ticks,
            synchronization=h.mean(-1),
            certainty=torch.sigmoid(self.certainty_head(h)),
            value=self.value_head(h),
        )
