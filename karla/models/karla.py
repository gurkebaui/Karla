"""
karla.py — Phase 3 Beta: Highspeed GPU Forward Pass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from dataclasses import dataclass, field
import logging
import math

from .l0_perception import L0Perception, L0PerceptionMock
from .l1_knowledge import L1KnowledgeMoE
from .l2_ctm import CTMHead, TickOutput

logger = logging.getLogger("Karla")

@dataclass
class KarlaOutput:
    logits: torch.Tensor              
    internal_ticks: int               
    certainty: torch.Tensor           
    value: torch.Tensor               
    loss: Optional[torch.Tensor] = None

class Karla(nn.Module):
    def __init__(
        self,
        l0_model_name: str = "nvidia/Cosmos-Reason2-2B",
        #l0_model_name: str = "Qwen/Qwen2.5-1.5B",
        l0_bits: int = 4,
        l1_num_experts: int = 64, # GPU-Only Speed!
        l1_top_k: int = 2,
        l1_expert_hidden_dim: int = 512,
        l2_hidden_dim: int = 512,
        l2_num_neurons: int = 256,
        l2_num_internal_ticks: int = 10,
        l2_nlm_history_length: int = 25,
        l2_nlm_hidden_dim: int = 32,
        l2_num_action_pairs: int = 512,
        l2_num_output_pairs: int = 512,
        l2_attn_heads: int = 8,
        vocab_size: int = 151936,
        use_mock_l0: bool = False,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        l0_hidden_dim = 2048 if l0_model_name == "nvidia/Cosmos-Reason2-2B" else 1536


        if use_mock_l0:
            self.l0 = L0PerceptionMock(hidden_size=l0_hidden_dim, vocab_size=vocab_size)
        else:
            self.l0 = L0Perception(model_name=l0_model_name, bits=l0_bits)

        self.l1 = L1KnowledgeMoE(
            d_model=l0_hidden_dim,
            num_experts=l1_num_experts,
            top_k=l1_top_k,
            expert_hidden_dim=l1_expert_hidden_dim
        )

        self.l2 = CTMHead(
            hidden_dim=l2_hidden_dim,
            num_neurons=l2_num_neurons,
            num_internal_ticks=l2_num_internal_ticks,
            l0_hidden_dim=l0_hidden_dim,
            nlm_history_length=l2_nlm_history_length,
            nlm_hidden_dim=l2_nlm_hidden_dim,
            num_action_pairs=l2_num_action_pairs,
            num_output_pairs=l2_num_output_pairs,
            attn_heads=l2_attn_heads,
        )

        self.l1_scale_raw = nn.Parameter(torch.tensor(-2.0))   
        self.ctm_scale_raw = nn.Parameter(torch.tensor(-0.4))  

        self.l1_optimizer = torch.optim.AdamW([
            {'params': self.l1.parameters(), 'lr': 0.005},
            {'params':[self.l1_scale_raw], 'lr': 0.05}
        ])

        logger.info(f"[Karla] Model initialized for Phase 3 (Fast Native Loss)")

    def ctm_scale(self) -> torch.Tensor: return F.softplus(self.ctm_scale_raw)
    def l1_scale(self) -> torch.Tensor: return F.softplus(self.l1_scale_raw)
    def get_multi_loss_config(self) -> dict: return self.l2.get_multi_loss_config()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        max_ticks: Optional[int] = None,
        collect_tick_outputs: bool = False,
    ) -> KarlaOutput:
        device = input_ids.device

        # 1. L0 (Qwen)
        l0_out = self.l0(input_ids, attention_mask)
        l0_hidden = l0_out.hidden_states.to(device)

        # 2. L1 (MoE)
        l1_features, l1_balance_loss = self.l1(l0_hidden)

        # 3. L2 (CTM)
        combined_input = l0_hidden + self.l1_scale() * l1_features
        l2_out = self.l2(combined_input, attention_mask=attention_mask, collect_tick_outputs=collect_tick_outputs)
        
        # 4. Fusion
        features = l0_hidden + self.l1_scale() * l1_features + self.ctm_scale() * l2_out.features
        
        # NATIVE HIGH-SPEED DECODING
        logits = self.l0.decode_head(features) if labels is None else torch.empty(1, device=device) 

        loss = None
        if labels is not None:
            # Schneller nativer Forward Pass durch den Head (VRAM ist dank 64 Experten entspannt)
            full_logits = self.l0.decode_head(features)
            
            shift_logits = full_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            if (shift_labels != -100).any():
                # Blitzschneller nativer PyTorch Loss
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.vocab_size), 
                    shift_labels.view(-1), 
                    ignore_index=-100
                )
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Multi-Loss für CTM
            if collect_tick_outputs and len(l2_out.tick_outputs) > 0 and (shift_labels != -100).any():
                multi_loss_cfg = self.get_multi_loss_config()
                weights = multi_loss_cfg['multi_loss_weights']
                
                multi_loss = torch.tensor(0.0, device=device)
                for i, tick_out in enumerate(l2_out.tick_outputs):
                    tick_feat = l0_hidden + self.l1_scale() * l1_features + self.ctm_scale() * tick_out.features
                    
                    t_logits = self.l0.decode_head(tick_feat)
                    shift_t_logits = t_logits[:, :-1, :].contiguous()
                    
                    t_loss = F.cross_entropy(
                        shift_t_logits.view(-1, self.vocab_size), 
                        shift_labels.view(-1), 
                        ignore_index=-100
                    )
                    
                    w = weights[i] if i < len(weights) else 1.0
                    multi_loss = multi_loss + w * t_loss
                
                loss = loss + 0.1 * multi_loss

            loss = loss + 0.01 * l1_balance_loss

        return KarlaOutput(
            logits=logits, 
            internal_ticks=l2_out.internal_ticks, 
            certainty=l2_out.certainty, 
            value=l2_out.value, 
            loss=loss
        )

    def update_memory(self, input_ids: torch.Tensor, base_lr: float = 0.05):
        self.l1.train()
        self.l1_optimizer.zero_grad()
        
        labels = input_ids.clone()
        outputs = self.forward(input_ids, labels=labels)
        loss = outputs.loss
        loss_val = loss.item()
        
        surprise_factor = min(max((loss_val - 0.3) / 4.0, 0.0), 1.0)
        dynamic_lr = base_lr * surprise_factor
        
        if surprise_factor <= 0.01:
            self.l1.eval()
            return loss_val
            
        for param_group in self.l1_optimizer.param_groups:
            if len(param_group['params']) == 1: 
                param_group['lr'] = dynamic_lr * 5.0 
            else:
                param_group['lr'] = dynamic_lr

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.l1.parameters(), max_norm=1.0)
        self.l1_optimizer.step()
        
        # Soft Retention (MEMORA) - Angepasst für schnelle ModuleList
        with torch.no_grad():
            # Wir iterieren durch unsere schnellen GPU-Experten und lassen ihr Wissen minimal verblassen
            for expert in self.l1.experts:
                expert.w1.weight.data.mul_(0.99999)
                expert.w2.weight.data.mul_(0.99999)
        
        self.l1.eval() 
        return loss_val

    def count_parameters(self) -> dict:
        counts = {
            "l0": sum(p.numel() for p in self.l0.parameters()), 
            "l1": sum(p.numel() for p in self.l1.parameters()), 
            "l2": sum(p.numel() for p in self.l2.parameters())
        }
        counts["total"] = sum(counts.values())
        counts["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return counts

def create_karla(config=None, use_mock=False, **kwargs):
    cfg = {}
    if config is not None:
        cfg = {
            "l0_model_name": getattr(config.l0, "model_name", "nvidia/Cosmos-Reason2-2B"),
            "l0_bits": getattr(config.l0, "bits", 4),
            "l2_hidden_dim": getattr(config.l2, "hidden_dim", 512),
            "l2_num_neurons": getattr(config.l2, "num_neurons", 256),
            "l2_num_internal_ticks": getattr(config.l2, "num_internal_ticks", 10)
        }
    cfg.update(kwargs)
    return Karla(use_mock_l0=use_mock, **cfg)