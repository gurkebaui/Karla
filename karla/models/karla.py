"""
karla.py — Phase 2: Knowledge MoE + CTM Architecture
====================================================
- L0: Frozen perception layer (Qwen 2.5-1.5B)
- L1: Knowledge MoE (System 1: Faktenabruf)
- L2: Parallel Sequence-Level CTM Head (System 2: Logik)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from dataclasses import dataclass, field
import logging

from .l0_perception import L0Perception, L0PerceptionMock
from .l1_knowledge import L1KnowledgeMoE
from .l2_ctm import CTMHead, TickOutput

logger = logging.getLogger("Karla")

class MemoryEfficientCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, lm_head_weight, lm_head_bias, labels, ignore_index=-100, chunk_size=32):
        B, S, D = features.shape
        V = lm_head_weight.shape[0]
        total_loss = 0.0
        total_valid = 0
        
        for i in range(0, S, chunk_size):
            end_i = min(i + chunk_size, S)
            chunk_features = features[:, i:end_i, :].contiguous()
            chunk_labels = labels[:, i:end_i].contiguous()
            
            chunk_logits = F.linear(
                chunk_features.float(), 
                lm_head_weight.float(), 
                lm_head_bias.float() if lm_head_bias is not None else None
            )
            
            valid_mask = chunk_labels != ignore_index
            if valid_mask.any():
                chunk_loss = F.cross_entropy(
                    chunk_logits.view(-1, V),
                    chunk_labels.view(-1),
                    ignore_index=ignore_index,
                    reduction='sum'
                )
                total_loss = total_loss + chunk_loss
                total_valid += valid_mask.sum().item()
        
        loss = (total_loss / total_valid) if total_valid > 0 else torch.tensor(0.0, device=features.device, requires_grad=True)
        ctx.save_for_backward(features, lm_head_weight, lm_head_bias, labels)
        ctx.ignore_index = ignore_index
        ctx.chunk_size = chunk_size
        ctx.total_valid = total_valid
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        features, lm_head_weight, lm_head_bias, labels = ctx.saved_tensors
        B, S, D = features.shape
        V = lm_head_weight.shape[0]
        ignore_index, chunk_size, total_valid = ctx.ignore_index, ctx.chunk_size, ctx.total_valid
        
        if total_valid == 0:
            return torch.zeros_like(features), None, None, None, None, None
        
        grad_features = torch.zeros_like(features)
        compute_dtype = features.dtype
        
        for i in range(0, S, chunk_size):
            end_i = min(i + chunk_size, S)
            chunk_features = features[:, i:end_i, :].contiguous()
            chunk_labels = labels[:, i:end_i].contiguous()
            
            chunk_logits = F.linear(
                chunk_features.float(), 
                lm_head_weight.float(), 
                lm_head_bias.float() if lm_head_bias is not None else None
            )
            
            chunk_logits_flat = chunk_logits.view(-1, V)
            chunk_labels_flat = chunk_labels.view(-1)
            
            probs = F.softmax(chunk_logits_flat, dim=-1)
            grad_logits = probs
            
            valid_mask = chunk_labels_flat != ignore_index
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            valid_labels = chunk_labels_flat[valid_mask]
            
            if len(valid_indices) > 0:
                grad_logits[valid_indices, valid_labels] -= 1.0
            
            grad_logits = grad_logits * grad_output / total_valid
            grad_logits[~valid_mask] = 0.0
            
            grad_chunk = grad_logits @ lm_head_weight.float()
            grad_features[:, i:end_i, :] = grad_chunk.view(B, end_i - i, D).to(compute_dtype)
            
            del chunk_logits, probs, grad_logits
        
        return grad_features, None, None, None, None, None

def memory_efficient_lm_loss(features, lm_head, labels, ignore_index=-100, chunk_size=32):
    return MemoryEfficientCrossEntropy.apply(features, lm_head.weight, lm_head.bias, labels, ignore_index, chunk_size)

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
        l0_model_name: str = "Qwen/Qwen2.5-1.5B",
        l0_bits: int = 4,
        l1_num_experts: int = 64,
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
        l0_hidden_dim = 1536

        if use_mock_l0:
            self.l0 = L0PerceptionMock(hidden_size=l0_hidden_dim, vocab_size=vocab_size)
        else:
            self.l0 = L0Perception(model_name=l0_model_name, bits=l0_bits)

        # === L1: Knowledge MoE (NEW!) ===
        self.l1 = L1KnowledgeMoE(
            d_model=l0_hidden_dim,
            num_experts=l1_num_experts,
            top_k=l1_top_k,
            expert_hidden_dim=l1_expert_hidden_dim
        )

        # === L2: CTM Head ===
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

        # Scales control Layer contributions
        self.l1_scale_raw = nn.Parameter(torch.tensor(-2.0))   # ≈ 0.127 (Startet schwach)
        self.ctm_scale_raw = nn.Parameter(torch.tensor(-0.4))  # ≈ 0.344

        self.l1_optimizer = torch.optim.AdamW([
            {'params': self.l1.parameters(), 'lr': 0.005},
            {'params': [self.l1_scale_raw], 'lr': 0.03} # Erlaubt L1, lauter zu werden!
        ])
        logger.info(f"[Karla] Model initialized for Phase 2 (L1 + L2 active)")

    def ctm_scale(self) -> torch.Tensor:
        return F.softplus(self.ctm_scale_raw)
        
    def l1_scale(self) -> torch.Tensor:
        return F.softplus(self.l1_scale_raw)

    def get_multi_loss_config(self) -> dict:
        return self.l2.get_multi_loss_config()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        max_ticks: Optional[int] = None,
        collect_tick_outputs: bool = False,
    ) -> KarlaOutput:
        device = input_ids.device

        # L0
        l0_out = self.l0(input_ids, attention_mask)
        l0_hidden = l0_out.hidden_states.to(device)

        # L1 (Knowledge Extraction)
        l1_features = self.l1(l0_hidden)

        # L2 (CTM "denkt" über L0 + L1 nach)
        combined_input = l0_hidden + self.l1_scale() * l1_features
        l2_out = self.l2(combined_input, attention_mask=attention_mask, collect_tick_outputs=collect_tick_outputs)
        ctm_features = l2_out.features 

        # Final Fusion: Qwen + Wissen + Nachdenken
        features = l0_hidden + self.l1_scale() * l1_features + self.ctm_scale() * ctm_features
        logits = self.l0.decode_head(features)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            if (shift_labels != -100).any():
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.vocab_size), 
                    shift_labels.view(-1), 
                    ignore_index=-100
                )
            else:
                loss = (shift_logits * 0.0).sum()

            # Multi-Loss für CTM
            if collect_tick_outputs and len(l2_out.tick_outputs) > 0 and (shift_labels != -100).any():
                multi_loss_cfg = self.get_multi_loss_config()
                weights = multi_loss_cfg['multi_loss_weights']
                
                multi_loss = torch.tensor(0.0, device=device)
                for i, tick_out in enumerate(l2_out.tick_outputs):
                    # Tick-Features mit L0 + L1 kombinieren
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

        return KarlaOutput(
            logits=logits, internal_ticks=l2_out.internal_ticks,
            certainty=l2_out.certainty, value=l2_out.value, loss=loss
        )
    

    def update_memory(self, input_ids: torch.Tensor, base_lr: float = 0.05):
        """
        Continual Learning Step (Nested Learning: Surprise-Based Plasticity)
        Das Modell ändert seine L1-Gewichte NUR, wenn es von der Information überrascht ist.
        Verhindert Catastrophic Forgetting!
        """
        self.l1.train()
        self.l1_optimizer.zero_grad()
        
        labels = input_ids.clone()
        outputs = self.forward(input_ids, labels=labels)
        loss = outputs.loss
        
        # ------------------------------------------------------------------
        # THE NESTED LEARNING MAGIC: Surprise-Based Plasticity
        # ------------------------------------------------------------------
        # Ein Loss von 0.0 bedeutet: "Ich weiß das schon perfekt."
        # Ein Loss > 3.0 bedeutet: "Das ist komplett neu für mich!"
        
        loss_val = loss.item()
        
        # Wir berechnen den "Surprise Factor" (zwischen 0.0 und 1.0)
        # Wenn der Loss unter 0.3 fällt, wird die Lernrate praktisch 0.
        surprise_factor = min(max((loss_val - 0.3) / 4.0, 0.0), 1.0)
        
        # Die Lernrate skaliert dynamisch mit der Überraschung!
        dynamic_lr = base_lr * surprise_factor
        
        # Wenn wir gar nicht überrascht sind, sparen wir uns den Backward-Pass (schützt altes Wissen!)
        if surprise_factor <= 0.01:
            self.l1.eval()
            return loss_val
            
        # Wende die dynamische Lernrate lokal auf diesen einen Denk-Zyklus an
        for param_group in self.l1_optimizer.param_groups:
            # Scale darf sich immer etwas schneller anpassen als die Gewichte
            if len(param_group['params']) == 1: # Das ist der l1_scale_raw
                param_group['lr'] = dynamic_lr * 5.0 
            else:
                param_group['lr'] = dynamic_lr

        # Update durchführen
        loss.backward()
        
        # Gradient Clipping nur für diesen Step, um Momentum-Explosionen zu verhindern
        torch.nn.utils.clip_grad_norm_(self.l1.parameters(), 1.0)
        
        self.l1_optimizer.step()
        
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
    cfg = {"l0_model_name": config.l0.model_name, "l0_bits": config.l0.bits, "l2_hidden_dim": config.l2.hidden_dim, "l2_num_neurons": config.l2.num_neurons, "l2_num_internal_ticks": config.l2.num_internal_ticks} if config is not None else {}
    cfg.update(kwargs)
    return Karla(use_mock_l0=use_mock, **cfg)