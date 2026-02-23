"""
l1_knowledge.py — Continuum Memory System (Knowledge MoE)
=========================================================
VRAM-Optimierte Token-Gathering Architektur.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeExpert(nn.Module):
    """Ein einzelnes Wissens-Cluster (Assoziativer Speicher)"""
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        # Normale Linear-Layer, brauchen extrem wenig VRAM (1536 x 512)
        self.w1 = nn.Linear(d_model, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))

class L1KnowledgeMoE(nn.Module):
    def __init__(
        self, 
        d_model: int = 1536, 
        num_experts: int = 64, 
        top_k: int = 2, 
        expert_hidden_dim: int = 512
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router: Projiziert Input auf die Experten
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # ModuleList hält alle Experten sauber getrennt im Speicher
        self.experts = nn.ModuleList([
            KnowledgeExpert(d_model, expert_hidden_dim) 
            for _ in range(num_experts)
        ])
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, D) von L0
        """
        B, S, D = x.shape
        x_flat = x.view(-1, D) # (B*S, D)
        
        # 1. Routing
        router_logits = self.router(x_flat) # (B*S, num_experts)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1) # (B*S, top_k)
        
        # 2. VRAM-Effizientes Expert-Routing (Token-Gathering)
        final_output = torch.zeros_like(x_flat)
        
        for k in range(self.top_k):
            expert_indices = selected_experts[:, k] # (B*S,)
            weights = routing_weights[:, k]         # (B*S,)
            
            # Gehe jeden Experten durch
            for expert_idx in range(self.num_experts):
                # Finde alle Tokens, die DIESEN Experten gewählt haben
                token_mask = (expert_indices == expert_idx)
                
                # Nur rechnen, wenn auch wirklich Tokens zu diesem Experten wollen
                if token_mask.any():
                    # Hole die Tokens (z.B. nur 40 Stück von 1024 -> extrem kleine Matrix!)
                    expert_input = x_flat[token_mask]
                    
                    # Rechne durch den Experten
                    expert_output = self.experts[expert_idx](expert_input)
                    
                    # Multipliziere mit Routing-Gewicht und adde zum finalen Output
                    final_output[token_mask] += expert_output * weights[token_mask].unsqueeze(-1)
                    
        final_output = self.norm(final_output)
        return final_output.view(B, S, D)