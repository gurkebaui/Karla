"""
l1_knowledge.py — Fast GPU-Only Knowledge MoE
=============================================
Volle Geschwindigkeit, PyTorch C++ Backend optimiert.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeExpert(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_model, bias=False)
        self.act = nn.SiLU()
        
    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

class L1KnowledgeMoE(nn.Module):
    def __init__(self, d_model=2048, num_experts=64, top_k=2, expert_hidden_dim=512, noise_std=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # PyTorch liebt ModuleLists, das macht den Backward-Pass rasend schnell
        self.experts = nn.ModuleList([KnowledgeExpert(d_model, expert_hidden_dim) for _ in range(num_experts)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        
        router_logits = self.router(x_flat)
        if self.training and self.noise_std > 0.0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std
            
        routing_probs = F.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        if self.training:
            top_1_experts = selected_experts[:, 0]
            expert_mask = F.one_hot(top_1_experts, num_classes=self.num_experts).float()
            load_balancing_loss = self.num_experts * torch.sum(expert_mask.mean(dim=0) * routing_probs.mean(dim=0))
        else:
            load_balancing_loss = torch.tensor(0.0, device=x.device)

        final_output = torch.zeros_like(x_flat)
        
        # Ultra-schnelle Schleife: Wir fassen Top-1 und Top-2 zusammen!
        for expert_idx in range(self.num_experts):
            mask_k0 = (selected_experts[:, 0] == expert_idx)
            mask_k1 = (selected_experts[:, 1] == expert_idx)
            mask_any = mask_k0 | mask_k1 # Alle Tokens, die DIESEN Experten brauchen
            
            if mask_any.any():
                expert_input = x_flat[mask_any]
                expert_output = self.experts[expert_idx](expert_input)
                
                # Gewichte korrekt addieren
                w0 = torch.where(mask_k0[mask_any], routing_weights[mask_any, 0], 0.0)
                w1 = torch.where(mask_k1[mask_any], routing_weights[mask_any, 1], 0.0)
                total_w = (w0 + w1).unsqueeze(-1)
                
                final_output[mask_any] += expert_output * total_w
                    
        return self.norm(final_output).view(B, S, D), load_balancing_loss