"""
L0 Perception Layer - Qwen 2.5-1.5B
=====================================
Frozen backbone + lm_head.
Provides: feature extraction, final norm, token decoding.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class L0Output:
    hidden_states: torch.Tensor      # (batch, seq_len, 1536) — post-norm
    last_hidden: torch.Tensor        # (batch, 1536)
    attention_mask: Optional[torch.Tensor] = None


class L0Perception(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        bits: int = 4,
        frozen: bool = True,
        hidden_size: int = 1536,
    ):
        super().__init__()
        self.model_name = model_name
        self.bits = bits
        self.hidden_size = hidden_size
        self.frozen = frozen
        self.model = None
        self.tokenizer = None
        self._initialized = False

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"[L0] Loading {self.model_name} (Full CausalLM, 4-bit)...")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.frozen:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            print("[L0] All parameters frozen")

        self._initialized = True
        print(f"[L0] Hidden size: {self.hidden_size}")
        print(f"[L0] LM Head: {self.model.lm_head.in_features} → {self.model.lm_head.out_features}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> L0Output:
        if not self._initialized:
            self._load_model()

        with torch.no_grad():
            # model.model = Qwen2Model (includes final RMSNorm)
            # Output is ALREADY post-norm
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )

        hidden_states = outputs.last_hidden_state.float()

        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1).long() - 1
            seq_lengths = seq_lengths.clamp(min=0)
            batch_idx = torch.arange(
                hidden_states.size(0), device=hidden_states.device
            )
            last_hidden = hidden_states[batch_idx, seq_lengths]
        else:
            last_hidden = hidden_states[:, -1, :]

        return L0Output(
            hidden_states=hidden_states,
            last_hidden=last_hidden,
            attention_mask=attention_mask,
        )

    def decode_head(self, features: torch.Tensor) -> torch.Tensor:
        """
        Qwen's frozen lm_head: features → logits.
        
        Handles both (B, D) and (B, S, D) inputs.
        
        IMPORTANT: lm_head is frozen but gradients flow THROUGH it
        back to features (and thus to L2/L1). This is how L2 learns
        what features to produce.
        """
        if not self._initialized:
            self._load_model()

        head = self.model.lm_head
        device = head.weight.device
        dtype = head.weight.dtype

        features = features.to(device=device, dtype=dtype)

        # No torch.no_grad()! Gradients must flow through the matmul.
        logits = head(features)

        return logits.float()


class L0PerceptionMock(nn.Module):
    """Mock for testing without Qwen"""

    def __init__(self, hidden_size: int = 1536, vocab_size: int = 151936):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self._initialized = True

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embedding(input_ids)
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden = hidden_states[batch_idx, seq_lengths]
        else:
            last_hidden = hidden_states[:, -1, :]
        return L0Output(hidden_states, last_hidden, attention_mask)

    def decode_head(self, features):
        return self.lm_head(features)


def create_l0_perception(use_mock=False, **kwargs):
    if use_mock:
        return L0PerceptionMock(**kwargs)
    return L0Perception(**kwargs)