"""
L0 Perception Layer - Qwen 2.5-1.5B (Frozen Feature Extractor)
==============================================================
Extracts semantic hidden states from frozen Qwen backbone.
Uses AutoModel (no lm_head) to save ~2.3GB VRAM.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class L0Output:
    """Output from L0 perception layer"""
    hidden_states: torch.Tensor      # (batch, seq_len, 1536) - full sequence
    last_hidden: torch.Tensor        # (batch, 1536) - last valid token
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
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"[L0] Loading {self.model_name} (Backbone only, 4-bit)...")

        # AutoModel = only transformer layers, NO lm_head
        self.model = AutoModel.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> L0Output:
        if not self._initialized:
            self._load_model()

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )

        # Full sequence hidden states
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, 1536)

        # Extract last valid token per sequence
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1  # 0-indexed
            batch_idx = torch.arange(
                hidden_states.size(0), device=hidden_states.device
            )
            last_hidden = hidden_states[batch_idx, seq_lengths]
        else:
            last_hidden = hidden_states[:, -1, :]

        # CRITICAL: Cast to float32 for numerical stability
        # Qwen 4-bit outputs float16, which causes NaN in downstream layers
        hidden_states = hidden_states.float()
        last_hidden = last_hidden.float()

        return L0Output(
            hidden_states=hidden_states,
            last_hidden=last_hidden,
            attention_mask=attention_mask,
        )

    def tokenize(self, text: str, max_length: int = 2048) -> Dict:
        if not self._initialized:
            self._load_model()
        return self.tokenizer(
            text,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )


class L0PerceptionMock(nn.Module):
    """Mock for testing without Qwen"""

    def __init__(self, hidden_size: int = 1536, vocab_size: int = 151936):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self._initialized = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> L0Output:
        hidden_states = self.embedding(input_ids)

        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1
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

    def tokenize(self, text: str, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


def create_l0_perception(use_mock: bool = False, **kwargs) -> nn.Module:
    if use_mock:
        return L0PerceptionMock(**kwargs)
    return L0Perception(**kwargs)