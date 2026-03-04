"""
L0 Perception Layer - Qwen 2.5 / Cosmos-Reason2
===============================================
Frozen backbone + lm_head.
Provides: feature extraction, final norm, token decoding.
Unterstützt jetzt auch Vision-Language Models (VLM) via AutoModelForVision2Seq!
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from dataclasses import dataclass

@dataclass
class L0Output:
    hidden_states: torch.Tensor      # (batch, seq_len, hidden_dim) — post-norm
    last_hidden: torch.Tensor        # (batch, hidden_dim)
    attention_mask: Optional[torch.Tensor] = None

class L0Perception(nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-2B", # Unser neues Basismodell!
        bits: int = 4,
        frozen: bool = True,
        hidden_size: int = 1536, # Qwen 1.5B hatte 1536. Cosmos 2B hat oft 2048. Passen wir unten dynamisch an!
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
        # WICHTIG: AutoModelForVision2Seq hinzugefügt!
        from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoTokenizer, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        print(f"[L0] Loading {self.model_name} (4-bit)...")

        # SMART LOADING: Probiere erst CausalLM, dann Vision2Seq
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        except ValueError:
            print(f"[L0] CausalLM schlug fehl. Lade als Vision2Seq (VLM) Model...")
            self.model = AutoModelForVision2Seq.from_pretrained(
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

        # Dynamisches Auslesen der Hidden Size (falls Cosmos 2048 nutzt statt 1536)
        if hasattr(self.model.config, "hidden_size"):
            self.hidden_size = self.model.config.hidden_size
        elif hasattr(self.model.config, "text_config"):
            self.hidden_size = self.model.config.text_config.hidden_size

        self._initialized = True
        print(f"[L0] Hidden size: {self.hidden_size}")
        
        # Hole den LM Head (Qwen3-VL packt ihn manchmal an andere Stellen)
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head:
            print(f"[L0] LM Head: {lm_head.in_features} → {lm_head.out_features}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> L0Output:
        if not self._initialized:
            self._load_model()

        with torch.no_grad():
            # =========================================================
            # DYNAMISCHER BACKBONE FINDER (Fix für Qwen3 / Cosmos)
            # Cosmos packt das Sprachmodell unter .language_model
            # Normale Modelle unter .model
            # =========================================================
            if hasattr(self.model, "language_model"):
                base_llm = self.model.language_model
            elif hasattr(self.model, "model"):
                base_llm = self.model.model
            else:
                base_llm = self.model

            # Output is ALREADY post-norm
            outputs = base_llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )

        # Bei einigen Modellen heißt es last_hidden_state, bei anderen gibt es ein Tuple zurück
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state.float()
        else:
            hidden_states = outputs[0].float()

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
        if not self._initialized:
            self._load_model()

        head = self.model.lm_head
        device = head.weight.device
        dtype = head.weight.dtype

        features = features.to(device=device, dtype=dtype)
        logits = head(features)

        return logits.float()

class L0PerceptionMock(nn.Module):
    """Mock for testing without downloading big models"""
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