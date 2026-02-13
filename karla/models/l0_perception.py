"""
L0 Perception Layer - Qwen 2.5-1.5B (Frozen Feature Extractor)
==============================================================
This layer is frozen and serves as the perception backbone.
It extracts semantic hidden states from input tokens.

Key Principle from Nested Learning Paper:
- L0 has update frequency of 0 (frozen weights)
- Acts as a "feature extractor" similar to biological sensory cortex
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class L0Output:
    """Output from L0 perception layer"""
    hidden_states: torch.Tensor  # (batch, seq_len, hidden_dim)
    last_hidden: torch.Tensor    # (batch, hidden_dim)
    attention_mask: Optional[torch.Tensor] = None


class L0Perception(nn.Module):
    """
    L0 Perception Layer - Frozen Qwen 2.5-1.5B Backbone
    
    This module loads Qwen 2.5-1.5B with 4-bit quantization and 
    freezes all parameters. It serves as the "body" that provides
    semantic representations to higher levels.
    
    Memory Usage: ~2GB VRAM with 4-bit quantization
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B",
        bits: int = 4,
        device_map: str = "auto",
        frozen: bool = True,
        hidden_size: int = 1536,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.bits = bits
        self.hidden_size = hidden_size
        self.frozen = frozen
        
        # Will be initialized in _load_model
        self.model = None
        self.tokenizer = None
        
        # Placeholder for model loading
        self._initialized = False
    
    def _load_model(self):
        """Load the model with quantization"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            # Quantization config for 4-bit
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            print(f"[L0] Loading {self.model_name} with {self.bits}-bit quantization...")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            
            # Ensure pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Freeze all parameters
            if self.frozen:
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.eval()
                print("[L0] All parameters frozen ✓")
            
            self._initialized = True
            print(f"[L0] Model loaded. Hidden size: {self.hidden_size}")
            
        except ImportError as e:
            raise ImportError(
                f"Required packages not installed. Run: pip install transformers bitsandbytes accelerate\n{e}"
            )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> L0Output:
        """
        Forward pass through the frozen backbone.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            
        Returns:
            L0Output with hidden states
        """
        if not self._initialized:
            self._load_model()
        
        # Get hidden states from the model
        with torch.no_grad():  # No gradients for frozen model
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Extract last hidden state
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        # Get last token hidden state (for sequence tasks)
        if attention_mask is not None:
            # Find last non-padded position
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden = hidden_states[batch_indices, seq_lengths]
        else:
            last_hidden = hidden_states[:, -1, :]
        
        return L0Output(
            hidden_states=hidden_states,
            last_hidden=last_hidden,
            attention_mask=attention_mask,
        )
    
    def tokenize(
        self,
        text: str,
        max_length: int = 2048,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text input"""
        if not self._initialized:
            self._load_model()
        
        return self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )
    
    def tokenize_batch(
        self,
        texts: list,
        max_length: int = 2048,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts"""
        if not self._initialized:
            self._load_model()
        
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text"""
        if not self._initialized:
            self._load_model()
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        if not self._initialized:
            self._load_model()
        return len(self.tokenizer)


class L0PerceptionMock(nn.Module):
    """
    Mock L0 Layer for testing without loading the full model.
    Useful for quick prototyping and CI/CD.
    """
    
    def __init__(self, hidden_size: int = 1536, vocab_size: int = 151936):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Simple embedding for testing
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
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden = hidden_states[batch_indices, seq_lengths]
        else:
            last_hidden = hidden_states[:, -1, :]
        
        return L0Output(
            hidden_states=hidden_states,
            last_hidden=last_hidden,
            attention_mask=attention_mask,
        )
    
    def tokenize(self, text: str, **kwargs) -> Dict[str, torch.Tensor]:
        # Mock tokenization
        tokens = [hash(c) % self.vocab_size for c in text[:100]]
        return {"input_ids": torch.tensor([tokens]), "attention_mask": torch.tensor([[1] * len(tokens)])}
    
    def tokenize_batch(self, texts: list, **kwargs) -> Dict[str, torch.Tensor]:
        all_ids = []
        all_masks = []
        max_len = 0
        
        for text in texts:
            tokens = [hash(c) % self.vocab_size for c in text[:100]]
            all_ids.append(tokens)
            all_masks.append([1] * len(tokens))
            max_len = max(max_len, len(tokens))
        
        # Pad
        for i in range(len(all_ids)):
            all_ids[i] = all_ids[i] + [0] * (max_len - len(all_ids[i]))
            all_masks[i] = all_masks[i] + [0] * (max_len - len(all_masks[i]))
        
        return {
            "input_ids": torch.tensor(all_ids),
            "attention_mask": torch.tensor(all_masks),
        }
    
    def decode(self, token_ids: torch.Tensor) -> str:
        return "<mock_output>"
    
    def get_vocab_size(self) -> int:
        return self.vocab_size


def create_l0_perception(
    use_mock: bool = False,
    **kwargs
) -> nn.Module:
    """Factory function to create L0 perception layer"""
    if use_mock:
        return L0PerceptionMock(**kwargs)
    return L0Perception(**kwargs)
