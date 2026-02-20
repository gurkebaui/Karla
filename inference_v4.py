#!/usr/bin/env python3
"""
inference_v4.py — Inference for Karla C1 with Pre-trained CTM
==============================================================

Updated for:
- Pre-trained CTM checkpoint compatibility
- Value head output display
- Plan/Answer mode support
- Better error handling

Usage:
    python inference_v4.py --interactive
    python inference_v4.py --prompt "What causes headaches?"
    python inference_v4.py --checkpoint checkpoints_ctm_pretrain/best_ctm.pt
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import logging
import time
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from karla.utils.config import KarlaConfig
from karla.models.karla import create_karla

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Inference-v4")


def load_checkpoint(model, checkpoint_path: str, device: torch.device):
    """
    Load checkpoint with support for multiple formats.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning(f"No checkpoint at {checkpoint_path}")
        return None
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Pre-trained CTM format
    if "ctm_state_dict" in ckpt:
        logger.info("  Detected pre-trained CTM format")
        ctm_state = ckpt["ctm_state_dict"]
        
        # Fix keys
        fixed_state = {}
        for key, value in ctm_state.items():
            if key.startswith("_l1_kv_proj"):
                new_key = key[1:]
                fixed_state[new_key] = value
            else:
                fixed_state[key] = value
        
        missing, unexpected = model.l2.load_state_dict(fixed_state, strict=False)
        logger.info(f"  CTM loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
        
        # Set CTM scale
        if "ctm_scale" in ckpt:
            pretrained_scale = ckpt["ctm_scale"]
            if pretrained_scale > 0:
                scale_raw = math.log(math.exp(pretrained_scale) - 1 + 1e-8)
                model.ctm_scale_raw.data = torch.tensor(scale_raw)
                scale_val = torch.sigmoid(torch.tensor(pretrained_scale)).item()
                logger.info(f"  CTM scale: {scale_val:.4f}")
        
        epoch = ckpt.get("epoch", "?")
        loss = ckpt.get("loss", "?")
        logger.info(f"  Pre-training: epoch={epoch}, loss={loss}")
        
        return {"type": "pretrained_ctm"}
    
    # Full model checkpoint
    if "model_state_dict" in ckpt:
        logger.info("  Detected full model checkpoint")
        state_dict = ckpt["model_state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"  Model loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
        
        step = ckpt.get("step", "?")
        reward = ckpt.get("reward", "?")
        logger.info(f"  Checkpoint info: step={step}, reward={reward}")
        
        if "config" in ckpt:
            cfg = ckpt["config"]
            logger.info(f"  Config: l1_scale={cfg.get('l1_scale', '?')}, ctm_scale={cfg.get('ctm_scale', '?')}")
        
        return {"type": "full", "step": step, "reward": reward}
    
    # Raw state dict
    logger.info("  Trying to load as raw state dict")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    logger.info(f"  Loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
    
    return {"type": "raw"}


class KarlaInference:
    """Inference engine for Karla C1."""
    
    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        logger.info("=" * 60)
        logger.info("KARLA C1 - INFERENCE v4")
        logger.info("=" * 60)
        
        # Config and tokenizer
        self.config = KarlaConfig()
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.l0.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Stop tokens
        self.stop_ids = {self.tokenizer.eos_token_id}
        try:
            im_end = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
            if im_end:
                self.stop_ids.add(im_end[0])
        except Exception:
            pass
        
        # Create model
        logger.info("Creating model...")
        self.model = create_karla(self.config)
        
        # Load checkpoint
        if checkpoint_path:
            load_checkpoint(self.model, checkpoint_path, self.device)
        
        # Move to device
        for name, module in self.model.named_children():
            if name != 'l0':
                module.to(self.device)
        
        self.model.l1_scale_raw.data = self.model.l1_scale_raw.data.to(self.device)
        self.model.ctm_scale_raw.data = self.model.ctm_scale_raw.data.to(self.device)
        self.model.eval()
        
        # Log info
        l1_scale = self.model.l1_scale().item()
        ctm_scale = self.model.ctm_scale().item()
        logger.info(f"L1 scale: {l1_scale:.4f}")
        logger.info(f"CTM scale: {ctm_scale:.4f}")
        
        counts = self.model.count_parameters()
        logger.info(f"Total params: {counts['total']:,}")
        logger.info(f"Trainable: {counts['trainable']:,}")
        logger.info(f"Device: {self.device}")
    
    def _format_prompt(self, user_input: str) -> str:
        messages = [{"role": "user", "content": user_input}]
        
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (AttributeError, TypeError):
            return (
                f"<|im_start|>user\n{user_input}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        print_stream: bool = True,
    ) -> dict:
        formatted = self._format_prompt(prompt)
        
        input_ids = self.tokenizer.encode(
            formatted, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        
        prompt_length = input_ids.shape[1]
        generated = input_ids[0].tolist()
        new_tokens = []
        
        total_ticks = 0
        total_certainty = 0.0
        total_value = 0.0
        start_time = time.time()
        
        if print_stream:
            print(f"\n{'─' * 60}")
            print(f"🧑 Prompt: {prompt}")
            print(f"{'─' * 60}")
            print("🤖 Karla: ", end="", flush=True)
        
        for step in range(max_new_tokens):
            current = torch.tensor([generated], dtype=torch.long, device=self.device)
            mask = torch.ones_like(current)
            
            outputs = self.model(current, mask)
            logits = outputs.logits[0, -1, :].float()
            
            # Repetition penalty
            if repetition_penalty != 1.0 and new_tokens:
                for tid in set(new_tokens[-50:]):
                    if logits[tid] > 0:
                        logits[tid] /= repetition_penalty
                    else:
                        logits[tid] *= repetition_penalty
            
            # Temperature
            if temperature <= 0:
                next_token = logits.argmax().item()
            else:
                logits = logits / temperature
                
                # Top-K
                if top_k > 0:
                    tk = min(top_k, logits.size(-1))
                    threshold = torch.topk(logits, tk).values[-1]
                    logits[logits < threshold] = float('-inf')
                
                # Top-P
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[remove] = float('-inf')
                    logits = sorted_logits.scatter(0, sorted_idx, sorted_logits)
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            
            # Stats
            total_ticks += outputs.internal_ticks
            total_certainty += outputs.certainty[0].item()
            total_value += outputs.value[0].item()
            
            # Stop
            if next_token in self.stop_ids:
                break
            
            generated.append(next_token)
            new_tokens.append(next_token)
            
            if print_stream:
                text = self.tokenizer.decode([next_token], skip_special_tokens=True)
                print(text, end="", flush=True)
        
        elapsed = time.time() - start_time
        n_gen = len(new_tokens)
        
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        if print_stream:
            print(f"\n{'─' * 60}")
        
        stats = {
            "prompt_tokens": prompt_length,
            "generated_tokens": n_gen,
            "time": elapsed,
            "tok_per_sec": n_gen / max(elapsed, 0.001),
            "avg_ticks": total_ticks / max(n_gen, 1),
            "avg_certainty": total_certainty / max(n_gen, 1),
            "avg_value": total_value / max(n_gen, 1),
        }
        
        print(f"\n📊 Stats:")
        print(f"   Tokens:     {stats['generated_tokens']}")
        print(f"   Time:       {stats['time']:.2f}s")
        print(f"   Speed:      {stats['tok_per_sec']:.1f} tok/s")
        print(f"   CTM Ticks:  {stats['avg_ticks']:.1f}")
        print(f"   Certainty:  {stats['avg_certainty']:.3f}")
        print(f"   Value:      {stats['avg_value']:.3f}")
        print()
        
        return {"prompt": prompt, "response": response, "stats": stats}
    
    def compare_checkpoints(self, prompt: str, paths: list):
        """Compare outputs from different checkpoints."""
        print(f"\n{'═' * 60}")
        print(f"  CHECKPOINT COMPARISON")
        print(f"  Prompt: {prompt}")
        print(f"{'═' * 60}\n")
        
        for path in paths:
            label = os.path.basename(path)
            print(f"▸ {label}")
            print(f"  {'─' * 50}")
            
            if not os.path.exists(path):
                print(f"  ⚠ Not found!\n")
                continue
            
            load_checkpoint(self.model, path, self.device)
            
            result = self.generate(
                prompt,
                max_new_tokens=200,
                temperature=0,  # deterministic
                print_stream=False,
            )
            
            text = result["response"][:500]
            print(f"  {text}")
            print()
    
    def interactive(self, **kwargs):
        print("\n" + "=" * 60)
        print("  KARLA C1 - Interactive Chat v4")
        print("  Commands: 'quit', 'greedy', 'creative', 'compare'")
        print("=" * 60 + "\n")
        
        gen_config = {
            "max_new_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_k": kwargs.get("top_k", 50),
            "top_p": kwargs.get("top_p", 0.9),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
        }
        
        while True:
            try:
                user = input("\n🧑 You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break
            
            if not user:
                continue
            
            if user.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            
            if user.lower() == "greedy":
                gen_config["temperature"] = 0
                print("  ✓ Greedy mode (temperature=0)")
                continue
            
            if user.lower() == "creative":
                gen_config["temperature"] = 1.0
                print("  ✓ Creative mode (temperature=1.0)")
                continue
            
            if user.lower() == "compare":
                prompt = input("  Prompt to compare: ").strip()
                if prompt:
                    paths = [
                        "checkpoints_ctm_pretrain/best_ctm.pt",
                        "checkpoints_rl/best_rl_model.pt",
                    ]
                    self.compare_checkpoints(prompt, paths)
                continue
            
            self.generate(prompt=user, print_stream=True, **gen_config)


def main():
    parser = argparse.ArgumentParser(description="Karla C1 Inference v4")
    
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--prompt", "-p", type=str, default=None)
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    
    args = parser.parse_args()
    
    engine = KarlaInference(checkpoint_path=args.checkpoint)
    
    if args.interactive:
        engine.interactive(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    
    elif args.prompt:
        engine.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    
    else:
        # Default test
        test_prompts = [
            "What is subarachnoid hemorrhage and how is it treated?",
            "A 45-year-old male presents with chest pain radiating to the left arm. What should be done?",
            "Explain how neural networks learn.",
        ]
        for prompt in test_prompts:
            engine.generate(prompt, max_new_tokens=200, temperature=0.7)


if __name__ == "__main__":
    main()
