#!/usr/bin/env python3
"""
inference_rl.py — Inference for POPE-RL trained Karla
=====================================================
Usage:
    python inference_rl.py --interactive
    python inference_rl.py --prompt "What causes headaches?"
    python inference_rl.py --checkpoint checkpoints_rl/best_rl_model.pt
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from karla.utils.config import KarlaConfig
from karla.models.karla import create_karla

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Inference")


class KarlaInference:
    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info("=" * 60)
        logger.info("KARLA C1 - INFERENCE (RL Checkpoint)")
        logger.info("=" * 60)

        # Tokenizer
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

        # Model
        logger.info("Creating model...")
        self.model = create_karla(self.config)

        # Load checkpoint
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location="cpu")

            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
                step = ckpt.get("step", "?")
                reward = ckpt.get("reward", "?")
                logger.info(f"  RL Step: {step}, Reward: {reward}")
            else:
                state_dict = ckpt

            self.model.load_state_dict(state_dict, strict=False)
            logger.info("  Checkpoint loaded")
        else:
            logger.warning(f"No checkpoint at {checkpoint_path}!")

        # Move to device
        for name, module in self.model.named_children():
            if name != 'l0':
                module.to(self.device)
        self.model.l1_scale.data = self.model.l1_scale.data.to(self.device)
        self.model.ctm_scale.data = self.model.ctm_scale.data.to(self.device)
        self.model.eval()

        # Show scale values (how much L1/CTM contribute)
        logger.info(f"  L1 scale: {self.model.l1_scale.item():.4f}")
        logger.info(f"  CTM scale: {self.model.ctm_scale.item():.4f}")
        logger.info(f"  (0 = pure Qwen, >0 = Karla contributes)")

        counts = self.model.count_parameters()
        logger.info(f"  Total params: {counts['total']:,}")
        logger.info(f"  Ready on {self.device}")

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
        start_time = time.time()

        if print_stream:
            print(f"\n{'─' * 60}")
            print(f"🧑 Prompt: {prompt}")
            print(f"{'─' * 60}")
            print("🤖 Karla: ", end="", flush=True)

        for step in range(max_new_tokens):
            current = torch.tensor(
                [generated], dtype=torch.long, device=self.device
            )
            mask = torch.ones_like(current)

            outputs = self.model(current, mask)

            # Last position logits
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
        }

        print(f"\n📊 Stats:")
        print(f"   Tokens:     {stats['generated_tokens']}")
        print(f"   Time:       {stats['time']:.2f}s")
        print(f"   Speed:      {stats['tok_per_sec']:.1f} tok/s")
        print(f"   CTM Ticks:  {stats['avg_ticks']:.1f}")
        print(f"   Certainty:  {stats['avg_certainty']:.3f}")
        print()

        return {"prompt": prompt, "response": response, "stats": stats}

    def compare_checkpoints(self, prompt: str, paths: list):
        """Compare outputs from different checkpoints side-by-side."""
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

            ckpt = torch.load(path, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state, strict=False)

            result = self.generate(
                prompt,
                max_new_tokens=200,
                temperature=0,  # deterministic
                print_stream=False,
            )
            # Print first 500 chars
            text = result["response"][:500]
            print(f"  {text}")
            print()

    def interactive(self, **kwargs):
        print("\n" + "=" * 60)
        print("  KARLA C1 - Interactive Chat (RL Model)")
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
                        "checkpoints/best_model.pt",       # SFT
                        "checkpoints_rl/best_rl_model.pt", # RL
                    ]
                    self.compare_checkpoints(prompt, paths)
                continue

            self.generate(prompt=user, print_stream=True, **gen_config)


def main():
    parser = argparse.ArgumentParser(description="Karla C1 Inference (RL)")

    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--prompt", "-p", type=str, default=None)
    parser.add_argument("--checkpoint", "-c", type=str,
                        default="checkpoints_rl/best_rl_model.pt")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)

    # Comparison mode
    parser.add_argument("--compare", action="store_true",
                        help="Compare SFT vs RL checkpoints")

    args = parser.parse_args()

    engine = KarlaInference(checkpoint_path=args.checkpoint)

    if args.compare:
        test_prompts = [
            "A patient presents with sudden severe headache and neck stiffness. What is the most likely diagnosis?",
            "Explain the difference between Type 1 and Type 2 diabetes.",
            "What are the main causes of heart failure?",
        ]
        paths = [
            "checkpoints/best_model.pt",
            "checkpoints_rl/best_rl_model.pt",
        ]
        for prompt in test_prompts:
            engine.compare_checkpoints(prompt, paths)

    elif args.interactive:
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
            "Explain how reinforcement learning works in simple terms.",
        ]
        for prompt in test_prompts:
            engine.generate(prompt, max_new_tokens=200, temperature=0.7)


if __name__ == "__main__":
    main()