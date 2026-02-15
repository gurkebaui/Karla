#!/usr/bin/env python3
"""
Karla C1 - Inference Script
============================
Usage:
    python inference.py --prompt "What is 2+2?"
    python inference.py --interactive
    python inference.py --prompt "Solve: 3x + 7 = 22" --max-tokens 200 --temperature 0.7
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
    """
    Autoregressive inference for Karla C1.
    
    The model predicts ONE token per forward pass.
    We loop: predict → append → predict → append ...
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pt",
        device: str = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info("=" * 60)
        logger.info("KARLA C1 - INFERENCE")
        logger.info("=" * 60)

        # === 1. Config & Tokenizer ===
        self.config = KarlaConfig()

        logger.info("Loading tokenizer...")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.l0.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.eos_token_id = self.tokenizer.eos_token_id
        # Qwen uses <|im_end|> as stop token in chat
        self.stop_token_ids = {self.eos_token_id}
        try:
            im_end_id = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
            if im_end_id:
                self.stop_token_ids.add(im_end_id[0])
        except Exception:
            pass

        logger.info(f"Vocab size: {len(self.tokenizer)}")
        logger.info(f"Stop tokens: {self.stop_token_ids}")

        # === 2. Model ===
        logger.info("Creating model...")
        self.model = create_karla(self.config)

        # Load checkpoint
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                epoch = checkpoint.get("epoch", "?")
                loss = checkpoint.get("loss", "?")
                step = checkpoint.get("global_step", "?")
                logger.info(f"  Epoch: {epoch}, Loss: {loss}, Step: {step}")
            else:
                # Raw state_dict (from epoch saves)
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=False)
            logger.info("  Checkpoint loaded successfully")
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}")
            logger.warning("Running with untrained model!")

        # Move to device
        for name, module in self.model.named_children():
            if name != 'l0':
                module.to(device)
    
    # Move top-level parameters (l1_scale, ctm_scale)
        self.model.l1_scale.data = self.model.l1_scale.data.to(device)
        self.model.ctm_scale.data = self.model.ctm_scale.data.to(device)


        self.model.eval()

        # Print param counts
        counts = self.model.count_parameters()
        logger.info(f"Total parameters: {counts['total']:,}")
        logger.info(f"Model ready on {self.device}")

    def _format_prompt(self, user_input: str) -> str:
        """Format as Qwen chat template."""
        messages = [
            {"role": "user", "content": user_input}
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return text
        except (AttributeError, TypeError):
            return (
                f"<|im_start|>user\n{user_input}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        generated_ids: list = None,
    ) -> int:
        """Sample next token from logits with various strategies."""
        logits = logits.float()

        # Repetition penalty
        if repetition_penalty != 1.0 and generated_ids:
            for token_id in set(generated_ids[-50:]):  # Last 50 tokens
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

        # Temperature
        if temperature <= 0:
            return logits.argmax().item()

        logits = logits / temperature

        # Top-K filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k).values[-1]
            logits[indices_to_remove] = float('-inf')

        # Top-P (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float('-inf')

            # Scatter back
            logits = sorted_logits.scatter(0, sorted_indices, sorted_logits)

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        return next_token

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        print_stream: bool = True,
        show_stats: bool = True,
    ) -> dict:
        """
        Generate text autoregressively.

        Args:
            prompt: User input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_k: Top-K filtering
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalize repeated tokens
            print_stream: Print tokens as they're generated
            show_stats: Show generation statistics

        Returns:
            Dict with generated text, tokens, stats
        """
        # Format prompt
        formatted = self._format_prompt(prompt)

        # Tokenize
        input_ids = self.tokenizer.encode(
            formatted, return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)

        prompt_length = input_ids.shape[1]
        generated_ids = input_ids[0].tolist()
        new_tokens = []

        total_ticks = 0
        total_certainty = 0.0
        start_time = time.time()

        if print_stream:
            print(f"\n{'─' * 60}")
            print(f"Prompt: {prompt}")
            print(f"{'─' * 60}")
            print("Response: ", end="", flush=True)

        for step in range(max_new_tokens):
            # Prepare input (full sequence so far)
            current_ids = torch.tensor(
                [generated_ids], dtype=torch.long, device=self.device
            )
            attention_mask = torch.ones_like(current_ids)

            # Forward pass
            outputs = self.model(
                input_ids=current_ids,
                attention_mask=attention_mask,
            )

            # Get logits for next token
            logits = outputs.logits[0, -1, :] # (vocab_size,)

            # Sample
            next_token = self._sample_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_ids=generated_ids,
            )

            # Track stats
            total_ticks += outputs.internal_ticks
            total_certainty += outputs.certainty[0].item()

            # Stop conditions
            if next_token in self.stop_token_ids:
                break

            # Append
            generated_ids.append(next_token)
            new_tokens.append(next_token)

            # Stream output
            if print_stream:
                token_text = self.tokenizer.decode(
                    [next_token], skip_special_tokens=True
                )
                print(token_text, end="", flush=True)

        elapsed = time.time() - start_time
        num_generated = len(new_tokens)

        # Decode full response
        response_text = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        )

        if print_stream:
            print(f"\n{'─' * 60}")

        # Stats
        stats = {
            "prompt_tokens": prompt_length,
            "generated_tokens": num_generated,
            "total_time": elapsed,
            "tokens_per_second": num_generated / max(elapsed, 0.001),
            "avg_ticks": total_ticks / max(num_generated, 1),
            "avg_certainty": total_certainty / max(num_generated, 1),
        }

        if show_stats:
            print(f"\n📊 Generation Stats:")
            print(f"   Prompt tokens:    {stats['prompt_tokens']}")
            print(f"   Generated tokens: {stats['generated_tokens']}")
            print(f"   Time:             {stats['total_time']:.2f}s")
            print(f"   Speed:            {stats['tokens_per_second']:.1f} tok/s")
            print(f"   Avg CTM ticks:    {stats['avg_ticks']:.1f}")
            print(f"   Avg certainty:    {stats['avg_certainty']:.3f}")
            print()

        return {
            "prompt": prompt,
            "response": response_text,
            "tokens": new_tokens,
            "stats": stats,
        }

    def interactive(self, **kwargs):
        """Interactive chat loop."""
        print("\n" + "=" * 60)
        print("  KARLA C1 - Interactive Mode")
        print("  Type 'quit' or 'exit' to stop")
        print("  Type 'config' to change generation settings")
        print("=" * 60 + "\n")

        gen_config = {
            "max_new_tokens": kwargs.get("max_tokens", 256),
            "temperature": kwargs.get("temperature", 0.7),
            "top_k": kwargs.get("top_k", 50),
            "top_p": kwargs.get("top_p", 0.9),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
        }

        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break

            if user_input.lower() == "config":
                print(f"\nCurrent settings:")
                for k, v in gen_config.items():
                    print(f"  {k}: {v}")
                print("\nChange setting: key=value (e.g. temperature=0.5)")
                try:
                    change = input("  > ").strip()
                    if "=" in change:
                        k, v = change.split("=", 1)
                        k = k.strip()
                        if k in gen_config:
                            gen_config[k] = type(gen_config[k])(v.strip())
                            print(f"  ✓ {k} = {gen_config[k]}")
                except Exception as e:
                    print(f"  Error: {e}")
                continue

            if user_input.lower() == "greedy":
                gen_config["temperature"] = 0
                print("  ✓ Switched to greedy decoding")
                continue

            # Generate
            print()
            self.generate(
                prompt=user_input,
                print_stream=True,
                show_stats=True,
                **gen_config,
            )


def main():
    parser = argparse.ArgumentParser(description="Karla C1 Inference")

    # Mode
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive chat mode")
    parser.add_argument("--prompt", "-p", type=str, default=None,
                        help="Single prompt to generate from")

    # Model
    parser.add_argument("--checkpoint", "-c", type=str,
                        default="checkpoints/best_model.pt",
                        help="Path to model checkpoint")

    # Generation
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-K sampling")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling threshold")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="Repetition penalty")

    # Batch prompts
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="File with one prompt per line")

    args = parser.parse_args()

    # Initialize
    engine = KarlaInference(checkpoint_path=args.checkpoint)

    if args.interactive:
        engine.interactive(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

    elif args.prompt:
        engine.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

    elif args.prompts_file:
        with open(args.prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]

        print(f"\nProcessing {len(prompts)} prompts...\n")
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}]")
            result = engine.generate(
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                print_stream=True,
            )
            results.append(result)

        # Summary
        avg_speed = sum(
            r["stats"]["tokens_per_second"] for r in results
        ) / len(results)
        print(f"\n{'=' * 60}")
        print(f"Average speed: {avg_speed:.1f} tok/s")

    else:
        # Default: run a few test prompts
        test_prompts = [
            "What is 2 + 2?",
            "Explain gravity in one sentence.",
            "If x + 5 = 12, what is x?",
        ]

        print("\n🧪 Running test prompts...\n")
        for prompt in test_prompts:
            engine.generate(
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.7,
            )


if __name__ == "__main__":
    main()