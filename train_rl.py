#!/usr/bin/env python3
"""
train_karla_pope_rl.py
=======================
POPE (Privileged On-Policy Exploration) + GRPO for Karla C1.

Based on: "POPE: Learning to Reason on Hard Problems via Privileged On-Policy Exploration"
(Qu et al., 2026, CMU)

Key ideas:
1. For hard problems, 50% of rollouts get an oracle prefix (guidance)
2. Model generates completions on-policy
3. Reward: binary correctness (does the answer match?)
4. GRPO: group-relative advantages, clipped policy gradient
5. L1 Engram learns via reward-weighted gradient hooks
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from karla.utils.config import KarlaConfig
from karla.models.karla import create_karla

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("POPE-RL")


# ============================================================
# 1. POPE Data Loader
# ============================================================

@dataclass
class POPEProblem:
    prompt: str
    reasoning: str  # oracle solution
    answer: str     # ground truth final answer
    difficulty: str
    is_guided: bool = False
    oracle_prefix: str = ""


class POPEDataset:
    """
    Loads problems and dynamically creates guided/unguided variants.
    Extracts ground truth from <answer> tags in reasoning field.
    """

    def __init__(
        self,
        data_path: str,
        guidance_ratio: float = 0.5,
        prefix_ratio_min: float = 0.10,
        prefix_ratio_max: float = 0.30,
    ):
        self.guidance_ratio = guidance_ratio
        self.prefix_ratio_min = prefix_ratio_min
        self.prefix_ratio_max = prefix_ratio_max
        self.problems: List[Dict] = []

        self._load(data_path)
        logger.info(f"[POPE] Loaded {len(self.problems)} problems from {data_path}")

        hard = sum(1 for p in self.problems if p.get('difficulty') == 'hard')
        has_answer = sum(1 for p in self.problems if p.get('answer'))
        logger.info(f"[POPE] Hard: {hard}, With answer: {has_answer}")

    def _extract_answer_from_reasoning(self, reasoning: str) -> str:
        """Extract answer from <answer>...</answer> tags in reasoning."""
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', reasoning, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # Clean up: take first sentence or first 200 chars
            # For comparison purposes
            if len(answer) > 300:
                answer = answer[:300]
            return answer
        return ""

    def _extract_thinking_from_reasoning(self, reasoning: str) -> str:
        """Extract just the <think> part for oracle prefix."""
        match = re.search(r'<think>\s*(.*?)\s*</think>', reasoning, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: everything before <answer>
        parts = reasoning.split('<answer>')
        if len(parts) > 1:
            return parts[0].replace('<think>', '').replace('</think>', '').strip()
        return reasoning

    def _load(self, path):
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    prompt = data.get('prompt') or data.get('question') or ''
                    reasoning = (
                        data.get('reasoning') or data.get('cot') or
                        data.get('response') or data.get('solution') or ''
                    )

                    # Extract answer from field OR from <answer> tags
                    answer = data.get('answer') or ''
                    if not answer and reasoning:
                        answer = self._extract_answer_from_reasoning(reasoning)

                    # Extract thinking part (for oracle prefix)
                    thinking = self._extract_thinking_from_reasoning(reasoning)

                    difficulty = data.get('difficulty', 'hard')

                    # Clean prompt: remove "Let's think step by step:"
                    prompt = prompt.replace("Let's think step by step:", "").strip()

                    if prompt and (answer or thinking):
                        self.problems.append({
                            'prompt': prompt,
                            'reasoning': thinking,  # Only the thinking part
                            'answer': answer,
                            'difficulty': difficulty,
                        })

    def sample_batch(self, batch_size: int) -> List[POPEProblem]:
        """Sample a batch with POPE guidance mixture."""
        batch = []
        selected = random.choices(self.problems, k=batch_size)

        for prob in selected:
            is_hard = prob['difficulty'] == 'hard'
            use_guidance = is_hard and random.random() < self.guidance_ratio

            oracle_prefix = ""
            if use_guidance:
                oracle_prefix = self._extract_prefix(prob['reasoning'])

            batch.append(POPEProblem(
                prompt=prob['prompt'],
                reasoning=prob['reasoning'],
                answer=prob['answer'],
                difficulty=prob['difficulty'],
                is_guided=use_guidance,
                oracle_prefix=oracle_prefix,
            ))

        return batch

    def _extract_prefix(self, reasoning: str) -> str:
        """Extract a random-length prefix (10-30%) of the oracle reasoning."""
        words = reasoning.split()
        if len(words) < 10:
            return reasoning

        ratio = random.uniform(self.prefix_ratio_min, self.prefix_ratio_max)
        cut = max(5, int(len(words) * ratio))
        return ' '.join(words[:cut])


class RewardFunction:
    """
    Reward for free-form reasoning tasks (not just math).
    
    Strategy:
    1. Extract key concepts from ground truth answer
    2. Check how many appear in model response
    3. Score based on concept coverage
    
    This is a soft reward — not binary — better suited for
    non-math reasoning where exact match is impossible.
    """

    @staticmethod
    def extract_key_concepts(text: str) -> List[str]:
        """
        Extract key noun phrases and technical terms from answer text.
        Simple approach: words longer than 4 chars, lowercased, deduplicated.
        """
        # Remove common stop words and short words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
            'have', 'been', 'from', 'that', 'this', 'with', 'they',
            'will', 'each', 'make', 'like', 'into', 'them', 'than',
            'then', 'also', 'such', 'when', 'what', 'which', 'their',
            'said', 'more', 'some', 'very', 'would', 'about', 'could',
            'these', 'other', 'given', 'patient', 'following',
        }

        # Extract meaningful words
        words = re.findall(r'[a-zA-Z]+', text.lower())
        concepts = []
        for w in words:
            if len(w) > 4 and w not in stop_words:
                concepts.append(w)

        # Also extract multi-word medical/technical terms
        # Look for capitalized phrases or specific patterns
        phrases = re.findall(
            r'(?:cerebral|subarachnoid|aneurysm|hemorrhage|vasospasm|'
            r'nimodipine|hydrocephalus|coiling|clipping|angiography|'
            r'blood pressure|hypertension|future tense|past tense|'
            r'present tense|sequence|implied|tense|actions|'
            r'diagnostic|management|treatment|monitoring)',
            text.lower()
        )
        concepts.extend(phrases)

        return list(set(concepts))

    @classmethod
    def score(cls, response: str, ground_truth: str) -> float:
        """
        Concept-coverage reward.
        
        Returns:
            1.0 if >= 60% of key concepts from GT appear in response
            0.5 if >= 30% coverage
            0.0 otherwise
            
        This creates a useful gradient signal for GRPO:
        - Some rollouts score 1.0 (good) → positive advantage
        - Some score 0.0 (bad) → negative advantage
        - The model learns to produce responses covering key concepts
        """
        if not ground_truth:
            return 0.0

        gt_concepts = cls.extract_key_concepts(ground_truth)
        if not gt_concepts:
            return 0.0

        response_lower = response.lower()

        # Count how many GT concepts appear in response
        hits = sum(1 for c in gt_concepts if c in response_lower)
        coverage = hits / len(gt_concepts)

        if coverage >= 0.6:
            return 1.0
        elif coverage >= 0.3:
            return 0.5
        else:
            return 0.0


# ============================================================
# 3. Rollout Generation
# ============================================================

POPE_SYSTEM_INSTRUCTION = (
    "You are given a problem and a partial solution. Your task is to carefully study "
    "the partial response, identify what reasoning or steps are already provided, and "
    "then complete the solution from where it left off. Ensure your response is logically "
    "consistent and leads to a complete and correct final answer.\n"
    "Important: Show your reasoning step-by-step, and present the final answer using "
    "\\boxed{your answer here}."
)


@dataclass
class Rollout:
    """One generated completion for a problem."""
    problem: POPEProblem
    prompt_ids: torch.Tensor     # (prompt_len,)
    response_ids: torch.Tensor   # (response_len,)
    log_probs: torch.Tensor      # (response_len,) — per-token log π(a|s)
    reward: float
    response_text: str


class RolloutGenerator:
    """
    Generates on-policy rollouts from Karla.
    
    For guided problems: prompt includes oracle prefix + POPE instruction.
    For unguided: just the problem.
    """

    def __init__(self, model, tokenizer, device, max_new_tokens=256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.eos_id = tokenizer.eos_token_id
        self.stop_ids = {self.eos_id}
        try:
            im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
            if im_end:
                self.stop_ids.add(im_end[0])
        except Exception:
            pass

    def _format_prompt(self, problem: POPEProblem) -> str:
        """Format prompt with optional POPE guidance."""
        if problem.is_guided and problem.oracle_prefix:
            # Guided: include system instruction + oracle prefix
            messages = [
                {"role": "system", "content": POPE_SYSTEM_INSTRUCTION},
                {"role": "user", "content": (
                    f"Problem: {problem.prompt}\n\n"
                    f"Partial Response: {problem.oracle_prefix}\n\n"
                    f"Continue solving the problem, starting from where the "
                    f"partial response ends."
                )},
            ]
        else:
            # Unguided: just the problem
            messages = [
                {"role": "user", "content": problem.prompt},
            ]

        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except (AttributeError, TypeError):
            if problem.is_guided:
                return (
                    f"<|im_start|>system\n{POPE_SYSTEM_INSTRUCTION}<|im_end|>\n"
                    f"<|im_start|>user\nProblem: {problem.prompt}\n"
                    f"Partial Response: {problem.oracle_prefix}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
            return (
                f"<|im_start|>user\n{problem.prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

    @torch.no_grad()
    def generate_rollouts(
        self,
        problems: List[POPEProblem],
        num_rollouts: int = 4,
        temperature: float = 0.8,
    ) -> List[List[Rollout]]:
        """
        Generate G rollouts per problem (on-policy).
        Returns: list of lists — rollouts[i][j] = j-th rollout for i-th problem.
        """
        self.model.eval()
        all_rollouts = []

        for prob in problems:
            prompt_text = self._format_prompt(prob)
            prompt_ids = self.tokenizer.encode(
                prompt_text, return_tensors="pt", add_special_tokens=False
            ).to(self.device)

            prompt_len = prompt_ids.shape[1]
            prob_rollouts = []

            for _ in range(num_rollouts):
                generated_ids = prompt_ids[0].tolist()
                response_ids = []
                log_probs = []

                for step in range(self.max_new_tokens):
                    input_ids = torch.tensor(
                        [generated_ids], dtype=torch.long, device=self.device
                    )
                    attention_mask = torch.ones_like(input_ids)

                    outputs = self.model(input_ids, attention_mask)
                    # Get logits for last position
                    next_logits = outputs.logits[0, -1, :].float()

                    # Sample with temperature
                    scaled = next_logits / max(temperature, 1e-8)
                    probs = F.softmax(scaled, dim=-1)
                    token = torch.multinomial(probs, 1).item()

                    # Log probability under current policy
                    log_p = F.log_softmax(next_logits, dim=-1)[token].item()

                    response_ids.append(token)
                    log_probs.append(log_p)
                    generated_ids.append(token)

                    if token in self.stop_ids:
                        break

                # Decode response
                response_text = self.tokenizer.decode(
                    response_ids, skip_special_tokens=True
                )

                # Score
                reward = RewardFunction.score(response_text, prob.answer)

                prob_rollouts.append(Rollout(
                    problem=prob,
                    prompt_ids=prompt_ids[0],
                    response_ids=torch.tensor(response_ids, dtype=torch.long),
                    log_probs=torch.tensor(log_probs, dtype=torch.float32),
                    reward=reward,
                    response_text=response_text,
                ))

            all_rollouts.append(prob_rollouts)

        self.model.train()
        return all_rollouts


# ============================================================
# 4. GRPO Loss (Section 2, Eq. 2 of POPE paper)
# ============================================================

class GRPOLoss:
    """
    Group Relative Policy Optimization.
    
    For each problem, we have G rollouts with rewards r_1, ..., r_G.
    Advantage A_i = r_i - mean(r) / max(std(r), eps)
    
    Loss = -E[ min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) ]
    
    Following POPE paper: ε_low = 0.0, ε_high = 5.0 (asymmetric clipping)
    """

    def __init__(
        self,
        clip_low: float = 0.0,     # POPE/DAPO style: no lower clip
        clip_high: float = 5.0,    # generous upper clip for rare positives
        kl_coeff: float = 0.0,     # no KL penalty (following POPE paper)
    ):
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.kl_coeff = kl_coeff

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        """
        Group-relative advantage normalization.
        A_i = (r_i - mean) / max(std, eps)
        
        If all rewards are 0, advantages are all 0 → no gradient (by design).
        """
        if not rewards:
            return []

        mean_r = sum(rewards) / len(rewards)
        var_r = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)
        std_r = max(var_r ** 0.5, 1e-8)

        return [(r - mean_r) / std_r for r in rewards]

    def compute_loss(
        self,
        model,
        rollouts: List[Rollout],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute GRPO loss for a group of rollouts (same problem).
        
        Returns:
            loss: scalar tensor
            stats: dict with metrics
        """
        if not rollouts:
            return torch.tensor(0.0, device=device), {}

        rewards = [r.reward for r in rollouts]
        advantages = self.compute_advantages(rewards)

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_tokens = 0
        stats = {
            'rewards': rewards,
            'advantages': advantages,
            'mean_reward': sum(rewards) / len(rewards),
            'num_correct': sum(1 for r in rewards if r > 0),
        }

        for rollout, advantage in zip(rollouts, advantages):
            if abs(advantage) < 1e-8:
                continue  # Zero advantage → no gradient (GRPO design)

            # Recompute log probs under current policy
            full_ids = torch.cat([
                rollout.prompt_ids,
                rollout.response_ids.to(device),
            ]).unsqueeze(0).to(device)

            attention_mask = torch.ones_like(full_ids)
            prompt_len = len(rollout.prompt_ids)

            outputs = model(full_ids, attention_mask)
            # logits: (1, seq, vocab)
            logits = outputs.logits

            # Get log probs for response tokens
            response_logits = logits[0, prompt_len - 1:-1, :]  # shifted by 1
            response_targets = rollout.response_ids.to(device)

            # Truncate to match
            min_len = min(len(response_logits), len(response_targets))
            if min_len == 0:
                continue

            response_logits = response_logits[:min_len]
            response_targets = response_targets[:min_len]

            current_log_probs = F.log_softmax(response_logits.float(), dim=-1)
            current_log_probs = current_log_probs.gather(
                1, response_targets.unsqueeze(1)
            ).squeeze(1)

            # Old log probs (from rollout generation)
            old_log_probs = rollout.log_probs[:min_len].to(device)

            # Importance ratio
            ratio = torch.exp(current_log_probs - old_log_probs)

            # Clipped ratio (asymmetric, POPE style)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - self.clip_low, 1.0 + self.clip_high
            )

            adv_tensor = torch.tensor(advantage, device=device, dtype=torch.float32)

            # Policy loss (minimize negative of objective)
            surr1 = ratio * adv_tensor
            surr2 = clipped_ratio * adv_tensor
            token_loss = -torch.min(surr1, surr2)

            total_loss = total_loss + token_loss.mean()
            num_tokens += min_len

        if num_tokens > 0:
            total_loss = total_loss / len([a for a in advantages if abs(a) > 1e-8])

        return total_loss, stats


# ============================================================
# 5. Main Training Loop
# ============================================================

def run_pope_rl(args):
    logger.info("=" * 60)
    logger.info("KARLA C1 - POPE-GRPO REINFORCEMENT LEARNING")
    logger.info("=" * 60)

    config = KarlaConfig()
    device = torch.device(config.training.device)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu} ({mem:.1f} GB)")

    # === Tokenizer ===
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.l0.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === Model ===
    logger.info("Loading model from SFT checkpoint...")
    model = create_karla(config)

    # Load SFT checkpoint
    ckpt_path = args.checkpoint or "checkpoints/best_model.pt"
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded SFT checkpoint: {ckpt_path}")
    else:
        logger.warning(f"No checkpoint at {ckpt_path}, using untrained model!")

    # Move to device
    for name, module in model.named_children():
        if name != 'l0':
            module.to(device)
    model.l1_scale.data = model.l1_scale.data.to(device)
    model.ctm_scale.data = model.ctm_scale.data.to(device)

    # === Dataset ===
    data_path = args.data_path or config.training.data_path
    dataset = POPEDataset(
        data_path,
        guidance_ratio=args.guidance_ratio,
        prefix_ratio_min=0.10,
        prefix_ratio_max=0.30,
    )

    # === Optimizer (only trainable params) ===
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # === Components ===
    rollout_gen = RolloutGenerator(
        model, tokenizer, device,
        max_new_tokens=args.max_response_tokens,
    )
    grpo = GRPOLoss(
        clip_low=args.clip_low,
        clip_high=args.clip_high,
    )

    # === Training Loop ===
    os.makedirs(args.output_dir, exist_ok=True)
    best_reward = 0.0

    logger.info(f"RL Steps: {args.num_steps}")
    logger.info(f"Problems per step: {args.problems_per_step}")
    logger.info(f"Rollouts per problem: {args.rollouts_per_problem}")
    logger.info(f"Guidance ratio: {args.guidance_ratio}")
    logger.info(f"Max response tokens: {args.max_response_tokens}")
    logger.info(f"Clip: low={args.clip_low}, high={args.clip_high}")

    for step in range(args.num_steps):
        step_start = time.time()

        # 1. Sample problems (POPE mixture)
        problems = dataset.sample_batch(args.problems_per_step)
        num_guided = sum(1 for p in problems if p.is_guided)

        # 2. Generate rollouts (on-policy)
        all_rollouts = rollout_gen.generate_rollouts(
            problems,
            num_rollouts=args.rollouts_per_problem,
            temperature=args.temperature,
        )

        # 3. Compute GRPO loss and update
        model.train()
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=device)
        total_reward = 0.0
        total_correct = 0
        total_rollouts = 0
        num_groups_with_signal = 0

        for group in all_rollouts:
            loss, stats = grpo.compute_loss(model, group, device)

            if loss.requires_grad and loss.item() != 0.0:
                total_loss = total_loss + loss
                num_groups_with_signal += 1

            total_reward += stats.get('mean_reward', 0.0)
            total_correct += stats.get('num_correct', 0)
            total_rollouts += len(group)

        # Average over groups with signal
        if num_groups_with_signal > 0:
            avg_loss = total_loss / num_groups_with_signal
            avg_loss.backward()

            # Clip gradients
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

        # 4. Update L1 engram memory
        model.update_memory()

        # 5. Logging
        avg_reward = total_reward / max(len(all_rollouts), 1)
        elapsed = time.time() - step_start

        if step % args.log_interval == 0:
            logger.info(
                f"Step {step:4d} | "
                f"Loss: {total_loss.item():.4f} | "
                f"Reward: {avg_reward:.3f} | "
                f"Correct: {total_correct}/{total_rollouts} | "
                f"Guided: {num_guided}/{len(problems)} | "
                f"Signal: {num_groups_with_signal}/{len(all_rollouts)} | "
                f"Time: {elapsed:.1f}s"
            )

        # 6. Save checkpoints
        if avg_reward > best_reward and total_correct > 0:
            best_reward = avg_reward
            path = os.path.join(args.output_dir, "best_rl_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'step': step,
                'reward': avg_reward,
            }, path)
            logger.info(f"New best model saved: reward={avg_reward:.3f}")

        if step > 0 and step % args.save_interval == 0:
            path = os.path.join(args.output_dir, f"rl_step_{step}.pt")
            torch.save(model.state_dict(), path)

    logger.info("=" * 60)
    logger.info(f"POPE-RL Training Complete!")
    logger.info(f"Best reward: {best_reward:.3f}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Karla POPE-GRPO RL Training")

    # Data
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="checkpoints_rl")

    # POPE
    parser.add_argument("--guidance-ratio", type=float, default=0.5)

    # GRPO
    parser.add_argument("--clip-low", type=float, default=0.0)
    parser.add_argument("--clip-high", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=1e-6)

    # Rollouts
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--problems-per-step", type=int, default=4)
    parser.add_argument("--rollouts-per-problem", type=int, default=4)
    parser.add_argument("--max-response-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)

    # Logging
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=50)

    args = parser.parse_args()
    run_pope_rl(args)


if __name__ == "__main__":
    main()