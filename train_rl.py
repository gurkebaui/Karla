#!/usr/bin/env python3
"""
train_rl.py — POPE (Privileged On-Policy Exploration) + GRPO for Karla C1.

Fixes:
- Old log_probs computed under SAME sampling distribution (temperature)
- PPO/GRPO clipping is SIGN-AWARE (min for A>=0, max for A<0)
- Avoid leaf requires_grad accumulation anti-pattern
- Move correct scale params (l1_scale_raw/ctm_scale_raw)
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from karla.utils.config import KarlaConfig
from karla.models.karla import create_karla

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("POPE-RL")


# ============================================================
# 1. POPE Data Loader
# ============================================================

@dataclass
class POPEProblem:
    prompt: str
    reasoning: str
    answer: str
    difficulty: str
    is_guided: bool = False
    oracle_prefix: str = ""


class POPEDataset:
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

    def _extract_answer_from_reasoning(self, reasoning: str) -> str:
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", reasoning, re.DOTALL | re.IGNORECASE)
        if match:
            ans = match.group(1).strip()
            return ans[:300] if len(ans) > 300 else ans
        return ""

    def _extract_thinking_from_reasoning(self, reasoning: str) -> str:
        match = re.search(r"<think>\s*(.*?)\s*</think>", reasoning, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        parts = reasoning.split("<answer>")
        if len(parts) > 1:
            return parts[0].replace("<think>", "").replace("</think>", "").strip()
        return reasoning

    def _load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            answers_found = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                prompt = data.get("prompt") or data.get("question") or ""
                reasoning_full = data.get("reasoning") or data.get("cot") or data.get("response") or data.get("solution") or ""
                answer = data.get("answer") or ""
                
                # Extract from <answer> tag if field is empty
                if not answer and reasoning_full:
                    answer = self._extract_answer_from_reasoning(reasoning_full)
                
                if answer:
                    answers_found += 1

                thinking = self._extract_thinking_from_reasoning(reasoning_full)
                difficulty = data.get("difficulty", "hard")

                prompt = prompt.replace("Let's think step by step:", "").strip()

                if prompt and (answer or thinking):
                    self.problems.append(
                        {"prompt": prompt, "reasoning": thinking, "answer": answer, "difficulty": difficulty}
                    )
            
            logger.info(f"[POPE] Extracted answers: {answers_found}/{len(self.problems)}")

    def sample_batch(self, batch_size: int) -> List[POPEProblem]:
        batch = []
        selected = random.choices(self.problems, k=batch_size)

        for prob in selected:
            is_hard = prob["difficulty"] == "hard"
            use_guidance = is_hard and (random.random() < self.guidance_ratio)

            oracle_prefix = ""
            if use_guidance:
                oracle_prefix = self._extract_prefix(prob["reasoning"])

            batch.append(
                POPEProblem(
                    prompt=prob["prompt"],
                    reasoning=prob["reasoning"],
                    answer=prob["answer"],
                    difficulty=prob["difficulty"],
                    is_guided=use_guidance,
                    oracle_prefix=oracle_prefix,
                )
            )

        return batch

    def _extract_prefix(self, reasoning: str) -> str:
        words = reasoning.split()
        if len(words) < 10:
            return reasoning
        ratio = random.uniform(self.prefix_ratio_min, self.prefix_ratio_max)
        cut = max(5, int(len(words) * ratio))
        return " ".join(words[:cut])


class RewardFunction:
    @staticmethod
    def extract_key_concepts(text: str) -> List[str]:
        stop_words = {
            "the","and","for","are","but","not","you","all","can","had","her","was","one","our",
            "out","has","have","been","from","that","this","with","they","will","each","make",
            "like","into","them","than","then","also","such","when","what","which","their","said",
            "more","some","very","would","about","could","these","other","given","patient","following",
        }
        words = re.findall(r"[a-zA-Z]+", (text or "").lower())
        concepts = [w for w in words if len(w) > 4 and w not in stop_words]

        phrases = re.findall(
            r"(?:cerebral|subarachnoid|aneurysm|hemorrhage|vasospasm|nimodipine|hydrocephalus|coiling|clipping|angiography|"
            r"blood pressure|hypertension|diagnostic|management|treatment|monitoring)",
            (text or "").lower(),
        )
        concepts.extend(phrases)
        return list(set(concepts))

    @classmethod
    def score(cls, response: str, ground_truth: str) -> float:
        if not ground_truth:
            return 0.0
        gt = cls.extract_key_concepts(ground_truth)
        if not gt:
            return 0.0
        r = (response or "").lower()
        hits = sum(1 for c in gt if c in r)
        coverage = hits / max(len(gt), 1)

        # keep your discrete reward (optional: make continuous later)
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
    problem: POPEProblem
    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    log_probs: torch.Tensor
    reward: float
    response_text: str


class RolloutGenerator:
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
        if problem.is_guided and problem.oracle_prefix:
            messages = [
                {"role": "system", "content": POPE_SYSTEM_INSTRUCTION},
                {"role": "user", "content": (
                    f"Problem: {problem.prompt}\n\n"
                    f"Partial Response: {problem.oracle_prefix}\n\n"
                    f"Continue solving the problem, starting from where the partial response ends."
                )},
            ]
        else:
            messages = [{"role": "user", "content": problem.prompt}]

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if problem.is_guided:
            return (
                f"<|im_start|>system\n{POPE_SYSTEM_INSTRUCTION}<|im_end|>\n"
                f"<|im_start|>user\nProblem: {problem.prompt}\nPartial Response: {problem.oracle_prefix}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        return f"<|im_start|>user\n{problem.prompt}<|im_end|>\n<|im_start|>assistant\n"

    @torch.no_grad()
    def generate_rollouts(self, problems: List[POPEProblem], num_rollouts: int = 4, temperature: float = 0.8):
        self.model.eval()
        all_rollouts = []

        for prob in problems:
            prompt_text = self._format_prompt(prob)
            prompt_ids = self.tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False).to(self.device)
            prob_rollouts = []

            for _ in range(num_rollouts):
                generated = prompt_ids[0].tolist()
                response_ids = []
                logps = []

                for _t in range(self.max_new_tokens):
                    cur = torch.tensor([generated], dtype=torch.long, device=self.device)
                    mask = torch.ones_like(cur)

                    out = self.model(cur, mask)
                    logits = out.logits[0, -1, :].float()

                    # Sample from TEMPERATURED distribution
                    scaled = logits / max(temperature, 1e-8)
                    probs = F.softmax(scaled, dim=-1)
                    tok = torch.multinomial(probs, 1).item()

                    # Store old logprob under SAME distribution as behavior policy
                    logp = torch.log(probs[tok].clamp(min=1e-12)).item()

                    response_ids.append(tok)
                    logps.append(logp)
                    generated.append(tok)

                    if tok in self.stop_ids:
                        break

                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                reward = RewardFunction.score(response_text, prob.answer)

                prob_rollouts.append(
                    Rollout(
                        problem=prob,
                        prompt_ids=prompt_ids[0].detach().cpu(),
                        response_ids=torch.tensor(response_ids, dtype=torch.long),
                        log_probs=torch.tensor(logps, dtype=torch.float32),
                        reward=reward,
                        response_text=response_text,
                    )
                )

            all_rollouts.append(prob_rollouts)

        self.model.train()
        return all_rollouts


# ============================================================
# 4. GRPO Loss (sign-aware PPO clipping)
# ============================================================

class GRPOLoss:
    def __init__(self, clip_low: float = 0.2, clip_high: float = 5.0):
        # IMPORTANT: clip_low should be >0 to allow ratio < 1 for negative advantages
        self.clip_low = clip_low
        self.clip_high = clip_high

    def compute_advantages(self, rewards: List[float]) -> List[float]:
        if not rewards:
            return []
        mean_r = sum(rewards) / len(rewards)
        var_r = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)
        std_r = max(var_r ** 0.5, 1e-8)
        return [(r - mean_r) / std_r for r in rewards]

    def compute_loss(self, model, rollouts: List[Rollout], device: torch.device, temperature: float):
        if not rollouts:
            return torch.zeros((), device=device), {}

        rewards = [r.reward for r in rollouts]
        advs = self.compute_advantages(rewards)

        total_loss = torch.zeros((), device=device)
        used = 0

        stats = {
            "rewards": rewards,
            "advantages": advs,
            "mean_reward": sum(rewards) / len(rewards),
            "num_correct": sum(1 for r in rewards if r > 0),
        }

        for rollout, adv in zip(rollouts, advs):
            if abs(adv) < 1e-8:
                continue

            full_ids = torch.cat([rollout.prompt_ids.to(device), rollout.response_ids.to(device)]).unsqueeze(0)
            attention_mask = torch.ones_like(full_ids)
            prompt_len = rollout.prompt_ids.numel()

            out = model(full_ids, attention_mask)
            logits = out.logits  # (1, seq, vocab)

            # response token logits (shifted)
            resp_logits = logits[0, prompt_len - 1 : -1, :].float()
            resp_tgt = rollout.response_ids.to(device)

            L = min(resp_logits.size(0), resp_tgt.numel())
            if L <= 0:
                continue
            resp_logits = resp_logits[:L]
            resp_tgt = resp_tgt[:L]

            # CURRENT logprobs under SAME temperature distribution
            scaled = resp_logits / max(temperature, 1e-8)
            log_probs_all = F.log_softmax(scaled, dim=-1)
            cur_logp = log_probs_all.gather(1, resp_tgt.unsqueeze(1)).squeeze(1)

            old_logp = rollout.log_probs[:L].to(device)
            ratio = torch.exp(cur_logp - old_logp)

            # PPO clip range
            clipped = torch.clamp(ratio, 1.0 - self.clip_low, 1.0 + self.clip_high)

            adv_t = torch.tensor(adv, device=device, dtype=torch.float32)

            surr1 = ratio * adv_t
            surr2 = clipped * adv_t

            # SIGN-AWARE:
            # For adv>=0 => min(surr1, surr2)
            # For adv<0  => max(surr1, surr2)
            if adv >= 0:
                obj = torch.min(surr1, surr2)
            else:
                obj = torch.max(surr1, surr2)

            loss = -obj.mean()
            total_loss = total_loss + loss
            used += 1

        if used > 0:
            total_loss = total_loss / used

        return total_loss, stats


# ============================================================
# 5. Main loop
# ============================================================

def run_pope_rl(args):
    logger.info("=" * 60)
    logger.info("KARLA C1 - POPE-GRPO RL")
    logger.info("=" * 60)

    config = KarlaConfig()
    device = torch.device(config.training.device)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.l0.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = create_karla(config)

    # optional load
    ckpt_path = args.checkpoint
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        logger.info(f"Loaded checkpoint: {ckpt_path}")

    # move non-L0 modules
    for name, module in model.named_children():
        if name != "l0":
            module.to(device)

    model.l1_scale_raw.data = model.l1_scale_raw.data.to(device)
    model.ctm_scale_raw.data = model.ctm_scale_raw.data.to(device)
    model.train()

    dataset = POPEDataset(args.data_path or config.training.data_path, guidance_ratio=args.guidance_ratio)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    rollout_gen = RolloutGenerator(model, tokenizer, device, max_new_tokens=args.max_response_tokens)
    grpo = GRPOLoss(clip_low=args.clip_low, clip_high=args.clip_high)

    os.makedirs(args.output_dir, exist_ok=True)
    best_reward = -1e9

    for step in range(args.num_steps):
        t0 = time.time()

        problems = dataset.sample_batch(args.problems_per_step)
        all_rollouts = rollout_gen.generate_rollouts(
            problems, num_rollouts=args.rollouts_per_problem, temperature=args.temperature
        )

        optimizer.zero_grad()

        total_loss = torch.zeros((), device=device)
        total_reward = 0.0
        total_correct = 0
        total_rollouts = 0
        groups_with_signal = 0

        for group in all_rollouts:
            loss, stats = grpo.compute_loss(model, group, device, temperature=args.temperature)
            if loss.item() != 0.0:
                total_loss = total_loss + loss
                groups_with_signal += 1

            total_reward += stats.get("mean_reward", 0.0)
            total_correct += stats.get("num_correct", 0)
            total_rollouts += len(group)

        if groups_with_signal > 0:
            avg_loss = total_loss / groups_with_signal
            avg_loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

        model.update_memory()

        avg_reward = total_reward / max(len(all_rollouts), 1)
        elapsed = time.time() - t0

        if step % args.log_interval == 0:
            logger.info(
                f"Step {step:4d} | loss {total_loss.item():.4f} | reward {avg_reward:.3f} | "
                f"correct {total_correct}/{total_rollouts} | "
                f"l1_scale {model.l1_scale().item():.4f} | ctm_scale {model.ctm_scale().item():.4f} | "
                f"time {elapsed:.1f}s"
            )

        if avg_reward > best_reward and total_correct > 0:
            best_reward = avg_reward
            path = os.path.join(args.output_dir, "best_rl_model.pt")
            torch.save({"model_state_dict": model.state_dict(), "step": step, "reward": avg_reward}, path)
            logger.info(f"New best saved: {path} (reward={avg_reward:.3f})")

        if step > 0 and step % args.save_interval == 0:
            path = os.path.join(args.output_dir, f"rl_step_{step}.pt")
            torch.save(model.state_dict(), path)

    logger.info(f"Done. Best reward={best_reward:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default="data/micro_pope_data.jsonl")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="checkpoints_rl")

    p.add_argument("--guidance-ratio", type=float, default=0.5)

    p.add_argument("--clip-low", type=float, default=0.2)
    p.add_argument("--clip-high", type=float, default=5.0)
    p.add_argument("--lr", type=float, default=1e-6)

    p.add_argument("--num-steps", type=int, default=500)
    p.add_argument("--problems-per-step", type=int, default=4)
    p.add_argument("--rollouts-per-problem", type=int, default=4)
    p.add_argument("--max-response-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)

    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--save-interval", type=int, default=50)

    args = p.parse_args()
    run_pope_rl(args)


if __name__ == "__main__":
    main()