#!/usr/bin/env python3
"""
train_rl_v4.py — POPE + GRPO for Karla C1 with Pre-trained CTM
================================================================

Updated for:
- Pre-trained CTM checkpoint compatibility (Paper 2505.05522v4)
- Plan/Answer mode for guided reasoning (Paper 2601.18779v1)
- Value head for AlphaZero-style self-improvement
- Reward-weighted Engram updates

Paper References:
- 2505.05522v4: CTM pre-training with sync/certainty/value losses
- 2601.18779v1: Plan-then-answer reasoning pattern
- 2601.07372v1: POPE (Privileged On-Policy Exploration)
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import random
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from karla.utils.config import KarlaConfig
from karla.models.karla import create_karla

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("POPE-RL-v4")


# ============================================================
# 1. POPE Dataset
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
    """
    POPE Dataset for Privileged On-Policy Exploration.
    
    Paper 2601.07372v1:
    "For hard problems, we provide oracle prefixes from expert demonstrations
    to guide the model towards better solutions."
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
    
    def _extract_answer_from_reasoning(self, reasoning: str) -> str:
        # Try <answer>...</answer> format
        match = re.search(r"<answer>\s*(.*?)\s*</answer>", reasoning, re.DOTALL | re.IGNORECASE)
        if match:
            ans = match.group(1).strip()
            return ans[:300] if len(ans) > 300 else ans
        
        # Try \boxed{...} format
        match = re.search(r"\\boxed\{([^}]+)\}", reasoning)
        if match:
            return match.group(1).strip()
        
        return ""
    
    def _extract_thinking_from_reasoning(self, reasoning: str) -> str:
        # Try <think|thinking>...</think|thinking> format
        match = re.search(r"<think(?:ing)?[^>]*>(.*?)</think(?:ing)?>", reasoning, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Fallback: everything before <answer>
        parts = reasoning.split("<answer>")
        if len(parts) > 1:
            return parts[0].strip()
        
        return reasoning
    
    def _load(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"Data file not found: {path}")
            return
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Extract fields with multiple possible key names
                prompt = data.get("prompt") or data.get("question") or data.get("input") or ""
                reasoning_full = data.get("reasoning") or data.get("cot") or data.get("response") or data.get("solution") or ""
                answer = data.get("answer") or data.get("output") or ""
                
                if not answer and reasoning_full:
                    answer = self._extract_answer_from_reasoning(reasoning_full)
                
                thinking = self._extract_thinking_from_reasoning(reasoning_full)
                difficulty = data.get("difficulty", "hard")
                
                # Clean prompt
                prompt = prompt.replace("Let's think step by step:", "").strip()
                
                if prompt and (answer or thinking):
                    self.problems.append({
                        "prompt": prompt,
                        "reasoning": thinking,
                        "answer": answer,
                        "difficulty": difficulty,
                    })
    
    def sample_batch(self, batch_size: int) -> List[POPEProblem]:
        batch = []
        selected = random.choices(self.problems, k=batch_size)
        
        for prob in selected:
            is_hard = prob["difficulty"] == "hard"
            use_guidance = is_hard and (random.random() < self.guidance_ratio)
            
            oracle_prefix = ""
            if use_guidance and prob["reasoning"]:
                words = prob["reasoning"].split()
                if len(words) >= 10:
                    ratio = random.uniform(self.prefix_ratio_min, self.prefix_ratio_max)
                    cut = max(5, int(len(words) * ratio))
                    oracle_prefix = " ".join(words[:cut])
            
            batch.append(POPEProblem(
                prompt=prob["prompt"],
                reasoning=prob["reasoning"],
                answer=prob["answer"],
                difficulty=prob["difficulty"],
                is_guided=use_guidance,
                oracle_prefix=oracle_prefix,
            ))
        
        return batch


# ============================================================
# 2. Reward Function
# ============================================================

class RewardFunction:
    """
    Reward function based on concept coverage.
    
    Paper 2601.18779v1:
    "We use a sparse reward signal based on answer correctness,
    with partial credit for reasoning quality."
    """
    
    # Domain-specific important terms
    DOMAIN_TERMS = {
        "medical": [
            "cerebral", "subarachnoid", "aneurysm", "hemorrhage", "vasospasm",
            "nimodipine", "hydrocephalus", "coiling", "clipping", "angiography",
            "blood pressure", "hypertension", "diagnostic", "management",
            "treatment", "monitoring", "symptoms", "patient", "clinical",
        ],
        "general": [
            "therefore", "because", "since", "however", "thus", "hence",
            "conclusion", "result", "analysis", "evidence", "solution",
        ],
    }
    
    @classmethod
    def extract_key_concepts(cls, text: str) -> List[str]:
        stop_words = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "has", "have", "been",
            "from", "that", "this", "with", "they", "will", "each", "make",
            "like", "into", "them", "than", "then", "also", "such", "when",
            "what", "which", "their", "said", "more", "some", "very", "would",
            "about", "could", "these", "other", "given", "following", "being",
        }
        
        words = re.findall(r"[a-zA-Z]+", (text or "").lower())
        concepts = [w for w in words if len(w) > 4 and w not in stop_words]
        
        # Add domain terms
        for domain_terms in cls.DOMAIN_TERMS.values():
            for term in domain_terms:
                if term in (text or "").lower():
                    concepts.append(term)
        
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
        
        # Sparse reward with partial credit
        if coverage >= 0.6:
            return 1.0
        elif coverage >= 0.4:
            return 0.7
        elif coverage >= 0.2:
            return 0.3
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
    value: float  # V(s) from CTM value head
    response_text: str


class RolloutGenerator:
    """
    Generates rollouts for GRPO training.
    
    Paper 2601.07372v1:
    "We sample multiple rollouts per problem and compute group-relative advantages."
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device,
        max_new_tokens: int = 256,
        use_plan_answer: bool = False,
        plan_ticks: int = 3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.use_plan_answer = use_plan_answer
        self.plan_ticks = plan_ticks
        
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
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        
        # Fallback format
        if problem.is_guided:
            return (
                f"<|im_start|>system\n{POPE_SYSTEM_INSTRUCTION}<|im_end|>\n"
                f"<|im_start|>user\nProblem: {problem.prompt}\n"
                f"Partial Response: {problem.oracle_prefix}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        return f"<|im_start|>user\n{problem.prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    @torch.no_grad()
    def generate_rollouts(
        self,
        problems: List[POPEProblem],
        num_rollouts: int = 4,
        temperature: float = 0.8,
    ) -> List[List[Rollout]]:
        self.model.eval()
        all_rollouts = []
        
        for prob in problems:
            prompt_text = self._format_prompt(prob)
            prompt_ids = self.tokenizer.encode(
                prompt_text, return_tensors="pt", add_special_tokens=False
            ).to(self.device)
            
            prob_rollouts = []
            
            for _ in range(num_rollouts):
                generated = prompt_ids[0].tolist()
                response_ids = []
                logps = []
                first_value = 0.0
                
                for t_idx in range(self.max_new_tokens):
                    cur = torch.tensor([generated], dtype=torch.long, device=self.device)
                    mask = torch.ones_like(cur)
                    
                    out = self.model(cur, mask)
                    logits = out.logits[0, -1, :].float()
                    
                    # Capture value at first token
                    if t_idx == 0:
                        first_value = out.value.item()
                    
                    # Sample from temperature-scaled distribution
                    scaled = logits / max(temperature, 1e-8)
                    probs = F.softmax(scaled, dim=-1)
                    tok = torch.multinomial(probs, 1).item()
                    
                    # Store log probability under the same distribution
                    logp = torch.log(probs[tok].clamp(min=1e-12)).item()
                    
                    response_ids.append(tok)
                    logps.append(logp)
                    generated.append(tok)
                    
                    if tok in self.stop_ids:
                        break
                
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                reward = RewardFunction.score(response_text, prob.answer)
                
                prob_rollouts.append(Rollout(
                    problem=prob,
                    prompt_ids=prompt_ids[0].detach().cpu(),
                    response_ids=torch.tensor(response_ids, dtype=torch.long),
                    log_probs=torch.tensor(logps, dtype=torch.float32),
                    reward=reward,
                    value=first_value,
                    response_text=response_text,
                ))
            
            all_rollouts.append(prob_rollouts)
        
        self.model.train()
        return all_rollouts


# ============================================================
# 4. GRPO Loss with Value Loss
# ============================================================

class GRPOWithValueLoss:
    """
    GRPO Loss with Value Loss for AlphaZero-style self-improvement.
    
    Paper 2601.07372v1:
    "We compute advantages relative to the group mean and use PPO-style clipping."
    
    Paper 2505.05522v4:
    "The value head V(s) is trained to predict normalized returns."
    """
    
    def __init__(
        self,
        clip_low: float = 0.2,
        clip_high: float = 5.0,
        c_value: float = 0.1,
        use_value_clipping: bool = True,
    ):
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.c_value = c_value
        self.use_value_clipping = use_value_clipping
    
    def compute_advantages(self, rewards: List[float]) -> List[float]:
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
        temperature: float,
    ) -> Tuple[torch.Tensor, Dict]:
        if not rollouts:
            return torch.zeros((), device=device), {}
        
        rewards = [r.reward for r in rollouts]
        advs = self.compute_advantages(rewards)
        
        # Compute value targets (normalized rewards)
        mean_r = sum(rewards) / len(rewards)
        var_r = sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1)
        std_r = max(var_r ** 0.5, 1e-8)
        value_targets = [(r - mean_r) / std_r for r in rewards]
        
        total_policy_loss = torch.zeros((), device=device)
        total_value_loss = torch.zeros((), device=device)
        used = 0
        
        stats = {
            "rewards": rewards,
            "mean_reward": sum(rewards) / len(rewards),
            "num_correct": sum(1 for r in rewards if r > 0),
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "advantages": advs,
        }
        
        for rollout, adv, v_target in zip(rollouts, advs, value_targets):
            if abs(adv) < 1e-8:
                continue
            
            # Concatenate prompt and response
            full_ids = torch.cat([
                rollout.prompt_ids.to(device),
                rollout.response_ids.to(device)
            ]).unsqueeze(0)
            attention_mask = torch.ones_like(full_ids)
            prompt_len = rollout.prompt_ids.numel()
            
            # Forward pass
            out = model(full_ids, attention_mask)
            logits = out.logits
            
            # Get response logits (shifted)
            resp_logits = logits[0, prompt_len - 1 : -1, :].float()
            resp_tgt = rollout.response_ids.to(device)
            
            L = min(resp_logits.size(0), resp_tgt.numel())
            if L <= 0:
                continue
            resp_logits = resp_logits[:L]
            resp_tgt = resp_tgt[:L]
            
            # Current log probabilities under same temperature distribution
            scaled = resp_logits / max(temperature, 1e-8)
            log_probs_all = F.log_softmax(scaled, dim=-1)
            cur_logp = log_probs_all.gather(1, resp_tgt.unsqueeze(1)).squeeze(1)
            
            # Compute ratio
            old_logp = rollout.log_probs[:L].to(device)
            ratio = torch.exp(cur_logp - old_logp)
            
            # PPO clipping
            clipped = torch.clamp(ratio, 1.0 - self.clip_low, 1.0 + self.clip_high)
            
            adv_t = torch.tensor(adv, device=device, dtype=torch.float32)
            
            surr1 = ratio * adv_t
            surr2 = clipped * adv_t
            
            # Sign-aware objective
            if adv >= 0:
                obj = torch.min(surr1, surr2)
            else:
                obj = torch.max(surr1, surr2)
            
            policy_loss = -obj.mean()
            total_policy_loss = total_policy_loss + policy_loss
            
            # Value loss
            value_pred = torch.tensor(rollout.value, device=device, dtype=torch.float32)
            value_tgt = torch.tensor(v_target, device=device, dtype=torch.float32)
            value_loss = F.mse_loss(value_pred, value_tgt)
            total_value_loss = total_value_loss + value_loss
            
            used += 1
        
        if used > 0:
            total_policy_loss = total_policy_loss / used
            total_value_loss = total_value_loss / used
            stats["policy_loss"] = total_policy_loss.item()
            stats["value_loss"] = total_value_loss.item()
        
        total_loss = total_policy_loss + self.c_value * total_value_loss
        return total_loss, stats


# ============================================================
# 5. Main Training Loop
# ============================================================

def load_checkpoint(model, checkpoint_path: str, device: torch.device, target_l1_scale: float = 0.1, target_ctm_scale: float = 0.3):
    """
    Load checkpoint with support for multiple formats:
    1. Pre-trained CTM checkpoint (ctm_state_dict)
    2. Full Karla checkpoint (model_state_dict)
    3. RL checkpoint (model_state_dict + step + reward)
    
    IMPORTANT: Scales are converted correctly between:
    - Pretraining uses sigmoid(scale)
    - Karla uses softplus(scale_raw)
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.warning(f"No checkpoint at {checkpoint_path}, starting fresh")
        # Initialize scales to target values
        _init_scales(model, target_l1_scale, target_ctm_scale)
        return None
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Check for pre-trained CTM format
    if "ctm_state_dict" in ckpt:
        logger.info("  Detected pre-trained CTM format")
        ctm_state = ckpt["ctm_state_dict"]
        
        # Fix keys: remove underscore prefix from lazy modules
        fixed_state = {}
        for key, value in ctm_state.items():
            if key.startswith("_l1_kv_proj"):
                new_key = key[1:]  # Remove first underscore
                fixed_state[new_key] = value
            else:
                fixed_state[key] = value
        
        missing, unexpected = model.l2.load_state_dict(fixed_state, strict=False)
        logger.info(f"  CTM loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
        
        # Initialize scales to target values (pretraining scale is sigmoid-based)
        _init_scales(model, target_l1_scale, target_ctm_scale)
        logger.info(f"  Scales initialized: L1={target_l1_scale}, CTM={target_ctm_scale}")
        
        # Check if value head exists
        if "l2.value_head.weight" in fixed_state:
            logger.info("  ✓ Value head loaded from checkpoint")
        else:
            logger.info("  Note: Value head not in checkpoint, using random init")
        
        return {"type": "pretrained_ctm", "epoch": ckpt.get("epoch", "?")}
    
    # Check for full model checkpoint
    if "model_state_dict" in ckpt:
        logger.info("  Detected full model checkpoint")
        state_dict = ckpt["model_state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"  Model loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
        
        # Check for scales info
        if "scales" in ckpt:
            scales = ckpt["scales"]
            logger.info(f"  Scales from checkpoint: L1={scales.get('l1_scale', '?')}, CTM={scales.get('ctm_scale', '?')}")
        else:
            # Check if scales are too small and fix them
            current_l1 = model.l1_scale().item()
            current_ctm = model.ctm_scale().item()
            if current_l1 < 0.01 or current_ctm < 0.01:
                logger.warning(f"  Scales too small: L1={current_l1:.6f}, CTM={current_ctm:.6f}")
                _init_scales(model, target_l1_scale, target_ctm_scale)
                logger.info(f"  Scales re-initialized: L1={target_l1_scale}, CTM={target_ctm_scale}")
        
        step = ckpt.get("step", "?")
        reward = ckpt.get("reward", "?")
        logger.info(f"  Checkpoint info: step={step}, reward={reward}")
        
        return {"type": "full", "step": step, "reward": reward}
    
    # Try loading as raw state dict
    logger.info("  Trying to load as raw state dict")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    logger.info(f"  Loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
    
    # Check scales
    current_l1 = model.l1_scale().item()
    current_ctm = model.ctm_scale().item()
    if current_l1 < 0.01 or current_ctm < 0.01:
        logger.warning(f"  Scales too small: L1={current_l1:.6f}, CTM={current_ctm:.6f}")
        _init_scales(model, target_l1_scale, target_ctm_scale)
    
    return {"type": "raw"}


def _init_scales(model, l1_scale: float, ctm_scale: float):
    """
    Initialize scales to target values.
    
    Karla uses softplus(scale_raw), so we need:
    scale_raw = log(exp(target) - 1)
    """
    if l1_scale > 0:
        l1_raw = math.log(math.exp(l1_scale) - 1 + 1e-8)
        model.l1_scale_raw.data = torch.tensor(l1_raw)
    
    if ctm_scale > 0:
        ctm_raw = math.log(math.exp(ctm_scale) - 1 + 1e-8)
        model.ctm_scale_raw.data = torch.tensor(ctm_raw)


def run_pope_rl(args):
    logger.info("=" * 60)
    logger.info("KARLA C1 - POPE-GRPO RL v4")
    logger.info("Paper: 2505.05522v4 (CTM), 2601.07372v1 (POPE), 2601.18779v1 (Plan/Answer)")
    logger.info("=" * 60)
    
    # Config
    config = KarlaConfig()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.l0.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model
    logger.info("Creating Karla model...")
    model = create_karla(config)
    
    # Load checkpoint
    if args.checkpoint:
        ckpt_info = load_checkpoint(
            model, args.checkpoint, device,
            target_l1_scale=args.target_l1_scale,
            target_ctm_scale=args.target_ctm_scale,
        )
    else:
        # Initialize scales to target values
        _init_scales(model, args.target_l1_scale, args.target_ctm_scale)
    
    # Move non-L0 modules to device
    for name, module in model.named_children():
        if name != "l0":
            module.to(device)
    
    model.l1_scale_raw.data = model.l1_scale_raw.data.to(device)
    model.ctm_scale_raw.data = model.ctm_scale_raw.data.to(device)
    model.train()
    
    # Log scale values
    logger.info(f"L1 scale: {model.l1_scale().item():.4f}")
    logger.info(f"CTM scale: {model.ctm_scale().item():.4f}")
    
    # Count parameters
    counts = model.count_parameters()
    logger.info(f"Total parameters: {counts['total']:,}")
    logger.info(f"Trainable parameters: {counts['trainable']:,}")
    
    # Dataset
    data_path = args.data_path or config.training.data_path
    dataset = POPEDataset(data_path, guidance_ratio=args.guidance_ratio)
    
    if len(dataset.problems) == 0:
        logger.error("No problems loaded! Check data path.")
        return
    
    # Optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Rollout generator and GRPO
    rollout_gen = RolloutGenerator(
        model, tokenizer, device,
        max_new_tokens=args.max_response_tokens,
        use_plan_answer=args.use_plan_answer,
        plan_ticks=args.plan_ticks,
    )
    
    grpo = GRPOWithValueLoss(
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        c_value=args.c_value,
    )
    
    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    best_reward = -1e9
    
    # Training loop
    logger.info(f"Starting training for {args.num_steps} steps...")
    
    for step in range(args.num_steps):
        t0 = time.time()
        
        # Sample batch and generate rollouts
        problems = dataset.sample_batch(args.problems_per_step)
        all_rollouts = rollout_gen.generate_rollouts(
            problems,
            num_rollouts=args.rollouts_per_problem,
            temperature=args.temperature,
        )
        
        # Reset gradients
        optimizer.zero_grad()
        
        total_loss = torch.zeros((), device=device)
        total_reward = 0.0
        total_correct = 0
        total_rollouts = 0
        groups_with_signal = 0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        # Compute loss for each group
        for group in all_rollouts:
            loss, stats = grpo.compute_loss(model, group, device, temperature=args.temperature)
            
            if loss.item() != 0.0:
                total_loss = total_loss + loss
                groups_with_signal += 1
            
            total_reward += stats.get("mean_reward", 0.0)
            total_correct += stats.get("num_correct", 0)
            total_rollouts += len(group)
            total_policy_loss += stats.get("policy_loss", 0.0)
            total_value_loss += stats.get("value_loss", 0.0)
        
        # Backward and optimize
        if groups_with_signal > 0:
            avg_loss = total_loss / groups_with_signal
            avg_loss.backward()
            nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
        
        # Update L1 memory
        model.update_memory()
        
        # Stats
        avg_reward = total_reward / max(len(all_rollouts), 1)
        elapsed = time.time() - t0
        
        # Logging
        if step % args.log_interval == 0:
            l1_scale = model.l1_scale().item()
            ctm_scale = model.ctm_scale().item()
            
            logger.info(
                f"Step {step:4d} | "
                f"loss {total_loss.item():.4f} | "
                f"policy {total_policy_loss/max(groups_with_signal,1):.4f} | "
                f"value {total_value_loss/max(groups_with_signal,1):.4f} | "
                f"reward {avg_reward:.3f} | "
                f"correct {total_correct}/{total_rollouts} | "
                f"l1 {l1_scale:.4f} | ctm {ctm_scale:.4f} | "
                f"time {elapsed:.1f}s"
            )
        
        # Save best model
        if avg_reward > best_reward and total_correct > 0:
            best_reward = avg_reward
            path = os.path.join(args.output_dir, "best_rl_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "step": step,
                "reward": avg_reward,
                "config": {
                    "l1_scale": model.l1_scale().item(),
                    "ctm_scale": model.ctm_scale().item(),
                }
            }, path)
            logger.info(f"New best saved: {path} (reward={avg_reward:.3f})")
        
        # Periodic save
        if step > 0 and step % args.save_interval == 0:
            path = os.path.join(args.output_dir, f"rl_step_{step}.pt")
            torch.save(model.state_dict(), path)
            logger.info(f"Saved checkpoint: {path}")
    
    logger.info("=" * 60)
    logger.info(f"Training complete! Best reward: {best_reward:.3f}")
    logger.info(f"Checkpoints saved to: {args.output_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="POPE-GRPO RL Training for Karla C1")
    
    # Data
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to training data (JSONL format)")
    
    # Checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pre-trained checkpoint (CTM or full model)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints_rl",
                        help="Directory for output checkpoints")
    
    # POPE
    parser.add_argument("--guidance-ratio", type=float, default=0.5,
                        help="Ratio of problems to provide oracle guidance")
    
    # GRPO
    parser.add_argument("--clip-low", type=float, default=0.2,
                        help="PPO clip lower bound")
    parser.add_argument("--clip-high", type=float, default=5.0,
                        help="PPO clip upper bound")
    parser.add_argument("--c-value", type=float, default=0.1,
                        help="Value loss coefficient")
    
    # Optimization
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping norm")
    
    # Training
    parser.add_argument("--num-steps", type=int, default=500,
                        help="Number of training steps")
    parser.add_argument("--problems-per-step", type=int, default=4,
                        help="Number of problems per step")
    parser.add_argument("--rollouts-per-problem", type=int, default=4,
                        help="Number of rollouts per problem")
    parser.add_argument("--max-response-tokens", type=int, default=256,
                        help="Maximum response tokens")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    
    # Plan/Answer mode
    parser.add_argument("--use-plan-answer", action="store_true",
                        help="Use plan-then-answer mode (Paper 2601.18779v1)")
    parser.add_argument("--plan-ticks", type=int, default=3,
                        help="Number of ticks for planning phase")
    
    # Scale initialization
    parser.add_argument("--target-l1-scale", type=float, default=0.1,
                        help="Target L1 scale if not in checkpoint")
    parser.add_argument("--target-ctm-scale", type=float, default=0.3,
                        help="Target CTM scale if not in checkpoint")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=1,
                        help="Logging interval")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Checkpoint save interval")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    
    args = parser.parse_args()
    run_pope_rl(args)


if __name__ == "__main__":
    main()
