import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional
import random
import os
import logging
import re

logger = logging.getLogger("Dataset")

_STEP_BY_STEP_PAT = re.compile(r"let'?s think step by step:?\s*$", re.IGNORECASE)


def _strip_step_by_step(prompt: str) -> str:
    prompt = (prompt or "").strip()
    prompt = _STEP_BY_STEP_PAT.sub("", prompt).strip()
    return prompt


def _extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


class ReasoningDataset(Dataset):
    """
    Dataset für Reasoning-Training mit POPE-Unterstützung.

    Fixes:
    - Exaktes Maskieren bis zum Assistant-Start (kein len//3 Heuristik-Bug)
    - Entfernt "Let's think step by step:" aus Prompts
    - Extrahiert Answer aus <answer> wenn answer-Feld leer
    - Sauberes Padding/Truncation Handling
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[Any] = None,
        max_length: int = 2048,
        pope_prefix_ratio: float = 0.3,
        pope_mixed_ratio: float = 0.5,
        use_pope: bool = True,
        mask_user_prompt: bool = True,
        keep_think_answer_tags: bool = True,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pope_prefix_ratio = pope_prefix_ratio
        self.pope_mixed_ratio = pope_mixed_ratio
        self.use_pope = use_pope
        self.mask_user_prompt = mask_user_prompt
        self.keep_think_answer_tags = keep_think_answer_tags

        self.examples: List[Dict[str, Any]] = []
        self._load_data()
        self._print_stats()

    def _load_data(self):
        if not os.path.exists(self.data_path):
            logger.warning(f"⚠️ Datei {self.data_path} fehlt → Fallback Daten.")
            self._create_fallback_data()
            return

        logger.info(f"📂 Lade {self.data_path}...")
        ext = os.path.splitext(self.data_path)[1].lower()
        if ext == ".json":
            self._load_json()
        else:
            self._load_jsonl()
        logger.info(f"   {len(self.examples)} Beispiele geladen")

    def _load_jsonl(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self._add_example(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"   Zeile {line_num} übersprungen: {e}")

    def _load_json(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                self._add_example(item)
        else:
            self._add_example(data)

    def _add_example(self, data: Dict[str, Any]):
        prompt = (
            data.get("prompt")
            or data.get("question")
            or data.get("instruction")
            or data.get("input")
            or data.get("problem")
            or ""
        )
        reasoning = (
            data.get("reasoning")
            or data.get("cot")
            or data.get("chain_of_thought")
            or data.get("response")
            or data.get("solution")
            or data.get("output")
            or data.get("thinking")
            or ""
        )
        answer = data.get("answer") or data.get("final_answer") or ""

        prompt = _strip_step_by_step(str(prompt))
        reasoning = str(reasoning).strip()
        answer = str(answer).strip()

        if not prompt or not reasoning:
            return

        # answer aus <answer> extrahieren falls Feld leer
        if not answer:
            extracted = _extract_tag(reasoning, "answer")
            if extracted:
                answer = extracted

        difficulty = data.get("difficulty", "medium")
        if not difficulty or difficulty == "unknown":
            reasoning_len = len(reasoning.split())
            if reasoning_len < 30:
                difficulty = "easy"
            elif reasoning_len < 100:
                difficulty = "medium"
            else:
                difficulty = "hard"

        self.examples.append(
            {
                "prompt": prompt,
                "reasoning": reasoning,
                "answer": answer,
                "difficulty": difficulty,
            }
        )

    def _create_fallback_data(self):
        fallback_examples = [
            {
                "prompt": "What is 2 + 2?",
                "reasoning": "<think>Add 2 and 2.</think>\n<answer>4</answer>",
                "answer": "4",
                "difficulty": "easy",
            },
            {
                "prompt": "If x + 5 = 12, what is x?",
                "reasoning": "<think>Subtract 5 from both sides.</think>\n<answer>7</answer>",
                "answer": "7",
                "difficulty": "medium",
            },
        ]
        for _ in range(100):
            for ex in fallback_examples:
                self.examples.append(ex.copy())
        logger.info(f"   {len(self.examples)} Fallback-Beispiele erstellt")

    def _print_stats(self):
        if not self.examples:
            return
        difficulty_counts = {}
        for ex in self.examples:
            d = ex.get("difficulty", "unknown")
            difficulty_counts[d] = difficulty_counts.get(d, 0) + 1

        logger.info("📊 Dataset Statistiken:")
        for d, c in sorted(difficulty_counts.items()):
            logger.info(f"   {d}: {c}")

    def __len__(self) -> int:
        return len(self.examples)

    def _create_oracle_prefix(self, assistant_text: str) -> str:
        words = assistant_text.split()
        if len(words) <= 6:
            return assistant_text
        cut = max(5, int(len(words) * self.pope_prefix_ratio))
        cut = min(cut, len(words) - 1)
        return " ".join(words[:cut])

    def _apply_chat_template_ids(self, messages, add_generation_prompt: bool) -> torch.Tensor:
        """
        Returns 1D token ids.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            out = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_tensors="pt",
            )
            return out[0]

        # Fallback manual
        parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        text = "".join(parts)
        enc = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        return enc["input_ids"][0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        prompt = example["prompt"]
        reasoning = example["reasoning"]
        answer = example.get("answer", "")
        difficulty = example.get("difficulty", "medium")

        assistant_content = reasoning

        # wenn reasoning kein <answer> enthält, aber answer vorhanden: anfügen
        if answer and ("<answer>" not in reasoning.lower()):
            assistant_content = f"{reasoning}\n\n<answer>\n{answer}\n</answer>"

        if not self.keep_think_answer_tags:
            think = _extract_tag(assistant_content, "think")
            ans = _extract_tag(assistant_content, "answer")
            if think or ans:
                assistant_content = f"{think}\n\nAnswer:\n{ans}".strip()

        # === POPE LOGIK ===
        use_guidance = False
        if self.use_pope:
            if difficulty == "hard":
                use_guidance = random.random() < self.pope_mixed_ratio
            elif difficulty == "medium":
                use_guidance = random.random() < self.pope_mixed_ratio * 0.5

        if use_guidance:
            prefix = self._create_oracle_prefix(assistant_content)
            user_content = f"{prompt}\n\n[Guidance] Start with:\n{prefix}"
        else:
            user_content = prompt

        # Mock-mode
        if self.tokenizer is None:
            return {
                "text": f"User: {user_content}\n\nAssistant: {assistant_content}",
                "input_ids": torch.zeros(16, dtype=torch.long),
                "attention_mask": torch.ones(16, dtype=torch.long),
                "labels": torch.zeros(16, dtype=torch.long),
            }

        messages_full = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        input_ids = self._apply_chat_template_ids(messages_full, add_generation_prompt=False)

        # === Exakter Prompt-Teil (User + assistant generation prompt) ===
        messages_prompt_only = [{"role": "user", "content": user_content}]
        prompt_ids = self._apply_chat_template_ids(messages_prompt_only, add_generation_prompt=True)
        prompt_len = int(prompt_ids.numel())

        # Truncate
        if input_ids.numel() > self.max_length:
            input_ids = input_ids[: self.max_length]

        seq_len = int(input_ids.numel())
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        labels = input_ids.clone()

        if self.mask_user_prompt:
            cut = min(prompt_len, seq_len)
            labels[:cut] = -100

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0

        # Pad
        if seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_dataloader(
    dataset: ReasoningDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )