#!/usr/bin/env python3
"""
Unified Dataset Loader for Karla C1 Training
============================================

Supports multiple data formats:
1. Messages format (chat_if.jsonl): {"messages": [{"role": "user", "content": ...}, ...]}
2. Instruction format (train.jsonl): {"instruction": ..., "output": ...}
3. Personality format (anime.json): [{"dialogue": ..., "trait": ...}, ...]

Features:
- Unified interface for all formats
- Automatic field mapping
- POPE (Privileged On-Policy Exploration) support
- Streaming for large files
- Proper train/val split
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Iterator, Tuple
import random
import os
import logging
import re
from dataclasses import dataclass
from enum import Enum
import itertools

logger = logging.getLogger("UnifiedDataset")


class DataFormat(Enum):
    MESSAGES = "messages"       # chat_if.jsonl
    INSTRUCTION = "instruction" # train.jsonl
    PERSONALITY = "personality" # anime.json


@dataclass
class TrainingExample:
    """Unified training example format."""
    prompt: str
    response: str
    reasoning: str = ""
    answer: str = ""
    difficulty: str = "medium"
    source: str = "unknown"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FormatDetector:
    """Automatically detect data format from sample."""

    @staticmethod
    def detect(sample: Dict) -> DataFormat:
        if "messages" in sample:
            return DataFormat.MESSAGES
        elif "instruction" in sample:
            return DataFormat.INSTRUCTION
        elif "dialogue" in sample:
            return DataFormat.PERSONALITY
        else:
            # Try to infer
            if "prompt" in sample or "question" in sample:
                return DataFormat.INSTRUCTION
            raise ValueError(f"Unknown format: keys = {list(sample.keys())}")


class MessageFormatter:
    """Convert various formats to unified messages format."""

    SYSTEM_INSTRUCTION_REASONING = (
        "You are a helpful AI assistant. When solving problems, "
        "think step by step and show your reasoning. "
        "Provide clear and accurate answers."
    )

    SYSTEM_INSTRUCTION_CODING = (
        "You are an expert programmer. When given a coding task, "
        "provide clean, efficient, and well-documented code. "
        "Explain your approach when appropriate."
    )

    @classmethod
    def format_messages(cls, example: TrainingExample, use_reasoning: bool = True) -> List[Dict]:
        """Convert TrainingExample to messages format."""
        messages = []

        # System message
        if "coding" in example.source.lower() or "programming" in example.source.lower():
            messages.append({"role": "system", "content": cls.SYSTEM_INSTRUCTION_CODING})
        elif use_reasoning and example.reasoning:
            messages.append({"role": "system", "content": cls.SYSTEM_INSTRUCTION_REASONING})

        # User message
        messages.append({"role": "user", "content": example.prompt})

        # Assistant message
        if example.reasoning and use_reasoning:
            # Format with reasoning
            response = f"бетHere's my analysis:\n\n{example.reasoning}\n\n**Answer:**\n{example.response}"
        else:
            response = example.response

        messages.append({"role": "assistant", "content": response})

        return messages


class DataParser:
    """Parse different data formats into TrainingExample."""

    @staticmethod
    def parse_messages_format(data: Dict, source: str = "messages") -> Optional[TrainingExample]:
        """Parse messages format (chat_if.jsonl)."""
        messages = data.get("messages", [])
        if not messages:
            return None

        # Extract user prompt
        prompt = ""
        for msg in messages:
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break

        if not prompt:
            return None

        # Extract assistant response
        response = ""
        reasoning = ""
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Try to extract reasoning
                think_match = re.search(r'бет(.*?)ллин', content, re.DOTALL)
                if think_match:
                    reasoning = think_match.group(1).strip()
                    # Remove thinking tags for response
                    response = re.sub(r'бет.*?ллин', '', content, flags=re.DOTALL).strip()
                else:
                    response = content
                break

        # Extract metadata
        metadata = {
            "uuid": data.get("uuid"),
            "license": data.get("license"),
            "tools": data.get("tools"),
            "capability_target": data.get("capability_target"),
            "used_in": data.get("used_in"),
        }

        # Determine difficulty based on response length
        difficulty = "medium"
        if len(response) < 200:
            difficulty = "easy"
        elif len(response) > 1000:
            difficulty = "hard"

        return TrainingExample(
            prompt=prompt,
            response=response,
            reasoning=reasoning,
            difficulty=difficulty,
            source=source,
            metadata=metadata,
        )

    @staticmethod
    def parse_instruction_format(data: Dict, source: str = "instruction") -> Optional[TrainingExample]:
        """Parse instruction format (train.jsonl)."""
        prompt = data.get("instruction") or data.get("prompt") or data.get("question", "")
        response = data.get("output") or data.get("response") or data.get("answer", "")

        if not prompt or not response:
            return None

        # Determine difficulty
        difficulty = "medium"
        word_count = len(response.split())
        if word_count < 50:
            difficulty = "easy"
        elif word_count > 300:
            difficulty = "hard"

        return TrainingExample(
            prompt=prompt,
            response=response,
            reasoning="",  # No explicit reasoning in this format
            difficulty=difficulty,
            source=source,
            metadata={},
        )

    @staticmethod
    def parse_personality_format(data: Dict, source: str = "personality") -> Optional[TrainingExample]:
        """Parse personality format (anime.json)."""
        dialogue = data.get("dialogue", "")
        trait = data.get("trait", "")

        if not dialogue:
            return None

        # For personality data, we create a prompt-response pair
        # Trait can be used as a style modifier
        if trait:
            prompt = f"Say something in a {trait} way."
            response = dialogue
        else:
            # Generate a generic prompt
            prompt = "Express yourself."
            response = dialogue

        return TrainingExample(
            prompt=prompt,
            response=response,
            reasoning="",
            difficulty="easy",
            source=source,
            metadata={"trait": trait},
        )


class UnifiedDataset(Dataset):
    """
    Unified dataset that handles multiple file formats.

    Features:
    - Auto-detects format from file content
    - Supports JSONL and JSON files
    - Memory-efficient streaming for large files
    - Optional POPE prefixes for hard problems
    - Proper tokenization and masking
    """

    def __init__(
        self,
        data_paths: List[str],
        tokenizer: Any = None,
        max_length: int = 2048,
        train_ratio: float = 0.95,
        split: str = "train",
        seed: int = 42,
        use_pope: bool = True,
        pope_prefix_ratio: float = 0.3,
        pope_hard_ratio: float = 0.5,
        mask_user_prompt: bool = True,
        max_samples_per_file: Optional[int] = None,
    ):
        self.data_paths = data_paths if isinstance(data_paths, list) else [data_paths]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_ratio = train_ratio
        self.split = split
        self.seed = seed
        self.use_pope = use_pope
        self.pope_prefix_ratio = pope_prefix_ratio
        self.pope_hard_ratio = pope_hard_ratio
        self.mask_user_prompt = mask_user_prompt
        self.max_samples_per_file = max_samples_per_file

        self.examples: List[TrainingExample] = []
        self._load_all_files()
        self._print_stats()

    def _load_all_files(self):
        """Load all data files."""
        random.seed(self.seed)

        for path in self.data_paths:
            if not os.path.exists(path):
                logger.warning(f"File not found: {path}")
                continue

            logger.info(f"Loading: {path}")
            file_examples = self._load_file(path)

            # Apply max samples limit
            if self.max_samples_per_file and len(file_examples) > self.max_samples_per_file:
                random.shuffle(file_examples)
                file_examples = file_examples[:self.max_samples_per_file]

            logger.info(f"  Loaded {len(file_examples)} examples")
            self.examples.extend(file_examples)

        # Shuffle all examples
        random.shuffle(self.examples)

        # Train/val split
        total = len(self.examples)
        train_size = int(total * self.train_ratio)

        if self.split == "train":
            self.examples = self.examples[:train_size]
        else:
            self.examples = self.examples[train_size:]

        logger.info(f"Split '{self.split}': {len(self.examples)} examples")

    def _load_file(self, path: str) -> List[TrainingExample]:
        """Load a single file, detecting format automatically."""
        examples = []

        ext = os.path.splitext(path)[1].lower()
        source = os.path.basename(path)

        if ext == ".json":
            # JSON array format
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        example = self._parse_item(item, source)
                        if example:
                            examples.append(example)
                else:
                    example = self._parse_item(data, source)
                    if example:
                        examples.append(example)
        else:
            # JSONL format (default)
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        example = self._parse_item(data, source)
                        if example:
                            examples.append(example)
                    except json.JSONDecodeError as e:
                        if line_num < 5:  # Only log first few errors
                            logger.warning(f"Parse error at line {line_num}: {e}")

        return examples

    def _parse_item(self, data: Dict, source: str) -> Optional[TrainingExample]:
        """Parse a single data item."""
        try:
            fmt = FormatDetector.detect(data)

            if fmt == DataFormat.MESSAGES:
                return DataParser.parse_messages_format(data, source)
            elif fmt == DataFormat.INSTRUCTION:
                return DataParser.parse_instruction_format(data, source)
            elif fmt == DataFormat.PERSONALITY:
                return DataParser.parse_personality_format(data, source)
        except Exception as e:
            logger.debug(f"Parse failed: {e}")

        return None

    def _print_stats(self):
        """Print dataset statistics."""
        if not self.examples:
            logger.warning("No examples loaded!")
            return

        stats = {
            "total": len(self.examples),
            "by_difficulty": {},
            "by_source": {},
            "avg_prompt_len": 0,
            "avg_response_len": 0,
        }

        total_prompt_len = 0
        total_response_len = 0

        for ex in self.examples:
            # Difficulty
            d = ex.difficulty
            stats["by_difficulty"][d] = stats["by_difficulty"].get(d, 0) + 1

            # Source
            s = ex.source
            stats["by_source"][s] = stats["by_source"].get(s, 0) + 1

            # Lengths
            total_prompt_len += len(ex.prompt.split())
            total_response_len += len(ex.response.split())

        stats["avg_prompt_len"] = total_prompt_len / len(self.examples)
        stats["avg_response_len"] = total_response_len / len(self.examples)

        logger.info("=" * 60)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total examples: {stats['total']}")
        logger.info(f"Avg prompt length: {stats['avg_prompt_len']:.1f} words")
        logger.info(f"Avg response length: {stats['avg_response_len']:.1f} words")
        logger.info("-" * 40)
        logger.info("By difficulty:")
        for d, c in sorted(stats["by_difficulty"].items()):
            logger.info(f"  {d}: {c} ({100*c/stats['total']:.1f}%)")
        logger.info("-" * 40)
        logger.info("By source:")
        for s, c in sorted(stats["by_source"].items()):
            logger.info(f"  {s}: {c} ({100*c/stats['total']:.1f}%)")
        logger.info("=" * 60)

    def _create_oracle_prefix(self, response: str) -> str:
        """Create POPE oracle prefix from response."""
        words = response.split()
        if len(words) <= 6:
            return response
        cut = max(5, int(len(words) * self.pope_prefix_ratio))
        return " ".join(words[:cut])

    def _apply_chat_template(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        """Apply chat template to messages."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        # Manual fallback
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        # Apply POPE for hard problems
        user_content = example.prompt
        if self.use_pope and example.difficulty == "hard":
            if random.random() < self.pope_hard_ratio:
                prefix = self._create_oracle_prefix(example.response)
                user_content = f"{example.prompt}\n\n[Hint] Start your answer with:\n{prefix}"

        # Create messages
        messages = MessageFormatter.format_messages(
            example,
            use_reasoning=bool(example.reasoning),
        )

        # Update user message if modified
        if user_content != example.prompt:
            for msg in messages:
                if msg["role"] == "user":
                    msg["content"] = user_content

        # Tokenize
        if self.tokenizer is None:
            return {
                "text": f"User: {user_content}\n\nAssistant: {example.response}",
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.ones(self.max_length, dtype=torch.long),
                "labels": torch.zeros(self.max_length, dtype=torch.long),
            }

        # Full sequence
        full_text = self._apply_chat_template(messages, add_generation_prompt=False)
        prompt_text = self._apply_chat_template(messages[:-1], add_generation_prompt=True)

        # Tokenize
        full_enc = self.tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
        prompt_enc = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")

        input_ids = full_enc["input_ids"][0]
        prompt_len = prompt_enc["input_ids"][0].numel()

        # Truncate
        if input_ids.numel() > self.max_length:
            input_ids = input_ids[:self.max_length]

        seq_len = input_ids.numel()
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        labels = input_ids.clone()

        # Mask user prompt
        if self.mask_user_prompt:
            cut = min(prompt_len, seq_len)
            labels[:cut] = -100

        # Pad
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0

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


class StreamingDataset(Iterator):
    """
    Memory-efficient streaming dataset for very large files.

    Yields batches on-the-fly without loading entire file into memory.
    Useful for the 6.7GB chat_if.jsonl file.
    """

    def __init__(
        self,
        data_paths: List[str],
        tokenizer: Any,
        batch_size: int = 4,
        max_length: int = 2048,
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
    ):
        self.data_paths = data_paths if isinstance(data_paths, list) else [data_paths]
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

        self.rng = random.Random(seed)
        self.buffer: List[Dict] = []
        self.file_handles: List = []
        self.exhausted = False

    def _fill_buffer(self):
        """Fill shuffle buffer from files."""
        while len(self.buffer) < self.shuffle_buffer_size:
            for path in self.data_paths:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                                example = self._process_item(data)
                                if example:
                                    self.buffer.append(example)
                                    if len(self.buffer) >= self.shuffle_buffer_size:
                                        return
                            except json.JSONDecodeError:
                                continue
                except FileNotFoundError:
                    continue

            # All files exhausted
            if len(self.buffer) == 0:
                self.exhausted = True
                return
            else:
                # Buffer not full but files exhausted
                return

    def _process_item(self, data: Dict) -> Optional[Dict]:
        """Process a single item into tokenized format."""
        try:
            fmt = FormatDetector.detect(data)

            if fmt == DataFormat.MESSAGES:
                example = DataParser.parse_messages_format(data)
            elif fmt == DataFormat.INSTRUCTION:
                example = DataParser.parse_instruction_format(data)
            else:
                return None

            if not example:
                return None

            messages = MessageFormatter.format_messages(example)
            full_text = self._apply_chat_template(messages, add_generation_prompt=False)
            prompt_text = self._apply_chat_template(messages[:-1], add_generation_prompt=True)

            full_enc = self.tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
            prompt_enc = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")

            input_ids = full_enc["input_ids"][0]
            prompt_len = prompt_enc["input_ids"][0].numel()

            if input_ids.numel() > self.max_length:
                input_ids = input_ids[:self.max_length]

            seq_len = input_ids.numel()
            attention_mask = torch.ones(seq_len, dtype=torch.long)
            labels = input_ids.clone()
            labels[:min(prompt_len, seq_len)] = -100

            # Pad
            pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
            if seq_len < self.max_length:
                pad_len = self.max_length - seq_len
                input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        except Exception:
            return None

    def _apply_chat_template(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def __iter__(self):
        while not self.exhausted:
            self._fill_buffer()

            if not self.buffer:
                return

            # Shuffle buffer
            self.rng.shuffle(self.buffer)

            # Yield batch
            batch = self.buffer[:self.batch_size]
            self.buffer = self.buffer[self.batch_size:]

            if len(batch) == self.batch_size:
                yield {
                    "input_ids": torch.stack([b["input_ids"] for b in batch]),
                    "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                    "labels": torch.stack([b["labels"] for b in batch]),
                }


def create_dataloaders(
    data_paths: List[str],
    tokenizer: Any,
    batch_size: int = 4,
    max_length: int = 2048,
    train_ratio: float = 0.95,
    num_workers: int = 0,
    use_pope: bool = True,
    max_samples_per_file: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_paths: List of paths to data files
        tokenizer: Tokenizer for encoding text
        batch_size: Batch size
        max_length: Maximum sequence length
        train_ratio: Ratio of data for training
        num_workers: Number of dataloader workers
        use_pope: Whether to use POPE prefixes for hard problems
        max_samples_per_file: Maximum samples to load per file (for debugging)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = UnifiedDataset(
        data_paths=data_paths,
        tokenizer=tokenizer,
        max_length=max_length,
        train_ratio=train_ratio,
        split="train",
        use_pope=use_pope,
        max_samples_per_file=max_samples_per_file,
    )

    val_dataset = UnifiedDataset(
        data_paths=data_paths,
        tokenizer=tokenizer,
        max_length=max_length,
        train_ratio=train_ratio,
        split="val",
        use_pope=False,  # No POPE in validation
        max_samples_per_file=max_samples_per_file,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


# ============================================================
# Test Script
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified Dataset Loader Test")
    parser.add_argument("--files", nargs="+", required=True, help="Data files to load")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples per file")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Use mock tokenizer for testing
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 1

        def __call__(self, text, **kwargs):
            words = text.split()
            ids = [hash(w) % 50000 for w in words]
            return {"input_ids": torch.tensor([ids])}

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            parts = []
            for m in messages:
                parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n")
            return "".join(parts)

    tokenizer = MockTokenizer()

    train_loader, val_loader = create_dataloaders(
        data_paths=args.files,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples_per_file=args.max_samples,
    )

    print("\nTesting train loader:")
    for batch in train_loader:
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        break

    print("\nTesting validation loader:")
    for batch in val_loader:
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        break

    print("\nDataset test complete!")
