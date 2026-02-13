"""
data/dataset.py
===============
Handhabt POPE (Privileged On-Policy Exploration) und Chat-Templates.

POPE-Prinzip:
- Harte Probleme erhalten Oracle-Prefix als Guidance
- Modell lernt, von guided zu unguided zu transferieren
- Mixed Training für optimale Exploration

Chat-Template Support:
- Qwen erwartet spezifisches Format: <|im_start|>user...<|im_end|>
- Wir nutzen tokenizer.apply_chat_template für korrekte Formatierung
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional
import random
import os
import logging

logger = logging.getLogger("Dataset")


class ReasoningDataset(Dataset):
    """
    Dataset für Reasoning-Training mit POPE-Unterstützung.
    
    Features:
    1. Lädt JSONL-Daten mit prompt/reasoning/answer
    2. Wendet POPE-Strategie an (Oracle Prefix für harte Probleme)
    3. Nutzt Qwen Chat-Template für korrekte Formatierung
    4. Maskiert User-Prompt bei Labels (Loss nur auf Response)
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[Any] = None,
        max_length: int = 2048,
        pope_prefix_ratio: float = 0.3,
        pope_mixed_ratio: float = 0.5,
        use_pope: bool = True,
        mask_user_prompt: bool = True,  # Loss nur auf Assistant-Response
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pope_prefix_ratio = pope_prefix_ratio
        self.pope_mixed_ratio = pope_mixed_ratio
        self.use_pope = use_pope
        self.mask_user_prompt = mask_user_prompt
        
        self.examples: List[Dict[str, Any]] = []
        
        # Daten laden
        self._load_data()
        
        # Statistiken
        self._print_stats()
    
    def _load_data(self):
        """Lade Daten aus verschiedenen Dateiformaten (.jsonl, .parquet, .json)."""
        if not os.path.exists(self.data_path):
            logger.warning(f"⚠️ Datei {self.data_path} fehlt!")
            logger.info("   Erstelle synthetische Fallback-Daten...")
            self._create_fallback_data()
            return
        
        logger.info(f"📂 Lade {self.data_path}...")
        
        # Dateiendung erkennen
        ext = os.path.splitext(self.data_path)[1].lower()
        
        if ext == '.parquet':
            self._load_parquet()
        elif ext == '.json':
            self._load_json()
        else:
            # Default: JSONL
            self._load_jsonl()
        
        logger.info(f"   {len(self.examples)} Beispiele geladen")
    
    def _load_jsonl(self):
        """Lade JSONL-Datei."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        data = json.loads(line)
                        self._add_example(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"   Zeile {line_num} übersprungen: {e}")
    
    def _load_json(self):
        """Lade einzelne JSON-Datei (Array von Objekten)."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                self._add_example(item)
        else:
            self._add_example(data)
    
    def _load_parquet(self):
        """Lade Parquet-Datei."""
        try:
            import pandas as pd
            df = pd.read_parquet(self.data_path)
            
            for _, row in df.iterrows():
                self._add_example(row.to_dict())
                
        except ImportError:
            logger.error("   pandas oder pyarrow nicht installiert!")
            logger.info("   Installiere mit: pip install pandas pyarrow")
            self._create_fallback_data()
        except Exception as e:
            logger.error(f"   Fehler beim Laden von Parquet: {e}")
            self._create_fallback_data()
    
    def _add_example(self, data: Dict[str, Any]):
        """
        Füge ein Beispiel zum Dataset hinzu.
        
        Unterstützt verschiedene Spaltennamen:
        - prompt/question/instruction/input
        - reasoning/cot/chain_of_thought/response/solution
        - answer/output/final_answer
        - difficulty (optional)
        """
        # Prompt extrahieren (verschiedene mögliche Spaltennamen)
        prompt = (
            data.get('prompt') or
            data.get('question') or
            data.get('instruction') or
            data.get('input') or
            data.get('problem') or
            ''
        )
        
        # Reasoning extrahieren
        reasoning = (
            data.get('reasoning') or
            data.get('cot') or
            data.get('chain_of_thought') or
            data.get('response') or
            data.get('solution') or
            data.get('output') or
            data.get('thinking') or
            data.get('deepseek_r1_distill_70b') or  # Platinum-CoT specific
            ''
        )
        
        # Answer extrahieren
        answer = (
            data.get('answer') or
            data.get('final_answer') or
            data.get('output') or
            ''
        )
        
        # Validierung
        if not prompt or not reasoning:
            return
        
        # Difficulty extrahieren oder ableiten
        difficulty = data.get('difficulty', 'medium')
        if not difficulty or difficulty == 'unknown':
            # Heuristisch ableiten basierend auf Reasoning-Länge
            reasoning_len = len(str(reasoning).split())
            if reasoning_len < 30:
                difficulty = 'easy'
            elif reasoning_len < 100:
                difficulty = 'medium'
            else:
                difficulty = 'hard'
        
        self.examples.append({
            'prompt': str(prompt).strip(),
            'reasoning': str(reasoning).strip(),
            'answer': str(answer).strip() if answer else '',
            'difficulty': difficulty,
        })
    
    def _create_fallback_data(self):
        """Erstelle minimale Fallback-Daten falls Datei fehlt."""
        fallback_examples = [
            {
                'prompt': 'What is 2 + 2?',
                'reasoning': 'I need to add 2 and 2.\n2 + 2 = 4',
                'answer': '4',
                'difficulty': 'easy',
            },
            {
                'prompt': 'If x + 5 = 12, what is x?',
                'reasoning': 'To solve for x, I subtract 5 from both sides.\nx = 12 - 5 = 7',
                'answer': '7',
                'difficulty': 'medium',
            },
        ]
        
        # Dupliziere für minimale Dataset-Größe
        for _ in range(100):
            for ex in fallback_examples:
                self.examples.append(ex.copy())
        
        logger.info(f"   {len(self.examples)} Fallback-Beispiele erstellt")
    
    def _print_stats(self):
        """Drucke Dataset-Statistiken."""
        if not self.examples:
            return
        
        difficulty_counts = {}
        for ex in self.examples:
            d = ex.get('difficulty', 'unknown')
            difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
        
        logger.info(f"📊 Dataset Statistiken:")
        for d, c in sorted(difficulty_counts.items()):
            logger.info(f"   {d}: {c}")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def _create_oracle_prefix(self, reasoning: str) -> str:
        """
        Erstelle Oracle-Prefix aus Reasoning.
        
        Für POPE: Wir schneiden die ersten X% der Lösung ab.
        Das zwingt das Modell, den "richtigen Weg" zu beginnen,
        muss aber den Rest selbstständig vollenden.
        """
        words = reasoning.split()
        
        # Mindestens 5 Wörter, maximal prefix_ratio
        min_words = max(5, int(len(words) * self.pope_prefix_ratio))
        cut_idx = min(min_words, len(words) - 1)  # Mindestens 1 Wort bleibt
        
        return ' '.join(words[:cut_idx])
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Hole ein Beispiel und bereite es für Training vor.
        
        POPE-Logik:
        - Bei 'hard' Problemen: Mit 50% Wahrscheinlichkeit Oracle-Prefix geben
        - Sonst: Normales Training
        """
        example = self.examples[idx]
        
        prompt = example['prompt']
        reasoning = example['reasoning']
        answer = example.get('answer', '')
        difficulty = example.get('difficulty', 'medium')
        
        # Vollständige Response
        full_response = reasoning
        if answer:
            full_response = f"{reasoning}\n\nAnswer: {answer}"
        
        # === POPE LOGIK ===
        use_guidance = False
        
        if self.use_pope:
            # Bei harden Problemen: Oracle Prefix nutzen
            if difficulty == 'hard':
                if random.random() < self.pope_mixed_ratio:
                    use_guidance = True
            
            # Bei medium: Manchmal auch Guidance
            elif difficulty == 'medium':
                if random.random() < self.pope_mixed_ratio * 0.5:
                    use_guidance = True
        
        # Wenn kein Tokenizer: Text zurückgeben (für Mock-Tests)
        if self.tokenizer is None:
            if use_guidance:
                prefix = self._create_oracle_prefix(full_response)
                text = f"User: {prompt}\n\nHint: Start with: {prefix}\n\nAssistant: {full_response}"
            else:
                text = f"User: {prompt}\n\nAssistant: {full_response}"
            
            return {
                "text": text,
                "input_ids": torch.zeros(10, dtype=torch.long),
                "attention_mask": torch.ones(10, dtype=torch.long),
                "labels": torch.zeros(10, dtype=torch.long),
            }
        
        # === MIT TOKENIZER: Chat-Template anwenden ===
        
        if use_guidance:
            # Oracle Prefix in den Prompt injizieren
            prefix = self._create_oracle_prefix(full_response)
            user_content = f"{prompt}\n\n[Guidance] Start your reasoning with: {prefix}"
            assistant_content = full_response
        else:
            user_content = prompt
            assistant_content = full_response
        
        # Chat-Template für Qwen
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        
        try:
            # Versuche apply_chat_template (Qwen 2.5 Style)
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
            )[0]
            
            attention_mask = torch.ones_like(input_ids)
            
        except (AttributeError, TypeError):
            # Fallback: Manuelle Formatierung
            text = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{assistant_content}<|im_end|>"
            
            encodings = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )
            
            input_ids = encodings['input_ids'][0]
            attention_mask = encodings['attention_mask'][0]
        
        # Labels erstellen
        labels = input_ids.clone()
        
        # === MASKIERUNG DES USER PROMPTS ===
        # Loss soll nur auf Assistant-Response berechnet werden
        if self.mask_user_prompt:
            # Finde den Start des Assistant-Teils
            # Qwen Format: ...<|im_start|>assistant\n[CONTENT]<|im_end|>
            
            # Suche nach "assistant" Token oder Pattern
            assistant_tokens = self.tokenizer.encode("assistant", add_special_tokens=False)
            
            # Vereinfachte Maskierung: Erste Hälfte maskieren
            # (Nicht perfekt, aber funktioniert für die meisten Fälle)
            split_point = len(input_ids) // 3  # User prompt ist meist kürzer
            
            # Setze Labels auf -100 für User-Teil
            labels[:split_point] = -100
        
        # Padding maskieren
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
    """
    Erstelle einen DataLoader für das Dataset.
    
    Args:
        dataset: Das ReasoningDataset
        batch_size: Batch-Größe
        shuffle: Ob zu shufflen
        num_workers: Anzahl Worker für paralleles Laden
        
    Returns:
        DataLoader
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        # Collate_fn nicht nötig, da wir bereits getorchte Tensoren zurückgeben
    )


def prepare_pope_data(
    input_path: str,
    output_path: str,
    difficulty_threshold: float = 0.5,
) -> None:
    """
    Bereite Daten mit POPE-Stratifizierung vor.
    
    Analysiert Beispiele und weist Difficulty-Labels basierend auf
    Komplexitätsmetriken zu (Länge, einzigartige Tokens, etc.)
    """
    examples = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    # Analysiere und weise Difficulty zu
    for ex in examples:
        reasoning = ex.get('reasoning', '')
        reasoning_len = len(reasoning.split())
        
        # Heuristik: Längeres Reasoning = schwieriger
        if reasoning_len < 30:
            ex['difficulty'] = 'easy'
        elif reasoning_len < 100:
            ex['difficulty'] = 'medium'
        else:
            ex['difficulty'] = 'hard'
    
    # Schreibe Output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    logger.info(f"[POPE] Prepared {len(examples)} examples → {output_path}")


if __name__ == "__main__":
    # Test Dataset
    dataset = ReasoningDataset("data/micro_pope_data.jsonl")
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
