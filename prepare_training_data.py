#!/usr/bin/env python3
"""
prepare_training_data.py — Prepare datasets for Karla training
===============================================================

Konvertiert verschiedene Dataset-Formate in ein einheitliches Format für Karla.

Input formats supported:
- JSONL with various field names
- JSON array
- Conversation format (messages array)

Output format:
{
    "prompt": "...",
    "reasoning": "..." (optional),
    "answer": "...",
}
"""

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Any
from pathlib import Path
import re


class DatasetConverter:
    """Konvertiert Datasets in einheitliches Format."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Feld-Mappings (werden automatisch erkannt)
        self.prompt_fields = ['prompt', 'question', 'input', 'instruction', 'user_query', 'problem']
        self.response_fields = ['response', 'answer', 'output', 'completion', 'assistant', 'solution', 'target']
        self.reasoning_fields = ['reasoning', 'cot', 'chain_of_thought', 'thinking', 'explanation', 'rationale']
        self.code_fields = ['code', 'solution_code', 'implementation', 'program']
        
        # Statistiken
        self.stats = {
            'total_processed': 0,
            'total_skipped': 0,
            'by_source': {},
        }
    
    def detect_format(self, sample: Dict) -> str:
        """Erkenne Sample-Format."""
        # Conversation format (messages array)
        if 'messages' in sample:
            return 'conversation'
        
        # Code format
        if any(f in sample for f in self.code_fields):
            return 'code'
        
        # Standard format
        if any(f in sample for f in self.prompt_fields):
            return 'standard'
        
        return 'unknown'
    
    def extract_field(self, sample: Dict, field_names: List[str]) -> Optional[str]:
        """Extrahiere Feld aus Sample."""
        for name in field_names:
            if name in sample and sample[name]:
                value = sample[name]
                if isinstance(value, str):
                    return value
                elif isinstance(value, list):
                    return ' '.join(str(x) for x in value)
                else:
                    return str(value)
        return None
    
    def convert_conversation(self, sample: Dict) -> Optional[Dict]:
        """Konvertiere Conversation-Format."""
        messages = sample.get('messages', [])
        if not messages:
            return None
        
        prompt_parts = []
        response_parts = []
        reasoning_parts = []
        
        current_role = None
        
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            
            if not content:
                continue
            
            if role in ['user', 'human']:
                prompt_parts.append(content)
            elif role in ['assistant', 'ai', 'bot']:
                response_parts.append(content)
            elif role in ['system']:
                prompt_parts.insert(0, f"[System: {content}]")
        
        if not prompt_parts or not response_parts:
            return None
        
        return {
            'prompt': '\n'.join(prompt_parts),
            'reasoning': '\n'.join(reasoning_parts) if reasoning_parts else '',
            'answer': '\n'.join(response_parts),
        }
    
    def convert_code(self, sample: Dict) -> Optional[Dict]:
        """Konvertiere Code-Format."""
        prompt = self.extract_field(sample, self.prompt_fields + ['problem', 'description', 'task'])
        code = self.extract_field(sample, self.code_fields + self.response_fields)
        
        if not prompt or not code:
            # Versuche alternative Extraktion
            if 'instruction' in sample:
                prompt = sample['instruction']
            if 'output' in sample:
                code = sample['output']
        
        if not prompt or not code:
            return None
        
        # Format code as answer
        answer = f"```python\n{code}\n```"
        
        return {
            'prompt': prompt,
            'reasoning': '',
            'answer': answer,
        }
    
    def convert_standard(self, sample: Dict) -> Optional[Dict]:
        """Konvertiere Standard-Format."""
        prompt = self.extract_field(sample, self.prompt_fields)
        response = self.extract_field(sample, self.response_fields)
        reasoning = self.extract_field(sample, self.reasoning_fields)
        
        if not prompt:
            return None
        
        # Wenn keine response, versuche aus anderen Feldern
        if not response:
            # Suche nach allen String-Feldern die nicht prompt sind
            for key, value in sample.items():
                if key not in self.prompt_fields and isinstance(value, str) and len(value) > 20:
                    response = value
                    break
        
        if not response:
            return None
        
        return {
            'prompt': prompt,
            'reasoning': reasoning or '',
            'answer': response,
        }
    
    def convert_sample(self, sample: Dict, source: str = 'unknown') -> Optional[Dict]:
        """Konvertiere ein einzelnes Sample."""
        format_type = self.detect_format(sample)
        
        if format_type == 'conversation':
            result = self.convert_conversation(sample)
        elif format_type == 'code':
            result = self.convert_code(sample)
        elif format_type == 'standard':
            result = self.convert_standard(sample)
        else:
            # Versuche alle Konverter
            for converter in [self.convert_conversation, self.convert_code, self.convert_standard]:
                result = converter(sample)
                if result:
                    break
        
        if result:
            result['source'] = source
            result['difficulty'] = 'medium'  # Default
            self.stats['total_processed'] += 1
            self.stats['by_source'][source] = self.stats['by_source'].get(source, 0) + 1
        else:
            self.stats['total_skipped'] += 1
        
        return result
    
    def process_jsonl(self, filepath: str, max_samples: int = None) -> List[Dict]:
        """Verarbeite JSONL-Datei."""
        results = []
        source = Path(filepath).stem
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    sample = json.loads(line)
                    converted = self.convert_sample(sample, source)
                    if converted:
                        results.append(converted)
                except json.JSONDecodeError:
                    continue
        
        return results
    
    def process_json(self, filepath: str, max_samples: int = None) -> List[Dict]:
        """Verarbeite JSON-Datei."""
        results = []
        source = Path(filepath).stem
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict):
            # Könnte ein einzelnes Sample oder verschachtelt sein
            if any(k in data for k in ['data', 'samples', 'examples', 'items']):
                samples = data.get('data') or data.get('samples') or data.get('examples') or data.get('items', [])
            else:
                samples = [data]
        else:
            return results
        
        if max_samples:
            samples = samples[:max_samples]
        
        for sample in samples:
            converted = self.convert_sample(sample, source)
            if converted:
                results.append(converted)
        
        return results
    
    def process_file(self, filepath: str, max_samples: int = None) -> List[Dict]:
        """Verarbeite Datei basierend auf Format."""
        print(f"Processing: {filepath}")
        
        # Erkenne Format
        with open(filepath, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            first_line = f.readline()
        
        try:
            json.loads(first_line)
            # Könnte JSONL oder JSON sein
            f.seek(0)
            second_line = f.readline()
            if second_line.strip():
                try:
                    json.loads(second_line)
                    return self.process_jsonl(filepath, max_samples)
                except:
                    pass
        except:
            pass
        
        # Versuche als JSON
        if first_char in ['[', '{']:
            return self.process_json(filepath, max_samples)
        
        # Fallback: JSONL
        return self.process_jsonl(filepath, max_samples)
    
    def save_output(self, data: List[Dict], output_path: str):
        """Speichere konvertierte Daten."""
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Saved {len(data)} samples to: {output_path}")
    
    def print_stats(self):
        """Zeige Statistiken."""
        print(f"\n{'='*50}")
        print("CONVERSION STATISTICS")
        print(f"{'='*50}")
        print(f"Total processed: {self.stats['total_processed']:,}")
        print(f"Total skipped: {self.stats['total_skipped']:,}")
        print(f"\nBy source:")
        for source, count in sorted(self.stats['by_source'].items()):
            print(f"  {source}: {count:,}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument('--chat', type=str, help='Chat/instruction dataset file')
    parser.add_argument('--code', type=str, help='Code dataset file')
    parser.add_argument('--max-chat', type=int, default=None, help='Max chat samples')
    parser.add_argument('--max-code', type=int, default=None, help='Max code samples')
    parser.add_argument('--output', type=str, default='data/training_data.jsonl', help='Output file')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle output')
    parser.add_argument('--train-split', type=float, default=0.95, help='Training split ratio')
    
    args = parser.parse_args()
    
    converter = DatasetConverter()
    all_data = []
    
    # Verarbeite Chat-Dataset
    if args.chat:
        chat_data = converter.process_file(args.chat, args.max_chat)
        print(f"  → Loaded {len(chat_data)} chat samples")
        all_data.extend(chat_data)
    
    # Verarbeite Code-Dataset
    if args.code:
        code_data = converter.process_file(args.code, args.max_code)
        print(f"  → Loaded {len(code_data)} code samples")
        all_data.extend(code_data)
    
    # Shuffle
    if args.shuffle:
        random.shuffle(all_data)
    
    # Split
    if args.train_split < 1.0:
        split_idx = int(len(all_data) * args.train_split)
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        # Save
        base, ext = os.path.splitext(args.output)
        converter.save_output(train_data, args.output)
        if val_data:
            converter.save_output(val_data, f"{base}_val{ext}")
    else:
        converter.save_output(all_data, args.output)
    
    converter.print_stats()


if __name__ == "__main__":
    main()
