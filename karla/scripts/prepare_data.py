#!/usr/bin/env python3
"""
scripts/prepare_data.py
=======================
Lädt lokale Datensätze und bereitet sie für Karla vor.

Unterstützte Formate:
- .jsonl (JSON Lines)
- .parquet (Apache Parquet)
- .json (JSON Array)

Usage:
    python karla/scripts/prepare_data.py --data-dir data --output data/train.jsonl
"""

import json
import os
import argparse
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataPrep")


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Lade JSONL-Datei."""
    samples = []
    logger.info(f"   Lade JSONL: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    if line_num < 5:  # Nur erste Fehler zeigen
                        logger.warning(f"   Zeile {line_num} übersprungen: {e}")
    
    logger.info(f"   ✓ {len(samples)} Beispiele geladen")
    return samples


def load_parquet(filepath: str) -> List[Dict[str, Any]]:
    """Lade Parquet-Datei."""
    samples = []
    logger.info(f"   Lade Parquet: {filepath}")
    
    try:
        import pandas as pd
        df = pd.read_parquet(filepath)
        
        for _, row in df.iterrows():
            samples.append(row.to_dict())
        
        logger.info(f"   ✓ {len(samples)} Beispiele geladen")
        
    except ImportError:
        logger.error("   ✗ pandas/pyarrow nicht installiert!")
        logger.info("   Installiere mit: pip install pandas pyarrow")
    except Exception as e:
        logger.error(f"   ✗ Fehler: {e}")
    
    return samples


def normalize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalisiere ein Sample in ein einheitliches Format.
    
    Erkennt verschiedene Spaltennamen:
    - prompt, question, instruction, input, problem
    - reasoning, cot, chain_of_thought, response, solution, thinking
    - answer, output, final_answer
    """
    # Prompt extrahieren
    prompt = (
        sample.get('prompt') or
        sample.get('question') or
        sample.get('instruction') or
        sample.get('input') or
        sample.get('problem') or
        ''
    )
    
    # Reasoning extrahieren (verschiedene mögliche Spaltennamen)
    reasoning = (
        sample.get('reasoning') or
        sample.get('cot') or
        sample.get('chain_of_thought') or
        sample.get('response') or
        sample.get('solution') or
        sample.get('output') or
        sample.get('thinking') or
        sample.get('deepseek_r1_distill_70b') or  # Platinum-CoT
        sample.get('generation') or
        ''
    )
    
    # Answer extrahieren
    answer = (
        sample.get('answer') or
        sample.get('final_answer') or
        sample.get('target') or
        ''
    )
    
    # Falls kein separates Reasoning, aber eine Antwort vorhanden
    if not reasoning and answer:
        reasoning = answer
        answer = ''
    
    # Difficulty ableiten falls nicht vorhanden
    difficulty = sample.get('difficulty', '')
    if not difficulty:
        reasoning_len = len(str(reasoning).split())
        if reasoning_len < 30:
            difficulty = 'easy'
        elif reasoning_len < 100:
            difficulty = 'medium'
        else:
            difficulty = 'hard'
    
    return {
        'prompt': str(prompt).strip(),
        'reasoning': str(reasoning).strip(),
        'answer': str(answer).strip() if answer else '',
        'difficulty': difficulty,
    }


def scan_and_load(data_dir: str) -> List[Dict[str, Any]]:
    """
    Scanne Verzeichnis nach Datendateien und lade sie alle.
    """
    all_samples = []
    
    if not os.path.exists(data_dir):
        logger.error(f"Verzeichnis nicht gefunden: {data_dir}")
        return all_samples
    
    logger.info(f"📂 Scanne Verzeichnis: {data_dir}")
    
    # Alle Dateien im Verzeichnis
    files = os.listdir(data_dir)
    
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.isfile(filepath):
            continue
        
        ext = os.path.splitext(filename)[1].lower()
        
        if ext == '.jsonl':
            samples = load_jsonl(filepath)
            for s in samples:
                normalized = normalize_sample(s)
                if normalized['prompt'] and normalized['reasoning']:
                    all_samples.append(normalized)
        
        elif ext == '.parquet':
            samples = load_parquet(filepath)
            for s in samples:
                normalized = normalize_sample(s)
                if normalized['prompt'] and normalized['reasoning']:
                    all_samples.append(normalized)
        
        elif ext == '.json':
            samples = load_jsonl(filepath)  # JSON kann wie JSONL behandelt werden
            for s in samples:
                normalized = normalize_sample(s)
                if normalized['prompt'] and normalized['reasoning']:
                    all_samples.append(normalized)
    
    return all_samples


def main():
    parser = argparse.ArgumentParser(description="Bereite Trainingsdaten vor")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Verzeichnis mit den Datendateien"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/micro_pope_data.jsonl",
        help="Output-Datei (JSONL)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximale Anzahl Samples (0 = alle)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("KARLA DATA PREPARATION")
    logger.info("=" * 50)
    
    # Daten laden
    all_samples = scan_and_load(args.data_dir)
    
    # Statistiken
    logger.info(f"\n📊 Gesamt: {len(all_samples)} Beispiele")
    
    difficulty_counts = {}
    for s in all_samples:
        d = s.get('difficulty', 'unknown')
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
    
    logger.info("   Difficulty Distribution:")
    for d, c in sorted(difficulty_counts.items()):
        logger.info(f"      {d}: {c}")
    
    # Limitieren falls gewünscht
    if args.max_samples > 0 and len(all_samples) > args.max_samples:
        import random
        random.shuffle(all_samples)
        all_samples = all_samples[:args.max_samples]
        logger.info(f"\n✂️  Limitiert auf {args.max_samples} Beispiele")
    
    # Speichern
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"\n💾 Speichere nach: {args.output}")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ Fertig! {len(all_samples)} Beispiele gespeichert")


if __name__ == "__main__":
    main()
