#!/usr/bin/env python3
"""
prepare_sft_data.py
===================
Lädt, filtert und normalisiert die SFT-Datensätze für Karla C1 / Cosmos.
Jetzt mit Multi-Language Support (Python, JS, C++)!
"""

import os
from datasets import load_dataset
import pandas as pd

os.makedirs("data", exist_ok=True)

# ==============================================================================
# 1. HERMES TOOL USE (Reasoning & Tools)
# ==============================================================================
print("Lade und formatiere Hermes Reasoning & Tool Use...")

def map_hermes(x):
    msgs = []
    # Tools als System-Prompt integrieren
    if x.get('tools'):
        msgs.append({"role": "system", "content": f"Available tools:\n{x['tools']}"})
        
    for c in x['conversations']:
        # Mappe die Rollennamen auf Standard ChatML
        role_map = {"human": "user", "user": "user", "system": "system"}
        role = role_map.get(c['from'], "assistant")
        msgs.append({"role": role, "content": str(c['value'])})
    return {"messages": msgs}

try:
    ds_hermes = load_dataset("interstellarninja/hermes_reasoning_tool_use", split="train")
    ds_hermes = ds_hermes.map(map_hermes, remove_columns=ds_hermes.column_names)
    ds_hermes.to_pandas().to_parquet("data/hermes_clean.parquet", index=False)
    print(f" -> Hermes gespeichert! ({len(ds_hermes)} Beispiele)\n")
except Exception as e:
    print(f"Fehler bei Hermes: {e}")


# ==============================================================================
# 2. GITHUB TOP CODE (Python, JavaScript, C++)
# ==============================================================================
print("Lade GitHub Top Code...")
ds_github = load_dataset("ronantakizawa/github-top-code", split="train")

# Wir definieren unsere Ziel-Sprachen und Dateinamen
languages = [
    ("Python", "github_python.parquet"),
    ("JavaScript", "github_javascript.parquet"),
    ("C++", "github_cpp.parquet")
]

# Helper-Funktion für das Mapping
def map_github_entry(x):
    lang = x.get('file_language', 'code')
    path = x.get('file_path', 'file')
    repo = x.get('repo_name', 'unknown-repo')
    
    prompt = f"Please write the complete {lang} code for `{path}` from the `{repo}` repository."
    
    return {"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": str(x['content'])}
    ]}

for lang_name, filename in languages:
    print(f"Filtere und speichere {lang_name}...")
    
    # Filter: Richtige Sprache UND gute Qualität (> 100 Sterne)
    # Bei C++ sind wir etwas gnädiger (> 50 Sterne), da es weniger Repos gibt als Python
    min_stars = 100 if lang_name != "C++" else 50
    
    ds_lang = ds_github.filter(
        lambda x: x["file_language"] == lang_name and x["repo_stars"] > min_stars
    )
    
    if len(ds_lang) > 0:
        ds_lang = ds_lang.map(map_github_entry, remove_columns=ds_github.column_names)
        ds_lang.to_pandas().to_parquet(f"data/{filename}", index=False)
        print(f" -> {filename} gespeichert! ({len(ds_lang)} Beispiele)")
    else:
        print(f" -> Warnung: Keine Beispiele für {lang_name} gefunden!")

print("\n=== ALLE DATEN VORBEREITET! ===")