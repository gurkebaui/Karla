#!/usr/bin/env python3
"""
SMRST-20 Half-Automatic Benchmark (Integrated Karla)
===================================================
Vergleicht Qwen2.5-1.5B mit Karla (CTM),
nutzt exakt denselben Inference-Pfad wie inference_phase1.py.
"""

import argparse
import torch
import torch.nn.functional as F
import json
import textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- Karla Imports (identisch zu deinem Script) ----
from karla.models.karla import create_karla
from karla.utils.config import KarlaConfig


# ================= BENCHMARK ================= #

TASKS = [
    {"id": "A1", "cat": "A", "prompt": "Drei Schalter A, B, C sind AUS.\nWenn A umgelegt wird, kippt B.\nWenn B umgelegt wird, kippt C.\nWenn C umgelegt wird, kippt A.\n\nSequenz: A → C → B → A\nWelche Schalter sind AN?", "gold": "A und B"},
    {"id": "A2", "cat": "A", "prompt": "Ein Zähler startet bei 0.\nWenn gerade: +3\nWenn ungerade: ×2\nRegeln dreimal anwenden.\nEndwert?", "gold": "6"},
    {"id": "A3", "cat": "A", "prompt": "Ein Licht ist AUS.\nNach jedem zweiten Knopfdruck wird der Zustand invertiert.\n5 Knopfdrücke.\nEndzustand?", "gold": "AN"},
    {"id": "A4", "cat": "A", "prompt": "X = 2\nWenn X > 3: X = X − 1\nWenn X ≤ 3: X = X + 2\nNach jeder dritten Anwendung: X = 1\n6 Anwendungen.\nEndwert?", "gold": "1"},
    {"id": "A5", "cat": "A", "prompt": "Operationen:\nF: +4\nG: ×2\nH: −3\nStartwert 1\nReihenfolge: G → H → F → G\nErgebnis?", "gold": "6"},

    {"id": "B1", "cat": "B", "prompt": "Alle Blargs sind Zorps.\nEinige Zorps sind keine Flibs.\nKein Flib ist ein Blarg.\nWelche Aussage MUSS wahr sein?", "gold": "Kein Blarg ist ein Flib"},
    {"id": "B2", "cat": "B", "prompt": "Genau eine Aussage ist wahr:\nA: B ist falsch\nB: C ist wahr\nC: A ist falsch\nWelche ist wahr?", "gold": "B"},
    {"id": "B3", "cat": "B", "prompt": "Alle roten Objekte sind schwer.\nEinige schwere Objekte sind nicht rund.\nKein rundes Objekt ist rot.\nMuss es schwere, nicht-runde Objekte geben?", "gold": "Ja"},
    {"id": "B4", "cat": "B", "prompt": "Mindestens ein Wächter lügt.\nJeder sagt: »Mindestens ein anderer Wächter lügt.«\nWie viele können ehrlich sein?", "gold": "Mehr als einer"},
    {"id": "B5", "cat": "B", "prompt": "X: »Genau eine von Y oder Z ist wahr.«\nY: »X ist wahr.«\nZ: »Y ist falsch.«\nIst X wahr oder falsch?", "gold": "Falsch"},

    {"id": "C1", "cat": "C", "prompt": "Wenn A, dann B.\nWenn B, dann C.\nA und C sind wahr.\nCounterfactual: A wäre falsch.\nIst C garantiert falsch?", "gold": "Nein"},
    {"id": "C2", "cat": "C", "prompt": "Wenn überlastet → Fehler.\nWenn Fehler → Neustart.\nEs gab einen Neustart.\nWar Überlastung garantiert?", "gold": "Nein"},
    {"id": "C3", "cat": "C", "prompt": "Wenn Alarm → Sirene.\nWenn Sirene → Polizei.\nNiemand ruft Polizei.\nKann Alarm aktiv gewesen sein?", "gold": "Nein"},
    {"id": "C4", "cat": "C", "prompt": "Ein Zug fährt, wenn der Schalter EIN ist.\nDer Zug fuhr nicht.\nCounterfactual: Schalter EIN.\nWäre der Zug garantiert gefahren?", "gold": "Ja"},
    {"id": "C5", "cat": "C", "prompt": "Wenn P, dann Q.\nWenn nicht Q, dann R.\nR ist falsch.\nWas folgt über P?", "gold": "P ist falsch"},

    {"id": "D1", "cat": "D", "prompt": "Der Code ist nicht 482.\nWenn der Code nicht 739 ist, dann ist er 482.\nWie lautet der Code?", "gold": "739"},
    {"id": "D2", "cat": "D", "prompt": "Objekt nicht links.\nWenn nicht rechts, dann links.\nWo ist es?", "gold": "Rechts"},
    {"id": "D3", "cat": "D", "prompt": "A ist falsch.\nWenn A falsch ist, dann ist B falsch.\nWas gilt für B?", "gold": "B ist falsch"},
    {"id": "D4", "cat": "D", "prompt": "Schlüssel nicht in Kiste 1.\nWenn nicht in Kiste 2, dann in Kiste 1.\nWo ist der Schlüssel?", "gold": "Kiste 2"},
    {"id": "D5", "cat": "D", "prompt": "Zahl ist nicht gerade.\nWenn sie nicht ungerade ist, ist sie 4.\nWelche Zahlen sind möglich?", "gold": "Ungerade"},
]


# ================= INFERENCE ================= #

def generate_karla(model, tokenizer, prompt, max_new_tokens=128, temperature=0.7, top_p=0.9):
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.size(1) > 512:
                input_ids = input_ids[:, -512:]

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(input_ids)

            logits = outputs.logits[0, -1] / temperature
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            remove = cum_probs > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits[sorted_indices[remove]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)[len(prompt):].strip()


def generate_qwen(model, tokenizer, prompt):
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=128, temperature=0.7, top_p=0.9, do_sample=True)
    return tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ================= MAIN ================= #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--karla-checkpoint", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Lade Tokenizer (Qwen shared)…")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

    print("Lade Qwen2.5-1.5B…")
    qwen = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Lade Karla…")
    karla = create_karla(KarlaConfig()).to(device)
    ckpt = torch.load(args.karla_checkpoint, map_location=device, weights_only=True)
    karla.load_state_dict(ckpt["model_state_dict"], strict=False)

    scores = {"qwen": {}, "karla": {}}

    print("\nScoring: 2 = korrekt | 1 = teilweise | 0 = falsch\n")

    for t in TASKS:
        print("=" * 80)
        print(f"{t['id']}  (Kategorie {t['cat']})\n")
        print(textwrap.fill(t["prompt"], 80))
        print("\nGold:", t["gold"])

        print("\n--- Qwen ---")
        qa = generate_qwen(qwen, tokenizer, t["prompt"])
        print(qa)
        scores["qwen"][t["id"]] = int(input("Score Qwen (0/1/2): "))

        print("\n--- Karla ---")
        ka = generate_karla(karla, tokenizer, t["prompt"])
        print(ka)
        scores["karla"][t["id"]] = int(input("Score Karla (0/1/2): "))

    with open("smrst20_results.json", "w") as f:
        json.dump(scores, f, indent=2)

    print("\nFertig. Ergebnisse gespeichert in smrst20_results.json")


if __name__ == "__main__":
    main()
