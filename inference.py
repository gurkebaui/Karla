#!/usr/bin/env python3
"""
inference.py — Karla Hacker / Boost inference
Boosts L1/CTM scales to reveal their influence.
"""

import argparse
import os
import sys
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from karla.utils.config import KarlaConfig
from karla.models.karla import create_karla


class KarlaHacker:
    def __init__(self, checkpoint_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"⚡ Loading Karla from {checkpoint_path}...")

        self.config = KarlaConfig()

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.l0.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = create_karla(self.config)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state, strict=False)

        # Move modules (not L0)
        for name, module in self.model.named_children():
            if name != "l0":
                module.to(self.device)

        # move raw scales
        self.model.l1_scale_raw.data = self.model.l1_scale_raw.data.to(self.device)
        self.model.ctm_scale_raw.data = self.model.ctm_scale_raw.data.to(self.device)

        self.model.eval()

        print(f"   L1 scale:  {self.model.l1_scale().item():.4f}")
        print(f"   CTM scale: {self.model.ctm_scale().item():.4f}")

    def _format(self, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    @torch.no_grad()
    def generate(self, prompt, boost_factor=1.0, temp=0.7, max_new=200):
        # Boost by shifting raw params
        orig_l1 = self.model.l1_scale_raw.item()
        orig_ctm = self.model.ctm_scale_raw.item()

        if boost_factor != 1.0:
            delta = math.log(boost_factor)
            self.model.l1_scale_raw.data.add_(delta)
            self.model.ctm_scale_raw.data.add_(delta)

        print(f"\n🧪 Generating with Boost x{boost_factor}")
        print(f"   Effective L1:  {self.model.l1_scale().item():.4f}")
        print(f"   Effective CTM: {self.model.ctm_scale().item():.4f}")
        print(f"   Prompt: {prompt}")
        print("-" * 60)

        text = self._format(prompt)
        input_ids = self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=False).to(self.device)

        generated = input_ids[0].tolist()

        print("🤖 ", end="", flush=True)
        for _ in range(max_new):
            cur = torch.tensor([generated], device=self.device)
            mask = torch.ones_like(cur)

            out = self.model(cur, mask)
            logits = out.logits[0, -1, :].float()

            if temp <= 0:
                tok = logits.argmax().item()
            else:
                probs = F.softmax(logits / temp, dim=-1)
                tok = torch.multinomial(probs, 1).item()

            if tok == self.tokenizer.eos_token_id:
                break

            generated.append(tok)
            print(self.tokenizer.decode([tok], skip_special_tokens=True), end="", flush=True)

        print("\n")

        # restore
        self.model.l1_scale_raw.data.fill_(orig_l1)
        self.model.ctm_scale_raw.data.fill_(orig_ctm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str, default="What are the symptoms of diabetes?")
    parser.add_argument("-b", "--boost", type=float, default=10.0)
    parser.add_argument("-t", "--temp", type=float, default=0.7)
    parser.add_argument("--ckpt", type=str, default="checkpoints_rl/best_rl_model.pt")
    args = parser.parse_args()

    ckpt = args.ckpt
    if not os.path.exists(ckpt):
        ckpt = "checkpoints/best_model.pt"

    hacker = KarlaHacker(ckpt)
    hacker.generate(args.prompt, boost_factor=1.0, temp=args.temp)
    hacker.generate(args.prompt, boost_factor=args.boost, temp=args.temp)


if __name__ == "__main__":
    main()