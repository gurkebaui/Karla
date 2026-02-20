#!/usr/bin/env python3
"""
check_checkpoint.py — Verify checkpoint contents
"""

import torch
import sys

def check_checkpoint(path):
    print(f"\n{'='*60}")
    print(f"Checking: {path}")
    print(f"{'='*60}")
    
    try:
        ckpt = torch.load(path, map_location="cpu")
        
        print(f"\nTop-level keys: {list(ckpt.keys())}")
        
        # Check for CTM scale
        if "ctm_scale" in ckpt:
            import math
            scale = ckpt["ctm_scale"]
            print(f"\n✓ CTM scale found: {scale:.4f}")
            print(f"  sigmoid(scale) ≈ {torch.sigmoid(torch.tensor(scale)).item():.4f}")
        else:
            print("\n⚠ No ctm_scale in checkpoint")
        
        # Check model state
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "ctm_state_dict" in ckpt:
            state = ckpt["ctm_state_dict"]
        else:
            state = ckpt
        
        # Check for CTM weights
        ctm_keys = [k for k in state.keys() if "l2." in k or k.startswith("synapse") or k.startswith("nlm")]
        print(f"\nCTM-related keys: {len(ctm_keys)}")
        
        if len(ctm_keys) > 0:
            print("Sample keys:")
            for k in list(ctm_keys)[:5]:
                print(f"  {k}")
        
        # Check for scale params
        if "ctm_scale_raw" in state:
            scale_raw = state["ctm_scale_raw"]
            import torch.nn.functional as F
            scale = F.softplus(torch.tensor(scale_raw)).item()
            print(f"\n✓ ctm_scale_raw: {scale_raw}")
            print(f"  softplus(scale_raw) = {scale:.4f}")
        else:
            print("\n⚠ No ctm_scale_raw in state")
        
        # Check for value head
        value_keys = [k for k in state.keys() if "value_head" in k]
        print(f"\nValue head keys: {len(value_keys)}")
        for k in value_keys:
            print(f"  {k}")
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY:")
        print(f"{'='*60}")
        
        if "ctm_scale" in ckpt or "ctm_scale_raw" in state:
            print("✓ This looks like a PRE-TRAINED checkpoint")
            print("  Use integrate_pretrained_ctm.py to load into Karla")
        elif "l2." in str(state.keys()):
            print("✓ This looks like a FULL KARLA checkpoint")
            print("  Load directly in inference.py")
        else:
            print("⚠ Checkpoint format unclear")
        
    except Exception as e:
        print(f"\n❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_checkpoint(sys.argv[1])
    else:
        # Check common checkpoints
        import os
        checkpoints = [
            "checkpoints_ctm_pretrain/best_ctm.pt",
            "checkpoints_rl/best_rl_model.pt",
            "karla_with_pretrained_ctm.pt",
        ]
        for ckpt in checkpoints:
            if os.path.exists(ckpt):
                check_checkpoint(ckpt)
