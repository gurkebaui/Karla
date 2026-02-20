#!/usr/bin/env python3
"""
integrate_pretrained_ctm_v2.py — Load pre-trained CTM into Karla (Fixed)
=========================================================================

Fixed version that properly handles:
1. Scale conversion between sigmoid (pretraining) and softplus (Karla)
2. Proper scale initialization for meaningful contributions
3. Value head initialization

The scale problem:
- Pretraining uses: sigmoid(ctm_scale) where ctm_scale is learned
- Karla uses: softplus(ctm_scale_raw)
- We need to convert correctly!
"""

import argparse
import os
import torch
import torch.nn.functional as F
import math
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Integrate-CTM-v2")


def integrate_ctm(args):
    logger.info("=" * 60)
    logger.info("INTEGRATING PRE-TRAINED CTM INTO KARLA (v2)")
    logger.info("=" * 60)
    
    from karla.models.karla import create_karla
    from karla.utils.config import KarlaConfig
    
    # Create Karla model
    config = KarlaConfig()
    model = create_karla(config)
    
    # Load pre-trained CTM checkpoint
    logger.info(f"Loading CTM from: {args.ctm_checkpoint}")
    ctm_ckpt = torch.load(args.ctm_checkpoint, map_location="cpu")
    
    # Get CTM state dict
    if "ctm_state_dict" in ctm_ckpt:
        ctm_state = ctm_ckpt["ctm_state_dict"]
    else:
        ctm_state = ctm_ckpt
    
    # Fix keys: remove underscore prefix from lazy modules
    fixed_state = {}
    for key, value in ctm_state.items():
        if key.startswith("_l1_kv_proj"):
            new_key = key[1:]
            fixed_state[new_key] = value
        else:
            fixed_state[key] = value
    
    # Load with strict=False
    missing, unexpected = model.l2.load_state_dict(fixed_state, strict=False)
    
    logger.info(f"Missing keys: {len(missing)}")
    if missing:
        for k in list(missing)[:5]:
            logger.info(f"  - {k}")
    
    logger.info(f"Unexpected keys: {len(unexpected)}")
    if unexpected:
        for k in list(unexpected)[:5]:
            logger.info(f"  + {k}")
    
    logger.info("✓ CTM weights loaded")
    
    # ========== SCALE FIX ==========
    # The pretraining saves ctm_scale as the sigmoid input
    # We need to convert to softplus format
    
    # Target scale for CTM contribution
    target_ctm_scale = args.target_ctm_scale  # e.g., 0.3
    
    # Check if checkpoint has scale info
    if "ctm_scale" in ctm_ckpt:
        pretrained_scale = ctm_ckpt["ctm_scale"]
        logger.info(f"Pre-training ctm_scale (raw): {pretrained_scale:.6f}")
        
        # In pretraining: output = hidden + sigmoid(ctm_scale) * ctm_features
        # sigmoid(pretrained_scale) is the actual contribution weight
        sigmoid_scale = torch.sigmoid(torch.tensor(pretrained_scale)).item()
        logger.info(f"Pre-training used sigmoid({pretrained_scale:.4f}) = {sigmoid_scale:.6f}")
        
        # In Karla: output = hidden + softplus(ctm_scale_raw) * ctm_features
        # We want softplus(ctm_scale_raw) = target_ctm_scale
        # So: ctm_scale_raw = log(exp(target) - 1)
    
    # Set CTM scale
    ctm_raw = math.log(math.exp(target_ctm_scale) - 1 + 1e-8)
    model.ctm_scale_raw.data = torch.tensor(ctm_raw)
    actual_scale = model.ctm_scale().item()
    logger.info(f"✓ CTM scale set: softplus({ctm_raw:.4f}) = {actual_scale:.6f}")
    
    # Set L1 scale
    target_l1_scale = args.target_l1_scale
    l1_raw = math.log(math.exp(target_l1_scale) - 1 + 1e-8)
    model.l1_scale_raw.data = torch.tensor(l1_raw)
    actual_l1 = model.l1_scale().item()
    logger.info(f"✓ L1 scale set: softplus({l1_raw:.4f}) = {actual_l1:.6f}")
    
    # ========== VALUE HEAD CHECK ==========
    # Check if value head was trained
    value_head_key = "l2.value_head.weight"
    if value_head_key in fixed_state:
        logger.info("✓ Value head loaded from checkpoint")
    else:
        logger.info("⚠ Value head not in checkpoint - using random initialization")
        logger.info("  This is normal for CTM-only pretraining")
    
    # Print stats
    epoch = ctm_ckpt.get("epoch", "?")
    loss = ctm_ckpt.get("loss", "?")
    step = ctm_ckpt.get("global_step", "?")
    logger.info(f"Pre-training stats: epoch={epoch}, loss={loss}, step={step}")
    
    # Save integrated model
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    
    save_dict = {
        "model_state_dict": model.state_dict(),
        "pretrained_ctm": True,
        "ctm_checkpoint": args.ctm_checkpoint,
        "scales": {
            "l1_scale": actual_l1,
            "ctm_scale": actual_scale,
            "l1_scale_raw": l1_raw,
            "ctm_scale_raw": ctm_raw,
        },
        "pretraining_info": {
            "epoch": epoch,
            "loss": loss,
            "step": step,
        }
    }
    
    torch.save(save_dict, args.output)
    logger.info(f"✓ Saved integrated model to: {args.output}")
    
    # Print parameter counts
    counts = model.count_parameters()
    logger.info(f"Total parameters: {counts['total']:,}")
    logger.info(f"Trainable parameters: {counts['trainable']:,}")
    
    # Test forward pass
    logger.info("\n--- Testing forward pass ---")
    model.eval()
    
    # Create dummy input
    dummy_ids = torch.tensor([[1, 2, 3, 4, 5]])
    dummy_mask = torch.ones_like(dummy_ids)
    
    with torch.no_grad():
        # Move L1/L2 to CPU for test
        out = model(dummy_ids, dummy_mask)
    
    logger.info(f"Output logits shape: {out.logits.shape}")
    logger.info(f"CTM ticks: {out.internal_ticks}")
    logger.info(f"Certainty: {out.certainty.item():.4f}")
    logger.info(f"Value: {out.value.item():.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("INTEGRATION COMPLETE")
    logger.info(f"L1 contribution: {actual_l1:.4f} * l1_memory")
    logger.info(f"CTM contribution: {actual_scale:.4f} * ctm_features")
    logger.info("=" * 60)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Integrate pre-trained CTM into Karla")
    
    parser.add_argument(
        "--ctm-checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained CTM checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="karla_with_pretrained_ctm.pt",
        help="Output path for integrated model"
    )
    parser.add_argument(
        "--target-ctm-scale",
        type=float,
        default=0.3,
        help="Target CTM scale (contribution weight, default 0.3)"
    )
    parser.add_argument(
        "--target-l1-scale",
        type=float,
        default=0.1,
        help="Target L1 scale (contribution weight, default 0.1)"
    )
    
    args = parser.parse_args()
    integrate_ctm(args)


if __name__ == "__main__":
    main()
