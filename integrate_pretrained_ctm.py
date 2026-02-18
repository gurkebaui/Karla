#!/usr/bin/env python3
"""
integrate_pretrained_ctm.py — Load pre-trained CTM into Karla
==============================================================

Nimmt ein pre-trained CTM checkpoint und integriert es in das 
vollständige Karla Modell für RL fine-tuning.

Usage:
    python integrate_pretrained_ctm.py --ctm-checkpoint checkpoints_ctm_pretrain/best_ctm.pt --output karla_with_pretrained_ctm.pt
"""

import argparse
import os
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Integrate-CTM")


def integrate_ctm(args):
    logger.info("=" * 60)
    logger.info("INTEGRATING PRE-TRAINED CTM INTO KARLA")
    logger.info("=" * 60)
    
    from karla.models.karla import create_karla
    from karla.utils.config import KarlaConfig
    
    # Create Karla model
    config = KarlaConfig()
    model = create_karla(config)
    
    # Load pre-trained CTM
    logger.info(f"Loading CTM from: {args.ctm_checkpoint}")
    ctm_ckpt = torch.load(args.ctm_checkpoint, map_location="cpu")
    
    # Load CTM weights
    if "ctm_state_dict" in ctm_ckpt:
        model.l2.load_state_dict(ctm_ckpt["ctm_state_dict"])
        logger.info("✓ CTM weights loaded")
    else:
        # Try direct load
        model.l2.load_state_dict(ctm_ckpt, strict=False)
        logger.info("✓ CTM weights loaded (direct)")
    
    # Set CTM scale if available
    if "ctm_scale" in ctm_ckpt:
        # The pretrainer uses sigmoid(scale), we use softplus(scale_raw)
        # sigmoid(x) ≈ softplus(x) for small x
        # So scale_raw ≈ log(exp(ctm_scale) - 1)
        pretrained_scale = ctm_ckpt["ctm_scale"]
        import math
        if pretrained_scale > 0:
            scale_raw = math.log(math.exp(pretrained_scale) - 1 + 1e-8)
            model.ctm_scale_raw.data = torch.tensor(scale_raw)
            logger.info(f"✓ CTM scale set to: {pretrained_scale:.4f}")
    
    # Print stats
    epoch = ctm_ckpt.get("epoch", "?")
    loss = ctm_ckpt.get("loss", "?")
    step = ctm_ckpt.get("global_step", "?")
    logger.info(f"Pre-training stats: epoch={epoch}, loss={loss}, step={step}")
    
    # Save integrated model
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "pretrained_ctm": True,
        "ctm_checkpoint": args.ctm_checkpoint,
    }, args.output)
    
    logger.info(f"✓ Saved integrated model to: {args.output}")
    
    # Print parameter counts
    counts = model.count_parameters()
    logger.info(f"Total parameters: {counts['total']:,}")
    logger.info(f"Trainable parameters: {counts['trainable']:,}")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctm-checkpoint", type=str, required=True, help="Path to pre-trained CTM")
    parser.add_argument("--output", type=str, default="karla_with_pretrained_ctm.pt", help="Output path")
    args = parser.parse_args()
    
    integrate_ctm(args)


if __name__ == "__main__":
    main()
