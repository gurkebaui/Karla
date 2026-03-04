#!/usr/bin/env python3
"""
Debug script to find NaN source in CTM training.
"""

import torch
import torch.nn as nn
import sys
import os

# Setup path
sys.path.insert(0, '/home/z/my-project')

def test_ctm_forward():
    """Test CTM forward pass with mock data."""
    print("=" * 60)
    print("Testing CTM Forward Pass")
    print("=" * 60)
    
    # Import modules
    from karla.models.l2_ctm import CTMHead
    
    # Create CTM head
    print("\n1. Creating CTMHead...")
    ctm = CTMHead(
        hidden_dim=512,
        num_neurons=256,
        num_internal_ticks=5,  # Reduced for testing
        l0_hidden_dim=1536,
        use_bitnet=True,
    )
    ctm.eval()  # Disable dropout
    
    # Create mock input
    B, S = 2, 16  # Small batch and sequence for testing
    l0_hidden = torch.randn(B, S, 1536) * 0.1
    l1_mem = torch.randn(B, S, 512) * 0.1
    
    print(f"   Input shapes: l0={l0_hidden.shape}, l1={l1_mem.shape}")
    print(f"   Input NaN check: l0={torch.isnan(l0_hidden).any()}, l1={torch.isnan(l1_mem).any()}")
    
    # Forward pass
    print("\n2. Running forward pass...")
    with torch.no_grad():
        try:
            output = ctm(l0_hidden, l1_mem)
            print(f"   Output shape: {output.features.shape}")
            print(f"   Features NaN: {torch.isnan(output.features).any()}")
            print(f"   Features range: [{output.features.min():.4f}, {output.features.max():.4f}]")
            print(f"   Certainty NaN: {torch.isnan(output.certainty).any()}")
            print(f"   Value NaN: {torch.isnan(output.value).any()}")
        except Exception as e:
            print(f"   ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n3. Testing with zeros input...")
    l0_zeros = torch.zeros(B, S, 1536)
    l1_zeros = torch.zeros(B, S, 512)
    with torch.no_grad():
        try:
            output = ctm(l0_zeros, l1_zeros)
            print(f"   Output shape: {output.features.shape}")
            print(f"   Features NaN: {torch.isnan(output.features).any()}")
        except Exception as e:
            print(f"   ERROR: {e}")
    
    print("\n4. Testing with large values...")
    l0_large = torch.randn(B, S, 1536) * 10.0
    l1_large = torch.randn(B, S, 512) * 10.0
    with torch.no_grad():
        try:
            output = ctm(l0_large, l1_large)
            print(f"   Output shape: {output.features.shape}")
            print(f"   Features NaN: {torch.isnan(output.features).any()}")
        except Exception as e:
            print(f"   ERROR: {e}")


def test_full_karla():
    """Test full Karla model with mock L0."""
    print("\n" + "=" * 60)
    print("Testing Full Karla Model")
    print("=" * 60)
    
    from karla.models.karla import Karla
    
    print("\n1. Creating Karla with mock L0...")
    model = Karla(use_mock_l0=True, l2_num_internal_ticks=5)
    model.eval()
    
    # Create mock input
    B, S = 2, 16
    input_ids = torch.randint(0, 1000, (B, S))
    labels = input_ids.clone()
    labels[:, -5:] = -100  # Simulate padding
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Valid labels: {(labels != -100).sum().item()}")
    
    print("\n2. Running forward pass...")
    with torch.no_grad():
        try:
            output = model(input_ids, labels=labels)
            print(f"   Logits shape: {output.logits.shape}")
            print(f"   Logits NaN: {torch.isnan(output.logits).any()}")
            if output.loss is not None:
                print(f"   Loss: {output.loss.item():.4f}")
                print(f"   Loss NaN: {torch.isnan(output.loss).any()}")
            else:
                print("   Loss: None")
        except Exception as e:
            print(f"   ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n3. Testing with all-padding labels...")
    labels_all_pad = torch.full((B, S), -100)
    with torch.no_grad():
        try:
            output = model(input_ids, labels=labels_all_pad)
            print(f"   Logits NaN: {torch.isnan(output.logits).any()}")
            if output.loss is not None:
                print(f"   Loss: {output.loss.item():.4f}")
                print(f"   Loss NaN: {torch.isnan(output.loss).any()}")
                if not torch.isnan(output.loss):
                    print("   ✓ FIX SUCCESSFUL: Loss is 0.0 instead of NaN!")
            else:
                print("   Loss: None")
        except Exception as e:
            print(f"   ERROR: {e}")


def test_gradient_flow():
    """Test gradient flow through CTM."""
    print("\n" + "=" * 60)
    print("Testing Gradient Flow")
    print("=" * 60)
    
    from karla.models.l2_ctm import CTMHead
    
    print("\n1. Creating CTMHead for gradient test...")
    ctm = CTMHead(
        hidden_dim=512,
        num_neurons=256,
        num_internal_ticks=3,  # Very few ticks for testing
        l0_hidden_dim=1536,
    )
    ctm.train()
    
    B, S = 1, 4
    l0_hidden = torch.randn(B, S, 1536, requires_grad=True) * 0.1
    l1_mem = torch.randn(B, S, 512, requires_grad=True) * 0.1
    
    print("\n2. Forward pass...")
    output = ctm(l0_hidden, l1_mem)
    
    print(f"   Output NaN: {torch.isnan(output.features).any()}")
    
    print("\n3. Computing loss...")
    target = torch.randn_like(output.features)
    loss = nn.functional.mse_loss(output.features, target)
    print(f"   Loss: {loss.item():.6f}")
    print(f"   Loss NaN: {torch.isnan(loss).any()}")
    
    print("\n4. Backward pass...")
    loss.backward()
    
    # Check gradients
    has_nan_grad = False
    for name, param in ctm.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"   NaN grad in: {name}")
                has_nan_grad = True
    
    if not has_nan_grad:
        print("   All gradients are finite!")
    
    print(f"   l0_hidden grad NaN: {torch.isnan(l0_hidden.grad).any() if l0_hidden.grad is not None else 'None'}")
    print(f"   l1_mem grad NaN: {torch.isnan(l1_mem.grad).any() if l1_mem.grad is not None else 'None'}")


if __name__ == "__main__":
    test_ctm_forward()
    test_full_karla()
    test_gradient_flow()
    print("\n" + "=" * 60)
    print("Debug Complete")
    print("=" * 60)
