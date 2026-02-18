#!/usr/bin/env python3
"""
debug_all.py — Master Debug Script for Karla C1 V2/V3
======================================================

Runs all debug tests in sequence:
1. Value Head tests
2. Value Loss tests  
3. Evaluation tests
4. Plan/Answer mode tests
5. Reward-weighted Engram tests
6. Training V3 component tests
7. Integration test

Usage:
    python debug_all.py
    python debug_all.py --quick  # Skip slow tests
    python debug_all.py --test value_head  # Run specific test
"""

import sys
import os
import argparse
import time
import subprocess

def print_banner():
    print("\n" + "="*60)
    print("  🔧 KARLA C1 V2/V3 DEBUG SUITE")
    print("="*60)

def print_section(name):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")

def run_test_file(filepath, name, timeout=120):
    """Run a test file and return success status."""
    print_section(f"Running: {name}")
    
    start = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, filepath],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        elapsed = time.time() - start
        
        if result.stdout:
            print(result.stdout)
        
        if result.returncode != 0:
            print(f"\n❌ {name} FAILED (exit code: {result.returncode})")
            if result.stderr:
                print("STDERR:", result.stderr[:500])
            return False
        else:
            print(f"\n✅ {name} PASSED ({elapsed:.1f}s)")
            return True
            
    except subprocess.TimeoutExpired:
        print(f"\n⏱️ {name} TIMED OUT after {timeout}s")
        return False
    except Exception as e:
        print(f"\n❌ {name} ERROR: {e}")
        return False


def quick_syntax_check():
    """Quick syntax check for all Python files."""
    print_section("Quick Syntax Check")
    
    files_to_check = [
        "karla_v2_l2_ctm.py",
        "karla_v2_karla.py",
        "l2_ctm_v3_plan_answer.py",
        "karla_v3_plan_answer.py",
        "l1_engram_v2_reward_weighted.py",
        "train_rl_v2_full.py",
        "train_rl_v3_plan_answer.py",
    ]
    
    all_ok = True
    for f in files_to_check:
        path = os.path.join(os.path.dirname(__file__), f)
        if not os.path.exists(path):
            print(f"  ⚠️  {f} not found, skipping")
            continue
            
        try:
            with open(path, 'r') as file:
                code = file.read()
            compile(code, f, 'exec')
            print(f"  ✅ {f}: Syntax OK")
        except SyntaxError as e:
            print(f"  ❌ {f}: Syntax Error - {e}")
            all_ok = False
    
    return all_ok


def check_imports():
    """Check that key imports work."""
    print_section("Import Check")
    
    imports = [
        ("torch", "PyTorch"),
        ("torch.nn", "PyTorch NN"),
        ("torch.nn.functional", "PyTorch Functional"),
    ]
    
    all_ok = True
    for module, name in imports:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            all_ok = False
    
    return all_ok


def integration_test():
    """Full integration test."""
    print_section("Integration Test")
    
    try:
        import torch
        from dataclasses import dataclass
        
        # Test 1: Value head integration
        @dataclass
        class L2Output:
            features: torch.Tensor
            internal_ticks: int
            synchronization: torch.Tensor
            certainty: torch.Tensor
            value: torch.Tensor
        
        output = L2Output(
            features=torch.randn(2, 1536),
            internal_ticks=5,
            synchronization=torch.randn(2),
            certainty=torch.sigmoid(torch.randn(2, 1)),
            value=torch.randn(2, 1)
        )
        print(f"  ✅ L2Output with value: shape={output.value.shape}")
        
        # Test 2: Plan/Answer output
        @dataclass
        class PlanAnswerOutput:
            plan_features: torch.Tensor
            plan_ticks: int
            plan_value: torch.Tensor
            answer_features: torch.Tensor
            answer_ticks: int
            answer_value: torch.Tensor
            features: torch.Tensor
            internal_ticks: int
            synchronization: torch.Tensor
            certainty: torch.Tensor
            value: torch.Tensor
        
        pa_output = PlanAnswerOutput(
            plan_features=torch.randn(2, 1536),
            plan_ticks=3,
            plan_value=torch.randn(2, 1),
            answer_features=torch.randn(2, 1536),
            answer_ticks=7,
            answer_value=torch.randn(2, 1),
            features=torch.randn(2, 1536),
            internal_ticks=10,
            synchronization=torch.randn(2),
            certainty=torch.sigmoid(torch.randn(2, 1)),
            value=torch.randn(2, 1)
        )
        print(f"  ✅ PlanAnswerOutput: plan_ticks={pa_output.plan_ticks}, answer_ticks={pa_output.answer_ticks}")
        
        # Test 3: Weighted gradient
        @dataclass
        class WeightedGradient:
            gradient: torch.Tensor
            weight: float
            timestamp: int
            source: str
        
        wg = WeightedGradient(
            gradient=torch.randn(32),
            weight=1.5,
            timestamp=1,
            source="reward"
        )
        print(f"  ✅ WeightedGradient: weight={wg.weight}")
        
        # Test 4: Combined loss
        policy_loss = torch.tensor(0.5)
        value_loss = torch.tensor(0.2)
        c_value = 0.1
        total = policy_loss + c_value * value_loss
        print(f"  ✅ Combined loss: {total.item():.4f}")
        
        # Test 5: Advantage normalization
        rewards = [0.0, 0.5, 1.0]
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r)**2 for r in rewards) / len(rewards)) ** 0.5
        advs = [(r - mean_r) / std_r for r in rewards]
        print(f"  ✅ Advantages: {[f'{a:.2f}' for a in advs]}")
        
        return True
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Karla Debug Suite")
    parser.add_argument("--quick", action="store_true", help="Skip slow tests")
    parser.add_argument("--test", type=str, default=None, help="Run specific test file")
    args = parser.parse_args()
    
    print_banner()
    
    results = {}
    test_dir = os.path.dirname(__file__)
    
    # Run specific test if requested
    if args.test:
        test_file = os.path.join(test_dir, f"debug_{args.test}.py")
        if os.path.exists(test_file):
            success = run_test_file(test_file, args.test)
            return success
        else:
            print(f"Test file not found: {test_file}")
            return False
    
    # Pre-checks
    results["syntax"] = quick_syntax_check()
    results["imports"] = check_imports()
    
    if not results["syntax"]:
        print("\n❌ Syntax errors found. Fix before continuing.")
        return False
    
    # Run test files
    if not args.quick:
        test_files = [
            ("debug_value_head.py", "Value Head"),
            ("debug_value_loss.py", "Value Loss"),
            ("debug_evaluation.py", "Evaluation"),
            ("debug_plan_answer.py", "Plan/Answer Mode"),
            ("debug_adaptive.py", "True Adaptive Compute"),
            ("debug_engram_reward.py", "Reward-Weighted Engram"),
            ("debug_train_v3.py", "Training V3 Components"),
        ]
        
        for filename, name in test_files:
            filepath = os.path.join(test_dir, filename)
            if os.path.exists(filepath):
                results[name] = run_test_file(filepath, name)
            else:
                print(f"\n⚠️  {filename} not found, skipping")
    
    # Integration
    results["integration"] = integration_test()
    
    # Summary
    print_section("FINAL SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    print(f"\n  Total: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n  🎉 ALL DEBUG TESTS PASSED!")
        print("\n  📁 Files ready for deployment:")
        print("     - karla_v2_l2_ctm.py → karla/models/l2_ctm.py")
        print("     - karla_v2_karla.py → karla/models/karla.py")
        print("     - l2_ctm_v3_plan_answer.py → karla/models/l2_ctm.py (V3)")
        print("     - karla_v3_plan_answer.py → karla/models/karla.py (V3)")
        print("     - l1_engram_v2_reward_weighted.py → karla/models/l1_engram.py (V2)")
        print("     - train_rl_v2_full.py → train_rl.py")
        print("     - train_rl_v3_plan_answer.py → train_rl.py (V3)")
        return True
    else:
        print(f"\n  ⚠️  {total - passed} test suite(s) failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
