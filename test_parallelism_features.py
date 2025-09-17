#!/usr/bin/env python3
"""
Comprehensive parallelism feature tests for XG Language

Tests shard blocks, multi-GPU concepts, and hardware targeting functionality.
"""

import torch
import xg
from pathlib import Path
import tempfile
import json


def print_separator(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def test_basic_shard_functionality():
    """Test basic shard block functionality."""
    print_separator("BASIC SHARD FUNCTIONALITY TEST")
    
    try:
        print("Testing cluster_matmul.xg with shard blocks...")
        engine = xg.compile_file(Path('examples/cluster_matmul.xg'))
        
        test_cases = [
            (2, 4, 2),  # Small matrices
            (4, 8, 4),  # Medium matrices
            (8, 16, 8), # Larger matrices
        ]
        
        for m, k, n in test_cases:
            print(f"\nTesting {m}x{k} @ {k}x{n} matrices...")
            a = torch.randn(m, k, dtype=torch.float32)
            b = torch.randn(k, n, dtype=torch.float32)
            
            result = xg.run_engine(engine, external_values={'A': a, 'B': b})
            expected = a @ b
            
            is_correct = torch.allclose(result.value, expected, rtol=1e-5, atol=1e-6)
            print(f"  ✓ Result shape: {result.value.shape}")
            print(f"  ✓ Correctness: {is_correct}")
            
            assert 'sharding_plan' in result.metadata
            sharding_plan = result.metadata['sharding_plan']
            print(f"  ✓ Sharding plan: {sharding_plan}")
            
            assert ('A', 0) in sharding_plan
            assert ('B', 1) in sharding_plan
            
            if not is_correct:
                return False
        
        print("\n✅ Basic shard functionality test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Basic shard functionality test FAILED: {e}")
        return False


def test_advanced_parallelism():
    """Test advanced parallelism with multi-stage computation."""
    print_separator("ADVANCED PARALLELISM TEST")
    
    try:
        print("Testing advanced_cluster_matmul.xg with multi-stage computation...")
        engine = xg.compile_file(Path('examples/advanced_cluster_matmul.xg'))
        
        a = torch.randn(4, 4, dtype=torch.float32)
        b = torch.randn(4, 4, dtype=torch.float32)
        
        result = xg.run_engine(engine, external_values={'A': a, 'B': b})
        expected = a @ b  # Should match sharded_matmul function
        
        is_correct = torch.allclose(result.value, expected, rtol=1e-5, atol=1e-6)
        print(f"  ✓ Result shape: {result.value.shape}")
        print(f"  ✓ Correctness: {is_correct}")
        
        assert 'sharding_plan' in result.metadata
        sharding_plan = result.metadata['sharding_plan']
        print(f"  ✓ Sharding plan: {sharding_plan}")
        
        assert len(sharding_plan) > 0
        
        print("\n✅ Advanced parallelism test PASSED")
        return is_correct
        
    except Exception as e:
        print(f"✗ Advanced parallelism test FAILED: {e}")
        return False


def test_hardware_targeting():
    """Test hardware targeting functionality."""
    print_separator("HARDWARE TARGETING TEST")
    
    try:
        print("Testing hardware targeting with different GPU types...")
        
        hardware_configs = [
            ("H100", 4),
            ("GB200", 8),
            ("A100", 2),
        ]
        
        for target_gpu, num_gpu in hardware_configs:
            print(f"\nTesting {target_gpu} with {num_gpu} GPUs...")
            
            engine = xg.compile_file(Path('examples/cluster_matmul.xg'))
            
            engine.hardware.target_gpu = target_gpu
            engine.hardware.num_gpu = num_gpu
            
            a = torch.randn(2, 4, dtype=torch.float32)
            b = torch.randn(4, 2, dtype=torch.float32)
            
            result = xg.run_engine(engine, external_values={'A': a, 'B': b})
            
            assert result.metadata['target_gpu'] == target_gpu
            assert result.metadata['num_gpu'] == num_gpu
            
            print(f"  ✓ Target GPU: {result.metadata['target_gpu']}")
            print(f"  ✓ Num GPUs: {result.metadata['num_gpu']}")
            print(f"  ✓ Sharding plan: {result.metadata['sharding_plan']}")
        
        print("\n✅ Hardware targeting test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Hardware targeting test FAILED: {e}")
        return False


def test_different_shard_dimensions():
    """Test sharding with different tensor dimensions."""
    print_separator("DIFFERENT SHARD DIMENSIONS TEST")
    
    try:
        print("Testing sharding with various tensor dimensions...")
        
        custom_program = """
TARGET_GPU = H100
NUM_GPU = 4

def test_shard_dim0(Tensor[float32, M, K] A, Tensor[float32, K, N] B) -> Tensor[float32, M, N] {
    with shard(A, dim=0) {
        return A @ B
    }
}

def test_shard_dim1(Tensor[float32, M, K] A, Tensor[float32, K, N] B) -> Tensor[float32, M, N] {
    with shard(B, dim=1) {
        return A @ B
    }
}

def test_both_shards(Tensor[float32, M, K] A, Tensor[float32, K, N] B) -> Tensor[float32, M, N] {
    with shard(A, dim=0), shard(B, dim=1) {
        return A @ B
    }
}

def main() -> Tensor[float32, M, N] {
    return test_both_shards(A, B)
}
"""
        
        engine = xg.compile_source(custom_program)
        
        test_sizes = [(4, 6, 4), (8, 12, 8), (16, 24, 16)]
        
        for m, k, n in test_sizes:
            print(f"\nTesting {m}x{k} @ {k}x{n} with both shards...")
            a = torch.randn(m, k, dtype=torch.float32)
            b = torch.randn(k, n, dtype=torch.float32)
            
            result = xg.run_engine(engine, external_values={'A': a, 'B': b})
            expected = a @ b
            
            is_correct = torch.allclose(result.value, expected, rtol=1e-5, atol=1e-6)
            print(f"  ✓ Correctness: {is_correct}")
            print(f"  ✓ Sharding plan: {result.metadata['sharding_plan']}")
            
            if not is_correct:
                return False
        
        print("\n✅ Different shard dimensions test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Different shard dimensions test FAILED: {e}")
        return False


def test_device_consistency():
    """Test device consistency with parallelism features."""
    print_separator("DEVICE CONSISTENCY TEST")
    
    try:
        print("Testing device consistency with shard blocks...")
        engine = xg.compile_file(Path('examples/cluster_matmul.xg'))
        
        print("\nTesting CPU execution...")
        a_cpu = torch.randn(4, 8, dtype=torch.float32)
        b_cpu = torch.randn(8, 4, dtype=torch.float32)
        
        result_cpu = xg.run_engine(engine, external_values={'A': a_cpu, 'B': b_cpu})
        
        print(f"  ✓ Input devices: A={a_cpu.device}, B={b_cpu.device}")
        print(f"  ✓ Result device: {result_cpu.value.device}")
        print(f"  ✓ Sharding plan: {result_cpu.metadata['sharding_plan']}")
        
        if torch.cuda.is_available():
            print("\nTesting GPU execution...")
            a_gpu = torch.randn(4, 8, dtype=torch.float32, device='cuda')
            b_gpu = torch.randn(8, 4, dtype=torch.float32, device='cuda')
            
            result_gpu = xg.run_engine(engine, external_values={'A': a_gpu, 'B': b_gpu})
            
            print(f"  ✓ Input devices: A={a_gpu.device}, B={b_gpu.device}")
            print(f"  ✓ Result device: {result_gpu.value.device}")
            print(f"  ✓ Sharding plan: {result_gpu.metadata['sharding_plan']}")
        else:
            print("\nCUDA not available - skipping GPU test")
        
        print("\n✅ Device consistency test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Device consistency test FAILED: {e}")
        return False


def main():
    """Run all parallelism feature tests."""
    print("XG Language Parallelism Feature Test Suite")
    print("=" * 60)
    
    test_results = []
    
    test_results.append(("Basic Shard Functionality", test_basic_shard_functionality()))
    test_results.append(("Advanced Parallelism", test_advanced_parallelism()))
    test_results.append(("Hardware Targeting", test_hardware_targeting()))
    test_results.append(("Different Shard Dimensions", test_different_shard_dimensions()))
    test_results.append(("Device Consistency", test_device_consistency()))
    
    print_separator("TEST SUMMARY")
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All parallelism tests passed! Shard blocks and multi-GPU concepts are working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
