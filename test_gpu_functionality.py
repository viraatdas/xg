#!/usr/bin/env python3

from pathlib import Path
import torch
import xg

def test_gpu_availability():
    print("=== GPU Availability Check ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available - will test CPU functionality")
    print()

def test_cpu_execution():
    print("=== Testing CPU execution ===")
    try:
        engine = xg.compile_file(Path('examples/cluster_matmul.xg'))
        a = torch.randn(2, 4, dtype=torch.float32)
        b = torch.randn(4, 2, dtype=torch.float32)
        
        result = xg.run_engine(engine, external_values={'A': a, 'B': b})
        print(f"✓ CPU execution successful")
        print(f"Result device: {result.value.device}")
        print(f"Correct result: {torch.allclose(result.value, a @ b)}")
        return True
    except Exception as e:
        print(f"✗ CPU execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_execution():
    print("=== Testing GPU execution ===")
    if not torch.cuda.is_available():
        print("CUDA not available - skipping GPU test")
        return True
        
    try:
        engine = xg.compile_file(Path('examples/cluster_matmul.xg'))
        device = torch.device('cuda')
        a = torch.randn(2, 4, dtype=torch.float32, device=device)
        b = torch.randn(4, 2, dtype=torch.float32, device=device)
        
        result = xg.run_engine(engine, external_values={'A': a, 'B': b})
        print(f"✓ GPU execution successful")
        print(f"Result device: {result.value.device}")
        print(f"Correct result: {torch.allclose(result.value, a @ b)}")
        return True
    except Exception as e:
        print(f"✗ GPU execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mixed_device_error():
    print("=== Testing mixed device error handling ===")
    if not torch.cuda.is_available():
        print("CUDA not available - skipping mixed device test")
        return True
        
    try:
        engine = xg.compile_file(Path('examples/cluster_matmul.xg'))
        a = torch.randn(2, 4, dtype=torch.float32)  # CPU
        b = torch.randn(4, 2, dtype=torch.float32, device='cuda')  # GPU
        
        result = xg.run_engine(engine, external_values={'A': a, 'B': b})
        print(f"✗ Mixed device test should have failed but didn't")
        return False
    except Exception as e:
        print(f"✓ Mixed device error correctly caught: {e}")
        return True

if __name__ == "__main__":
    test_gpu_availability()
    success1 = test_cpu_execution()
    success2 = test_gpu_execution()
    success3 = test_mixed_device_error()
    
    if success1 and success2 and success3:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
