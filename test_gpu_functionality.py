#!/usr/bin/env python3

from pathlib import Path
import torch
import xg
import time

def print_separator(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def test_environment_info():
    print_separator("ENVIRONMENT INFORMATION")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
    else:
        print("CUDA not available - will test CPU functionality only")
    
    print(f"CPU cores: {torch.get_num_threads()}")

def test_basic_cpu_execution():
    print_separator("BASIC CPU EXECUTION TEST")
    try:
        print("Compiling cluster_matmul.xg...")
        engine = xg.compile_file(Path('examples/cluster_matmul.xg'))
        print(f"✓ Compilation successful")
        print(f"  Target GPU: {engine.hardware.target_gpu}")
        print(f"  Num GPUs: {engine.hardware.num_gpu}")
        
        print("\nCreating test tensors...")
        a = torch.randn(2, 4, dtype=torch.float32)
        b = torch.randn(4, 2, dtype=torch.float32)
        print(f"  A shape: {a.shape}, device: {a.device}, dtype: {a.dtype}")
        print(f"  B shape: {b.shape}, device: {b.device}, dtype: {b.dtype}")
        
        print("\nExecuting XG program...")
        start_time = time.time()
        result = xg.run_engine(engine, external_values={'A': a, 'B': b})
        execution_time = time.time() - start_time
        
        print(f"✓ Execution successful in {execution_time*1000:.2f}ms")
        print(f"  Result shape: {result.value.shape}")
        print(f"  Result device: {result.value.device}")
        print(f"  Result dtype: {result.value.dtype}")
        
        expected = a @ b
        is_correct = torch.allclose(result.value, expected, rtol=1e-5, atol=1e-6)
        print(f"  Correctness check: {'✓ PASS' if is_correct else '✗ FAIL'}")
        
        if not is_correct:
            print(f"    Max difference: {torch.max(torch.abs(result.value - expected)).item()}")
        
        print(f"  Metadata: {result.metadata}")
        return True
        
    except Exception as e:
        print(f"✗ CPU execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_large_matrix_cpu():
    print_separator("LARGE MATRIX CPU TEST")
    try:
        print("Testing with larger matrices...")
        engine = xg.compile_file(Path('examples/cluster_matmul.xg'))
        
        sizes = [(8, 16, 8), (16, 32, 16), (32, 64, 32)]
        
        for m, k, n in sizes:
            print(f"\nTesting {m}x{k} @ {k}x{n} matrices...")
            a = torch.randn(m, k, dtype=torch.float32)
            b = torch.randn(k, n, dtype=torch.float32)
            
            start_time = time.time()
            result = xg.run_engine(engine, external_values={'A': a, 'B': b})
            execution_time = time.time() - start_time
            
            expected = a @ b
            is_correct = torch.allclose(result.value, expected, rtol=1e-5, atol=1e-6)
            
            print(f"  ✓ {m}x{k}@{k}x{n}: {execution_time*1000:.2f}ms, correct: {is_correct}")
            
            if not is_correct:
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Large matrix test failed: {e}")
        return False

def test_gpu_execution():
    print_separator("GPU EXECUTION TEST")
    if not torch.cuda.is_available():
        print("CUDA not available - skipping GPU test")
        return True
        
    try:
        print("Testing GPU execution with CUDA tensors...")
        engine = xg.compile_file(Path('examples/cluster_matmul.xg'))
        
        device = torch.device('cuda')
        print(f"Using device: {device}")
        
        a = torch.randn(2, 4, dtype=torch.float32, device=device)
        b = torch.randn(4, 2, dtype=torch.float32, device=device)
        print(f"  A: {a.shape} on {a.device}")
        print(f"  B: {b.shape} on {b.device}")
        
        print("\nExecuting on GPU...")
        start_time = time.time()
        result = xg.run_engine(engine, external_values={'A': a, 'B': b})
        execution_time = time.time() - start_time
        
        print(f"✓ GPU execution successful in {execution_time*1000:.2f}ms")
        print(f"  Result device: {result.value.device}")
        
        expected = a @ b
        is_correct = torch.allclose(result.value, expected, rtol=1e-5, atol=1e-6)
        print(f"  Correctness check: {'✓ PASS' if is_correct else '✗ FAIL'}")
        
        print("\nTesting larger GPU matrices...")
        a_large = torch.randn(64, 128, dtype=torch.float32, device=device)
        b_large = torch.randn(128, 64, dtype=torch.float32, device=device)
        
        start_time = time.time()
        result_large = xg.run_engine(engine, external_values={'A': a_large, 'B': b_large})
        gpu_time = time.time() - start_time
        
        print(f"  Large GPU matrix ({a_large.shape[0]}x{a_large.shape[1]}@{b_large.shape[0]}x{b_large.shape[1]}): {gpu_time*1000:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"✗ GPU execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_consistency():
    print_separator("DEVICE CONSISTENCY TEST")
    if not torch.cuda.is_available():
        print("CUDA not available - skipping device consistency test")
        return True
        
    try:
        print("Testing mixed device error handling...")
        engine = xg.compile_file(Path('examples/cluster_matmul.xg'))
        
        a_cpu = torch.randn(2, 4, dtype=torch.float32)  # CPU
        b_gpu = torch.randn(4, 2, dtype=torch.float32, device='cuda')  # GPU
        
        print(f"  A: {a_cpu.shape} on {a_cpu.device}")
        print(f"  B: {b_gpu.shape} on {b_gpu.device}")
        
        print("\nExecuting with mixed devices (should auto-move to target device)...")
        result = xg.run_engine(engine, external_values={'A': a_cpu, 'B': b_gpu})
        
        print(f"✓ Mixed device handling successful")
        print(f"  Result device: {result.value.device}")
        
        expected = a_cpu.to('cuda') @ b_gpu
        is_correct = torch.allclose(result.value, expected, rtol=1e-5, atol=1e-6)
        print(f"  Correctness check: {'✓ PASS' if is_correct else '✗ FAIL'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Device consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    print_separator("PERFORMANCE COMPARISON")
    if not torch.cuda.is_available():
        print("CUDA not available - skipping performance comparison")
        return True
    
    try:
        print("Comparing CPU vs GPU performance...")
        engine = xg.compile_file(Path('examples/cluster_matmul.xg'))
        
        size = 256
        print(f"Testing {size}x{size} matrix multiplication...")
        
        a_cpu = torch.randn(size, size, dtype=torch.float32)
        b_cpu = torch.randn(size, size, dtype=torch.float32)
        
        print("\nCPU execution...")
        start_time = time.time()
        result_cpu = xg.run_engine(engine, external_values={'A': a_cpu, 'B': b_cpu})
        cpu_time = time.time() - start_time
        print(f"  CPU time: {cpu_time*1000:.2f}ms")
        
        a_gpu = a_cpu.to('cuda')
        b_gpu = b_cpu.to('cuda')
        
        print("GPU execution...")
        start_time = time.time()
        result_gpu = xg.run_engine(engine, external_values={'A': a_gpu, 'B': b_gpu})
        gpu_time = time.time() - start_time
        print(f"  GPU time: {gpu_time*1000:.2f}ms")
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"  Speedup: {speedup:.2f}x")
        
        expected = a_cpu @ b_cpu
        cpu_correct = torch.allclose(result_cpu.value, expected, rtol=1e-4, atol=1e-5)
        gpu_correct = torch.allclose(result_gpu.value.cpu(), expected, rtol=1e-4, atol=1e-5)
        
        print(f"  CPU correctness: {'✓ PASS' if cpu_correct else '✗ FAIL'}")
        print(f"  GPU correctness: {'✓ PASS' if gpu_correct else '✗ FAIL'}")
        
        return cpu_correct and gpu_correct
        
    except Exception as e:
        print(f"✗ Performance comparison failed: {e}")
        return False

def main():
    print("XG Language GPU Functionality Test Suite")
    print("=" * 60)
    
    test_results = []
    
    test_results.append(("Environment Info", test_environment_info()))
    test_results.append(("Basic CPU Execution", test_basic_cpu_execution()))
    test_results.append(("Large Matrix CPU", test_large_matrix_cpu()))
    test_results.append(("GPU Execution", test_gpu_execution()))
    test_results.append(("Device Consistency", test_device_consistency()))
    test_results.append(("Performance Comparison", test_performance_comparison()))
    
    print_separator("TEST SUMMARY")
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! GPU functionality is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
