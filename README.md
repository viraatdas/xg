# XG - eXla Gpu pl (OpenAI level infrastructure for training + inference)

- A GPU-first programming language built for clusters running at massive scale.  
- Ensures safety, determinism, and peak efficiency.
- Meant for enterprises to reduce costly mistake of running unoptimized GPU code


## Example syntax
### Hardware declarations
```python
TARGET_GPU = GB200
NUM_GPU = 8
```


### Strictly typed function
```python
def add(int64 a, int64 b) -> int64 {
    return a + b
}
```

### Matrix multiplication with shape/type checks
```python
def multMat(Tensor[float32, M, K] a, Tensor[float32, K, N] b) -> Tensor[float32, M, N] {
    return a @ b
}
```

### Division with explicit error handling
```
def safe_div(float32 a, float32 b) -> Result[float32] {
    if b == 0 { return DivByZero }
    return Ok(a / b)
}
```



### Entry point
```python
def main() {
    x = add(10, 20)
    y = multMat(A, B)
    z = safe_div(1.0, 0.0) # forces explicit handling
}
```


### Multi-GPU block
```python
with shard(A, dim=0), shard(B, dim=1):
    C = A @ B
```

## Examples

A full program that shards a matrix multiplication across GPUs lives in [`examples/cluster_matmul.xg`](examples/cluster_matmul.xg).
You can compile and execute it from Python with:

```python
from pathlib import Path

import torch
import xg

engine = xg.compile_file(Path("examples/cluster_matmul.xg"))
a = torch.randn(2, 4, dtype=torch.float32)
b = torch.randn(4, 2, dtype=torch.float32)

result = xg.run_engine(engine, external_values={"A": a, "B": b})
print(result.value)
print(result.metadata)
```

## How to run?
1. Locally for testing
    - `xgc model.xg --target=H100 --out build/engine.xge`
        - The compiler lowers code into an intermediate representation (IR).
        - Generates CUDA/PTX (or HIP/SYCL for other vendors).
        - Runs test workloads, validates shapes and dtypes.
    - `xgrun build/engine.xge`: run it on dev
        - This executes with automatic correctness checks.
2. Deployment mode (cluster GPU)
    - Change the target GPU in code or at runtime:
        -   ```
            TARGET_GPU = GB200
            NUM_GPU = 1024
            ```
    - Deploy to the cluster: `xgc train.xg --target=GB200 --num-gpu=1024 --out build/engine.xge`
        - Compiler queries GB200’s hardware profile (tensor cores, memory hierarchy, NVLink/NVSwitch).
        - Re-specializes kernels and communication patterns.
        - Benchmarks kernel variants on the cluster.
        - Saves tuned kernels in a cluster cache keyed by (op, shape, dtype, GPU).
    - `xgrun build/engine.xge`:
        - Multi-GPU sharding/replication happens automatically.
        - Compiler-inserted collectives (allreduce, allgather) are overlapped with compute.
        - Expensive mistakes (bad sharding, unsafe ops, silent NaNs) are caught at compile time.



## PyTorch Support

XG is designed to integrate directly with PyTorch.  
- Developers can import existing PyTorch models and operations into XG.  
- Training and inference pipelines written in PyTorch can be compiled into XG engines with no kernel-level changes.  
- XG automatically lowers PyTorch graphs to its IR, applies optimizations (fusion, tuning, sharding), and re-specializes kernels for the target GPU cluster.  
- This ensures that organizations already invested in PyTorch can adopt XG seamlessly, without rewriting their models.  

Example workflow:
1. Train or prototype a model in PyTorch on a single GPU.  
2. Export or wrap the model with XG.  
3. XG compiles and deploys the model across hundreds or thousands of GPUs with guaranteed safety and optimized kernels.  

This makes XG a drop-in accelerator for enterprises that rely on PyTorch today, while still providing a path to write pure XG code for maximum safety and performance.

```python
import torch
import xg

model = MyTorchModel()
engine = xg.compile(model, target="GB200", num_gpu=1024)

output = engine(inputs)
```


## Rules

1. Compilation error if operations are attempted between incompatible datatypes.  
2. Casting must be explicit. No implicit promotions.  
3. Smart compile-time checks:  
   - Division by zero must be handled explicitly.  
   - Out-of-bounds access prevented at compile time if shapes are known.  
4. Programs start with a `def main() {}` entrypoint.  
5. No exceptions; all errors must be represented and handled explicitly.  

## Features

- Math-first syntax: developers describe operations, not kernels.  
- Safety guarantees: explicit typing, no implicit casts, compile-time error handling for unsafe operations.  
- Performance portability: `TARGET_GPU` is a hint; code can run on other GPUs with auto-tuning at deployment.  
- Multi-GPU primitives: `NUM_GPU=8` automatically enables sharding/replication and inserts collectives.  
- Automatic kernel generation: compiler generates multiple candidate kernels, benchmarks them, and caches the fastest.  
- Graph-level optimizations: operator fusion, layout selection, quantization policies (fp16, bf16, int8, etc.).  
- Deterministic execution: fixed RNG streams, shape-specialized engine caching.  
- Built for scale: optimized for clusters with hundreds or thousands of GPUs, where runtime inefficiency or safety errors would result in enormous costs.  

## Execution Model

XG runs in two modes:  

### Development mode
- Write code once, run locally on any GPU (e.g., 4090, A100).  
- Compiler lowers to a portable IR.  
- Execution validates shapes, datatypes, and correctness.  

### Deployment mode
- Set `TARGET_GPU = GB200` (or other hardware).  
- Compiler re-specializes code for the deployment GPU:  
  - Retunes kernels for hardware characteristics.  
  - Inserts optimal communication primitives for `NUM_GPU`.  
  - Caches tuned kernels keyed by `(op, shape, dtype, GPU)`.  

### Example workflow
1. Developer writes XG code on a 4090 or A100.  
2. Compiler generates generic IR and runs locally for testing.  
3. At deployment on a GB200 cluster:  
   - Compiler queries GB200 capabilities (tensor cores, NVLink/NVSwitch).  
   - Re-specializes kernels, benchmarks variants, picks fastest.  
   - Deploys a tuned “engine” binary.  
4. Runtime loads from cache on subsequent runs, ensuring consistent peak performance at cluster scale.  


## Positioning

Today’s GPU programming options each have gaps:  

- CUDA / HIP / SYCL → low-level, brittle across GPU generations, requires kernel expertise.  
- Triton → simplifies GPU programming, but developers must still author kernels.  
- PyTorch Inductor / XLA → graph compilers, but hidden inside frameworks and not portable across vendors.  
- TVM / Halide → research compilers, not GPU-native languages and difficult to integrate.  

XG is different:  
- A language, not a library.  
- Portable across GPU vendors and GPU generations.  
- Built for both training and inference, with safety and determinism guarantees.  
- Multi-GPU parallelism as a primitive, not a framework bolt-on.  
- Designed for enterprises running clusters at scale, where correctness and efficiency prevent millions of dollars in wasted compute.  

## Slogan

Write once. Run at peak performance. Anywhere.
