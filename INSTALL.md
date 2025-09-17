# XG Language Installation Guide

XG is a high-performance language for GPU cluster computing with automatic parallelization and sharding capabilities.

## Quick Install

Install XG Language from PyPI (once published):

```bash
pip install xg-lang
```

## Development Install

For development or to get the latest features:

```bash
git clone https://github.com/viraatdas/xg.git
cd xg
pip install -e .
```

This will install XG in editable mode with all dependencies.

## Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- Optional: CUDA-capable GPU for GPU acceleration

## Verify Installation

After installation, verify that the CLI tools are available:

```bash
# Check compiler
xgc --help

# Check runtime
xgrun --help
```

## Basic Usage

### 1. Create an XG Program

Create a file `hello.xg`:

```xg
TARGET_GPU = H100
NUM_GPU = 4

def matmul_example(Tensor[float32, M, K] A, Tensor[float32, K, N] B) -> Tensor[float32, M, N] {
    with shard(A, dim=0), shard(B, dim=1) {
        return A @ B
    }
}

def main() -> Tensor[float32, 2, 2] {
    return matmul_example(A, B)
}
```

### 2. Compile the Program

```bash
# Compile for H100 with 4 GPUs
xgc hello.xg --target=H100 --num-gpu=4 --out hello.xge --verbose

# Or just check compilation without saving
xgc hello.xg --check-only
```

### 3. Create External Values

Create `values.json` with input tensors:

```json
{
  "A": {
    "tensor": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
    "dtype": "float32"
  },
  "B": {
    "tensor": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
    "dtype": "float32"
  }
}
```

### 4. Run the Program

```bash
# Run compiled engine
xgrun hello.xge --external values.json --verbose

# Or run source directly
xgrun hello.xg --external values.json --verbose --profile
```

## CLI Reference

### xgc (Compiler)

```bash
xgc program.xg [OPTIONS]
```

Options:
- `--target TARGET`: Target hardware (H100, GB200, A100, CPU)
- `--num-gpu NUM`: Number of GPUs to target
- `--out FILE`: Output engine file (.xge)
- `--verbose`: Enable verbose output
- `--check-only`: Only check compilation, don't save

### xgrun (Runtime)

```bash
xgrun program.xge [OPTIONS]
```

Options:
- `--external FILE`: JSON file with external values
- `--verbose`: Enable verbose output
- `--profile`: Enable execution profiling
- `--device DEVICE`: Force device (cpu, cuda, auto)

## Examples

The repository includes several examples:

- `examples/cluster_matmul.xg`: Basic matrix multiplication with sharding
- `examples/advanced_cluster_matmul.xg`: Multi-stage computation example

Run examples:

```bash
# Basic example
xgrun examples/cluster_matmul.xg --external values.json --verbose

# Advanced example
xgrun examples/advanced_cluster_matmul.xg --external values.json --verbose
```

## Parallelism Features

XG provides automatic parallelization through:

### Shard Blocks

```xg
with shard(tensor, dim=0) {
    // Operations are automatically distributed
    return tensor @ other_tensor
}
```

### Hardware Declarations

```xg
TARGET_GPU = GB200  // Target specific GPU architecture
NUM_GPU = 8         // Number of GPUs to use
```

### Multi-Stage Computation

```xg
def multi_stage(Tensor[float32, M, K] X, Tensor[float32, K, N] Y, Tensor[float32, N, P] Z) -> Tensor[float32, M, P] {
    with shard(X, dim=0), shard(Y, dim=1) {
        intermediate = X @ Y
    }
    
    with shard(intermediate, dim=0), shard(Z, dim=1) {
        return intermediate @ Z
    }
}
```

## Troubleshooting

### CUDA Issues

If you encounter CUDA-related issues:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install CUDA support
pip install xg-lang[cuda]
```

### Import Errors

If you get import errors after installation:

```bash
# Reinstall in development mode
pip uninstall xg-lang
pip install -e .
```

### CLI Commands Not Found

If `xgc` or `xgrun` commands are not found:

```bash
# Check if they're in your PATH
which xgc
which xgrun

# Reinstall package
pip install --force-reinstall xg-lang
```

## Development

For contributing to XG Language:

```bash
# Clone repository
git clone https://github.com/viraatdas/xg.git
cd xg

# Install development dependencies
pip install -e .[dev]

# Run tests
python -m pytest tests/ -v

# Run CLI tests
python test_cli_functionality.py

# Run parallelism tests
python test_parallelism_features.py
```

## Support

- GitHub Issues: https://github.com/viraatdas/xg/issues
- Documentation: https://github.com/viraatdas/xg#readme
