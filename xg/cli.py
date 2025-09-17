#!/usr/bin/env python3
"""
XG Language Command Line Interface

Provides xgc (compiler) and xgrun (runtime) commands for the XG language.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json

import torch

from .compiler import compile_file, compile_source
from .runtime import run_engine
from .engine import Engine
from .errors import XGTypeError, XGSafetyError, XGRuntimeError


def create_xgc_parser() -> argparse.ArgumentParser:
    """Create argument parser for xgc (XG compiler)."""
    parser = argparse.ArgumentParser(
        prog='xgc',
        description='XG Language Compiler - Compile XG source files to executable engines'
    )
    
    parser.add_argument(
        'source',
        type=Path,
        help='XG source file to compile (.xg)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='CPU',
        help='Target hardware (e.g., H100, GB200, A100, CPU)'
    )
    
    parser.add_argument(
        '--num-gpu',
        type=int,
        default=1,
        help='Number of GPUs to target'
    )
    
    parser.add_argument(
        '--out', '-o',
        type=Path,
        help='Output engine file (.xge). Defaults to <source>.xge'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check compilation, do not save engine'
    )
    
    return parser


def create_xgrun_parser() -> argparse.ArgumentParser:
    """Create argument parser for xgrun (XG runtime)."""
    parser = argparse.ArgumentParser(
        prog='xgrun',
        description='XG Language Runtime - Execute compiled XG engines'
    )
    
    parser.add_argument(
        'engine',
        type=Path,
        help='Compiled XG engine file (.xge) or source file (.xg)'
    )
    
    parser.add_argument(
        '--external',
        type=Path,
        help='JSON file containing external values for the program'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable execution profiling'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='Force execution device (auto detects based on availability)'
    )
    
    return parser


def xgc_main(args: Optional[list] = None) -> int:
    """Main entry point for xgc compiler."""
    parser = create_xgc_parser()
    parsed_args = parser.parse_args(args)
    
    try:
        source_file = parsed_args.source
        if not source_file.exists():
            print(f"Error: Source file '{source_file}' not found", file=sys.stderr)
            return 1
        
        if not source_file.suffix == '.xg':
            print(f"Error: Source file must have .xg extension", file=sys.stderr)
            return 1
        
        if parsed_args.verbose:
            print(f"Compiling {source_file}...")
            print(f"Target: {parsed_args.target}")
            print(f"GPUs: {parsed_args.num_gpu}")
        
        engine = compile_file(source_file)
        
        engine.hardware.target_gpu = parsed_args.target if parsed_args.target != 'CPU' else None
        engine.hardware.num_gpu = parsed_args.num_gpu
        
        if parsed_args.verbose:
            print(f"✓ Compilation successful")
            print(f"  Functions: {list(engine.function_map().keys())}")
            print(f"  Target GPU: {engine.hardware.target_gpu}")
            print(f"  Num GPUs: {engine.hardware.num_gpu}")
        
        if not parsed_args.check_only:
            output_file = parsed_args.out or source_file.with_suffix('.xge')
            
            engine.save(output_file)
            
            if parsed_args.verbose:
                print(f"✓ Engine saved to {output_file}")
            else:
                print(f"Compiled {source_file} -> {output_file}")
        else:
            print(f"✓ Compilation check passed for {source_file}")
        
        return 0
        
    except (XGTypeError, XGSafetyError) as e:
        print(f"Compilation error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def xgrun_main(args: Optional[list] = None) -> int:
    """Main entry point for xgrun runtime."""
    parser = create_xgrun_parser()
    parsed_args = parser.parse_args(args)
    
    try:
        engine_file = parsed_args.engine
        if not engine_file.exists():
            print(f"Error: Engine file '{engine_file}' not found", file=sys.stderr)
            return 1
        
        if engine_file.suffix == '.xge':
            if parsed_args.verbose:
                print(f"Loading engine from {engine_file}...")
            engine = Engine.load(engine_file)
        elif engine_file.suffix == '.xg':
            if parsed_args.verbose:
                print(f"Compiling and running {engine_file}...")
            engine = compile_file(engine_file)
        else:
            print(f"Error: Engine file must have .xge or .xg extension", file=sys.stderr)
            return 1
        
        external_values = {}
        if parsed_args.external:
            if not parsed_args.external.exists():
                print(f"Error: External values file '{parsed_args.external}' not found", file=sys.stderr)
                return 1
            
            with open(parsed_args.external, 'r') as f:
                external_data = json.load(f)
            
            for key, value in external_data.items():
                if isinstance(value, dict) and 'tensor' in value:
                    tensor_data = value['tensor']
                    dtype = getattr(torch, value.get('dtype', 'float32'))
                    external_values[key] = torch.tensor(tensor_data, dtype=dtype)
                else:
                    external_values[key] = value
        
        if parsed_args.device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU", file=sys.stderr)
        
        if parsed_args.verbose:
            print(f"Executing engine...")
            print(f"  Target GPU: {engine.hardware.target_gpu}")
            print(f"  Num GPUs: {engine.hardware.num_gpu}")
            print(f"  External values: {list(external_values.keys())}")
            print(f"  CUDA available: {torch.cuda.is_available()}")
        
        import time
        start_time = time.time()
        result = run_engine(engine, external_values=external_values)
        execution_time = time.time() - start_time
        
        if parsed_args.verbose or parsed_args.profile:
            print(f"✓ Execution completed in {execution_time*1000:.2f}ms")
            print(f"  Result type: {type(result.value)}")
            if hasattr(result.value, 'shape'):
                print(f"  Result shape: {result.value.shape}")
                print(f"  Result device: {result.value.device}")
            print(f"  Metadata: {result.metadata}")
        
        if isinstance(result.value, torch.Tensor):
            print(f"Result tensor: shape={result.value.shape}, device={result.value.device}")
            if result.value.numel() <= 16:  # Only print small tensors
                print(result.value)
            else:
                print(f"Tensor too large to display ({result.value.numel()} elements)")
        else:
            print(f"Result: {result.value}")
        
        return 0
        
    except XGRuntimeError as e:
        print(f"Runtime error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    script_name = Path(sys.argv[0]).name
    if script_name == 'xgc' or 'xgc' in script_name:
        sys.exit(xgc_main())
    elif script_name == 'xgrun' or 'xgrun' in script_name:
        sys.exit(xgrun_main())
    else:
        print("Error: Unknown command. Use 'xgc' or 'xgrun'", file=sys.stderr)
        sys.exit(1)
