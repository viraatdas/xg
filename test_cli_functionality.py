#!/usr/bin/env python3
"""
Comprehensive CLI functionality tests for XG Language

Tests both xgc (compiler) and xgrun (runtime) CLI tools with various configurations.
"""

import subprocess
import json
import tempfile
import os
from pathlib import Path
import pytest


def run_command(cmd, cwd=None, input_data=None):
    """Run a shell command and return result."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=cwd, input=input_data
    )
    return result.returncode, result.stdout, result.stderr


def test_xgc_help():
    """Test xgc help command."""
    returncode, stdout, stderr = run_command("xgc --help")
    assert returncode == 0
    assert "XG Language Compiler" in stdout
    assert "--target" in stdout
    assert "--num-gpu" in stdout


def test_xgrun_help():
    """Test xgrun help command."""
    returncode, stdout, stderr = run_command("xgrun --help")
    assert returncode == 0
    assert "XG Language Runtime" in stdout
    assert "--external" in stdout
    assert "--verbose" in stdout


def test_xgc_compilation():
    """Test basic compilation with xgc."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        returncode, stdout, stderr = run_command(
            f"xgc examples/cluster_matmul.xg --target=A100 --num-gpu=2 --out {tmpdir}/test.xge --verbose"
        )
        
        assert returncode == 0, f"Compilation failed: {stderr}"
        assert "Compilation successful" in stdout
        assert "Target GPU: A100" in stdout
        assert "Num GPUs: 2" in stdout
        assert (tmpdir / "test.xge").exists()


def test_xgc_check_only():
    """Test compilation check without saving."""
    returncode, stdout, stderr = run_command(
        "xgc examples/cluster_matmul.xg --check-only --verbose"
    )
    
    assert returncode == 0, f"Check failed: {stderr}"
    assert "Compilation check passed" in stdout


def test_xgrun_with_engine_file():
    """Test execution with compiled engine file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        returncode, stdout, stderr = run_command(
            f"xgc examples/cluster_matmul.xg --out {tmpdir}/test.xge"
        )
        assert returncode == 0, f"Compilation failed: {stderr}"
        
        external_values = {
            "A": {"tensor": [[1.0, 2.0], [3.0, 4.0]], "dtype": "float32"},
            "B": {"tensor": [[5.0, 6.0], [7.0, 8.0]], "dtype": "float32"}
        }
        external_file = tmpdir / "external.json"
        with open(external_file, 'w') as f:
            json.dump(external_values, f)
        
        returncode, stdout, stderr = run_command(
            f"xgrun {tmpdir}/test.xge --external {external_file} --verbose"
        )
        
        assert returncode == 0, f"Execution failed: {stderr}"
        assert "Execution completed" in stdout
        assert "Result tensor" in stdout
        assert "sharding_plan" in stdout


def test_xgrun_direct_source():
    """Test execution directly from source file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        external_values = {
            "A": {"tensor": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "dtype": "float32"},
            "B": {"tensor": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "dtype": "float32"}
        }
        external_file = tmpdir / "external.json"
        with open(external_file, 'w') as f:
            json.dump(external_values, f)
        
        returncode, stdout, stderr = run_command(
            f"xgrun examples/cluster_matmul.xg --external {external_file} --verbose --profile"
        )
        
        assert returncode == 0, f"Execution failed: {stderr}"
        assert "Compiling and running" in stdout
        assert "Execution completed" in stdout
        assert "sharding_plan" in stdout


def test_parallelism_sharding_plans():
    """Test that parallelism features record correct sharding plans."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        external_values = {
            "A": {"tensor": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], "dtype": "float32"},
            "B": {"tensor": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], "dtype": "float32"}
        }
        external_file = tmpdir / "external.json"
        with open(external_file, 'w') as f:
            json.dump(external_values, f)
        
        returncode, stdout, stderr = run_command(
            f"xgrun examples/cluster_matmul.xg --external {external_file} --verbose"
        )
        
        assert returncode == 0, f"Execution failed: {stderr}"
        assert "sharding_plan" in stdout
        assert "('A', 0)" in stdout
        assert "('B', 1)" in stdout


def test_advanced_parallelism():
    """Test advanced parallelism example."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        external_values = {
            "A": {"tensor": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]], "dtype": "float32"},
            "B": {"tensor": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]], "dtype": "float32"}
        }
        external_file = tmpdir / "external.json"
        with open(external_file, 'w') as f:
            json.dump(external_values, f)
        
        returncode, stdout, stderr = run_command(
            f"xgrun examples/advanced_cluster_matmul.xg --external {external_file} --verbose"
        )
        
        assert returncode == 0, f"Execution failed: {stderr}"
        assert "Execution completed" in stdout
        assert "sharding_plan" in stdout


def test_different_targets():
    """Test compilation with different hardware targets."""
    targets = ["H100", "GB200", "A100", "CPU"]
    
    for target in targets:
        returncode, stdout, stderr = run_command(
            f"xgc examples/cluster_matmul.xg --target={target} --check-only --verbose"
        )
        
        assert returncode == 0, f"Target {target} failed: {stderr}"
        assert f"Target: {target}" in stdout


def test_error_handling():
    """Test CLI error handling."""
    returncode, stdout, stderr = run_command("xgc nonexistent.xg")
    assert returncode != 0
    assert "not found" in stderr
    
    returncode, stdout, stderr = run_command("xgc README.md")
    assert returncode != 0
    assert ".xg extension" in stderr
    
    returncode, stdout, stderr = run_command("xgrun nonexistent.xge")
    assert returncode != 0
    assert "not found" in stderr


if __name__ == "__main__":
    import sys
    
    print("=== XG CLI Functionality Tests ===")
    
    tests = [
        test_xgc_help,
        test_xgrun_help,
        test_xgc_compilation,
        test_xgc_check_only,
        test_xgrun_with_engine_file,
        test_xgrun_direct_source,
        test_parallelism_sharding_plans,
        test_advanced_parallelism,
        test_different_targets,
        test_error_handling,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
            print(f"✓ {test.__name__} PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("🎉 All CLI tests passed!")
