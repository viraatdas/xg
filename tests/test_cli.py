"""
CLI integration tests for XG Language

Tests CLI tools integration with pytest framework.
"""

import pytest
import subprocess
import tempfile
import json
from pathlib import Path


def run_cli_command(cmd):
    """Helper to run CLI commands and return result."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def test_xgc_available():
    """Test that xgc command is available."""
    returncode, stdout, stderr = run_cli_command("xgc --help")
    assert returncode == 0
    assert "XG Language Compiler" in stdout


def test_xgrun_available():
    """Test that xgrun command is available."""
    returncode, stdout, stderr = run_cli_command("xgrun --help")
    assert returncode == 0
    assert "XG Language Runtime" in stdout


def test_cli_compilation_basic():
    """Test basic CLI compilation functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        returncode, stdout, stderr = run_cli_command(
            f"xgc examples/cluster_matmul.xg --out {tmpdir}/test.xge"
        )
        
        assert returncode == 0, f"Compilation failed: {stderr}"
        assert (tmpdir / "test.xge").exists()


def test_cli_execution_basic():
    """Test basic CLI execution functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        external_values = {
            "A": {"tensor": [[1.0, 2.0], [3.0, 4.0]], "dtype": "float32"},
            "B": {"tensor": [[5.0, 6.0], [7.0, 8.0]], "dtype": "float32"}
        }
        external_file = tmpdir / "external.json"
        with open(external_file, 'w') as f:
            json.dump(external_values, f)
        
        returncode, stdout, stderr = run_cli_command(
            f"xgrun examples/cluster_matmul.xg --external {external_file}"
        )
        
        assert returncode == 0, f"Execution failed: {stderr}"
        assert "Result tensor" in stdout


def test_cli_parallelism_metadata():
    """Test that CLI execution includes parallelism metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        external_values = {
            "A": {"tensor": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "dtype": "float32"},
            "B": {"tensor": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "dtype": "float32"}
        }
        external_file = tmpdir / "external.json"
        with open(external_file, 'w') as f:
            json.dump(external_values, f)
        
        returncode, stdout, stderr = run_cli_command(
            f"xgrun examples/cluster_matmul.xg --external {external_file} --verbose"
        )
        
        assert returncode == 0, f"Execution failed: {stderr}"
        assert "sharding_plan" in stdout
        assert "Target GPU" in stdout
        assert "Num GPUs" in stdout


def test_cli_error_handling():
    """Test CLI error handling."""
    returncode, stdout, stderr = run_cli_command("xgc nonexistent.xg")
    assert returncode != 0
    
    returncode, stdout, stderr = run_cli_command("xgrun nonexistent.xge")
    assert returncode != 0


def test_cli_check_only():
    """Test compilation check-only mode."""
    returncode, stdout, stderr = run_cli_command(
        "xgc examples/cluster_matmul.xg --check-only"
    )
    
    assert returncode == 0, f"Check failed: {stderr}"


def test_cli_different_targets():
    """Test CLI with different hardware targets."""
    targets = ["H100", "GB200", "A100"]
    
    for target in targets:
        returncode, stdout, stderr = run_cli_command(
            f"xgc examples/cluster_matmul.xg --target={target} --check-only"
        )
        
        assert returncode == 0, f"Target {target} failed: {stderr}"
