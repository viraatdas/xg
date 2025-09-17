import json
from pathlib import Path
import textwrap

import pytest

torch = pytest.importorskip("torch")

from xg.compiler import compile_source, compile_file
from xg.errors import XGTypeError, XGSafetyError, XGRuntimeError
from xg.runtime import ExecutionResult, run_engine


def compile_and_run(source: str, *, target: str | None = None, num_gpu: int | None = None, external=None) -> ExecutionResult:
    engine = compile_source(source, target=target, num_gpu=num_gpu)
    return run_engine(engine, external_values=external or {})


def test_add_function_executes_integer_addition():
    source = textwrap.dedent(
        """
        TARGET_GPU = H100
        NUM_GPU = 8

        def add(int64 a, int64 b) -> int64 {
            return a + b
        }

        def main() -> int64 {
            return add(10, 20)
        }
        """
    )

    result = compile_and_run(source)
    assert result.value == 30
    assert result.metadata["target_gpu"] == "H100"
    assert result.metadata["num_gpu"] == 8


def test_type_mismatch_is_caught_at_compile_time():
    source = textwrap.dedent(
        """
        def bad(int64 a, float32 b) -> int64 {
            return a + b
        }

        def main() -> int64 {
            return bad(1, 2.0)
        }
        """
    )

    with pytest.raises(XGTypeError):
        compile_source(source)


def test_matrix_multiplication_type_checks_shapes():
    source = textwrap.dedent(
        """
        def multMat(Tensor[float32, M, K] a, Tensor[float32, K, N] b) -> Tensor[float32, M, N] {
            return a @ b
        }

        def main() -> Tensor[float32, 2, 3] {
            return multMat(A, B)
        }
        """
    )

    a = torch.randn(2, 4, dtype=torch.float32)
    b = torch.randn(4, 3, dtype=torch.float32)

    result = compile_and_run(source, external={"A": a, "B": b})
    assert torch.allclose(result.value, a @ b)


def test_matrix_multiplication_incompatible_shapes_fail_compile():
    source = textwrap.dedent(
        """
        def multMat(Tensor[float32, M, K] a, Tensor[float32, J, N] b) -> Tensor[float32, M, N] {
            return a @ b
        }

        def main() -> Tensor[float32, 2, 3] {
            return multMat(A, B)
        }
        """
    )

    with pytest.raises(XGTypeError):
        compile_source(source)


def test_division_by_zero_literal_is_rejected():
    source = textwrap.dedent(
        """
        def unsafe(float32 a) -> float32 {
            return a / 0.0
        }

        def main() -> float32 {
            return unsafe(1.0)
        }
        """
    )

    with pytest.raises(XGSafetyError):
        compile_source(source)


def test_safe_division_requires_explicit_handling():
    source = textwrap.dedent(
        """
        def safe_div(float32 a, float32 b) -> Result[float32] {
            if b == 0 {
                return DivByZero
            }
            return Ok(a / b)
        }

        def main() -> Result[float32] {
            return safe_div(4.0, 2.0)
        }
        """
    )

    result = compile_and_run(source)
    assert result.value.ok is True
    assert pytest.approx(result.value.value) == 2.0

    engine = compile_source(source)
    zero_result = run_engine(engine, external_values={})
    assert zero_result.value.ok is True

    # A follow-up run where denominator is zero should return an error Result
    def zero_main_source(base: str) -> str:
        return base.replace("return safe_div(4.0, 2.0)", "return safe_div(4.0, 0.0)")

    engine_zero = compile_source(zero_main_source(source))
    zero_exec = run_engine(engine_zero)
    assert zero_exec.value.ok is False
    assert zero_exec.value.error == "DivByZero"


def test_runtime_shape_mismatch_raises():
    source = textwrap.dedent(
        """
        def multMat(Tensor[float32, M, K] a, Tensor[float32, K, N] b) -> Tensor[float32, M, N] {
            return a @ b
        }

        def main() -> Tensor[float32, 2, 3] {
            return multMat(A, B)
        }
        """
    )

    a = torch.randn(2, 4, dtype=torch.float32)
    b = torch.randn(5, 3, dtype=torch.float32)

    engine = compile_source(source)
    with pytest.raises(XGRuntimeError):
        run_engine(engine, external_values={"A": a, "B": b})


def test_shard_block_records_plan():
    source = textwrap.dedent(
        """
        TARGET_GPU = GB200
        NUM_GPU = 16

        def compute(Tensor[float32, M, K] A, Tensor[float32, K, N] B) -> Tensor[float32, M, N] {
            with shard(A, dim=0), shard(B, dim=1) {
                return A @ B
            }
        }

        def main() -> Tensor[float32, 2, 3] {
            return compute(A, B)
        }
        """
    )

    a = torch.randn(2, 4, dtype=torch.float32)
    b = torch.randn(4, 3, dtype=torch.float32)

    result = compile_and_run(source, external={"A": a, "B": b})
    assert torch.allclose(result.value, a @ b)
    assert ("A", 0) in result.metadata["sharding_plan"]
    assert ("B", 1) in result.metadata["sharding_plan"]
    assert result.metadata["target_gpu"] == "GB200"
    assert result.metadata["num_gpu"] == 16


def test_cli_round_trip(tmp_path: Path):
    source_path = tmp_path / "model.xg"
    engine_path = tmp_path / "engine.xge"
    source = textwrap.dedent(
        """
        def add(int64 a, int64 b) -> int64 {
            return a + b
        }

        def main() -> int64 {
            return add(2, 3)
        }
        """
    )
    source_path.write_text(source)

    engine = compile_file(source_path, target="H100", num_gpu=1)
    assert engine_path.exists() is False

    engine.save(engine_path)
    assert engine_path.exists()

    loaded = engine.__class__.load(engine_path)
    result = run_engine(loaded)
    assert result.value == 5

    payload = json.loads(engine_path.read_text())
    assert payload["target_gpu"] == "H100"
    assert payload["num_gpu"] == 1


def test_cluster_matmul_example_executes():
    example_path = Path(__file__).resolve().parent.parent / "examples" / "cluster_matmul.xg"
    engine = compile_file(example_path)

    a = torch.randn(2, 4, dtype=torch.float32)
    b = torch.randn(4, 2, dtype=torch.float32)

    result = run_engine(engine, external_values={"A": a, "B": b})

    assert torch.allclose(result.value, a @ b)
    assert result.metadata["target_gpu"] == "GB200"
    assert result.metadata["num_gpu"] == 8
    assert ("A", 0) in result.metadata["sharding_plan"]
    assert ("B", 1) in result.metadata["sharding_plan"]


def test_function_map_caching():
    source = textwrap.dedent(
        """
        def add(int64 a, int64 b) -> int64 {
            return a + b
        }

        def main() -> int64 {
            return add(1, 2)
        }
        """
    )
    engine = compile_source(source)
    
    map1 = engine.function_map()
    map2 = engine.function_map()
    
    assert map1 is map2
    assert "add" in map1
    assert "main" in map1
