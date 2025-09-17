from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from . import ast
from .engine import Engine
from .errors import XGRuntimeError
from .types import ResultType, TensorType, Type, VOID


KNOWN_ERRORS = {"DivByZero"}


@dataclass
class ResultValue:
    ok: bool
    value: Any = None
    error: Optional[str] = None

    @classmethod
    def ok(cls, value: Any) -> "ResultValue":
        return cls(ok=True, value=value)

    @classmethod
    def error(cls, name: str) -> "ResultValue":
        return cls(ok=False, error=name)


@dataclass
class ExecutionResult:
    value: Any
    metadata: Dict[str, Any]


@dataclass
class RuntimeContext:
    engine: Engine
    external: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class CallFrame:
    function: ast.FunctionDef
    return_type: Type
    env: Dict[str, Any] = field(default_factory=dict)
    dim_bindings: Dict[str, int] = field(default_factory=dict)


class ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value


class Interpreter:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.functions = engine.function_map()
        self.hardware = engine.hardware
        if "main" not in self.functions:
            raise XGRuntimeError("Program is missing a main function")

    def run_main(self, external_values: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        metadata: Dict[str, Any] = {
            "target_gpu": self.hardware.target_gpu,
            "num_gpu": self.hardware.num_gpu,
            "sharding_plan": [],
        }
        context = RuntimeContext(engine=self.engine, external=external_values or {}, metadata=metadata)
        result = self._call_function("main", [], context)
        return ExecutionResult(value=result, metadata=metadata)

    def _call_function(self, name: str, args: Sequence[Any], context: RuntimeContext) -> Any:
        if name not in self.functions:
            raise XGRuntimeError(f"Unknown function {name}")
        func = self.functions[name]
        expected_params = func.params
        if len(args) != len(expected_params):
            raise XGRuntimeError(f"Function {name} expects {len(expected_params)} arguments")
        frame = CallFrame(function=func, return_type=func.return_type or VOID)
        for param, arg in zip(expected_params, args):
            validated = self._validate_runtime_type(param.typ, arg, frame.dim_bindings)
            frame.env[param.name] = validated
        try:
            for stmt in func.body:
                self._execute_statement(stmt, frame, context)
        except ReturnSignal as signal:
            result = signal.value
        else:
            result = None
        return self._validate_return_value(frame.return_type, result, frame.dim_bindings)

    def _execute_statement(self, stmt: ast.Statement, frame: CallFrame, context: RuntimeContext) -> None:
        if isinstance(stmt, ast.Assignment):
            value = self._evaluate_expression(stmt.expression, frame, context)
            frame.env[stmt.target] = value
        elif isinstance(stmt, ast.Return):
            if stmt.expression is None:
                raise ReturnSignal(None)
            value = self._evaluate_expression(stmt.expression, frame, context)
            raise ReturnSignal(value)
        elif isinstance(stmt, ast.If):
            condition = self._evaluate_expression(stmt.condition, frame, context)
            if not isinstance(condition, bool):
                raise XGRuntimeError("if condition must be boolean")
            body = stmt.then_body if condition else (stmt.else_body or [])
            for body_stmt in body:
                self._execute_statement(body_stmt, frame, context)
        elif isinstance(stmt, ast.With):
            self._execute_with(stmt, frame, context)
        elif isinstance(stmt, ast.ExpressionStatement):
            self._evaluate_expression(stmt.expression, frame, context)
        else:  # pragma: no cover - defensive
            raise XGRuntimeError(f"Unsupported statement {stmt}")

    def _execute_with(self, stmt: ast.With, frame: CallFrame, context: RuntimeContext) -> None:
        for ctx in stmt.contexts:
            if not isinstance(ctx.expression, ast.Call):
                raise XGRuntimeError("with context must be a call")
            call = ctx.expression
            if not isinstance(call.func, ast.Identifier) or call.func.name != "shard":
                raise XGRuntimeError("Unsupported with context")
            if len(call.args) != 1:
                raise XGRuntimeError("shard expects exactly one tensor argument")
            tensor_expr = call.args[0]
            if not isinstance(tensor_expr, ast.Identifier):
                raise XGRuntimeError("shard argument must be an identifier")
            tensor_value = self._evaluate_expression(tensor_expr, frame, context)
            if not isinstance(tensor_value, torch.Tensor):
                raise XGRuntimeError("shard can only target tensors")
            dim_kw = None
            for name, value_expr in call.kwargs:
                if name == "dim":
                    dim_kw = value_expr
            if dim_kw is None:
                raise XGRuntimeError("shard requires a dim keyword argument")
            dim_value = self._evaluate_expression(dim_kw, frame, context)
            if not isinstance(dim_value, int):
                raise XGRuntimeError("shard dim must be an integer")
            if dim_value < 0 or dim_value >= tensor_value.dim():
                raise XGRuntimeError("shard dim out of range")
            entry = (tensor_expr.name, dim_value)
            plan = context.metadata.setdefault("sharding_plan", [])
            if entry not in plan:
                plan.append(entry)
        for body_stmt in stmt.body:
            self._execute_statement(body_stmt, frame, context)

    def _evaluate_expression(self, expr: ast.Expression, frame: CallFrame, context: RuntimeContext) -> Any:
        if isinstance(expr, ast.Number):
            return expr.value
        if isinstance(expr, ast.Identifier):
            if expr.name in frame.env:
                return frame.env[expr.name]
            if expr.name in context.external:
                return context.external[expr.name]
            if expr.name in KNOWN_ERRORS:
                return ResultValue.error(expr.name)
            raise XGRuntimeError(f"Unknown identifier {expr.name}")
        if isinstance(expr, ast.BinaryOp):
            left = self._evaluate_expression(expr.left, frame, context)
            right = self._evaluate_expression(expr.right, frame, context)
            if expr.op == "+":
                return left + right
            if expr.op == "-":
                return left - right
            if expr.op == "*":
                return left * right
            if expr.op == "/":
                return left / right
            if expr.op == "@":
                if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                    raise XGRuntimeError("@ operator requires tensors")
                return left @ right
            if expr.op == "==":
                return left == right
            raise XGRuntimeError(f"Unsupported operator {expr.op}")
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Identifier) and expr.func.name == "Ok":
                if len(expr.args) != 1:
                    raise XGRuntimeError("Ok expects one argument")
                value = self._evaluate_expression(expr.args[0], frame, context)
                return ResultValue.ok(value)
            if isinstance(expr.func, ast.Identifier):
                func_name = expr.func.name
                args = [self._evaluate_expression(arg, frame, context) for arg in expr.args]
                if func_name == "shard":
                    raise XGRuntimeError("shard cannot be called as a function")
                if expr.kwargs:
                    raise XGRuntimeError("Keyword arguments are not supported in calls")
                return self._call_function(func_name, args, context)
            raise XGRuntimeError("Unsupported call expression")
        raise XGRuntimeError(f"Unsupported expression {expr}")

    def _validate_return_value(self, expected: Type, value: Any, dim_bindings: Dict[str, int]) -> Any:
        if expected is VOID:
            return None
        if value is None:
            raise XGRuntimeError("Missing return value")
        return self._validate_runtime_type(expected, value, dim_bindings)

    def _validate_runtime_type(self, expected: Type, value: Any, dim_bindings: Dict[str, int]) -> Any:
        if expected is VOID:
            return value
        if isinstance(expected, ResultType):
            if not isinstance(value, ResultValue):
                raise XGRuntimeError("Expected a Result value")
            if value.ok:
                inner = self._validate_runtime_type(expected.inner, value.value, dim_bindings)
                return ResultValue.ok(inner)
            return ResultValue.error(value.error or "UnknownError")
        if isinstance(expected, TensorType):
            if not isinstance(value, torch.Tensor):
                raise XGRuntimeError("Expected tensor value")
            self._validate_tensor(expected, value, dim_bindings)
            return value
        if isinstance(expected, Type):
            primitive_name = getattr(expected, "name", None)
            if primitive_name in {"int64", "int32"}:
                if not isinstance(value, int):
                    raise XGRuntimeError("Expected integer value")
                return int(value)
            if primitive_name in {"float32", "float64"}:
                if not isinstance(value, (int, float)):
                    raise XGRuntimeError("Expected floating point value")
                return float(value)
            if primitive_name == "bool":
                if not isinstance(value, bool):
                    raise XGRuntimeError("Expected boolean value")
                return value
        return value

    def _validate_tensor(self, expected: TensorType, value: torch.Tensor, dim_bindings: Dict[str, int]) -> None:
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int64": torch.int64,
            "int32": torch.int32,
        }
        expected_dtype = dtype_map.get(expected.dtype.name)
        if expected_dtype is None:
            raise XGRuntimeError(f"Unsupported tensor dtype {expected.dtype.name}")
        if value.dtype != expected_dtype:
            raise XGRuntimeError("Tensor dtype mismatch")
        if value.dim() != len(expected.dims):
            raise XGRuntimeError("Tensor rank mismatch")
        for size, dim in zip(value.shape, expected.dims):
            if dim.is_literal:
                if size != dim.value:
                    raise XGRuntimeError("Tensor dimension mismatch")
            else:
                bound = dim_bindings.get(dim.value)
                if bound is None:
                    dim_bindings[dim.value] = size
                elif bound != size:
                    raise XGRuntimeError("Tensor dimension binding conflict")


def run_engine(engine: Engine, external_values: Optional[Dict[str, Any]] = None) -> ExecutionResult:
    interpreter = Interpreter(engine)
    return interpreter.run_main(external_values)
