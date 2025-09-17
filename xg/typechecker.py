from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from . import ast
from .errors import XGSafetyError, XGTypeError
from .types import (
    Dimension,
    ErrorType,
    PrimitiveType,
    ResultType,
    TensorType,
    Type,
    UnknownType,
    VOID,
    get_primitive,
    is_numeric,
)


BOOL = get_primitive("bool")
INT64 = get_primitive("int64")
FLOAT32 = get_primitive("float32")


KNOWN_ERRORS = {"DivByZero"}


@dataclass
class HardwareSettings:
    target_gpu: Optional[str] = None
    num_gpu: Optional[int] = None


@dataclass
class FunctionContext:
    name: str
    env: Dict[str, Type]
    return_type: Type
    guard_nonzero: Set[str] = field(default_factory=set)
    dim_bindings: Dict[str, Dimension] = field(default_factory=dict)


class TypeChecker:
    def __init__(self, program: ast.Program):
        self.program = program
        self.functions: Dict[str, ast.FunctionDef] = {}
        self.hardware = HardwareSettings()

    def run(self) -> HardwareSettings:
        self._collect_top_level()
        if "main" not in self.functions:
            raise XGTypeError("Program must define a main function")
        for func in self.functions.values():
            self._check_function(func)
        return self.hardware

    def _collect_top_level(self) -> None:
        for stmt in self.program.statements:
            if isinstance(stmt, ast.FunctionDef):
                if stmt.name in self.functions:
                    raise XGTypeError(f"Function {stmt.name} defined multiple times")
                self.functions[stmt.name] = stmt
            elif isinstance(stmt, ast.Assignment):
                self._process_hardware_assignment(stmt)
            elif isinstance(stmt, ast.ExpressionStatement):
                # Evaluate to ensure expression is well-formed even at top-level
                self._infer_expression(stmt.expression, FunctionContext(name="<top>", env={}, return_type=VOID))

    def _process_hardware_assignment(self, stmt: ast.Assignment) -> None:
        if stmt.target == "TARGET_GPU":
            value = self._extract_identifier_literal(stmt.expression)
            self.hardware.target_gpu = value
        elif stmt.target == "NUM_GPU":
            if not isinstance(stmt.expression, ast.Number) or not isinstance(stmt.expression.value, int):
                raise XGTypeError("NUM_GPU must be assigned an integer literal")
            self.hardware.num_gpu = stmt.expression.value
        else:
            # For other assignments we simply ensure expression can be typed
            self._infer_expression(stmt.expression, FunctionContext(name="<top>", env={}, return_type=VOID))

    def _extract_identifier_literal(self, expr: ast.Expression) -> str:
        if isinstance(expr, ast.Identifier):
            return expr.name
        raise XGTypeError("Hardware identifiers must be simple names")

    def _check_function(self, func: ast.FunctionDef) -> None:
        env = {param.name: param.typ for param in func.params}
        return_type = func.return_type or VOID
        context = FunctionContext(name=func.name, env=env, return_type=return_type)
        for stmt in func.body:
            self._check_statement(stmt, context)
        if return_type is not VOID and func.name == "main":
            # Ensure main returns something when typed
            pass

    def _check_statement(self, stmt: ast.Statement, context: FunctionContext) -> None:
        if isinstance(stmt, ast.Assignment):
            expr_type = self._infer_expression(stmt.expression, context)
            existing = context.env.get(stmt.target)
            if existing is not None:
                self._ensure_assignable(existing, expr_type, context, stmt.expression)
            context.env[stmt.target] = expr_type if existing is None else existing
        elif isinstance(stmt, ast.Return):
            if stmt.expression is None:
                if context.return_type is not VOID:
                    raise XGTypeError(f"Function {context.name} must return {context.return_type}")
                return
            expr_type = self._infer_expression(stmt.expression, context)
            self._ensure_assignable(context.return_type, expr_type, context, stmt.expression)
        elif isinstance(stmt, ast.If):
            cond_type = self._infer_expression(stmt.condition, context)
            if cond_type != BOOL:
                raise XGTypeError("if condition must be boolean")
            for body_stmt in stmt.then_body:
                self._check_statement(body_stmt, context)
            if stmt.else_body:
                for body_stmt in stmt.else_body:
                    self._check_statement(body_stmt, context)
            self._maybe_register_guard(stmt, context)
        elif isinstance(stmt, ast.With):
            self._check_with(stmt, context)
        elif isinstance(stmt, ast.ExpressionStatement):
            self._infer_expression(stmt.expression, context)
        else:  # pragma: no cover - defensive
            raise XGTypeError(f"Unsupported statement: {stmt}")

    def _maybe_register_guard(self, stmt: ast.If, context: FunctionContext) -> None:
        cond = stmt.condition
        guard_name: Optional[str] = None
        if isinstance(cond, ast.BinaryOp) and cond.op == "==":
            if isinstance(cond.left, ast.Identifier) and isinstance(cond.right, ast.Number) and cond.right.value == 0:
                guard_name = cond.left.name
            elif isinstance(cond.right, ast.Identifier) and isinstance(cond.left, ast.Number) and cond.left.value == 0:
                guard_name = cond.right.name
        if guard_name and self._branch_returns_error(stmt.then_body):
            context.guard_nonzero.add(guard_name)

    def _branch_returns_error(self, body: List[ast.Statement]) -> bool:
        if len(body) != 1:
            return False
        stmt = body[0]
        if not isinstance(stmt, ast.Return) or stmt.expression is None:
            return False
        expr = stmt.expression
        return isinstance(expr, ast.Identifier) and expr.name in KNOWN_ERRORS

    def _check_with(self, stmt: ast.With, context: FunctionContext) -> None:
        for ctx in stmt.contexts:
            if not isinstance(ctx.expression, ast.Call):
                raise XGTypeError("with statements must use shard(...) contexts")
            call = ctx.expression
            if not isinstance(call.func, ast.Identifier) or call.func.name != "shard":
                raise XGTypeError("Unsupported with context")
            if len(call.args) != 1:
                raise XGTypeError("shard expects a tensor variable")
            tensor_expr = call.args[0]
            tensor_type = self._infer_expression(tensor_expr, context)
            if not isinstance(tensor_type, TensorType):
                raise XGTypeError("shard can only be applied to tensors")
            dim_kw = None
            for name, value in call.kwargs:
                if name == "dim":
                    dim_kw = value
            if dim_kw is None:
                raise XGTypeError("shard requires a dim keyword argument")
            if not isinstance(dim_kw, ast.Number) or not isinstance(dim_kw.value, int):
                raise XGTypeError("shard dim must be an integer literal")
            dim_index = dim_kw.value
            if dim_index < 0 or dim_index >= len(tensor_type.dims):
                raise XGTypeError("shard dim out of range for tensor")
        for body_stmt in stmt.body:
            self._check_statement(body_stmt, context)

    def _infer_expression(self, expr: ast.Expression, context: FunctionContext) -> Type:
        if isinstance(expr, ast.Number):
            return FLOAT32 if isinstance(expr.value, float) else INT64
        if isinstance(expr, ast.Identifier):
            if expr.name in context.env:
                return context.env[expr.name]
            if expr.name in KNOWN_ERRORS:
                return ErrorType(expr.name)
            return UnknownType(expr.name)
        if isinstance(expr, ast.BinaryOp):
            return self._infer_binary(expr, context)
        if isinstance(expr, ast.Call):
            return self._infer_call(expr, context)
        raise XGTypeError(f"Unsupported expression: {expr}")

    def _infer_binary(self, expr: ast.BinaryOp, context: FunctionContext) -> Type:
        if expr.op == "==":
            left = self._infer_expression(expr.left, context)
            right = self._infer_expression(expr.right, context)
            self._ensure_comparable(left, right, expr)
            return BOOL
        if expr.op == "/":
            right_type = self._infer_expression(expr.right, context)
            self._enforce_division_safety(expr.right, context)
            left_type = self._infer_expression(expr.left, context)
            if left_type != right_type or left_type != FLOAT32:
                raise XGTypeError("Division requires float32 operands with explicit safety handling")
            return FLOAT32
        if expr.op == "@":
            left_type = self._infer_expression(expr.left, context)
            right_type = self._infer_expression(expr.right, context)
            if not isinstance(left_type, TensorType) or not isinstance(right_type, TensorType):
                raise XGTypeError("@ operator requires tensors")
            if left_type.dtype != right_type.dtype:
                raise XGTypeError("Tensor dtypes must match for matmul")
            if not left_type.dims or not right_type.dims:
                raise XGTypeError("Tensors must have rank >= 2 for matmul")
            inner_left = left_type.dims[-1]
            inner_right = right_type.dims[0]
            if inner_left.is_symbolic and inner_right.is_symbolic and inner_left.value != inner_right.value:
                raise XGTypeError("Tensor inner dimensions must align for matmul")
            self._unify_dimensions(inner_left, inner_right, context, expr)
            result_dims = left_type.dims[:-1] + right_type.dims[1:]
            return TensorType(dtype=left_type.dtype, dims=result_dims)
        if expr.op in {"+", "-", "*"}:
            left_type = self._infer_expression(expr.left, context)
            right_type = self._infer_expression(expr.right, context)
            if left_type != right_type or not is_numeric(left_type):
                raise XGTypeError(f"Operator {expr.op} requires matching numeric types")
            return left_type
        raise XGTypeError(f"Unsupported operator {expr.op}")

    def _infer_call(self, expr: ast.Call, context: FunctionContext) -> Type:
        if isinstance(expr.func, ast.Identifier):
            name = expr.func.name
            if name == "Ok":
                if len(expr.args) != 1 or expr.kwargs:
                    raise XGTypeError("Ok expects a single argument")
                inner = self._infer_expression(expr.args[0], context)
                return ResultType(inner)
            if name == "shard":
                raise XGTypeError("shard context may only appear inside with blocks")
            if name in self.functions:
                return self._check_function_call(name, expr, context)
        raise XGTypeError("Unknown function call")

    def _check_function_call(self, name: str, call: ast.Call, context: FunctionContext) -> Type:
        func_def = self.functions[name]
        if len(call.args) != len(func_def.params):
            raise XGTypeError(f"Function {name} expects {len(func_def.params)} arguments")
        if call.kwargs:
            raise XGTypeError("Keyword arguments are not supported for function calls")
        for arg_expr, param in zip(call.args, func_def.params):
            arg_type = self._infer_expression(arg_expr, context)
            self._ensure_assignable(param.typ, arg_type, context, arg_expr)
        return func_def.return_type or VOID

    def _ensure_assignable(self, expected: Type, actual: Type, context: FunctionContext, node: ast.Expression) -> None:
        if isinstance(actual, UnknownType):
            context.env[actual.name] = expected
            return
        if expected == actual:
            return
        if isinstance(expected, PrimitiveType) and isinstance(actual, PrimitiveType) and expected == actual:
            return
        if isinstance(expected, TensorType) and isinstance(actual, TensorType):
            self._ensure_tensor_compatible(expected, actual, context, node)
            return
        if isinstance(expected, ResultType):
            if isinstance(actual, ResultType):
                self._ensure_assignable(expected.inner, actual.inner, context, node)
                return
            if isinstance(actual, ErrorType):
                return
        if expected is VOID and actual is VOID:
            return
        raise XGTypeError(f"Cannot assign {actual} to {expected}")

    def _ensure_tensor_compatible(
        self, expected: TensorType, actual: TensorType, context: FunctionContext, node: ast.Expression
    ) -> None:
        if expected.dtype != actual.dtype:
            raise XGTypeError("Tensor dtypes must match")
        if len(expected.dims) != len(actual.dims):
            raise XGTypeError("Tensor ranks must match")
        for exp_dim, act_dim in zip(expected.dims, actual.dims):
            self._unify_dimensions(exp_dim, act_dim, context, node)

    def _unify_dimensions(self, expected: Dimension, actual: Dimension, context: FunctionContext, node: ast.Expression) -> None:
        resolved_expected = self._resolve_dimension(expected, context)
        resolved_actual = self._resolve_dimension(actual, context)
        if resolved_expected.is_literal and resolved_actual.is_literal:
            if resolved_expected.value != resolved_actual.value:
                raise XGTypeError("Tensor dimensions are incompatible")
            return
        if resolved_expected.is_literal and resolved_actual.is_symbolic:
            context.dim_bindings[resolved_actual.value] = resolved_expected
            return
        if resolved_expected.is_symbolic and resolved_actual.is_literal:
            context.dim_bindings[resolved_expected.value] = resolved_actual
            return
        if resolved_expected.is_symbolic and resolved_actual.is_symbolic:
            context.dim_bindings[resolved_actual.value] = resolved_expected
            return
        raise XGTypeError("Unsupported dimension combination")

    def _resolve_dimension(self, dim: Dimension, context: FunctionContext) -> Dimension:
        current = dim
        visited: Set[str] = set()
        while current.is_symbolic and current.value in context.dim_bindings:
            if current.value in visited:
                break
            visited.add(current.value)
            current = context.dim_bindings[current.value]
        return current

    def _ensure_comparable(self, left: Type, right: Type, expr: ast.BinaryOp) -> None:
        if isinstance(left, PrimitiveType) and isinstance(right, PrimitiveType):
            if left == right:
                return
            if is_numeric(left) and is_numeric(right):
                return
        if isinstance(left, TensorType) and isinstance(right, TensorType):
            if left.dtype != right.dtype or len(left.dims) != len(right.dims):
                raise XGTypeError("Tensor comparison requires matching shape")
            return
        if isinstance(left, UnknownType) or isinstance(right, UnknownType):
            return
        raise XGTypeError(f"Cannot compare types {left} and {right}")

    def _enforce_division_safety(self, denominator: ast.Expression, context: FunctionContext) -> None:
        if isinstance(denominator, ast.Number):
            if denominator.value == 0:
                raise XGSafetyError("Division by zero is not allowed")
            return
        if isinstance(denominator, ast.Identifier):
            if denominator.name in context.guard_nonzero:
                return
            raise XGSafetyError(f"Division involving {denominator.name} requires explicit zero check")
        raise XGSafetyError("Division requires explicit error handling")


def type_check(program: ast.Program) -> HardwareSettings:
    checker = TypeChecker(program)
    return checker.run()
