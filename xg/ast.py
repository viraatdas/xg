from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .types import Type, type_from_dict, type_to_dict


class Node:
    def to_ir(self) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    @classmethod
    def from_ir(cls, payload: Dict[str, Any]) -> "Node":  # pragma: no cover - abstract
        raise NotImplementedError


class Statement(Node):
    pass


class Expression(Node):
    pass


@dataclass
class Program(Node):
    statements: List[Statement]

    def to_ir(self) -> Dict[str, Any]:
        return {"node": "Program", "statements": [stmt.to_ir() for stmt in self.statements]}

    @classmethod
    def from_ir(cls, payload: Dict[str, Any]) -> "Program":
        return cls(statements=[statement_from_ir(item) for item in payload["statements"]])


@dataclass
class Parameter:
    typ: Type
    name: str

    def to_ir(self) -> Dict[str, Any]:
        return {"type": type_to_dict(self.typ), "name": self.name}

    @classmethod
    def from_ir(cls, payload: Dict[str, Any]) -> "Parameter":
        return cls(typ=type_from_dict(payload["type"]), name=payload["name"])


@dataclass
class Assignment(Statement):
    target: str
    expression: Expression

    def to_ir(self) -> Dict[str, Any]:
        return {"node": "Assignment", "target": self.target, "expression": self.expression.to_ir()}

    @classmethod
    def from_ir(cls, payload: Dict[str, Any]) -> "Assignment":
        return cls(target=payload["target"], expression=expression_from_ir(payload["expression"]))


@dataclass
class Return(Statement):
    expression: Optional[Expression]

    def to_ir(self) -> Dict[str, Any]:
        return {
            "node": "Return",
            "expression": None if self.expression is None else self.expression.to_ir(),
        }

    @classmethod
    def from_ir(cls, payload: Dict[str, Any]) -> "Return":
        expr = payload["expression"]
        return cls(expression=None if expr is None else expression_from_ir(expr))


@dataclass
class If(Statement):
    condition: Expression
    then_body: List[Statement]
    else_body: Optional[List[Statement]] = None

    def to_ir(self) -> Dict[str, Any]:
        return {
            "node": "If",
            "condition": self.condition.to_ir(),
            "then": [stmt.to_ir() for stmt in self.then_body],
            "else": None if self.else_body is None else [stmt.to_ir() for stmt in self.else_body],
        }

    @classmethod
    def from_ir(cls, payload: Dict[str, Any]) -> "If":
        return cls(
            condition=expression_from_ir(payload["condition"]),
            then_body=[statement_from_ir(item) for item in payload["then"]],
            else_body=None
            if payload["else"] is None
            else [statement_from_ir(item) for item in payload["else"]],
        )


@dataclass
class WithContext:
    expression: Expression

    def to_ir(self) -> Dict[str, Any]:
        return {"expression": self.expression.to_ir()}

    @classmethod
    def from_ir(cls, payload: Dict[str, Any]) -> "WithContext":
        return cls(expression=expression_from_ir(payload["expression"]))


@dataclass
class With(Statement):
    contexts: List[WithContext]
    body: List[Statement]

    def to_ir(self) -> Dict[str, Any]:
        return {
            "node": "With",
            "contexts": [ctx.to_ir() for ctx in self.contexts],
            "body": [stmt.to_ir() for stmt in self.body],
        }

    @classmethod
    def from_ir(cls, payload: Dict[str, Any]) -> "With":
        return cls(
            contexts=[WithContext.from_ir(item) for item in payload["contexts"]],
            body=[statement_from_ir(item) for item in payload["body"]],
        )


@dataclass
class ExpressionStatement(Statement):
    expression: Expression

    def to_ir(self) -> Dict[str, Any]:
        return {"node": "ExpressionStatement", "expression": self.expression.to_ir()}

    @classmethod
    def from_ir(cls, payload: Dict[str, Any]) -> "ExpressionStatement":
        return cls(expression=expression_from_ir(payload["expression"]))


@dataclass
class FunctionDef(Statement):
    name: str
    params: List[Parameter]
    return_type: Optional[Type]
    body: List[Statement]

    def to_ir(self) -> Dict[str, Any]:
        return {
            "node": "FunctionDef",
            "name": self.name,
            "params": [param.to_ir() for param in self.params],
            "return_type": None if self.return_type is None else type_to_dict(self.return_type),
            "body": [stmt.to_ir() for stmt in self.body],
        }

    @classmethod
    def from_ir(cls, payload: Dict[str, Any]) -> "FunctionDef":
        return cls(
            name=payload["name"],
            params=[Parameter.from_ir(item) for item in payload["params"]],
            return_type=None if payload["return_type"] is None else type_from_dict(payload["return_type"]),
            body=[statement_from_ir(item) for item in payload["body"]],
        )


@dataclass
class Number(Expression):
    value: Union[int, float]

    def to_ir(self) -> Dict[str, Any]:
        return {"node": "Number", "value": self.value}

    @classmethod
    def from_ir(cls, payload: Dict[str, Any]) -> "Number":
        return cls(value=payload["value"])


@dataclass
class Identifier(Expression):
    name: str

    def to_ir(self) -> Dict[str, Any]:
        return {"node": "Identifier", "name": self.name}

    @classmethod
    def from_ir(cls, payload: Dict[str, Any]) -> "Identifier":
        return cls(name=payload["name"])


@dataclass
class BinaryOp(Expression):
    left: Expression
    op: str
    right: Expression

    def to_ir(self) -> Dict[str, Any]:
        return {"node": "BinaryOp", "op": self.op, "left": self.left.to_ir(), "right": self.right.to_ir()}

    @classmethod
    def from_ir(cls, payload: Dict[str, Any]) -> "BinaryOp":
        return cls(
            left=expression_from_ir(payload["left"]),
            op=payload["op"],
            right=expression_from_ir(payload["right"]),
        )


@dataclass
class Call(Expression):
    func: Expression
    args: List[Expression] = field(default_factory=list)
    kwargs: List[Tuple[str, Expression]] = field(default_factory=list)

    def to_ir(self) -> Dict[str, Any]:
        return {
            "node": "Call",
            "func": self.func.to_ir(),
            "args": [arg.to_ir() for arg in self.args],
            "kwargs": [{"name": name, "value": value.to_ir()} for name, value in self.kwargs],
        }

    @classmethod
    def from_ir(cls, payload: Dict[str, Any]) -> "Call":
        return cls(
            func=expression_from_ir(payload["func"]),
            args=[expression_from_ir(item) for item in payload["args"]],
            kwargs=[(item["name"], expression_from_ir(item["value"])) for item in payload["kwargs"]],
        )


ExpressionDispatch = {
    "Number": Number,
    "Identifier": Identifier,
    "BinaryOp": BinaryOp,
    "Call": Call,
}


StatementDispatch = {
    "Assignment": Assignment,
    "Return": Return,
    "If": If,
    "With": With,
    "ExpressionStatement": ExpressionStatement,
    "FunctionDef": FunctionDef,
}


def expression_from_ir(payload: Dict[str, Any]) -> Expression:
    node_type = payload["node"]
    if node_type not in ExpressionDispatch:
        raise ValueError(f"Unknown expression node: {node_type}")
    cls = ExpressionDispatch[node_type]
    return cls.from_ir(payload)


def statement_from_ir(payload: Dict[str, Any]) -> Statement:
    node_type = payload["node"]
    if node_type not in StatementDispatch:
        raise ValueError(f"Unknown statement node: {node_type}")
    cls = StatementDispatch[node_type]
    return cls.from_ir(payload)
