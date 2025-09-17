from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


DimensionValue = Union[int, str]


@dataclass(frozen=True)
class Dimension:
    """Represents either a symbolic or literal tensor dimension."""

    value: DimensionValue

    @property
    def is_symbolic(self) -> bool:
        return isinstance(self.value, str)

    @property
    def is_literal(self) -> bool:
        return isinstance(self.value, int)

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return str(self.value)


class Type:
    """Base class for all types."""

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def __str__(self) -> str:  # pragma: no cover - debug helper
        raise NotImplementedError


@dataclass(frozen=True)
class PrimitiveType(Type):
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "primitive", "name": self.name}

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return self.name


@dataclass(frozen=True)
class TensorType(Type):
    dtype: PrimitiveType
    dims: Tuple[Dimension, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "tensor",
            "dtype": self.dtype.to_dict(),
            "dims": [dim.value for dim in self.dims],
        }

    def __str__(self) -> str:  # pragma: no cover
        dims = ", ".join(str(d.value) for d in self.dims)
        return f"Tensor[{self.dtype}, {dims}]"


@dataclass(frozen=True)
class ResultType(Type):
    inner: Type

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "result", "inner": self.inner.to_dict()}

    def __str__(self) -> str:  # pragma: no cover
        return f"Result[{self.inner}]"


@dataclass(frozen=True)
class VoidType(Type):
    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "void"}

    def __str__(self) -> str:  # pragma: no cover
        return "void"


@dataclass(frozen=True)
class ErrorType(Type):
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "error", "name": self.name}

    def __str__(self) -> str:  # pragma: no cover
        return f"Error[{self.name}]"


@dataclass(frozen=True)
class UnknownType(Type):
    name: str

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "unknown", "name": self.name}

    def __str__(self) -> str:  # pragma: no cover
        return f"Unknown[{self.name}]"


PrimitiveTable: Dict[str, PrimitiveType] = {}


def get_primitive(name: str) -> PrimitiveType:
    if name not in PrimitiveTable:
        PrimitiveTable[name] = PrimitiveType(name)
    return PrimitiveTable[name]


VOID = VoidType()


def dimension_from_token(token: Union[str, int]) -> Dimension:
    if isinstance(token, int):
        return Dimension(token)
    if isinstance(token, str) and token.isdigit():
        return Dimension(int(token))
    return Dimension(str(token))


def type_from_dict(payload: Dict[str, Any]) -> Type:
    kind = payload["kind"]
    if kind == "primitive":
        return get_primitive(payload["name"])
    if kind == "tensor":
        dtype = type_from_dict(payload["dtype"])
        if not isinstance(dtype, PrimitiveType):  # pragma: no cover - defensive
            raise TypeError("Tensor dtype must be primitive")
        dims = tuple(dimension_from_token(dim) for dim in payload["dims"])
        return TensorType(dtype=dtype, dims=dims)
    if kind == "result":
        return ResultType(inner=type_from_dict(payload["inner"]))
    if kind == "void":
        return VOID
    if kind == "error":
        return ErrorType(payload["name"])
    if kind == "unknown":
        return UnknownType(payload["name"])
    raise ValueError(f"Unknown type payload: {payload}")


def type_to_dict(typ: Type) -> Dict[str, Any]:
    return typ.to_dict()


def is_numeric(typ: Type) -> bool:
    return isinstance(typ, PrimitiveType) and typ.name in {"int64", "float32", "float64", "int32"}


def is_integer(typ: Type) -> bool:
    return isinstance(typ, PrimitiveType) and typ.name.startswith("int")


def is_float(typ: Type) -> bool:
    return isinstance(typ, PrimitiveType) and typ.name.startswith("float")
