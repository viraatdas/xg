class XGError(Exception):
    """Base class for all XG language errors."""


class XGParseError(XGError):
    """Raised when the source program cannot be parsed."""


class XGTypeError(XGError):
    """Raised when type checking fails."""


class XGSafetyError(XGError):
    """Raised when safety rules are violated at compile time."""


class XGRuntimeError(XGError):
    """Raised when runtime validation fails."""
