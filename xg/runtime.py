from __future__ import annotations

from typing import Dict, Optional

from .engine import Engine
from .interpreter import ExecutionResult, Interpreter, ResultValue


def run_engine(engine: Engine, external_values: Optional[Dict[str, object]] = None) -> ExecutionResult:
    interpreter = Interpreter(engine)
    return interpreter.run_main(external_values)


__all__ = ["run_engine", "ExecutionResult", "ResultValue"]
