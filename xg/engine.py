from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from . import ast
from .typechecker import HardwareSettings, TypeChecker


@dataclass
class Engine:
    program: ast.Program
    hardware: HardwareSettings

    def to_dict(self) -> Dict[str, object]:
        return {
            "target_gpu": self.hardware.target_gpu,
            "num_gpu": self.hardware.num_gpu,
            "program": self.program.to_ir(),
        }

    def save(self, path: Path) -> None:
        payload = self.to_dict()
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Engine":
        payload = json.loads(path.read_text())
        program = ast.Program.from_ir(payload["program"])
        hardware = HardwareSettings(target_gpu=payload.get("target_gpu"), num_gpu=payload.get("num_gpu"))
        return cls(program=program, hardware=hardware)

    def function_map(self) -> Dict[str, ast.FunctionDef]:
        functions: Dict[str, ast.FunctionDef] = {}
        for stmt in self.program.statements:
            if isinstance(stmt, ast.FunctionDef):
                functions[stmt.name] = stmt
        return functions

    def ensure_checked(self) -> None:
        checker = TypeChecker(self.program)
        checker.run()
