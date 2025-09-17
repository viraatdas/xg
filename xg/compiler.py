from __future__ import annotations

from pathlib import Path
from typing import Optional

from .engine import Engine
from .parser import parse_source
from .typechecker import HardwareSettings, TypeChecker


def compile_source(source: str, *, target: Optional[str] = None, num_gpu: Optional[int] = None) -> Engine:
    program = parse_source(source)
    checker = TypeChecker(program)
    hardware = checker.run()
    effective_target = target if target is not None else hardware.target_gpu
    effective_num_gpu = num_gpu if num_gpu is not None else hardware.num_gpu
    if effective_num_gpu is None:
        effective_num_gpu = 1
    final_hardware = HardwareSettings(target_gpu=effective_target, num_gpu=effective_num_gpu)
    return Engine(program=program, hardware=final_hardware)


def compile_file(path: Path, *, target: Optional[str] = None, num_gpu: Optional[int] = None) -> Engine:
    source = path.read_text()
    return compile_source(source, target=target, num_gpu=num_gpu)
