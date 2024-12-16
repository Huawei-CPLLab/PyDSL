import ast
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

import pytest

from pydsl.compiler import ToMLIR
from pydsl.frontend import compile
from pydsl.macro import CallMacro, Compiled, Evaluated, Uncompiled
from pydsl.protocols import SubtreeOut, ToMLIRBase
from pydsl.type import Number, UInt32, get_operator, supports_operator


def test_Compiled_ArgCompiler():
    @CallMacro.generate()
    def bind(visitor: ToMLIRBase, a: Compiled) -> None:
        assert isinstance(a, Number), "a wasn't compiled"

    @compile()
    def _() -> None:
        bind(5)


def test_Evaluated_ArgCompiler():
    @CallMacro.generate()
    def bind(visitor: ToMLIRBase, a: Evaluated) -> None:
        assert a == 5, "a wasn't evaluated"

    @compile()
    def _() -> None:
        bind(5)


def test_Uncompiled_ArgCompiler():
    @CallMacro.generate()
    def bind(visitor: ToMLIRBase, a: Uncompiled) -> None:
        assert isinstance(a, ast.Constant), "a wasn't uncompiled"

    @compile()
    def _() -> None:
        bind(5)


def test_ArgCompiler_fmap():
    @CallMacro.generate()
    def bind(
        visitor: ToMLIRBase,
        i: Compiled,
        *li: list[Compiled],
        **di: dict[Compiled],
    ) -> None:
        assert i == li[0]
        assert tuple(li) == (
            di["a"],
            di["b"],
            di["c"],
            di["d"],
        )

    @compile()
    def _() -> None:
        a: UInt32 = 5
        b: UInt32 = 6
        c: UInt32 = 7
        d: UInt32 = 8
        bind(a, a, b, c, d, b=b, a=a, d=d, c=c)
