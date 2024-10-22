from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

# Lowerable is needed for SubtreeOut
from pydsl.type import get_operator, supports_operator, Lowerable

from mlir.dialects import transform
from mlir.dialects.transform import loop, structured
from mlir.dialects.transform import validator as taffine
from mlir.ir import IndexType, IntegerAttr, OpView, UnitAttr

from pydsl.macro import ArgRep, CallMacro, Compiled, Evaluated, Uncompiled
from pydsl.protocols import SubtreeOut, lower_single, ToMLIRBase


class Transform(ABC):
    tag_attribute: str = ""

    @abstractmethod
    def target_op_type() -> str:
        pass

    @abstractmethod
    def apply_transform(op) -> None:
        pass

    def tag(self, op, i) -> None:
        self.tag_attribute = f"transform{i}"
        op.attributes[self.tag_attribute] = UnitAttr.get()


class LoopCoalesce(Transform):
    def target_op_type() -> str:
        return "scf.for"

    def apply_transform(op) -> None:
        loop.LoopCoalesceOp(transform.OperationType.get("scf.for"), op)


@CallMacro.generate()
def tag(
    visitor: "ToMLIRBase", mlir: Compiled, attr_name: Evaluated[str]
) -> SubtreeOut:
    target = get_operator(mlir)

    if type(attr_name) is not str:
        raise TypeError("Attribute name is not a string")

    target.attributes[attr_name] = UnitAttr.get()
    return mlir


@CallMacro.generate()
def int_attr(
    visitor: "ToMLIRBase",
    mlir: Compiled,
    attr_name: Evaluated[str],
    value: Evaluated[int],
) -> SubtreeOut:
    target = get_operator(mlir)

    if type(attr_name) is not str:
        raise TypeError("Attribute name is not a string")
    target.attributes[attr_name] = IntegerAttr.get(IndexType.get(), value)

    return mlir


class match_tag(CallMacro):
    def argreps() -> list[ArgRep]:
        raise NotImplementedError()

    def _on_Call(visitor: "ToMLIRBase", args: list[Any]) -> OpView:
        raise NotImplementedError()


class fuse_into(CallMacro):
    def argreps() -> list[ArgRep]:
        raise NotImplementedError()

    def _on_Call(visitor: "ToMLIRBase", args: list[Any]) -> OpView:
        raise NotImplementedError()


class fuse(CallMacro):
    def argreps() -> list[ArgRep]:
        raise NotImplementedError()

    def _on_Call(visitor: "ToMLIRBase", args: list[Any]) -> OpView:
        raise NotImplementedError()
    
class skew(CallMacro):
    def argreps() -> list[ArgRep]:
        raise NotImplementedError()

    def _on_Call(visitor: "ToMLIRBase", args: list[Any]) -> OpView:
        raise NotImplementedError()


class tile(CallMacro):
    def argreps() -> list[ArgRep]:
        raise NotImplementedError()

    def _on_Call(visitor: "ToMLIRBase", args: list[Any]) -> OpView:
        raise NotImplementedError()


class reorder(CallMacro):
    def argreps() -> list[ArgRep]:
        raise NotImplementedError()

    def _on_Call(visitor: "ToMLIRBase", args: list[Any]) -> OpView:
        raise NotImplementedError()


class unroll(CallMacro):
    def argreps() -> list["unroll.ArgType"]:
        raise NotImplementedError()

    def _on_Call(visitor: "ToMLIRBase", args: list[Any]) -> OpView:
        raise NotImplementedError()


class distribute(CallMacro):
    def argreps() -> list[ArgRep]:
        raise NotImplementedError()

    def _on_Call(visitor: "ToMLIRBase", args: list[Any]) -> OpView:
        raise NotImplementedError()


class get_loop(CallMacro):
    def argreps() -> list[ArgRep]:
        raise NotImplementedError()

    def _on_Call(visitor: "ToMLIRBase", args: list[Any]) -> OpView:
        raise NotImplementedError()


class parallel(CallMacro):
    def argreps() -> list["parallel.ArgType"]:
        raise NotImplementedError()

    def _on_Call(visitor: "ToMLIRBase", args: list[Any]) -> OpView:
        raise NotImplementedError()


class vectorize(CallMacro):
    def argreps() -> list["vectorize.ArgType"]:
        raise NotImplementedError()

    def _on_Call(visitor: "ToMLIRBase", args: list[Any]) -> OpView:
        raise NotImplementedError()


@CallMacro.generate()
def recursively(
    visitor: "ToMLIRBase", targets: Uncompiled, func: Evaluated[Callable]
) -> Any:
    if not isinstance(targets, Iterable):
        targets = [targets]

    # This function filters out all elements that are not operators and pass
    # only operators into func
    def func_with_filter(x):
        return func(x) if supports_operator(x) else x

    return [
        visitor.visit_with_interception(t, func_with_filter) for t in targets
    ]
