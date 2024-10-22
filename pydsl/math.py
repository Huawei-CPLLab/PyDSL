from typing import Any, List, TYPE_CHECKING
from mlir.ir import OpView

from mlir.ir import *
from mlir.dialects.math import SqrtOp, ExpOp, PowFOp, AbsFOp
from pydsl.protocols import ToMLIRBase

from pydsl.macro import ArgRep, CallMacro


class sqrt(CallMacro):
    def argreps() -> List["sqrt.ArgType"]:
        return [ArgRep.COMPILED]

    def _on_Call(visitor: "ToMLIRBase", args: List[Any]) -> OpView:
        operand = args[0]
        return type(operand)(SqrtOp(operand.value))


class pow(CallMacro):
    def argreps() -> List["sqrt.ArgType"]:
        return [ArgRep.COMPILED, ArgRep.COMPILED]

    def _on_Call(visitor: "ToMLIRBase", args: List[Any]) -> OpView:
        operand, exponent = args
        return type(operand)(PowFOp(lhs=operand.value, rhs=exponent.value))


class exp(CallMacro):
    def argreps() -> List["exp.ArgType"]:
        return [ArgRep.COMPILED]

    def _on_Call(visitor: "ToMLIRBase", args: List[Any]) -> OpView:
        operand = args[0]
        return (type(operand))(ExpOp(operand.value))


class abs(CallMacro):
    def argreps() -> List["abs.ArgType"]:
        return [ArgRep.COMPILED]

    def _on_Call(visitor: "ToMLIRBase", args: List[Any]) -> OpView:
        operand = args[0]
        return (type(operand))(AbsFOp(operand.value))
