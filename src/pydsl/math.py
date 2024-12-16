from mlir.dialects.math import ExpOp, PowFOp, SqrtOp

from pydsl.macro import CallMacro, Compiled
from pydsl.protocols import ToMLIRBase, lower_single
from pydsl.type import Number


@CallMacro.generate()
def sqrt(visitor: ToMLIRBase, operand: Compiled):
    return type(operand)(SqrtOp(lower_single(operand)))


@CallMacro.generate()
def pow(visitor: ToMLIRBase, operand: Compiled, exponent: Compiled):
    return type(operand)(
        PowFOp(lhs=lower_single(operand), rhs=lower_single(exponent))
    )


@CallMacro.generate()
def exp(visitor: ToMLIRBase, operand: Compiled):
    return type(operand)(ExpOp(lower_single(operand)))


@CallMacro.generate()
def abs(visitor: ToMLIRBase, operand: Compiled):
    return operand.op_abs()


inf = Number(float("inf"))
