from typing import Any
from pydsl.macro import CallMacro, Compiled
from pydsl.protocols import lower_single, SubtreeOut, ToMLIRBase
from pydsl.type import Int, Float, Sign

import mlir.dialects.arith as arith

# TODO: we should think more about how to sturcture this file. Right now, it is
# temporary and mostly exists to support max and min. See
# https://github.com/Huawei-CPLLab/PyDSL/issues/46


def _cast(a: Any, b: Any) -> tuple[Any, Any]:
    """
    General function for determining the result type of a binary operator.
    Casts both elements to the result type. Currently, we just cast to the
    type of the first operand, which is not ideal. See
    https://github.com/Huawei-CPLLab/PyDSL/issues/46
    """

    return a, type(a)(b)


# TODO: code of min and max is mostly the same, maybe generate the functions
# like how we generate operators in type.py or linalg.py


@CallMacro.generate()
def max(visitor: ToMLIRBase, a: Compiled, b: Compiled) -> SubtreeOut:
    a, b = _cast(a, b)
    av, bv = lower_single(a), lower_single(b)
    rett = type(a)

    if isinstance(a, Int):
        if a.sign == Sign.SIGNED:
            return rett(arith.MaxSIOp(av, bv))
        else:
            return rett(arith.MaxUIOp(av, bv))
    elif isinstance(a, Float):
        # TODO: there is also MaxNumFOp
        return rett(arith.MaximumFOp(av, bv))
    else:
        raise TypeError(f"cannot take max of {rett.__qualname__}")


@CallMacro.generate()
def min(visitor: ToMLIRBase, a: Compiled, b: Compiled) -> SubtreeOut:
    a, b = _cast(a, b)
    av, bv = lower_single(a), lower_single(b)
    rett = type(a)

    if isinstance(a, Int):
        if a.sign == Sign.SIGNED:
            return rett(arith.MinSIOp(av, bv))
        else:
            return rett(arith.MinUIOp(av, bv))
    elif isinstance(a, Float):
        # TODO: there is also MinNumFOp
        return rett(arith.MinimumFOp(av, bv))
    else:
        raise TypeError(f"cannot take min of {rett.__qualname__}")
