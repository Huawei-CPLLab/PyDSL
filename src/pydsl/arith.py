from typing import Any
from pydsl.macro import CallMacro, Compiled
from pydsl.protocols import lower_single, SubtreeOut, ToMLIRBase
from pydsl.type import Int, Float, Sign
from pydsl.vector import Vector

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


@CallMacro.generate()
def trunc(
    visitor: ToMLIRBase,
    a: Compiled,
    truncated_type: Compiled,
    *,
    round_mode: Compiled = None,
) -> SubtreeOut:
    a_type = type(a)
    out_type = truncated_type
    if isinstance(a, Vector):
        out_type = Vector.get(a.shape, truncated_type)
        a_type = a.element_type

    if truncated_type.width >= a_type.width:
        raise TypeError("truncated type must be smaller than called type.")

    if issubclass(a_type, Int):
        out = arith.TruncIOp(lower_single(out_type), lower_single(a))
    elif issubclass(a_type, Float):
        out = arith.TruncFOp(lower_single(out_type), lower_single(a))
    else:
        raise TypeError(f"cannot take trunc of {a_type.__qualname__}")
    if round_mode is not None:
        out.attributes["round_mode"] = lower_single(round_mode)
    return (out_type)(out)


@CallMacro.generate()
def vadd(visitor: ToMLIRBase, a: Compiled, b: Compiled) -> SubtreeOut:
    rett = type(a)

    if not isinstance(a, Vector):
        raise TypeError(f"NOT a vector addition operation")
    if type(a) != type(b):
        raise TypeError(f"VADD type {type(a)} does not match {type(b)}")

    a_type = a.element_type
    if issubclass(a_type, Int):
        op = arith.addi(lower_single(a), lower_single(b))
    elif issubclass(a_type, Float):
        op = arith.addf(lower_single(a), lower_single(b))
    else:
        raise TypeError(f"unsupported vector addition type: {a_type}")
    return rett(op)


@CallMacro.generate()
def vmul(visitor: ToMLIRBase, a: Compiled, b: Compiled) -> SubtreeOut:
    rett = type(a)

    if not isinstance(a, Vector):
        raise TypeError(f"NOT a vector multiplication operation")
    if type(a) != type(b):
        raise TypeError(f"VMUL type {type(a)} does not match {type(b)}")

    a_type = a.element_type
    if issubclass(a_type, Int):
        op = arith.muli(lower_single(a), lower_single(b))
    elif issubclass(a_type, Float):
        op = arith.mulf(lower_single(a), lower_single(b))
    else:
        raise TypeError(f"unsupported vector multiplication type: {a_type}")
    return rett(op)
