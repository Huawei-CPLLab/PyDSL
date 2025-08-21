from __future__ import annotations

import ast
import collections.abc as cabc
import ctypes
import math
import numbers
import operator
import sys
import typing
from collections import namedtuple
from enum import Enum, auto
from functools import cache, reduce
from math import log2
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

import mlir.dialects.index as mlir_index
import mlir.ir as mlir
from mlir.dialects import arith, transform
from mlir.dialects import math as mlirmath
from mlir.dialects.transform.extras import OpHandle
from mlir.ir import (
    F16Type,
    F32Type,
    F64Type,
    IndexType,
    IntegerType,
    Operation,
    OpView,
    Value,
)

from pydsl.macro import CallMacro, MethodType, Uncompiled
from pydsl.protocols import ToMLIRBase, ArgContainer

if TYPE_CHECKING:
    # This is for imports for type hinting purposes only and which can result
    # in cyclic imports.
    from pydsl.frontend import CTypeTree


def iscompiled(x: Any) -> bool:
    """
    TODO: This is terrible and ugly code just to get things out of the way.

    Ideally, there should be a supertype of all PyDSL values called Value.
    """
    # Number is not lowerable but it is still returned as a SubtreeOut
    return isinstance(x, Number) or isinstance(x, Lowerable)


class Supportable(type):
    """
    A metaclass that modifies the behavior of initialization such that if a
    single variable that is passed in supports the class being initiated,
    then immediate casting of that variable is performed instead.

    This behavior is in likeness to built-in types like str, which uses __str__
    method to indicate a class' ability to be casted into a str.

    Implementers are required to specify their supporter class and caster class
    """

    def __init__(cls, name, bases, namespace, *args, **kwargs):
        if not hasattr(cls, "supporter"):
            raise TypeError(
                f"{cls.__qualname__} is an instance of Supportable, but did "
                f"not specify a supporter protocol"
            )

        if not hasattr(cls, "caster"):
            raise TypeError(
                f"{cls.__qualname__} is an instance of Supportable, but did "
                f"not specify a cast operation in the supporter protocol"
            )

        super().__init__(name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        match args:
            case (cls.supporter(),):
                # If there's exactly one argument and it's a supporter of this
                # class, have the rep cast itself
                (rep,) = args
                return cls.caster(cls, rep)
            case _:
                # Initialize the class as normal
                return super().__call__(*args, **kwargs)


@runtime_checkable
class Lowerable(Protocol):
    def lower(self) -> tuple[Value]: ...

    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]: ...


def lower(
    v: Lowerable | type | OpView | Value | mlir.Type,
) -> tuple[Value] | tuple[mlir.Type]:
    """
    Convert a `Lowerable` type, type instance, and other MLIR objects into its
    lowest MLIR representation, as a tuple.

    This function is *not* idempotent.

    Specific behavior:
    - If `v` is a `Lowerable` type, a `mlir.ir.Type` is returned.
    - If `v` is a `Lowerable` type instance, a `mlir.ir.Value` is returned.
    - If `v` is an `mlir.ir.OpView` type instance, then its results (of type
      `mlir.ir.Value`) are returned.
    - If `v` is already an `mlir.ir.Value` or `mlir.ir.Type`, `v` is returned
      enclosed in a tuple.
    - If `v` is not any of the types above, `TypeError` will be raised.

    For example:
    - `lower(Index)` should be equivalent to `(IndexType.get(),)`.
    - `lower(Index(5))` should be equivalent to
      `(ConstantOp(IndexType.get(), 5).results,)`.
    - ```lower(UInt8(4).op_add(UInt8(5)))``` should be equivalent to ::

        tuple(AddIOp(
            ConstantOp(IntegerType.get_signless(8), 4), )
            ConstantOp(IntegerType.get_signless(8), 5)).results)
    """
    match v:
        case OpView():
            return tuple(v.results)
        case Value() | mlir.Type():
            return (v,)
        case type() if issubclass(v, Lowerable):
            # Lowerable class
            return v.lower_class()
        case _ if issubclass(type(v), Lowerable):
            # Lowerable class instance
            return v.lower()
        case _:
            raise TypeError(f"{v} is not Lowerable")


def get_operator(x):
    target = x

    if issubclass(type(target), Lowerable):
        target = lower_single(target)

    if isinstance(target, Value):
        target = target.owner

    if not (
        issubclass(type(target), OpView) or issubclass(type(target), Operation)
    ):
        raise TypeError(f"{x} cannot be cast into an operator")

    return target


def supports_operator(x):
    try:
        get_operator(x)
        return True
    except TypeError:
        return False


def lower_single(
    v: Lowerable | type | OpView | Value | mlir.Type,
) -> mlir.Type | Value:
    """
    lower with the return value stripped of its tuple.
    Lowered output tuple must have length of exactly 1. Otherwise,
    `ValueError` is raised.

    This function is idempotent.
    """

    res = lower(v)
    if len(res) != 1:
        raise ValueError(f"lowering expected single element, got {res}")
    return res[0]


def lower_flatten(li):
    """
    Apply lower to each element of the list, then unpack the resulting tuples
    within the list.
    """
    # Uses map-reduce
    # Map:    lower each element
    # Reduce: flatten the resulting list of tuples into a list of its
    #         constituents
    return reduce(lambda a, b: a + [*b], map(lower, li), [])


class Sign(Enum):
    SIGNED = auto()
    UNSIGNED = auto()


AnyInt = typing.TypeVar("AnyInt", bound="Int")


@runtime_checkable
class SupportsInt(Protocol):
    def Int(self, target_type: type[AnyInt]) -> AnyInt: ...


class Int(metaclass=Supportable):
    supporter = SupportsInt

    def caster(cls, rep):
        return rep.Int(cls)

    width: int = None
    sign: Sign = None
    value: Value

    def __init_subclass__(
        cls, /, width: int, sign: Sign = Sign.SIGNED, **kwargs
    ) -> None:
        super().__init_subclass__()
        cls.width = width
        cls.sign = sign

    def __init__(self, rep: Any) -> None:
        # WARNING: There is no good way to enforce that the OpView type passed
        # in has the right sign.
        # This class is technically low-level enough that it's possible to
        # construct the wrong sign with this function.
        # This is because MLIR by default uses signless for many dialects, and
        # it's up to the language to enforce signs.
        # Users who never touch type implementation won't need to worry, but
        # those who develop type classes can potentially
        # use the wrong sign when wrapping their MLIR OpView back into a
        # language type.

        if not all([self.width, self.sign]):
            raise TypeError(
                f"attempted to initialize {type(self).__name__} without "
                f"defined size or sign"
            )

        if isinstance(rep, OpView):
            rep = rep.result

        match rep:
            # the rep must be real and close enough to an integer
            case numbers.Real() if math.isclose(rep, int(rep)):
                rep = int(rep)

                if not self.in_range(rep):
                    raise ValueError(
                        f"{rep} is out of range for {type(self).__name__}"
                    )

                self.value = arith.ConstantOp(
                    self.lower_class()[0], rep
                ).result

            case Value():
                self._init_from_mlir_value(rep)

            case _:
                raise TypeError(
                    f"{rep} cannot be casted as {type(self).__name__}"
                )

    def lower(self) -> tuple[Value]:
        return (self.value,)

    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]:
        return (IntegerType.get_signless(cls.width),)

    @classmethod
    def val_range(cls) -> tuple[int, int]:
        match cls.sign:
            case Sign.SIGNED:
                return (-(1 << (cls.width - 1)), (1 << (cls.width - 1)) - 1)
            case Sign.UNSIGNED:
                return (0, (1 << cls.width) - 1)
            case _:
                AssertionError("unimplemented sign")

    @classmethod
    def in_range(cls, val) -> bool:
        return cls.val_range()[0] <= val <= cls.val_range()[1]

    def _init_from_mlir_value(self, rep) -> None:
        if (rep_type := type(rep.type)) is not IntegerType:
            raise TypeError(
                f"{rep_type.__name__} cannot be casted as a "
                f"{type(self).__name__}"
            )
        if (width := rep.type.width) != self.width:
            raise TypeError(
                f"{type(self).__name__} expected to have width of "
                f"{self.width}, got {width}"
            )
        if not rep.type.is_signless:
            raise TypeError(
                f"ops passed into {type(self).__name__} must have "
                f"signless result, but was signed or unsigned"
            )

        self.value = rep

    @classmethod
    def _try_casting(cls, val) -> None:
        return cls(val)

    def op_abs(self) -> "Int":
        return self._try_casting(mlirmath.AbsIOp(self.value))

    # TODO: figure out how to do unsigned -> signed conversion
    # TODO: these arith operators should have automatic width-expansion
    def op_add(self, rhs: SupportsInt) -> "Int":
        rhs = self._try_casting(rhs)
        return self._try_casting(arith.AddIOp(self.value, rhs.value))

    op_radd = op_add  # commutative

    def op_sub(self, rhs: SupportsInt) -> "Int":
        rhs = self._try_casting(rhs)
        return self._try_casting(arith.SubIOp(self.value, rhs.value))

    def op_rsub(self, lhs: SupportsInt) -> "Int":
        lhs = self._try_casting(lhs)
        # note that operators are reversed
        return self._try_casting(arith.SubIOp(lhs.value, self.value))

    def op_mul(self, rhs: SupportsInt) -> "Int":
        rhs = self._try_casting(rhs)
        return self._try_casting(arith.MulIOp(self.value, rhs.value))

    op_rmul = op_mul  # commutative

    def op_neg(self) -> "Int":
        # Integer negation and bitwise not are operations that are
        # not present in arith dialect, for reasons related to peepholes:
        # https://discourse.llvm.org/t/arith-noti/4844/3
        return self._try_casting(
            arith.SubIOp(
                arith.ConstantOp(lower_single(type(self)), 0), self.value
            )
        )

    def op_invert(self) -> "Int":
        # Integer negation and bitwise not are operations that are
        # not present in arith dialect, for reasons related to peepholes:
        # https://discourse.llvm.org/t/arith-noti/4844/3
        return self._try_casting(
            arith.SubIOp(
                arith.ConstantOp(lower_single(type(self)), -1), self.value
            ),
        )

    def op_pos(self) -> "Int":
        return self._try_casting(self.value)

    # TODO: op_truediv cannot be implemented right now as it returns floating
    # points

    def op_floordiv(self, rhs: SupportsInt) -> "Int":
        rhs = self._try_casting(rhs)
        # assertion ensures that self and rhs have the same sign
        op = (
            arith.FloorDivSIOp if (self.sign == Sign.SIGNED) else arith.DivUIOp
        )
        return self._try_casting(op(self.value, rhs.value))

    def op_rfloordiv(self, lhs: SupportsInt) -> "Int":
        lhs = self._try_casting(lhs)
        # assertion ensures that self and rhs have the same sign
        op = (
            arith.FloorDivSIOp if (self.sign == Sign.SIGNED) else arith.DivUIOp
        )
        return self._try_casting(op(lhs.value, self.value))

    def _compare_with_pred(self, rhs: SupportsInt, pred: arith.CmpIPredicate):
        return Bool(
            arith.CmpIOp(pred, self.value, self._try_casting(rhs).value)
        )

    def op_lt(self, rhs: SupportsInt) -> "Bool":
        rhs = self._try_casting(rhs)

        match self.sign:
            case Sign.SIGNED:
                pred = arith.CmpIPredicate.slt
            case Sign.UNSIGNED:
                pred = arith.CmpIPredicate.ult

        return self._compare_with_pred(rhs, pred)

    def op_le(self, rhs: SupportsInt) -> "Bool":
        rhs = self._try_casting(rhs)

        match self.sign:
            case Sign.SIGNED:
                pred = arith.CmpIPredicate.sle
            case Sign.UNSIGNED:
                pred = arith.CmpIPredicate.ule

        return self._compare_with_pred(rhs, pred)

    def op_eq(self, rhs: SupportsInt) -> "Bool":
        rhs = self._try_casting(rhs)

        return self._compare_with_pred(rhs, arith.CmpIPredicate.eq)

    def op_ne(self, rhs: SupportsInt) -> "Bool":
        rhs = self._try_casting(rhs)

        return self._compare_with_pred(rhs, arith.CmpIPredicate.ne)

    def op_gt(self, rhs: SupportsInt) -> "Bool":
        rhs = self._try_casting(rhs)

        match self.sign:
            case Sign.SIGNED:
                pred = arith.CmpIPredicate.sgt
            case Sign.UNSIGNED:
                pred = arith.CmpIPredicate.ugt

        return self._compare_with_pred(rhs, pred)

    def op_ge(self, rhs: SupportsInt) -> "Bool":
        rhs = self._try_casting(rhs)

        match self.sign:
            case Sign.SIGNED:
                pred = arith.CmpIPredicate.sge
            case Sign.UNSIGNED:
                pred = arith.CmpIPredicate.uge

        return self._compare_with_pred(rhs, pred)

    @classmethod
    def CType(cls) -> tuple[type]:
        ctypes_map = {
            (Sign.SIGNED, 1): ctypes.c_bool,
            (Sign.SIGNED, 8): ctypes.c_int8,
            (Sign.SIGNED, 16): ctypes.c_int16,
            (Sign.SIGNED, 32): ctypes.c_int32,
            (Sign.SIGNED, 64): ctypes.c_int64,
            (Sign.UNSIGNED, 1): ctypes.c_bool,
            (Sign.UNSIGNED, 8): ctypes.c_uint8,
            (Sign.UNSIGNED, 16): ctypes.c_uint16,
            (Sign.UNSIGNED, 32): ctypes.c_uint32,
            (Sign.UNSIGNED, 64): ctypes.c_uint64,
        }

        if (key := (cls.sign, cls.width)) in ctypes_map:
            return (ctypes_map[key],)

        raise TypeError(f"{cls.__name__} does not have a corresponding ctype")

    @classmethod
    def to_CType(cls, arg_cont: ArgContainer, pyval: Any):
        try:
            pyval = int(pyval)
        except Exception as e:
            raise TypeError(
                f"{pyval} cannot be converted into an Int ctype"
            ) from e

        if not cls.in_range(pyval):
            lo, hi = cls.val_range()
            raise ValueError(
                f"{pyval} cannot fit into {cls.__qualname__}, must be in "
                f"the range [{lo}, {hi}]"
            )

        arg_cont.add_arg(pyval)
        return (pyval,)

    @classmethod
    def from_CType(cls, arg_cont: ArgContainer, cval: "CTypeTree"):
        return int(cval[0])

    @CallMacro.generate(method_type=MethodType.CLASS_ONLY)
    def on_Call(visitor: ToMLIRBase, cls: type[Self], rep: Uncompiled) -> Any:
        match rep:
            case ast.Constant():
                return cls(rep.value)
            case _:
                return cls(visitor.visit(rep))

    def Int(self, target_type: type[AnyInt]) -> AnyInt:
        if target_type.sign != self.sign:
            raise TypeError(
                "Int cannot be casted into another Int with differing signs"
            )

        if target_type.width < self.width:
            raise TypeError(
                f"Int of width {self.width} cannot be casted into width "
                f"{target_type.width}. Width must be extended"
            )

        if target_type.width == self.width:
            return target_type._try_casting(self.value)

        if target_type.width > self.width:
            match self.sign:
                case Sign.SIGNED:
                    new_val = arith.ExtSIOp(
                        lower_single(target_type), lower_single(self)
                    )
                case Sign.UNSIGNED:
                    new_val = arith.ExtUIOp(
                        lower_single(target_type), lower_single(self)
                    )

            return target_type._try_casting(new_val)

    F = typing.TypeVar("F", bound="Float")

    def Float(self, target_type: type[F]) -> F:
        match self.sign:
            case Sign.SIGNED:
                return target_type(
                    arith.sitofp(lower_single(target_type), lower_single(self))
                )

            case Sign.UNSIGNED:
                return target_type(
                    arith.uitofp(lower_single(target_type), lower_single(self))
                )


class UInt8(Int, width=8, sign=Sign.UNSIGNED):
    pass


class UInt16(Int, width=16, sign=Sign.UNSIGNED):
    pass


class UInt32(Int, width=32, sign=Sign.UNSIGNED):
    pass


class UInt64(Int, width=64, sign=Sign.UNSIGNED):
    pass


class SInt8(Int, width=8, sign=Sign.SIGNED):
    pass


class SInt16(Int, width=16, sign=Sign.SIGNED):
    pass


class SInt32(Int, width=32, sign=Sign.SIGNED):
    pass


class SInt64(Int, width=64, sign=Sign.SIGNED):
    pass


# It's worth noting that Python treat bool as an integer, meaning that e.g.
# (1 + True) == 2. It is also i1 in MLIR.
# To reflect this behavior, Bool inherits all integer operator overloading
# functions

# TODO: Bool currently does not accept anything except for Python value.
# It should also support ops returning i1


@runtime_checkable
class SupportsBool(Protocol):
    def Bool(self) -> "Bool": ...


class Bool(Int, width=1, sign=Sign.UNSIGNED):
    supporter = SupportsBool

    def caster(cls, rep):
        return rep.Bool()

    def __init__(self, rep: Any) -> None:
        match rep:
            case bool():
                lit_as_bool = 1 if rep else 0
                self.value = arith.ConstantOp(
                    IntegerType.get_signless(1), lit_as_bool
                ).result
            case _:
                return super().__init__(rep)

    def Bool(self) -> "Bool":
        return self

    def op_and(self, rhs: SupportsBool) -> "Bool":
        rhs = Bool(rhs)
        return Bool(arith.AndIOp(self.value, rhs.value))

    def op_or(self, rhs: SupportsBool) -> "Bool":
        rhs = Bool(rhs)
        return Bool(arith.OrIOp(self.value, rhs.value))

    def op_not(self) -> "Bool":
        # MLIR doesn't seem to have bitwise not
        return Bool(
            arith.SelectOp(
                self.value,
                arith.ConstantOp(IntegerType.get_signless(1), 0).result,
                arith.ConstantOp(IntegerType.get_signless(1), 1).result,
            )
        )


AnyFloat = typing.TypeVar("AnyFloat", bound="Float")


@runtime_checkable
class SupportsFloat(Protocol):
    def Float(self, target_type: type[AnyFloat]) -> AnyFloat: ...


class Float(metaclass=Supportable):
    supporter = SupportsFloat

    def caster(cls, rep):
        return rep.Float(cls)

    width: int
    mlir_type: mlir.Type
    value: Value

    def __init_subclass__(
        cls, /, width: int, mlir_type: mlir.Type, **kwargs
    ) -> None:
        super().__init_subclass__(**kwargs)
        cls.width = width
        cls.mlir_type = mlir_type

    def __init__(self, rep: Any) -> None:
        if not all([self.width, self.mlir_type]):
            raise TypeError(
                "attempted to initialize Float without defined width or "
                "mlir_type"
            )

        # TODO: Code duplication in many classes. Consider a superclass?
        if isinstance(rep, OpView):
            rep = rep.result

        match rep:
            case float() | int() | bool():
                rep = float(rep)
                self.value = arith.ConstantOp(
                    self.lower_class()[0], rep
                ).result

            case Value():
                if (rep_type := type(rep.type)) is not self.mlir_type:
                    raise TypeError(
                        f"{rep_type.__name__} cannot be casted as a "
                        f"{type(self).__name__}"
                    )

                self.value = rep

            case _:
                raise TypeError(
                    f"{rep} cannot be casted as a {type(self).__name__}"
                )

    def lower(self) -> tuple[Value]:
        return (self.value,)

    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]:
        return (cls.mlir_type.get(),)

    def _same_type_assertion(self, val):
        if type(self) is not type(val):
            raise TypeError(
                f"{type(self).__name__} cannot be added with "
                f"{type(val).__name__}"
            )

    @classmethod
    def _try_casting(cls, val) -> None:
        return cls(val)

    def op_abs(self) -> "Float":
        return self._try_casting(mlirmath.AbsFOp(self.value))

    def op_add(self, rhs: "Float") -> "Float":
        rhs = self._try_casting(rhs)
        return self._try_casting(arith.AddFOp(self.value, rhs.value))

    op_radd = op_add  # commutative

    def op_sub(self, rhs: "Float") -> "Float":
        rhs = self._try_casting(rhs)
        return self._try_casting(arith.SubFOp(self.value, rhs.value))

    def op_rsub(self, rhs: "Float") -> "Float":
        lhs = self._try_casting(rhs)
        # note that operators are reversed
        return self._try_casting(arith.SubFOp(lhs.value, self.value))

    def op_mul(self, rhs: "Float") -> "Float":
        rhs = self._try_casting(rhs)
        return self._try_casting(arith.MulFOp(self.value, rhs.value))

    op_rmul = op_mul  # commutative

    def op_truediv(self, rhs: "Float") -> "Float":
        rhs = self._try_casting(rhs)
        return self._try_casting(arith.DivFOp(self.value, rhs.value))

    def op_rtruediv(self, lhs: "Float") -> "Float":
        lhs = self._try_casting(lhs)
        return self._try_casting(arith.DivFOp(lhs.value, self.value))

    def op_neg(self) -> "Float":
        return self._try_casting(arith.NegFOp(self.value))

    def op_pos(self) -> "Float":
        return self._try_casting(self.value)

    def op_pow(self, rhs: "Float") -> "Float":
        rhs = self._try_casting(rhs)
        return self._try_casting(mlirmath.PowFOp(self.value))

    def _compare_with_pred(
        self, rhs: SupportsFloat, pred: arith.CmpFPredicate
    ):
        return Bool(arith.CmpFOp(pred, self.value, type(self)(rhs).value))

    def op_lt(self, rhs: SupportsInt) -> "Bool":
        rhs = self._try_casting(rhs)
        return self._compare_with_pred(rhs, arith.CmpFPredicate.OLT)

    def op_le(self, rhs: SupportsInt) -> "Bool":
        rhs = self._try_casting(rhs)
        return self._compare_with_pred(rhs, arith.CmpFPredicate.OLE)

    def op_eq(self, rhs: SupportsInt) -> "Bool":
        rhs = self._try_casting(rhs)
        return self._compare_with_pred(rhs, arith.CmpFPredicate.OEQ)

    def op_ne(self, rhs: SupportsInt) -> "Bool":
        rhs = self._try_casting(rhs)
        return self._compare_with_pred(rhs, arith.CmpFPredicate.ONE)

    def op_gt(self, rhs: SupportsInt) -> "Bool":
        rhs = self._try_casting(rhs)
        return self._compare_with_pred(rhs, arith.CmpFPredicate.OGT)

    def op_ge(self, rhs: SupportsInt) -> "Bool":
        rhs = self._try_casting(rhs)
        return self._compare_with_pred(rhs, arith.CmpFPredicate.OGE)

    # TODO: floordiv cannot be implemented so far. float -> int
    # needs floor ops.

    @classmethod
    def CType(cls) -> tuple[type]:
        ctypes_map = {
            32: ctypes.c_float,
            64: ctypes.c_double,
            80: ctypes.c_longdouble,
        }

        if (key := cls.width) in ctypes_map:
            return (ctypes_map[key],)

        raise TypeError(f"{cls.__name__} does not have a corresponding ctype.")

    out_CType = CType

    @classmethod
    def to_CType(cls, arg_cont: ArgContainer, pyval: float | int | bool):
        try:
            pyval = float(pyval)
        except Exception as e:
            raise TypeError(
                f"{pyval} cannot be converted into a "
                f"{cls.__name__} ctype. Reason: {e}"
            )

        arg_cont.add_arg(pyval)
        return (pyval,)

    @classmethod
    def from_CType(cls, arg_cont: ArgContainer, cval: "CTypeTree"):
        return float(cval[0])

    @CallMacro.generate(method_type=MethodType.CLASS_ONLY)
    def on_Call(
        visitor: "ToMLIRBase", cls: type[Self], rep: Uncompiled
    ) -> Any:
        match rep:
            case ast.Constant():
                return cls(rep.value)
            case _:
                return cls(visitor.visit(rep))

    def Float(self, target_type: type[AnyFloat]) -> AnyFloat:
        if target_type.width > self.width:
            return target_type(
                arith.extf(lower_single(target_type), lower_single(self))
            )

        if target_type.width == self.width:
            return target_type(self.value)

        if target_type.width < self.width:
            return target_type(
                arith.truncf(lower_single(target_type), lower_single(self))
            )


class F16(Float, width=16, mlir_type=F16Type):
    pass


class F32(Float, width=32, mlir_type=F32Type):
    pass


class F64(Float, width=64, mlir_type=F64Type):
    pass


# TODO: make this aware of compilation target rather than always making
# the target the current machine this runs on
def get_index_width() -> int:
    s = log2(sys.maxsize + 1) + 1
    assert (
        s.is_integer()
    ), "the compiler cannot determine the index size of the current "
    f"system. sys.maxsize yielded {sys.maxsize}"

    return int(s)


@runtime_checkable
class SupportsIndex(Protocol):
    def Index(self) -> "Index": ...


# TODO: for now, you can only do limited math on Index
# division requires knowledge of whether Index is signed or unsigned
# everything will be assumed to be unsigned for now...
class Index(Int, width=get_index_width(), sign=Sign.UNSIGNED):
    supporter = SupportsIndex

    def caster(cls, rep):
        return rep.Index()

    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]:
        return (IndexType.get(),)

    AnyInt = typing.TypeVar("I", bound="Int")

    def Int(self, cls: type[AnyInt]) -> AnyInt:
        if self.sign != cls.sign:
            raise TypeError(
                "attempt to cast Index to an Int of different sign"
            )

        op = {
            Sign.SIGNED: arith.index_cast,
            Sign.UNSIGNED: arith.index_castui,
        }

        return cls(
            op[cls.sign](IntegerType.get_signless(cls.width), self.value)
        )

    def Index(self) -> "Index":
        return self

    F = typing.TypeVar("F", bound="Float")

    def Float(self, target_type: type[F]) -> F:
        # There does not seem to exist any operation that takes
        # Index -> target Float.

        # We instead do Index -> widest UInt -> target Float.
        return target_type(
            arith.UIToFPOp(
                lower_single(target_type),
                mlir_index.CastUOp(
                    IntegerType.get_signless(64), lower_single(self)
                ),
            )
        )

    def _init_from_mlir_value(self, rep) -> None:
        if (rep_type := type(rep.type)) is not IndexType:
            raise TypeError(
                f"{rep_type.__name__} cannot be casted as a "
                f"{type(self).__name__}"
            )

        self.value = rep

    def _same_type_assertion(self, val):
        if type(self) is not type(val):
            raise TypeError(
                f"{type(self).__name__} cannot be added with "
                f"{type(val).__name__}"
            )

    def _try_casting(self, val) -> None:
        return type(self)(val)

    def op_add(self, rhs: SupportsIndex) -> "Index":
        rhs = self._try_casting(rhs)
        return type(self)(mlir_index.AddOp(self.value, rhs.value))

    op_radd = op_add  # commutative

    def op_sub(self, rhs: SupportsIndex) -> "Index":
        rhs = self._try_casting(rhs)
        return type(self)(mlir_index.SubOp(self.value, rhs.value))

    def op_rsub(self, lhs: SupportsIndex) -> "Index":
        lhs = self._try_casting(lhs)
        return type(self)(mlir_index.SubOp(lhs.value, self.value))

    def op_mul(self, rhs: SupportsIndex) -> "Index":
        rhs = self._try_casting(rhs)
        return type(self)(mlir_index.MulOp(self.value, rhs.value))

    op_rmul = op_mul  # commutative

    def op_truediv(self, rhs: SupportsIndex) -> "Index":
        raise NotImplementedError()  # TODO

    def op_rtruediv(self, rhs: SupportsIndex) -> "Index":
        raise NotImplementedError()  # TODO

    def op_floordiv(self, rhs: SupportsIndex) -> "Index":
        rhs = self._try_casting(rhs)
        return type(self)(mlir_index.FloorDivSOp(self.value, rhs.value))

    def op_rfloordiv(self, lhs: SupportsIndex) -> "Index":
        lhs = self._try_casting(lhs)
        return type(self)(mlir_index.FloorDivSOp(lhs.value, self.value))

    # TODO: maybe these should be unsigned ops. Actually, why do we treat Index
    # as an unsigned type and not a signed type?
    def op_ceildiv(self, rhs: SupportsIndex) -> "Index":
        rhs = self._try_casting(rhs)
        return type(self)(mlir_index.CeilDivSOp(self.value, rhs.value))

    def op_rceildiv(self, lhs: SupportsIndex) -> "Index":
        lhs = self._try_casting(lhs)
        return type(self)(mlir_index.CeilDivSOp(lhs.value, self.value))

    @classmethod
    def CType(cls) -> tuple[type]:
        # TODO: this needs to be different depending on the platform.
        # On Python, you use sys.maxsize. However, we should let the
        # user choose in CompilationSetting

        return (ctypes.c_size_t,)

    @classmethod
    def PolyCType(cls) -> tuple[type]:
        return (ctypes.c_int,)


# TODO: this class should be renamed to TransformAnyOp to avoid confusion
# with MLIR's OpView subclasses
class AnyOp:
    value: OpHandle

    def __init__(self, rep: transform.AnyOpType) -> None:
        self.value = rep

    def lower(self) -> tuple[transform.AnyOpType]:
        return (self.value,)

    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]:
        return (transform.AnyOpType.get(),)


# FIXME: this class is still work-in-progress and is untested
# eventually will probably change to a Operation class which can be subclassed
# for each operator that exists in MLIR
class ForOp:
    value: OpHandle

    # Side note: OpHandle is the type of output of a transform operation
    # TODO: AnyOp should be taken out of here and into a casting mechanism
    # instead
    def __init__(self, rep: OpHandle | AnyOp) -> None:
        if isinstance(rep, AnyOp):
            rep = lower_single(rep)

        match rep.type:
            case transform.AnyOpType.get():
                self.value = transform.CastOp(
                    transform.OperationType("scf.for"), rep
                )
            case transform.OperationType.get("scf.for"):
                self.value = rep
            case _:
                raise TypeError(
                    f"{type(rep)} is not castable to a `scf.for` operation "
                    f"type"
                )

    def lower(self) -> tuple[transform.OperationType]:
        return (self.value,)

    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]:
        return (transform.OperationType.get("scf.for"),)


NumberLike: typing.TypeAlias = typing.Union["Number", Int, Float, Index]


class Number:
    """
    A class that represents a generic number constant whose exact
    representation at runtime is evaluated lazily. As long as this type isn't
    used by an MLIR operator, it will only exist at compile-time.

    This type supports any value that is an instance of numbers.Number.

    All numeric literals in PyDSL evaluates to this type.

    See _NumberMeta for how its dunder functions are dynamically generated.
    """

    value: numbers.Number
    """
    The internal representation of the number.
    """

    def __init__(self, rep: numbers.Number):
        self.value = rep

    AnyInt = typing.TypeVar("I", bound="Int")

    def Int(self, target_type: type[AnyInt]) -> AnyInt:
        return target_type(self.value)

    F = typing.TypeVar("F", bound="Float")

    def Float(self, target_type: type[F]) -> F:
        return target_type(self.value)

    def Index(self) -> Index:
        return Index(self.value)

    def Bool(self) -> Bool:
        return Bool(self.value)


# These are for unary operators in Number class
UnNumberOp = namedtuple("UnNumberOp", "dunder_name, internal_op")
un_number_op = {
    UnNumberOp("op_neg", operator.neg),
    UnNumberOp("op_not", operator.not_),
    UnNumberOp("op_pos", operator.pos),
    UnNumberOp("op_abs", operator.abs),
    UnNumberOp("op_truth", operator.truth),
    UnNumberOp("op_floor", math.floor),
    UnNumberOp("op_ceil", math.ceil),
    UnNumberOp("op_round", round),
    UnNumberOp("op_invert", operator.invert),
}

for tup in un_number_op:

    def method_gen(tup):
        """
        This function exists simply to allow a unique generic_unary_op to be
        generated whose variables are bound to the arguments of this function
        rather than the variable of the for loop.
        """
        # TODO: why is the above useful? What's wrong with binding to for loop
        # variables?
        _, internal_op = tup

        # perform the unary operation on the underlying value
        def generic_unary_op(self: Number) -> Number:
            return Number(internal_op(self.value))

        return generic_unary_op

    ldunder_name, internal_op = tup
    setattr(Number, ldunder_name, method_gen(tup))

# These are for binary operators in Number
BinNumberOp = namedtuple(
    "BinNumberOp", "ldunder_name, internal_op, rdunder_name"
)
bin_number_op = {
    BinNumberOp("op_add", operator.add, "op_radd"),
    BinNumberOp("op_sub", operator.sub, "op_rsub"),
    BinNumberOp("op_mul", operator.mul, "op_rmul"),
    BinNumberOp("op_truediv", operator.truediv, "op_rtruediv"),
    BinNumberOp("op_pow", operator.pow, "op_rpow"),
    BinNumberOp("op_divmod", divmod, "op_rdivmod"),
    BinNumberOp("op_floordiv", operator.floordiv, "op_rfloordiv"),
    BinNumberOp("op_mod", operator.mod, "op_rmod"),
    BinNumberOp("op_lshift", operator.lshift, "op_rlshift"),
    BinNumberOp("op_rshift", operator.rshift, "op_rrshift"),
    BinNumberOp("op_and", operator.and_, "op_rand"),
    BinNumberOp("op_xor", operator.xor, "op_rxor"),
    BinNumberOp("op_or", operator.or_, "op_ror"),
    BinNumberOp("op_lt", operator.lt, "op_gt"),
    BinNumberOp("op_le", operator.le, "op_ge"),
    BinNumberOp("op_eq", operator.le, "op_eq"),
    BinNumberOp("op_ge", operator.ge, "op_le"),
    BinNumberOp("op_gt", operator.gt, "op_lt"),
}


for tup in bin_number_op:
    """
    This dynamically add left-hand dunder operations to Number without
    repeatedly writing the code in generic_op.

    In order for this to work, new methods must be generated by returning
    a function where all of its variables are bound to arguments of its nested
    function (in this case, method_gen).
    """

    def method_gen(tup):
        """
        This function exists simply to allow a unique generic_bin_op to be
        generated whose variables are bound to the arguments of this function
        rather than the variable of the for loop.
        """
        _, internal_op, rdunder_name = tup

        def generic_bin_op(self: Number, rhs: NumberLike) -> NumberLike:
            # if RHS is also a Number
            if isinstance(rhs, Number):
                # perform the binary operation on the underlying values
                return Number(internal_op(self.value, rhs.value))

            # otherwise use RHS's implementation instead
            return getattr(rhs, rdunder_name)(self)

        return generic_bin_op

    ldunder_name, internal_op, rdunder_name = tup
    setattr(Number, ldunder_name, method_gen(tup))


DTypes = typing.TypeVarTuple("DTypes")


class Tuple(typing.Generic[*DTypes]):
    """
    While tuple is not an MLIR type, it is still present in the language
    syntax-wise.

    This class mainly allows users to express multiple returns without having
    to rely on Python's built-in tuple type, which does not contain the
    necessary information for casting to Python CType.

    While it also allows users to group data together, this grouping only
    exists during compile-time and is not reflected in the code.
    As such, indexing cannot be performed, which makes this grouping rather
    useless.

    TODO: Below are some future design considerations:

    If tuple type was to become available and indexable at run-time, one
    would need to think of how to deal with a tuple with different types.
    Depending on the index, the return type of the operation can differ, which
    does not play nicely with MLIR's static type nature.

    One could also opt for a restrictive version of tuple that only allows
    index values known at compile-time so that the type can be inferred.

    One could also opt for a tuple that only accepts a single type, but that
    may be too restrictive to be useful.
    """

    _default_subclass_name = "TupleUnnamedSubclass"
    dtypes: tuple[type]
    value: tuple

    @staticmethod
    @cache
    def class_factory(
        dtypes: tuple[type], name=_default_subclass_name
    ) -> type["Tuple"]:
        """
        Create a new subclass of Tuple dynamically
        """
        if not isinstance(dtypes, cabc.Iterable):
            raise TypeError(
                f"MemRef requires dtypes to be iterable, got {type(dtypes)}"
            )

        if any([issubclass(t, Tuple) for t in dtypes]):
            raise TypeError("Tuples cannot be nested")

        return type(
            name,
            (Tuple,),
            {"dtypes": tuple(dtypes)},
        )

    # TODO: this will temporarily perform casting of other tuples
    # but later we will need to do a major refactor that shifts this behavior
    # to a dedicated member function called `cast`.
    def __init__(self, iterable: cabc.Iterable | Tuple | mlir.OpView):
        # this is the very bad casting code mentioned in the todo
        if isinstance(iterable, Tuple):
            iterable = iterable.value
        elif isinstance(iterable, mlir.OpView):
            iterable = iterable.results

        error = TypeError(
            f"Tuple with dtypes {[t.__name__ for t in self.dtypes]} "
            f"received "
            f"{[type(v).__name__ for v in iterable]} types as values"
        )

        # If input is a tuple of a single tuple, stripe it down
        # Note that we do not allow tuples in tuples
        if len(iterable) == 1 and isinstance(iterable[0], Tuple):
            iterable = iterable.value

        if len(self.dtypes) != len(iterable):
            raise error

        # Check if input values match the dtype accepted by the tuple
        # If not, attempt element-wise casting
        try:
            casted = tuple([
                v if isinstance(v, t) else t(v)
                for v, t in zip(iterable, self.dtypes)
            ])
        except TypeError as casting_error:
            error.add_note(
                f"attempted casting failed with error message: "
                f"{repr(casting_error)}"
            )
            raise error from casting_error

        self.value = casted

    @classmethod
    def check_lowerable(cls) -> None | typing.Never:
        if not all([isinstance(t, Lowerable) for t in cls.dtypes]):
            raise TypeError(
                f"lowering a Tuple requires all of its dtypes to be "
                f"lowerable, got {cls.dtypes}"
            )

    def lower(self):
        self.check_lowerable()
        return lower_flatten(self.value)

    @classmethod
    def lower_class(cls) -> type:
        # Since MLIR FuncOps in MLIR accept tuple returns using plain Python
        # tuples
        cls.check_lowerable()
        return lower_flatten(cls.dtypes)

    @classmethod
    def on_class_getitem(
        cls, visitor: ToMLIRBase, slice: ast.AST
    ) -> type["Tuple"]:
        # TODO: this looks boilerplatey, maybe a helper function that takes
        # in a typing.Generic and do automatic binding of arguments?
        match slice:
            case ast.Tuple(elts=elts):
                args = [visitor.resolve_type_annotation(e) for e in elts]
            case t:
                args = [visitor.resolve_type_annotation(t)]

        dtypes = tuple(args)

        return cls.class_factory(dtypes)

    @classmethod
    def CType(cls) -> tuple[mlir.Type]:
        return tuple([d.CType() for d in cls.dtypes])

    @classmethod
    def to_CType(cls, arg_cont: ArgContainer, *_) -> typing.Never:
        raise TypeError("function arguments cannot have type Tuple")

    @classmethod
    def from_CType(cls, arg_cont: ArgContainer, ct: "CTypeTree") -> tuple:
        return tuple(
            t.from_CType(arg_cont, sub_ct)
            for t, sub_ct in zip(cls.dtypes, ct, strict=False)
        )

    @staticmethod
    def from_values(visitor: ToMLIRBase, *values):
        cls = Tuple.class_factory(tuple([type(v) for v in values]))
        return cls(values)

    def as_iterable(self: Self, visitor: "ToMLIRBase"):
        return self.value

    # TODO: maybe need dedicated PolyCType?


class Slice:
    """
    An object to represent a Python slice [lo:hi:step].
    If some of the arguments are missing (e.g. [::3]) they will be
    stored as None.
    Note that most MLIR functions that take in slice-like inputs have
    a different set of arguments from Python: they want lo, size, step.
    """

    lo: Index | None
    hi: Index | None
    step: Index | None

    def __init__(self, lo: Index | None, hi: Index | None, step: Index | None):
        self.lo = lo
        self.hi = hi
        self.step = step

    def get_args(self, max_size: SupportsIndex) -> tuple[Index, Index, Index]:
        """
        Returns [offset, size, step], which can be used for MLIR functions.
        Returns 0, max_size, 1 instead of None values, respectively.
        """
        lo = Index(0) if self.lo is None else self.lo
        hi = Index(max_size) if self.hi is None else self.hi
        step = Index(1) if self.step is None else self.step
        size = hi.op_sub(lo).op_ceildiv(step)
        return (lo, size, step)
