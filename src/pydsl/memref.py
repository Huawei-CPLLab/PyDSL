import ast
import collections.abc as cabc
import ctypes
import typing
from ctypes import POINTER, c_void_p
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

import mlir.ir as mlir
import numpy
from mlir.dialects import affine, memref
from mlir.ir import MemRefType, OpView, Value

from pydsl.affine import AffineContext, AffineMapExpr, AffineMapExprWalk
from pydsl.macro import CallMacro, Compiled, Evaluated
from pydsl.protocols import SubtreeOut, ToMLIRBase, lower
from pydsl.type import (
    Index,
    Lowerable,
    SupportsIndex,
    Tuple,
    lower_flatten,
    lower_single,
)

if TYPE_CHECKING:
    from pydsl.frontend import CTypeTree

# magic value that equates to MLIR's "?" dimension symbol
DYNAMIC: Final = -9223372036854775808

# used for the virtual static-typing in PyDSL
Dynamic: Final = -9223372036854775808

RawRMRD: typing.TypeAlias = tuple[c_void_p | int]

# based on example in PEP 646: https://peps.python.org/pep-0646/
DType = typing.TypeVar("DType")
Shape = typing.TypeVarTuple("Shape")


@runtime_checkable
class SupportsRMRD(Protocol):
    """
    Classes that supports converting to a RankedMemRefDescriptor
    """

    def RankedMemRefDescriptor(
        self, target_type: type["MemRef"]
    ) -> "RankedMemRefDescriptor": ...


@dataclass
class RankedMemRefDescriptor:
    """
    Python representation of the lowered structure of a ranked memref in
    LLVMIR.

    See https://mlir.llvm.org/docs/TargetLLVMIR/#ranked-memref-types for
    details.
    """

    allocated_ptr: c_void_p
    """
    The pointer to the data buffer as allocated, referred to as
    "allocated pointer". This is only useful for deallocating the memref.
    """

    aligned_ptr: c_void_p
    "The pointer to the properly aligned data pointer that the memref indexes."

    offset: int
    """
    A lowered converted index-type integer containing the distance in number
    of elements between the beginning of the (aligned) buffer and the first
    element to be accessed through the memref.
    """

    shape: tuple[int]
    """
    An array containing as many converted index-type integers as the rank of
    the memref: the array represents the size, in number of elements, of the
    memref along the given dimension.
    """

    strides: tuple[int]
    """
    A second array containing as many converted index-type integers as the
    rank of memref: the second array represents the "stride" (in tensor
    abstraction sense), i.e. the number of consecutive elements of the
    underlying buffer one needs to jump over to get to the next logically
    indexed element.
    """

    # This allows RankedMemRefDescriptors to be splatted
    def __iter__(self):
        arr_index_t = Index.CType()[0] * self.rank()
        yield from (
            self.allocated_ptr,
            self.aligned_ptr,
            self.offset,
            arr_index_t(*self.shape),
            arr_index_t(*self.strides),
        )

    def rank(self) -> int:
        assert len(self.shape) == len(
            self.strides
        ), "rank of a RankedMemRefDescriptor is inconsistent!"

        return len(self.shape)

    def assert_support(self, cls: type["MemRef"]) -> None:
        if not (cls.shape == self.shape):
            raise TypeError(
                f"RankedMemRefDescriptor {self.__qualname__} does not match "
                f"{cls.__qualname__}"
            )

    def out_CType_of_MemRef(cls: type["MemRef"]) -> tuple:
        return (RankedMemRefDescriptor.generate_struct(cls.rank()),)

    def CType_of_MemRef(cls: type["MemRef"]) -> tuple:
        return (
            c_void_p,
            c_void_p,
            Index.CType()[0],
            (Index.CType()[0] * len(cls.shape)),
            (Index.CType()[0] * len(cls.shape)),
        )

    def as_CType(self) -> "CTypeTree":
        return (*self,)

    @classmethod
    def from_CType(cls, ct: "CTypeTree") -> "RankedMemRefDescriptor":
        allo, align, offset, shape, strides = ct
        return RankedMemRefDescriptor(
            allocated_ptr=allo,
            aligned_ptr=align,
            offset=offset,
            shape=tuple(shape),
            strides=tuple(strides),
        )


class UsesRMRD:
    """
    A mixin class for adding CType support for classes that eventually lowers
    down to a ranked MemRef descriptor in LLVM C calling convention.

    This mostly exists to reduce code duplication.
    """

    # These fields must be present for this to be used.
    shape: tuple[int]
    element_type: Lowerable

    @classmethod
    def CType(cls) -> tuple[mlir.Type]:
        # TODO: this assumes that the shape is ranked
        return RankedMemRefDescriptor.CType_of_MemRef(cls)

    @classmethod
    def to_CType(
        cls, pyval: typing.Union[tuple, list, numpy.ndarray, "SupportsRMRD"]
    ) -> RawRMRD:
        """
        Accepts any value that is numpy.ndarray or convertible
        to RankedMemRefDescriptor.

        The elements are restricted as such:
        - if the expected element's CType is int or float, then the input must
          be int or float, respectively
        - if the expected element's CType is a tuple of CTypes, then an error
          is thrown
        """
        match pyval:
            case tuple() | list():
                raise TypeError(
                    f"{type(pyval)} cannot be casted into a "
                    f"{cls.__qualname__}. Supported types include "
                    f"numpy.ndarray"
                )
            case numpy.ndarray():
                return cls._ndarray_to_CType(pyval)
            case SupportsRMRD():
                return pyval.RankedMemRefDescriptor(cls).as_CType()
            case _:
                raise TypeError(
                    f"{type(pyval)} cannot be casted into a "
                    f"{cls.__qualname__}"
                )

    @classmethod
    def from_CType(cls, ct: "CTypeTree") -> numpy.ndarray:
        rmd = RankedMemRefDescriptor.from_CType(ct)

        return numpy.ctypeslib.as_array(
            # This requires element_type to be representable with a single
            # ctypes element
            ctypes.cast(rmd.aligned_ptr, POINTER(cls.element_type.CType()[0])),
            shape=rmd.shape,
        )

    @classmethod
    def PolyCType(cls) -> tuple[mlir.Type]:
        # TODO: this assumes that the shape is ranked
        return (c_void_p,)

    @classmethod
    def to_PolyCType(
        cls, pyval: typing.Union[tuple, list, numpy.ndarray, "SupportsRMRD"]
    ) -> RawRMRD:
        # the 0th index is the pointer to the memory location, which is the
        # only thing Poly accepts
        return (cls.to_CType(pyval)[0],)

    @classmethod
    def from_PolyCType(cls, ct: "CTypeTree") -> numpy.ndarray:
        raise RuntimeError(
            "PolyCType MemRef pointers cannot be converted into NumPy NDArray"
        )

    @classmethod
    def _arraylike_to_CType(cls, li) -> RawRMRD:
        return cls._ndarray_to_CType(numpy.asarray(li))

    @classmethod
    def rank(cls) -> int:
        return len(cls.shape)

    @classmethod
    def same_shape(cls, x: numpy.ndarray) -> bool:
        """
        Returns true if x's shape is the same as the MemRef's
        """
        return all([
            xi == clsi or clsi == DYNAMIC  # "?" accepts any shape
            for xi, clsi in zip(x.shape, cls.shape, strict=False)
        ])

    @classmethod
    def _ndarray_to_CType(cls, a: numpy.ndarray) -> RawRMRD:
        ndtype = cls.element_type.CType()
        if len(ndtype) > 1:  # TODO: ideally, this tuple is also flattened
            raise TypeError(
                f"The element type of a {cls.__qualname__} cannot be "
                f"composite CType"
            )

        if (actual_dt := numpy.ctypeslib.as_ctypes_type(a.dtype)) is not (
            expected_dt := ndtype[0]
        ):
            raise TypeError(
                f"{cls.__qualname__} expect ndarray with dtype {expected_dt}, "
                f"got {actual_dt}"
            )

        if not cls.same_shape(a):
            raise TypeError(
                f"attempted to pass array with shape {a.shape} into a MemRef "
                f"of shape {cls.shape}"
            )

        rmd = RankedMemRefDescriptor(
            allocated_ptr=a.ctypes.data_as(c_void_p),
            aligned_ptr=a.ctypes.data_as(c_void_p),
            offset=0,
            shape=a.shape,
            strides=[s // a.strides[-1] for s in a.strides],
        )

        return rmd.as_CType()


class MemRef(typing.Generic[DType, *Shape], UsesRMRD):
    """
    TODO: this MemRef abstraction currently only supports ranked MemRefs.

    TODO: the element types (i.e. DType) allowed by MLIR's MemRef is currently
    hard-coded and constrained to int, float, index, complex, vector, and other
      MemRefs.
    For now and for simplicity, vector and MemRef types will not be accepted as
    DType, as they are composite types.
    Hard-coding is necessary as the author is unaware of any API in MLIR Python
    binding that allows one to check whether an MLIR type implements the
    MemRefElementTypeInterface.

    This means that:
    - When passing Python list-likes into the program, MemRef will try to cast
    it to the ctype of the MemRef's element type, to the discretion of NumPy.
    - If MemRef's element type has a ctype that is a tuple of ctypes (e.g.
    MemRef itself, as in a MemRef of MemRefs), it will cast the list passed in
    into a binary, to the discretion of NumPy. It will not enforce whether the
    bitstring is of the right format. The source of the binary is responsible
    for providing a format that the LLVM-lowered IR can parse

    The official docs provides information on what elements are supported:
    https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype.

    If one wishes to have an array of composite, struct-like element types,
    one should abstract above this class and treat the array as a MemRef of
    bytes.
    """

    value: Value
    shape: tuple[int] = None
    element_type: Lowerable = None
    _default_subclass_name = "AnnonymousMemRefSubclass"
    _supported_mlir_type = [
        mlir.IntegerType,
        mlir.F16Type,
        mlir.F32Type,
        mlir.F64Type,
        mlir.IndexType,
        mlir.ComplexType,
    ]

    @staticmethod
    @cache
    def class_factory(
        shape: tuple[int], element_type, name=_default_subclass_name
    ):
        """
        Create a new subclass of MemRef dynamically with the specified
        dimensions and type
        """
        # TODO: this check cannot be done right now because types can't be
        # lowered outside of MLIR context
        # if len(lower(element_type)) != 1: raise TypeError(f"MemRef cannot
        # store composite types, got {element_type} which lowers to
        # {lower(element_type)}")

        if not isinstance(shape, cabc.Iterable):
            raise TypeError(
                f"MemRef requires shape to be iterable, got {type(shape)}"
            )

        return type(
            name,
            (MemRef,),
            {
                "shape": tuple(shape),
                "element_type": element_type,
            },
        )

    # Convenient alias
    get = class_factory

    def __init__(self, rep: OpView | Value) -> None:
        mlir_element_type = lower_single(self.element_type)
        if not any([
            type(mlir_element_type) is t for t in self._supported_mlir_type
        ]):
            raise NotImplementedError(
                f"having a MemRef with DType {self.element_type.__qualname__} "
                f"is not supported"
            )

        if isinstance(rep, OpView):
            rep = rep.result

        if (rep_type := type(rep.type)) is not MemRefType:
            raise TypeError(f"{rep_type} cannot be casted as a MemRef")

        if not all([
            self.shape == tuple(rep.type.shape),
            lower_single(self.element_type) == rep.type.element_type,
        ]):
            raise TypeError(
                f"expected shape {'x'.join([str(sh) for sh in self.shape])}"
                f"x{lower_single(self.element_type)}, got OpView with shape "
                f"{'x'.join([str(sh) for sh in rep.type.shape])}"
                f"x{rep.type.element_type}"
            )

        self.value = rep

    def on_getitem(
        self: typing.Self, visitor: "ToMLIRBase", slice: ast.AST
    ) -> SubtreeOut:
        def cons_affine_load(am: AffineMapExpr):
            am = am.lowered()
            return self.element_type(
                affine.AffineLoadOp(
                    lower_single(self.element_type),
                    lower_single(self),
                    indices=[*am.dims, *am.syms],
                    map=am.map,
                )
            )

        if AffineContext in visitor.context_stack:
            key = AffineMapExprWalk.compile(slice, visitor.scope_stack)
            return cons_affine_load(key)

        key = visitor.visit(slice)

        match key:
            case SupportsIndex():
                key = Index(key)
                return self.element_type(
                    memref.LoadOp(lower_single(self), [lower_single(key)])
                )

            case Tuple():
                key = [Index(k) for k in key.value]
                return self.element_type(
                    memref.LoadOp(lower_single(self), lower_flatten(key))
                )

            case AffineMapExpr():
                return cons_affine_load(key)

            case _:
                raise TypeError(
                    f"{type(key)} cannot be used to index a MemRef"
                )

    def on_setitem(
        self: typing.Self,
        visitor: "ToMLIRBase",
        slice: ast.AST,
        value: ast.AST | SubtreeOut,
    ) -> SubtreeOut:
        value = self.element_type(visitor.visit(value))

        def cons_affine_store(am: AffineMapExpr):
            am = am.lowered()
            return affine.AffineStoreOp(
                lower_single(value),
                lower_single(self),
                indices=[*am.dims, *am.syms],
                map=am.map,
            )

        if AffineContext in visitor.context_stack:
            key = AffineMapExprWalk.compile(slice, visitor.scope_stack)
            return cons_affine_store(key)

        key = visitor.visit(slice)

        match key:
            case SupportsIndex():
                key = Index(key)
                return memref.StoreOp(
                    lower_single(value),
                    lower_single(self),
                    [lower_single(key)],
                )

            case Tuple():
                key = [Index(k) for k in key.value]
                return memref.StoreOp(
                    lower_single(value), lower_single(self), lower_flatten(key)
                )

            case AffineMapExpr():
                return cons_affine_store(key)

            case _:
                raise TypeError(
                    f"{type(key)} cannot be used to index a MemRef"
                )

    @classmethod
    def on_class_getitem(
        cls, visitor: ToMLIRBase, slice: ast.AST
    ) -> SubtreeOut:
        # TODO: this looks boilerplatey, maybe a helper function that takes
        # in a typing.Generic and do automatic binding of arguments?
        match slice:
            case ast.Tuple(elts=elts):
                args = [visitor.resolve_type_annotation(e) for e in elts]
            case t:
                args = [visitor.resolve_type_annotation(t)]

        if len(args) < 2:
            raise TypeError(
                f"MemRef expected at least 2 Generic arguments, got {args}"
            )

        dtype = args[0]
        shape = args[1:]

        return cls.class_factory(tuple(shape), dtype)

    def lower(self) -> tuple[Value]:
        return (self.value,)

    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]:
        if not all([cls.shape, cls.element_type]):
            e = TypeError(
                "attempted to lower MemRef without defined dims or type"
            )
            if (clsname := cls.__name__) != MemRef._default_subclass_name:
                e.add_note(f"hint: class name is {clsname}")
            raise e

        return (
            MemRefType.get(list(cls.shape), lower_single(cls.element_type)),
        )

    @classmethod
    def CType(cls) -> tuple[mlir.Type]:
        # TODO: this assumes that the shape is ranked
        return RankedMemRefDescriptor.CType_of_MemRef(cls)

    @classmethod
    def to_CType(
        cls, pyval: typing.Union[tuple, list, numpy.ndarray, "SupportsRMRD"]
    ) -> RawRMRD:
        """
        Accepts any value that is numpy.ndarray, tuple, list, or convertible
        to RankedMemRefDescriptor.

        The elements are restricted as such:
        - if the expected element's CType is int or float, then the input must
          be int or float, respectively
        - if the expected element's CType is a tuple of CTypes, then an error
          is thrown
        - if the expected element's CType is a struct, then
        - NOTE: string list is not supported
        """
        match pyval:
            case tuple() | list():
                raise TypeError(
                    f"{type(pyval)} cannot be casted into a MemRef. "
                    f"Supported types include numpy.ndarray"
                )
            case numpy.ndarray():
                return cls._ndarray_to_CType(pyval)
            case SupportsRMRD():
                return pyval.RankedMemRefDescriptor(cls).as_CType()
            case _:
                raise TypeError(
                    f"{type(pyval)} cannot be casted into a "
                    f"{cls.__qualname__}"
                )

    @classmethod
    def from_CType(cls, ct: "CTypeTree") -> numpy.ndarray:
        rmd = RankedMemRefDescriptor.from_CType(ct)

        return numpy.ctypeslib.as_array(
            # This requires element_type to be representable with a single
            # ctypes element
            ctypes.cast(rmd.aligned_ptr, POINTER(cls.element_type.CType()[0])),
            shape=rmd.shape,
        )

    @classmethod
    def PolyCType(cls) -> tuple[mlir.Type]:
        # TODO: this assumes that the shape is ranked
        return (c_void_p,)

    @classmethod
    def to_PolyCType(
        cls, pyval: typing.Union[tuple, list, numpy.ndarray, "SupportsRMRD"]
    ) -> RawRMRD:
        # the 0th index is the pointer to the memory location, which is the
        # only thing Poly accepts
        return (cls.to_CType(pyval)[0],)

    @classmethod
    def from_PolyCType(cls, ct: "CTypeTree") -> numpy.ndarray:
        raise RuntimeError(
            "PolyCType memref pointers cannot be converted into NumPy NDArray"
        )

    @classmethod
    def _arraylike_to_CType(cls, li) -> RawRMRD:
        return cls._ndarray_to_CType(numpy.asarray(li))

    @classmethod
    def rank(cls) -> int:
        return len(cls.shape)

    @classmethod
    def same_shape(cls, x: numpy.ndarray) -> bool:
        """
        Returns true if x's shape is the same as the MemRef's
        """
        if len(x.shape) != len(cls.shape):
            return False

        return all([
            xi == clsi or clsi == DYNAMIC  # "?" accepts any shape
            for xi, clsi in zip(x.shape, cls.shape)
        ])

    @classmethod
    def _ndarray_to_CType(cls, a: numpy.ndarray) -> RawRMRD:
        ndtype = cls.element_type.CType()
        if len(ndtype) > 1:  # TODO: ideally, this tuple is also flattened
            raise TypeError(
                "The element type of a MemRef cannot be composite CType"
            )

        if (actual_dt := numpy.ctypeslib.as_ctypes_type(a.dtype)) is not (
            expected_dt := ndtype[0]
        ):
            raise TypeError(
                f"MemRef expect ndarray with dtype {expected_dt}, got "
                f"{actual_dt}"
            )

        if not cls.same_shape(a):
            raise TypeError(
                f"attempted to pass array with shape {a.shape} into a MemRef "
                f"of shape {cls.shape}"
            )

        rmd = RankedMemRefDescriptor(
            allocated_ptr=a.ctypes.data_as(c_void_p),
            aligned_ptr=a.ctypes.data_as(c_void_p),
            offset=0,
            shape=a.shape,
            strides=[s // a.strides[-1] for s in a.strides],
        )

        return rmd.as_CType()


# Convenient alias
MemRefFactory = MemRef.class_factory


def verify_memory(mem: MemRef):
    if not isinstance(mem, MemRef):
        raise TypeError(
            f"the type being allocated must be a subclass of MemRef, got {mem}"
        )


def verify_memory_type(mtype: type[MemRef]):
    if not issubclass(mtype, MemRef):
        raise TypeError(
            f"the type being allocated must be a subclass of MemRef, got "
            f"{mtype}"
        )


def verify_dynamics_val(mtype: type[MemRef], dynamics_val: Tuple) -> None:
    dynamics_val = lower(dynamics_val)

    if not isinstance(dynamics_val, cabc.Iterable):
        raise TypeError(f"{repr(dynamics_val)} is not iterable")

    if (actual_dyn := len(dynamics_val)) != (
        target_dyn := mtype.shape.count(DYNAMIC)
    ):
        raise ValueError(
            f"MemRef has {target_dyn} dynamic dimensions to be filled, "
            f"but alloca received {actual_dyn}"
        )


@CallMacro.generate()
def alloca(
    visitor: ToMLIRBase, mtype: Compiled, dynamics_val: Compiled = None
) -> SubtreeOut:
    if dynamics_val is None:
        dynamics_val = Tuple.from_values(visitor, *())

    verify_memory_type(mtype)
    verify_dynamics_val(mtype, dynamics_val)
    dynamics_val = [lower_single(Index(i)) for i in lower(dynamics_val)]

    # TODO: not sure what the third argument do
    return mtype(memref.alloca(lower_single(mtype), dynamics_val, []))


@CallMacro.generate()
def alloc(
    visitor: ToMLIRBase, mtype: Compiled, dynamics_val: Compiled = None
) -> SubtreeOut:
    if dynamics_val is None:
        dynamics_val = Tuple.from_values(visitor, *())

    verify_memory_type(mtype)
    verify_dynamics_val(mtype, dynamics_val)
    dynamics_val = [lower_single(Index(i)) for i in lower(dynamics_val)]

    # TODO: not sure what the third argument do
    return mtype(memref.alloc(lower_single(mtype), dynamics_val, []))


@CallMacro.generate()
def dealloc(visitor: ToMLIRBase, mem: Evaluated) -> None:
    verify_memory(mem)
    return memref.dealloc(lower_single(mem))
