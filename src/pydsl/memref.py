import ast
import ctypes
import typing
from collections.abc import Callable, Iterable
from ctypes import POINTER, c_void_p
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Final

import mlir.ir as mlir
import numpy as np
from mlir.dialects import affine, memref
from mlir.ir import (
    DenseI64ArrayAttr,
    MemRefType,
    OpView,
    StridedLayoutAttr,
    Value,
)

from pydsl.affine import AffineContext, AffineMapExpr, AffineMapExprWalk
from pydsl.macro import CallMacro, Compiled, Evaluated
from pydsl.protocols import (
    ArgContainer,
    canonicalize_args,
    SubtreeOut,
    ToMLIRBase,
    lower,
)
from pydsl.type import (
    Index,
    Lowerable,
    Slice,
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

RawRMRD: typing.TypeAlias = "CTypeTree"
RuntimeMemrefShape = list[int | Value]

# based on example in PEP 646: https://peps.python.org/pep-0646/
DType = typing.TypeVar("DType")
Shape = typing.TypeVarTuple("Shape")


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

    def CType_of_MemRef(cls: type["MemRef"]) -> tuple:
        return (
            c_void_p,
            c_void_p,
            Index.CType()[0],
            (Index.CType()[0] * len(cls.shape)),
            (Index.CType()[0] * len(cls.shape)),
        )

    @classmethod
    def to_CType(
        cls, arg_cont: ArgContainer, pyval: "RankedMemRefDescriptor"
    ) -> "CTypeTree":
        arg_cont.add_arg(pyval)
        return (*pyval,)

    @classmethod
    def from_CType(
        cls, arg_cont: ArgContainer, ct: "CTypeTree"
    ) -> "RankedMemRefDescriptor":
        alloc, align, offset, shape, strides = ct
        return RankedMemRefDescriptor(
            allocated_ptr=c_void_p(alloc),
            aligned_ptr=c_void_p(align),
            offset=offset,
            shape=tuple(shape),
            strides=tuple(strides),
        )


def are_shapes_compatible(arr1: Iterable[int], arr2: Iterable[int]) -> bool:
    """
    Returns whether arr1 and arr2 have the same elements, excluding positions
    where at least one of the values is DYNAMIC.
    """
    return len(arr1) == len(arr2) and all(
        a == b or a == DYNAMIC or b == DYNAMIC for a, b in zip(arr1, arr2)
    )


class UsesRMRD:
    """
    A mixin class for adding CType support for classes that eventually lower
    down to a ranked MemRef descriptor in LLVM C calling convention.

    This mostly exists to reduce code duplication.

    strides == None indicates that the default layout is used.
    That is, row major order.
    """

    # These fields must be present for this superclass to be used.
    # The values of shape and strides are mostly only used for type
    # checking, i.e. making sure a compatible ndarray was passed in.
    # len(shape), element_type, and offset are actually used.
    shape: tuple[int]
    element_type: Lowerable
    offset: int
    strides: tuple[int] | None

    @classmethod
    def CType(cls) -> tuple[mlir.Type]:
        # TODO: this assumes that the shape is ranked
        return RankedMemRefDescriptor.CType_of_MemRef(cls)

    @classmethod
    def to_CType(
        cls, arg_cont: ArgContainer, pyval: tuple | list | np.ndarray
    ) -> RawRMRD:
        """
        Accepts any value that is numpy.ndarray.
        """
        match pyval:
            case tuple() | list():
                raise TypeError(
                    f"{type(pyval)} cannot be casted into a "
                    f"{cls.__qualname__}. Supported types include numpy.ndarray"
                )
            case np.ndarray():
                return cls._ndarray_to_CType(arg_cont, pyval)
            case _:
                raise TypeError(
                    f"{type(pyval)} cannot be casted into a {cls.__qualname__}"
                )

    @classmethod
    def from_CType(cls, arg_cont: ArgContainer, ct: "CTypeTree") -> np.ndarray:
        # This requires element_type to be representable with a single ctypes element
        rmd = RankedMemRefDescriptor.from_CType(arg_cont, ct)
        element_ctype = cls.element_type.CType()[0]
        element_size = ctypes.sizeof(element_ctype)
        ptr = rmd.aligned_ptr  # No nice way to do this in one line it seems
        ptr.value += rmd.offset * element_size

        max_size = (
            sum((rmd.shape[i] - 1) * rmd.strides[i] for i in range(rmd.rank()))
            + 1
        )
        byte_strides = [s * element_size for s in rmd.strides]
        # Load as a 1D array first, then apply the correct shape and strides
        flat_arr = np.ctypeslib.as_array(
            ctypes.cast(ptr, POINTER(element_ctype)), shape=(max_size,)
        )
        arr = np.lib.stride_tricks.as_strided(
            flat_arr, shape=rmd.shape, strides=byte_strides
        )

        # If arr overlaps with any ndarray that was a aargument of the
        # function, construct it from that instead so we have a pointer to the
        # original ndarray and it doesn't deallocate the memory
        base_arr = arg_cont.get_overlap(arr)

        if base_arr is not None:
            # Use the buffer base_arr.data then apply the correct offset,
            # shape, and strides. buf has a pointer to base_arr, and the new
            # ndarray will also have this pointer when constructed from buf
            buf = base_arr.data
            base_ptr = base_arr.ctypes.data_as(c_void_p)
            offs = int(ptr.value) - int(base_ptr.value)
            arr = np.ndarray(rmd.shape, arr.dtype, buf, offs, byte_strides)
            assert id(arr.base) == id(base_arr)

        return arr

    @classmethod
    def same_shape(cls, x: np.ndarray) -> bool:
        """
        Returns true if x's shape is the same as the MemRef's.
        """
        return are_shapes_compatible(cls.shape, x.shape)

    @classmethod
    def same_strides(cls, x: np.ndarray) -> bool:
        """
        Returns true if x's strides are the same as the MemRef's.
        """
        if cls.strides is None:
            # If default layout, check if ndarray is also row-major order
            return x.flags["C_CONTIGUOUS"]
        else:
            # numpy arrays store strides by number of bytes, but PyDSL and MLIR
            # use number of elements
            cls_byte_strides = [
                s * x.itemsize if s != DYNAMIC else DYNAMIC
                for s in cls.strides
            ]
            return are_shapes_compatible(cls_byte_strides, x.strides)

    @classmethod
    def _ndarray_to_CType(
        cls, arg_cont: ArgContainer, a: np.ndarray
    ) -> RawRMRD:
        ndtype = cls.element_type.CType()
        if len(ndtype) > 1:  # TODO: ideally, this tuple is also flattened
            raise TypeError(
                f"The element type of a {cls.__qualname__} cannot be "
                f"composite CType"
            )

        if (actual_dt := np.ctypeslib.as_ctypes_type(a.dtype)) is not (
            expected_dt := ndtype[0]
        ):
            raise TypeError(
                f"{cls.__qualname__} expects ndarray with dtype "
                f"{expected_dt.__qualname__}, got {actual_dt.__qualname__}"
            )

        if not cls.same_shape(a):
            raise TypeError(
                f"attempted to pass array with shape {a.shape} into a MemRef "
                f"of shape {cls.shape}"
            )

        if not cls.same_strides(a):
            raise TypeError(
                f"attempted to pass array with strides {a.strides} into a "
                f"MemRef with strides {cls.strides}. The array has itemsize "
                f"{a.itemsize}. The strides of the array should be "
                f"{a.itemsize} times the strides of the MemRef."
            )

        act_offset = cls.offset if cls.offset != DYNAMIC else 0
        # No nice way to do this in one line it seems
        act_ptr = a.ctypes.data_as(c_void_p)
        act_ptr.value -= act_offset * a.itemsize

        # Actual shape and strides are determined by shape and strides of ndarray.
        # We throw an error earlier if they don't match the shape and
        # strides of this class.
        rmd = RankedMemRefDescriptor(
            allocated_ptr=a.ctypes.data_as(c_void_p),
            aligned_ptr=act_ptr,
            offset=act_offset,
            shape=a.shape,
            strides=[s // a.itemsize for s in a.strides],
        )

        arg_cont.add_arg(a)
        return RankedMemRefDescriptor.to_CType(arg_cont, rmd)

    @classmethod
    def PolyCType(cls) -> tuple[mlir.Type]:
        # TODO: this assumes that the shape is ranked
        return (c_void_p,)

    @classmethod
    def to_PolyCType(cls, pyval: tuple | list | np.ndarray) -> RawRMRD:
        # the 0th index is the pointer to the memory location, which is the
        # only thing Poly accepts
        return (cls.to_CType(pyval)[0],)

    @classmethod
    def from_PolyCType(cls, ct: "CTypeTree") -> np.ndarray:
        raise RuntimeError(
            "PolyCType MemRef pointers cannot be converted into NumPy NDArray"
        )

    @classmethod
    def _arraylike_to_CType(cls, arg_cont: ArgContainer, li) -> RawRMRD:
        return cls._ndarray_to_CType(arg_cont, np.asarray(li))

    @classmethod
    def rank(cls) -> int:
        return len(cls.shape)


class MemRef(typing.Generic[DType, *Shape], UsesRMRD):
    """
    TODO: this MemRef abstraction currently only supports ranked MemRefs and
    only StridedLayout and default layout.

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

    # Only strides is allowed to be None. If anything else remains None,
    # something has gone wrong.
    value: Value
    shape: tuple[int] = None
    element_type: Lowerable = None
    offset: int = None
    strides: tuple[int] | None = None

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
    @canonicalize_args
    @cache
    def class_factory(
        shape: tuple[int],
        element_type,
        *,
        offset: int = 0,
        strides: tuple[int] | None = None,
        name=_default_subclass_name,
    ):
        """
        Create a new subclass of MemRef with the specified dimensions and type.

        If strides is None, a MemRef with the default layout will be created.
        See https://mlir.llvm.org/docs/Dialects/Builtin/#layout.
        If strides is not None, the layout will be a StridedLayout specified
        by offset and strides.
        Note that in this case, strides are absolute, not relative to the
        next dimension.
        For example, MemRef.class_factory((4, 16), F32) and
        MemRef.class_factory((4, 16), F32, strides=(16, 1)) return MemRef types with
        the same indexing scheme (although technically, MLIR considers them to
        be different layouts, since the default layout is implicitly an
        affine map layout).
        General affine map layouts are currently not supported.
        """
        # TODO: this check cannot be done right now because types can't be
        # lowered outside of MLIR context
        # if len(lower(element_type)) != 1: raise TypeError(f"MemRef cannot
        # store composite types, got {element_type} which lowers to
        # {lower(element_type)}")

        if not isinstance(shape, Iterable):
            raise TypeError(
                f"MemRef requires shape to be iterable, got {type(shape)}"
            )

        return type(
            name,
            (MemRef,),
            {
                "shape": tuple(shape),
                "element_type": element_type,
                "offset": int(offset),
                "strides": None if strides is None else tuple(strides),
            },
        )

    # Convenient alias
    get = class_factory

    @classmethod
    def get_fully_dynamic(cls, element_type, rank: int):
        """
        Quick alias for returning a MemRef type where shape,
        offset, and strides are all dynamic.
        """
        dyn_list = tuple([DYNAMIC] * rank)
        return cls.class_factory(
            dyn_list, element_type, offset=DYNAMIC, strides=dyn_list
        )

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
                f"x{lower_single(self.element_type)}, got representation with shape "
                f"{'x'.join([str(sh) for sh in rep.type.shape])}"
                f"x{rep.type.element_type}"
            )

        cls_is_strided = self.strides is not None
        rep_is_strided = isinstance(rep.type.layout, StridedLayoutAttr)

        if cls_is_strided and not rep_is_strided:
            raise TypeError(
                "MemRef has strided layout but representation does not"
            )

        if not cls_is_strided and rep_is_strided:
            raise TypeError(
                "representation has strided layout but MemRef does not"
            )

        if cls_is_strided and not all([
            self.offset == rep.type.layout.offset,
            self.strides == tuple(rep.type.layout.strides),
        ]):
            raise TypeError(
                f"expected layout with offset = {self.offset},"
                f"strides = {self.strtides}, got representation with"
                f"offset = {rep.type.layout.offset}, strides = "
                f"{tuple(rep.type.layout.strides)}"
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

        key_st = visitor.visit(slice)

        if isinstance(key_st, AffineMapExpr):
            return cons_affine_load(key_st)

        key_list = subtree_to_slices(visitor, key_st)
        dim = self.rank()

        if len(key_list) == dim and all(
            isinstance(key, SupportsIndex) for key in key_list
        ):
            key_list = lower_flatten([Index(key) for key in key_list])
            return self.element_type(
                memref.LoadOp(lower_single(self), key_list)
            )

        lo_list, size_list, step_list = slices_to_mlir_format(
            key_list, self.runtime_shape
        )
        result_type = self.get_fully_dynamic(self.element_type, dim)
        dynamic_i64_attr = DenseI64ArrayAttr.get([DYNAMIC] * dim)
        rep = memref.SubViewOp(
            result_type.lower_class()[0],
            lower_single(self),
            lo_list,
            size_list,
            step_list,
            dynamic_i64_attr,
            dynamic_i64_attr,
            dynamic_i64_attr,
        )
        return result_type(rep)

    def on_setitem(
        self: typing.Self,
        visitor: "ToMLIRBase",
        slice: ast.AST,
        value: ast.AST,
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

        key_st = visitor.visit(slice)

        if isinstance(key_st, AffineMapExpr):
            return cons_affine_store(key_st)

        key_list = subtree_to_slices(visitor, key_st)
        dim = self.rank()

        if len(key_list) == dim and all(
            isinstance(key, SupportsIndex) for key in key_list
        ):
            key_list = lower_flatten([Index(key) for key in key_list])
            return memref.StoreOp(
                lower_single(value), lower_single(self), key_list
            )

        if len(key_list) != dim:
            raise IndexError(
                f"number of indices must be the same as the rank of a MemRef when storing: `"
                f"rank is {dim}, but number of indices is {len(key_list)}"
            )

        raise TypeError("cannot store to a slice of a MemRef")

    @classmethod
    def __class_getitem__(cls, args: tuple):
        if not isinstance(args, tuple):
            args = (args,)

        return cls.class_factory(tuple(args[1:]), args[0])

    @classmethod
    def on_class_getitem(
        cls, visitor: ToMLIRBase, slice: ast.AST
    ) -> SubtreeOut:
        match slice:
            case ast.Tuple(elts=elts):
                args = tuple(visitor.resolve_type_annotation(e) for e in elts)
            case t:
                args = (visitor.resolve_type_annotation(t),)

        # Equivalent to cls.__class_getitem__(args)
        return cls[args]

    def lower(self) -> tuple[Value]:
        return (self.value,)

    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]:
        if cls.shape is None or cls.element_type is None:
            e = TypeError(
                "attempted to lower MemRef without defined dims or type"
            )
            if (clsname := cls.__name__) != MemRef._default_subclass_name:
                e.add_note(f"hint: class name is {clsname}")
            raise e

        if cls.strides is None:
            return (
                MemRefType.get(
                    list(cls.shape), lower_single(cls.element_type)
                ),
            )
        else:
            layout = StridedLayoutAttr.get(cls.offset, list(cls.strides))
            return (
                MemRefType.get(
                    list(cls.shape), lower_single(cls.element_type), layout
                ),
            )

    @property
    def runtime_shape(self) -> RuntimeMemrefShape:
        """
        Return the shape of the memref as it exists at runtime.

        If one of the dimension sizes is dynamic, a memref.dim operator is
        returned instead for that dimension.
        """
        return [
            (
                d
                if d != DYNAMIC
                else memref.DimOp(lower_single(self), lower_single(Index(i)))
            )
            for i, d in enumerate(self.shape)
        ]


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


def verify_dynamic_sizes(mtype: type[MemRef], dynamic_sizes: Tuple) -> None:
    dynamic_sizes = lower(dynamic_sizes)

    # TODO: does this check do anything, since lower returns a tuple?
    if not isinstance(dynamic_sizes, Iterable):
        raise TypeError(f"{repr(dynamic_sizes)} is not iterable")

    if (actual_dyn := len(dynamic_sizes)) != (
        target_dyn := mtype.shape.count(DYNAMIC)
    ):
        raise ValueError(
            f"MemRef has {target_dyn} dynamic dimensions to be filled, "
            f"but alloc/alloca received {actual_dyn}"
        )


def verify_dynamic_symbols(
    mtype: type[MemRef], dynamic_symbols: Tuple
) -> None:
    dynamic_symbols = lower(dynamic_symbols)

    # TODO: does this check do anything, since lower returns a tuple?
    if not isinstance(dynamic_symbols, Iterable):
        raise TypeError(f"{repr(dynamic_symbols)} is not iterable")

    if (actual_dyn := len(dynamic_symbols)) != (
        target_dyn := 0
        if mtype.strides is None
        else mtype.strides.count(DYNAMIC)
    ):
        raise ValueError(
            f"MemRef has {target_dyn} dynamic strides to be filled, "
            f"but alloc/alloca received {actual_dyn}"
        )


def _alloc_generic(
    visitor: ToMLIRBase,
    mtype: Compiled,
    dynamic_sizes: Compiled,
    dynamic_symbols: Compiled,
    alloc_func: Callable[..., SubtreeOut],
) -> SubtreeOut:
    """
    Does the logic required for alloc/alloca. It was silly having
    two functions that differed by only one character. alloc_func
    should be memref.alloc or memref.alloca.
    """
    if dynamic_sizes is None:
        dynamic_sizes = Tuple.from_values(visitor, *())

    if dynamic_symbols is None:
        dynamic_symbols = Tuple.from_values(visitor, *())

    verify_memory_type(mtype)

    if mtype.strides is not None:
        raise NotImplementedError(
            "allocating MemRefs with a strided layout is currently"
            "not supported, since there seems to be no way to lower"
            "the resulting MLIR to LLVMIR. Allocate a MemRef with"
            "consecutive memory instead"
        )

    verify_dynamic_sizes(mtype, dynamic_sizes)
    verify_dynamic_symbols(mtype, dynamic_symbols)
    dynamic_sizes = [lower_single(Index(i)) for i in lower(dynamic_sizes)]
    dynamic_symbols = [lower_single(Index(i)) for i in lower(dynamic_symbols)]

    return mtype(
        alloc_func(lower_single(mtype), dynamic_sizes, dynamic_symbols)
    )


@CallMacro.generate()
def alloca(
    visitor: ToMLIRBase,
    mtype: Compiled,
    dynamic_sizes: Compiled = None,
    dynamic_symbols: Compiled = None,
) -> SubtreeOut:
    return _alloc_generic(
        visitor, mtype, dynamic_sizes, dynamic_symbols, memref.alloca
    )


@CallMacro.generate()
def alloc(
    visitor: ToMLIRBase,
    mtype: Compiled,
    dynamic_sizes: Compiled = None,
    dynamic_symbols: Compiled = None,
) -> SubtreeOut:
    return _alloc_generic(
        visitor, mtype, dynamic_sizes, dynamic_symbols, memref.alloc
    )


@CallMacro.generate()
def dealloc(visitor: ToMLIRBase, mem: Evaluated) -> None:
    verify_memory(mem)
    return memref.dealloc(lower_single(mem))


def slices_to_mlir_format(
    key_list: list[Slice | SupportsIndex], runtime_shape: RuntimeMemrefShape
) -> tuple[list[Value], list[Value], list[Value]]:
    """
    Given a list of slices/indices, converts the slices to MLIR format and
    infers missing bounds/dimensions based on the dimensions of the tensor/memref.
    3 lists will be returned: [offsets], [sizes], [strides], which can be
    passed to MLIR functions like tensor.extract_slice and memref.subview.
    If key_list is shorter than runtime_shape, assume the entirety of the
    remaining dimensions should be included ([:]).
    There is currently no bounds checking!
    Negative strides or indices are not supported (even though [3:-2:-1] can
    be valid in normal Python) and result in undefined behaviour!
    """

    dim = len(runtime_shape)

    if len(key_list) > dim:
        raise IndexError(
            f"number of subscripts {len(key_list)} is greater than number"
            f"of dimensions {dim}"
        )

    while len(key_list) < dim:
        key_list.append(Slice(None, None, None))

    lo_list = []
    size_list = []
    step_list = []

    for i in range(dim):
        key = key_list[i]
        if isinstance(key, SupportsIndex):
            lo_list.append(lower_single(Index(key)))
            size_list.append(lower_single(Index(1)))
            step_list.append(lower_single(Index(1)))
        elif isinstance(key, Slice):
            lo, size, step = key.get_args(Index(runtime_shape[i]))
            lo_list.append(lower_single(lo))
            size_list.append(lower_single(size))
            step_list.append(lower_single(step))
        else:
            raise TypeError(f"{type(key)} cannot be used as a subscript")

    return (lo_list, size_list, step_list)


def subtree_to_slices(
    visitor: "ToMLIRBase", key: SubtreeOut
) -> list[Slice | SupportsIndex]:
    match key:
        case SupportsIndex():
            return [key]
        case Tuple():
            return list(key.as_iterable(visitor))
        case Slice():
            return [key]
        case _:
            raise TypeError(f"{type(key)} cannot be used as a subscript")
