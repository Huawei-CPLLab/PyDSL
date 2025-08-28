import ast
from collections.abc import Iterable
from functools import cache
import typing

from mlir.dialects import tensor
import mlir.ir as mlir
from mlir.ir import DenseI64ArrayAttr, OpView, RankedTensorType, Value

from pydsl.func import InlineFunction
from pydsl.macro import CallMacro, Compiled, Evaluated, MethodType
from pydsl.memref import (
    assert_shapes_compatible,
    UsesRMRD,
    RuntimeMemrefShape,
    slices_to_mlir_format,
    split_static_dynamic_dims,
    subtree_to_slices,
)
from pydsl.protocols import canonicalize_args, SubtreeOut, ToMLIRBase
from pydsl.type import (
    Index,
    Lowerable,
    lower_flatten,
    lower_single,
    SupportsIndex,
    Tuple,
)

DYNAMIC = -9223372036854775808

# used for the virtual static-typing in PyDSL
Dynamic = typing.Literal[-9223372036854775808]

# based on example in PEP 646: https://peps.python.org/pep-0646/
# TODO: these currently are unused
DType = typing.TypeVar("DType")
Shape = typing.TypeVarTuple("Shape")


class Tensor(typing.Generic[DType, *Shape], UsesRMRD):
    """
    TODO: this Tensor type is limited to ranked versions for now.
    """

    value: Value
    shape: tuple[int]
    element_type: Lowerable
    offset: int
    strides: tuple[int] | None

    _default_subclass_name = "AnnonymousTensorSubclass"
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
        shape: Iterable[int], element_type, *, name=_default_subclass_name
    ):
        """
        Create a new subclass of Tensor dynamically with the specified
        dimensions and type
        """
        # TODO: this check cannot be done right now because types can't be
        # lowered outside of MLIR context
        # if len(lower(element_type)) != 1: raise TypeError(f"Tensor cannot
        # store composite types, got {element_type} which lowers to
        # {lower(element_type)}")

        if not isinstance(shape, Iterable):
            raise TypeError(
                f"Tensor requires shape to be iterable, got {type(shape)}"
            )

        return type(
            name,
            (Tensor,),
            {
                "shape": tuple(shape),
                "element_type": element_type,
                # tensor seems to get lowererd to memref<..., strided<[?, ?, ?], offset: ?>>
                "offset": DYNAMIC,
                "strides": tuple([DYNAMIC] * len(shape)),
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
                f"having a Tensor with DType {self.element_type.__qualname__} "
                f"is not supported"
            )

        if isinstance(rep, OpView):
            rep = rep.result

        if (rep_type := type(rep.type)) is not RankedTensorType:
            raise TypeError(f"{rep_type} cannot be casted as a Tensor")

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

    def lower(self) -> tuple[Value]:
        return (self.value,)

    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]:
        if cls.shape is None or cls.element_type is None:
            e = TypeError(
                "attempted to lower Tensor without defined dims or type"
            )
            if (clsname := cls.__name__) != Tensor._default_subclass_name:
                e.add_note(f"hint: class name is {clsname}")
            raise e

        return (
            RankedTensorType.get(
                list(cls.shape), lower_single(cls.element_type)
            ),
        )

    # TODO: potential dead code. MLIR already compute all the dims for us if we
    # pass the input tensor as the output tensor as well. I feel this can still
    # be useful if the end PyDSL user wants to get the shape though.
    # Update: on_getitem and on_setitem use this function now.
    @property
    def runtime_shape(self) -> RuntimeMemrefShape:
        """
        Return the shape of the tensor as it exists at runtime.

        If one of the dimension sizes is dynamic, a tensor.dim operator is
        returned instead for that dimension.
        """
        return [
            (
                d
                if d != DYNAMIC
                else tensor.DimOp(lower_single(self), lower_single(Index(i)))
            )
            for i, d in enumerate(self.shape)
        ]

    def on_getitem(
        self: typing.Self, visitor: "ToMLIRBase", slice: ast.AST
    ) -> SubtreeOut:
        key_list = subtree_to_slices(visitor, visitor.visit(slice))
        dim = len(self.shape)

        # If all indices are integers, not slices, do an extract op
        if len(key_list) == dim and all(
            isinstance(key, SupportsIndex) for key in key_list
        ):
            key_list = lower_flatten([Index(key) for key in key_list])
            rep = tensor.extract(lower_single(self), key_list)
            return self.element_type(rep)

        # Otherwise, do an extract_slice op
        lo_list, size_list, step_list = slices_to_mlir_format(
            key_list, self.runtime_shape
        )

        # We make the result a tensor with all dynamic dimensions
        result_type = TensorFactory(tuple([DYNAMIC] * dim), self.element_type)
        dynamic_i64_attr = DenseI64ArrayAttr.get([DYNAMIC] * dim)

        rep = tensor.extract_slice(
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
        value_st = visitor.visit(value)
        key_list = subtree_to_slices(visitor, visitor.visit(slice))
        dst_dim = len(self.shape)

        # If all indices are integers, not slices, do an insert op
        if len(key_list) == dst_dim and all(
            isinstance(key, SupportsIndex) for key in key_list
        ):
            value_mlir = lower_single(self.element_type(value_st))
            key_list = lower_flatten([Index(key) for key in key_list])
            rep = tensor.insert(value_mlir, lower_single(self), key_list)
            self.value = rep
            return rep

        # Otherwise, do an insert_slice op
        lo_list, size_list, step_list = slices_to_mlir_format(
            key_list, self.runtime_shape
        )
        src_dim = len(value_st.shape)

        if dst_dim != src_dim:
            raise TypeError(
                "trying to insert_slice with tensors of different ranks"
            )

        # We make all offsets and strides dynamic
        dynamic_i64_attr = DenseI64ArrayAttr.get([DYNAMIC] * src_dim)

        # We use static dimensions for the shape of the source tensor whenever possible.
        # This is necessary to not cause an error (can't use tensor of static shape
        # if the op expects dynamic shape).
        src_size_i64_attr = DenseI64ArrayAttr.get(value_st.shape)
        size_list = [
            size_list[i]
            for i in range(src_dim)
            if value_st.shape[i] == DYNAMIC
        ]

        rep = tensor.insert_slice(
            lower_single(value_st),
            lower_single(self),
            lo_list,
            size_list,
            step_list,
            dynamic_i64_attr,
            src_size_i64_attr,
            dynamic_i64_attr,
        )
        self.value = rep
        return rep

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

    @CallMacro.generate(method_type=MethodType.INSTANCE)
    def cast(
        visitor: ToMLIRBase, self: typing.Self, shape: Evaluated
    ) -> typing.Self:
        """
        Convert a tensor from one type to an equivalent type without changing
        any data elements. The resulting tensor type will have the same element
        type. shape is the shape of the new tensor and must be known at compile
        time. For any constant dimensions of shape, the input tensor must
        actually have that dimension at runtime, otherwise the operation is
        invalid.

        Note: this function only returns a tensor with the updated type, it
        does not modify the type of the input tensor.

        Example:
        ```
        def f(t1: Tensor[F32, DYNAMIC, 32, 5]) -> Tensor[F32, 64, 32, DYNAMIC]:
            # Only valid if the first dimension of t1 is always 64
            t2 = t1.cast((64, 32, DYNAMIC))
            return t2
        ```
        """

        shape = tuple(shape)

        if not all(isinstance(x, int) for x in shape):
            raise TypeError(
                f"shape should be a tuple of integers known at compile time ",
                f"got {repr(shape)}",
            )

        assert_shapes_compatible(self.shape, shape)

        result_type = self.class_factory(shape, self.element_type)
        rep = tensor.cast(lower_single(result_type), lower_single(self))
        return result_type(rep)


# Convenient alias
TensorFactory = Tensor.class_factory


@CallMacro.generate()
def empty(
    visitor: ToMLIRBase, shape: Compiled, dtype: Evaluated
) -> SubtreeOut:
    if not isinstance(shape, Tuple):
        raise TypeError(f"shape should be a Tuple, got {type(shape)}")

    shape = shape.as_iterable(visitor)
    static_shape, dynamic_sizes = split_static_dynamic_dims(shape)

    t_type = TensorFactory(tuple(static_shape), dtype)
    return t_type(
        tensor.empty(lower_single(t_type), lower_flatten(dynamic_sizes))
    )


# This is not at the top of the file to avoid circular import
import pydsl.linalg as linalg


@InlineFunction.generate()
def full(shape, val, dtype) -> typing.Any:
    res = empty(shape, dtype)
    return linalg.fill(res, val)


@InlineFunction.generate()
def zeros(shape, dtype) -> typing.Any:
    return full(shape, 0, dtype)


@InlineFunction.generate()
def ones(shape, dtype) -> typing.Any:
    return full(shape, 1, dtype)
