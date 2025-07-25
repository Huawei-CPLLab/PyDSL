import collections.abc as cabc
import ast
import typing
from functools import cache
from pydsl.macro import CallMacro, Compiled, Evaluated

import mlir.dialects.tensor as mlir_tensor
import mlir.ir as mlir
from mlir.ir import DenseI64ArrayAttr, OpView, RankedTensorType, Value
import mlir.dialects._tensor_ops_gen as tensor_ops_gen

from pydsl.memref import (
    UsesRMRD,
    RuntimeMemrefShape,
    slices_to_mlir_format,
    subtree_to_slices,
)

from pydsl.type import (
    Index,
    Lowerable,
    lower,
    lower_flatten,
    lower_single,
    SupportsIndex,
    Tuple,
)
from pydsl.protocols import SubtreeOut, ToMLIRBase

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
    shape: tuple[int] = None
    element_type: Lowerable = None
    offset: int = None
    strides: tuple[int] = None
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
    @cache
    def class_factory(
        shape: tuple[int], element_type, *, name=_default_subclass_name
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

        if not isinstance(shape, cabc.Iterable):
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
        if not all([cls.shape, cls.element_type]):
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
                else mlir_tensor.DimOp(
                    lower_single(self), lower_single(Index(i))
                )
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
            rep = tensor_ops_gen.extract(lower_single(self), key_list)
            return self.element_type(rep)

        # Otherwise, do an extract_slice op
        lo_list, size_list, step_list = slices_to_mlir_format(
            key_list, self.runtime_shape
        )

        # We make the result a tensor with all dynamic dimensions
        result_type = TensorFactory(tuple([DYNAMIC] * dim), self.element_type)
        dynamic_i64_attr = DenseI64ArrayAttr.get([DYNAMIC] * dim)

        rep = tensor_ops_gen.extract_slice(
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
            rep = tensor_ops_gen.insert(
                value_mlir, lower_single(self), key_list
            )
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

        rep = tensor_ops_gen.insert_slice(
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


# Convenient alias
TensorFactory = Tensor.class_factory


def verify_tensor_type(t_type: type[Tensor]):
    if not issubclass(t_type, Tensor):
        raise TypeError(
            f"the type being allocated must be a subclass of Tensor, got "
            f"{t_type}"
        )


def verify_dynamics_val(t_type: type[Tensor], dynamics_val: Tuple) -> None:
    dynamics_val = lower(dynamics_val)

    if not isinstance(dynamics_val, cabc.Iterable):
        raise TypeError(f"{repr(dynamics_val)} is not iterable")

    if (actual_dyn := len(dynamics_val)) != (
        target_dyn := t_type.shape.count(DYNAMIC)
    ):
        raise ValueError(
            f"Tensor has {target_dyn} dynamic dimensions to be filled, "
            f"but emptyOp received {actual_dyn}"
        )


@CallMacro.generate()
def empty(
    visitor: ToMLIRBase, t_type: Evaluated, dynamics_val: Compiled = None
) -> SubtreeOut:
    if dynamics_val is None:
        dynamics_val = Tuple.from_values(visitor, *())
    verify_tensor_type(t_type)
    verify_dynamics_val(t_type, dynamics_val)
    dynamics_val = [lower_single(Index(i)) for i in lower(dynamics_val)]
    idx = 0
    orig_shape = t_type.shape
    shape = [0] * len(orig_shape)
    for i in range(len(orig_shape)):
        if orig_shape[i] == DYNAMIC:
            shape[i] = dynamics_val[idx]
            idx += 1
        else:
            shape[i] = orig_shape[i]
    return t_type(mlir_tensor.empty(shape, lower_single(t_type.element_type)))
