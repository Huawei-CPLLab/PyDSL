import collections.abc as cabc
import typing
from ctypes import c_void_p
from functools import cache
from pydsl.macro import CallMacro, Compiled, Evaluated

from mlir.dialects import arith
import mlir.dialects.tensor as mlir_tensor
import mlir.ir as mlir
from mlir.ir import (
    OpView,
    RankedTensorType,
    Value,
)

from pydsl.memref import UsesRMRD
from pydsl.type import Index, Lowerable, lower_single, Tuple
from pydsl.protocols import SubtreeOut, ToMLIRBase, lower

DYNAMIC = -9223372036854775808

# used for the virtual static-typing in PyDSL
Dynamic = typing.Literal[-9223372036854775808]
RawRMRD = tuple[c_void_p | int]
RuntimeTensorShape = list[int | Value]

# based on example in PEP 646: https://peps.python.org/pep-0646/
# TODO: these currently are unused
DType = typing.TypeVar("DType")
Shape = typing.TypeVarTuple("Shape")


class Tensor(typing.Generic[DType, *Shape], UsesRMRD):
    """
    TODO: this Tensor type is fairly bare-bone right now. It's meant mostly
    to demonstrate other operations. It's also limited to ranked versions
    of rank type for now
    """

    value: Value
    shape: tuple[int] = None
    element_type: Lowerable = None
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
        shape: tuple[int], element_type, name=_default_subclass_name
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
    @property
    def runtime_shape(self) -> RuntimeTensorShape:
        """
        Return the shape of the tensor as it exists at runtime.

        If one of the dimension size is dynamic, a tensor.dim operator is
        returned instead for that dimension.
        """
        return [
            (
                d
                if d != DYNAMIC
                else mlir_tensor.DimOp(self.value, lower_single(Index(i)))
            )
            for i, d in enumerate(self.shape)
        ]


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
            f"Temspr has {target_dyn} dynamic dimensions to be filled, "
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
