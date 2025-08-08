import ast
from collections.abc import Iterable
from functools import cache

import mlir.ir as mlir
from mlir.ir import OpView, Value, VectorType

from pydsl.type import Lowerable
from pydsl.protocols import lower_single, SubtreeOut, ToMLIRBase


class Vector:
    """
    Class representing an MLIR Vector type. Copied from Tensor, extremely
    bare-bone right now and no operations are supported. Created since we need
    this downstream. In the future, we can use vectors to properly support
    the vector dialect.
    """

    value: Value
    shape: tuple[int]
    element_type: Lowerable

    _default_subclass_name = "AnnonymousVectorSubclass"
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
        Create a new subclass of Vector dynamically with the specified
        dimensions and type.
        """
        # TODO: this check cannot be done right now because types can't be
        # lowered outside of MLIR context
        # if len(lower(element_type)) != 1: raise TypeError(f"Vector cannot
        # store composite types, got {element_type} which lowers to
        # {lower(element_type)}")

        if not isinstance(shape, Iterable):
            raise TypeError(
                f"Vector requires shape to be iterable, got {type(shape)}"
            )

        return type(
            name,
            (Vector,),
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
                f"having a Vector with DType {self.element_type.__qualname__} "
                f"is not supported"
            )

        if isinstance(rep, OpView):
            rep = rep.result

        if (rep_type := type(rep.type)) is not VectorType:
            raise TypeError(f"{rep_type} cannot be casted as a Vector")

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
                "attempted to lower Vector without defined dims or type"
            )
            if (clsname := cls.__name__) != Vector._default_subclass_name:
                e.add_note(f"hint: class name is {clsname}")
            raise e

        return (
            VectorType.get(list(cls.shape), lower_single(cls.element_type)),
        )

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
VectorFactory = Vector.class_factory
