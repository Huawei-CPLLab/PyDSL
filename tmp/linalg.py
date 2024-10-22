import typing
from python_bindings_mlir.memref import MemRef
from python_bindings_mlir.tensor import Tensor

# Compiled TypeAlias needs Lowerable
from python_bindings_mlir.type import Float, Int, Sign, Lowerable

from mlir.dialects import linalg as mlir_linalg
from mlir.dialects.linalg import DefinedOpCallable
from mlir.dialects.linalg.opdsl.lang.comprehension import TypeFn, BinaryFn
import mlir.dialects.linalg as linalg

from python_bindings_mlir.common import CallMacro, Compiled
from python_bindings_mlir.helpers import lower_single, ToMLIRBase


def _gen_elementwise_unary_macro(op: DefinedOpCallable) -> CallMacro:
    """
    Uses a template macro to create any macro wrappers for elementwise unary fn
    supported by linalg dialect.
    """

    # The template macro
    @CallMacro.generate()
    def op_macro(visitor: "ToMLIRBase", x: Compiled) -> Tensor | MemRef:
        if not issubclass(x.element_type, Float):
            raise TypeError(
                f"Elementwise linalg operation {op.op_name} expected Float "
                f"element type, got {x.element_type.__name__}"
            )

        match x:
            case Tensor():
                # Tensors are immutable, so a new Tensor is always made
                tensor = x
                return type(tensor)(
                    op(lower_single(tensor), outs=[lower_single(tensor)])
                )
            case MemRef():
                # MemRefs are mutable, so they are always written in-place
                memref = x
                op(lower_single(memref), outs=[lower_single(memref)])
                return memref
            case _:
                raise Exception(
                    f"Elementwise linalg operation {op.op_name} expected "
                    f"MemRef or Tensor, got {type(x).__name__}"
                )

    return op_macro


# Define elementwise unary operators
exp = _gen_elementwise_unary_macro(mlir_linalg.exp)
log = _gen_elementwise_unary_macro(mlir_linalg.log)
abs = _gen_elementwise_unary_macro(mlir_linalg.abs)
ceil = _gen_elementwise_unary_macro(mlir_linalg.ceil)
floor = _gen_elementwise_unary_macro(mlir_linalg.floor)
negf = _gen_elementwise_unary_macro(mlir_linalg.negf)
round = _gen_elementwise_unary_macro(mlir_linalg.round)
sqrt = _gen_elementwise_unary_macro(mlir_linalg.sqrt)
rsqrt = _gen_elementwise_unary_macro(mlir_linalg.rsqrt)
square = _gen_elementwise_unary_macro(mlir_linalg.square)
tanh = _gen_elementwise_unary_macro(mlir_linalg.tanh)
erf = _gen_elementwise_unary_macro(mlir_linalg.erf)

# TODO: For some reason, reciprocal doesn't have an unary op. The Python
# binding should be modified somewhat...

BinaryTypeDecision = typing.Callable[
    [Tensor | MemRef], tuple[BinaryFn, TypeFn]
]


# Elementwise binary macros are defined using linalg.elemwise_binary instead
# of specific linalg operators. This is because some operations such as
# max_unsigned and fexp are only supported by elemwise_binary.
def _gen_elementwise_binary_macro(
    type_decision: BinaryTypeDecision,
) -> CallMacro:
    @CallMacro.generate()
    def op_macro(
        visitor: "ToMLIRBase", x: Compiled, y: Compiled
    ) -> Tensor | MemRef:
        if x.shape != y.shape:
            raise Exception(
                "operands with differing shapes used in elementwise binary "
                "linalg operation. Shape must be the same"
            )

        if x.element_type != y.element_type:
            raise Exception(
                "operands with differing element types used in elementwise "
                "binary linalg operation. Element types must be the same"
            )

        # Get the respective fn and typefn from type_decision
        fn, typefn = type_decision(x)

        match x, y:
            case Tensor(), Tensor():
                # Tensors are immutable, so a new Tensor is always made
                return type(x)(
                    linalg.elemwise_binary(
                        lower_single(x),
                        lower_single(y),
                        outs=[lower_single(x)],
                        fun=fn,
                        cast=typefn,
                    )
                )
            case MemRef(), MemRef():
                # MemRefs are mutable, so they are always written in-place
                # TODO: writing is always done to the LHS for now
                linalg.elemwise_binary(
                    lower_single(x),
                    lower_single(y),
                    outs=[lower_single(x)],
                    fun=fn,
                    cast=typefn,
                )
                return x
            case _:
                raise Exception(
                    f"elementwise linalg operation {fn.fn_name} expected "
                    f"MemRef or Tensor, got {type(x).__name__}"
                )

    return op_macro


def _float_and_int(
    fn_signed: DefinedOpCallable, fn_unsigned: DefinedOpCallable
) -> BinaryTypeDecision:
    """
    TypeDecision generator for elementwise binary operations that
    support both Floats and Ints.

    Result of this function can be passed into _gen_elementwise_binary_macro.
    """

    def payload(x):
        t = x.element_type
        if issubclass(t, Float):
            return fn_signed, TypeFn.cast_signed
        elif issubclass(t, Int) and t.sign == Sign.SIGNED:
            return fn_signed, TypeFn.cast_signed
        elif issubclass(t, Int) and t.sign == Sign.UNSIGNED:
            return fn_unsigned, TypeFn.cast_unsigned
        else:
            raise TypeError(
                f"{x.element_type.__qualname__} is not supported in this "
                f"elementwise binary linalg operations. Only Float and Int "
                f"are supported"
            )

    return payload


def _float_only(fn: DefinedOpCallable) -> BinaryTypeDecision:
    """
    TypeDecision generator for elementwise binary operations that
    only support Floats.

    Result of this function can be passed into _gen_elementwise_binary_macro.
    """

    def payload(x):
        t = x.element_type
        if issubclass(t, Float):
            return fn, TypeFn.cast_signed
        else:
            raise TypeError(
                f"{x.element_type.__qualname__} is not supported in this "
                f"elementwise binary linalg operations. Only Float is "
                f"supported"
            )

    return payload


# Define elementwise binary operators
add = _gen_elementwise_binary_macro(_float_and_int(BinaryFn.add, BinaryFn.add))
sub = _gen_elementwise_binary_macro(_float_and_int(BinaryFn.sub, BinaryFn.sub))
mul = _gen_elementwise_binary_macro(_float_and_int(BinaryFn.mul, BinaryFn.mul))
div = _gen_elementwise_binary_macro(
    _float_and_int(BinaryFn.div, BinaryFn.div_unsigned)
)
max = _gen_elementwise_binary_macro(
    _float_and_int(BinaryFn.max_signed, BinaryFn.max_unsigned)
)
min = _gen_elementwise_binary_macro(
    _float_and_int(BinaryFn.min_signed, BinaryFn.min_unsigned)
)
powf = _gen_elementwise_binary_macro(_float_only(BinaryFn.powf))