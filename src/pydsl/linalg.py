from collections.abc import Callable

import mlir.dialects.linalg as linalg
from mlir.dialects import linalg as mlir_linalg
from mlir.dialects.linalg import DefinedOpCallable
from mlir.dialects.linalg.opdsl.lang.comprehension import BinaryFn, TypeFn

from pydsl.macro import CallMacro, Compiled
from pydsl.memref import MemRef
from pydsl.protocols import ToMLIRBase, lower_single
from pydsl.tensor import Tensor, TensorFactory

# Compiled TypeAlias needs Lowerable
from pydsl.type import Float, Int, Sign


# TODO: it seems we currently only support Float element_type for unary ops.
# TODO: currently, all unary ops only take one input operand, and generate the
# output operand automatically, doing the operation in-place in case of a
# MemRef. It might be useful to also allow the user to specify a different
# output operand? They can currently achieve the same behaviour with a copy
# and possibly a cast, but I don't know if it gets optimized the same way
# (e.g. linalg.exp(tensor<?xf32>) -> tensor<?xf64>).
def _gen_elementwise_unary_macro(op: DefinedOpCallable) -> CallMacro:
    """
    Uses a template macro to create any macro wrappers for elementwise unary fn
    supported by linalg dialect.
    """

    # The template macro
    @CallMacro.generate()
    def op_macro(visitor: ToMLIRBase, x: Compiled) -> Tensor | MemRef:
        if not isinstance(x, (Tensor, MemRef)):
            raise TypeError(
                f"linalg elementwise unary operation expects an argument of "
                f"type Tensor or MemRef, got {type(x).__qualname__}"
            )

        if not issubclass(x.element_type, Float):
            raise TypeError(
                f"linalg elementwise unary operation {op.op_name} expects"
                f"Float element type, got {x.element_type.__qualname__}"
            )

        if isinstance(x, Tensor):
            # Return a new tensor, since tensors are SSA in MLIR
            rep = op(lower_single(x), outs=[lower_single(x)])
            return type(x)(rep)
        else:
            # Return x, since memrefs are modified in-place and op returns
            # nothing useful
            op(lower_single(x), outs=[lower_single(x)])
            return x

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

# TODO: For some reason, reciprocal doesn't have a unary op. The Python
# binding should be modified somewhat...


BinaryTypeDecision = Callable[
    [[Tensor | MemRef], [Tensor | MemRef], [Tensor | MemRef]],
    tuple[BinaryFn, TypeFn],
]
"""
A function that takes in the three arguments of an elementwise binary op
(in1, in2, out), and returns the appropriate signed/unsigned function and cast
to apply. Both input operands will be cast to the element_type of the output
operand. Type checking to make sure all 3 element_types are appropriate should
also be done here.

The cast part specifies whether to use a signed or unsigned cast if element
types differ. Examples from documentation of TypeFn:
- cast_signed(I32 -> I64) -> `arith.ExtSIOp`
- cast_unsigned(I32 -> I64) -> `arith.ExtUIOp`
"""


def _float_and_int(
    fn_signed: DefinedOpCallable, fn_unsigned: DefinedOpCallable
) -> BinaryTypeDecision:
    """
    TypeDecision generator for elementwise binary operations that
    support both Floats and Ints.

    Result of this function can be passed into _gen_elementwise_binary_macro.
    """

    def payload(
        x: Tensor | MemRef, y: Tensor | MemRef, out: Tensor | MemRef
    ) -> tuple[BinaryFn, TypeFn]:
        for arg in (x, y, out):
            t = arg.element_type
            if not issubclass(t, (Float, Int)):
                raise TypeError(
                    f"this linalg elementwise binary operation only supports "
                    f"arguments with element type Float or Int, got "
                    f"{t.__qualname__}"
                )

        t = out.element_type

        if issubclass(t, Float):
            return fn_signed, TypeFn.cast_signed
        elif issubclass(t, Int) and t.sign == Sign.SIGNED:
            return fn_signed, TypeFn.cast_signed
        elif issubclass(t, Int) and t.sign == Sign.UNSIGNED:
            return fn_unsigned, TypeFn.cast_unsigned
        else:
            assert (
                False
            ), "Already checked type of t, this should be uncreachable"

    return payload


def _float_only(fn: DefinedOpCallable) -> BinaryTypeDecision:
    """
    TypeDecision generator for elementwise binary operations that only support
    Floats.

    Result of this function can be passed into _gen_elementwise_binary_macro.
    """

    def payload(
        x: Tensor | MemRef, y: Tensor | MemRef, out: Tensor | MemRef
    ) -> tuple[BinaryFn, TypeFn]:
        for arg in (x, y, out):
            t = arg.element_type
            if not issubclass(t, (Float, Int)):
                raise TypeError(
                    f"this linalg elementwise binary operation only supports "
                    f"arguments with element type Float or Int, got "
                    f"{t.__qualname__}"
                )

        t = out.element_type

        if issubclass(t, Float):
            return fn, TypeFn.cast_signed
        else:
            raise TypeError(
                f"this linalg elementwise binary operation only supports "
                f"output element type Float, got {t.__qualname__}"
            )

    return payload


# Elementwise binary macros are defined using linalg.elemwise_binary instead
# of specific linalg operators. This is because some operations such as
# max_unsigned and fexp are only supported by elemwise_binary.
#
# We currently allow different element types, I believe MLIR will cast
# everything to the element type of the output operand.
#
# TODO: add a proper docstring readable by the user to write down this casting
# behaviour (don't want to write same docstrict for all individual ops, like
# add, max, etc., but only that is exposed to the user). Maybe possible once
# we add generic elementwise binary?
def _gen_elementwise_binary_macro(
    type_decision: BinaryTypeDecision,
) -> CallMacro:
    @CallMacro.generate()
    def op_macro(
        visitor: ToMLIRBase, x: Compiled, y: Compiled, *, out: Compiled
    ) -> Tensor | MemRef:
        # This check must be done first, otherwise x.shape, y.element_type fail
        for arg in (x, y, out):
            if not isinstance(arg, (Tensor, MemRef)):
                raise TypeError(
                    f"linalg elementwise binary operation expects arguments "
                    f"of type Tensor or MemRef, got {type(arg).__qualname__}"
                )

        is_x_tensor = isinstance(x, Tensor)
        is_y_tensor = isinstance(x, Tensor)
        is_out_tensor = isinstance(x, Tensor)

        if is_x_tensor != is_y_tensor or is_x_tensor != is_out_tensor:
            raise TypeError(
                f"arguments to elementwise binary operation must be all "
                f"Tensor or all MemRef, got {type(x).__qualname__}, "
                f"{type(y).__qualname__}, {type(out).__qualname__}"
            )

        if x.shape != y.shape or x.shape != out.shape:
            raise TypeError(
                f"linalg elementwise binary operation expects arguments with "
                f"the same shape, got arguments with shapes {x.shape}, "
                f"{y.shape}, {out.shape}"
            )

        # Get the respective fn and typefn from type_decision
        fn, typefn = type_decision(x, y, out)

        rep = linalg.elemwise_binary(
            lower_single(x),
            lower_single(y),
            outs=[lower_single(out)],
            fun=fn,
            cast=typefn,
        )

        if is_out_tensor:
            # A new tensor needs to be returned, only the shape and element
            # type of out is used
            return type(out)(rep)
        else:
            # MemRefs are modified in-place
            return out

    return op_macro


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


@CallMacro.generate()
def batch_matmul(visitor: "ToMLIRBase", x: Compiled, y: Compiled):
    raise NotImplementedError(
        "batch_matmul implementation is not quite complete"
    )

    if not x.element_type == y.element_type:
        raise Exception(
            "operands with differing element types used in matmul"
            "linalg operation. Element types must be the same"
        )
    if not (len(x.shape) == len(y.shape) == 3):
        raise Exception("operands need to have rank of 3")
    if not (x.shape[0] == y.shape[0]):
        raise Exception("all operands need to have the same batch number")
    if x.shape[2] != y.shape[1]:
        raise Exception(
            "operands need to be in batch matmul form. e.g (b,m,k)x(b,k,n) = (b,m,n)"
        )

    t = x.element_type

    # TODO: what is typeFn? it is never used
    if issubclass(t, Float):
        typeFn = TypeFn.cast_signed
    elif issubclass(t, Int) and t.sign == Sign.SIGNED:
        typeFn = TypeFn.cast_signed
    elif issubclass(t, Int) and t.sign == Sign.UNSIGNED:
        typeFn = TypeFn.cast_unsigned
    else:
        raise TypeError(
            f"{x.element_type.__qualname__} is not supported in this "
            f"linalg.batch_matmul operations. Only Float and Int "
            f"are supported"
        )

    match x, y:
        case Tensor(), Tensor():
            ret_shape = [x.shape[0], x.shape[1], y.shape[2]]
            ret_tensor = TensorFactory(tuple(ret_shape), x.element_type)
            ret_runtime_shape = [
                x.runtime_shape[0],
                x.runtime_shape[1],
                y.runtime_shape[2],
            ]
            ret = ret_tensor(
                # TODO needs tensor.empty
                build_empty_tensor(
                    ret_runtime_shape, lower_single(x.element_type)
                )
            )
            return type(ret)(
                linalg.batch_matmul(
                    lower_single(x), lower_single(y), outs=[lower_single(ret)]
                )
            )
        # TODO: support memref operands
        case _:
            raise Exception(
                f"batch_matmul operation  expected "
                f"Tensor, got {type(x).__name__}"
            )


@CallMacro.generate()
def fill(visitor: "ToMLIRBase", c: Compiled, x: Compiled):
    """
    Fill a MemRef/Tensor with the single value c.
    If x is a MemRef, it is modified in-place.
    If x is a Tensor, a new Tensor is returned.
    """

    if not isinstance(x, (Tensor, MemRef)):
        raise TypeError(
            f"linalg.fill expects Tensor or MemRef, got {type(x).__qualname__}"
        )

    # MLIR also supports casting, but we must cast in PyDSL anyway, to deal
    # with the case when c is a Python constant expression
    c = x.element_type(c)

    if isinstance(x, Tensor):
        rep = mlir_linalg.fill(lower_single(c), outs=[lower_single(x)])
        return type(x)(rep)
    else:
        mlir_linalg.fill(lower_single(c), outs=[lower_single(x)])
        return x
