from collections.abc import Callable, Iterable

import mlir.dialects.linalg as mlir_linalg
from mlir.dialects.linalg import DefinedOpCallable
from mlir.dialects.linalg.opdsl.lang.comprehension import BinaryFn, TypeFn
from mlir.ir import InsertionPoint

from pydsl.macro import CallMacro, Compiled, Evaluated
from pydsl.memref import MemRef, are_dims_compatible, DYNAMIC
from pydsl.protocols import ToMLIRBase, lower, lower_single
from pydsl.tensor import Tensor, TensorFactory

# Compiled TypeAlias needs Lowerable
from pydsl.type import Float, Int, Sign


def verify_memref_tensor_types(*args):
    """
    Checks that the elements of args are either all MemRef or all Tensor.
    Raises a TypeError otherwise.
    """

    arg_type_str = ", ".join(type(arg).__qualname__ for arg in args)
    if not all(isinstance(arg, (MemRef, Tensor)) for arg in args):
        raise TypeError(
            f"linalg operation expects arguments of type MemRef or "
            f"Tensor, got {arg_type_str}"
        )

    is_tensor_arr = tuple(isinstance(arg, Tensor) for arg in args)
    if not all(is_tensor_arr[0] == is_t for is_t in is_tensor_arr):
        raise TypeError(
            f"linalg operation expected arguments to be all MemRef or all "
            f"Tensor, got {arg_type_str}"
        )


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
        verify_memref_tensor_types(x)

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
        verify_memref_tensor_types(x, y, out)

        if x.shape != y.shape or x.shape != out.shape:
            raise TypeError(
                f"linalg elementwise binary operation expects arguments with "
                f"the same shape, got arguments with shapes {x.shape}, "
                f"{y.shape}, {out.shape}"
            )

        # Get the respective fn and typefn from type_decision
        fn, typefn = type_decision(x, y, out)

        rep = mlir_linalg.elemwise_binary(
            lower_single(x),
            lower_single(y),
            outs=[lower_single(out)],
            fun=fn,
            cast=typefn,
        )

        if isinstance(out, Tensor):
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
def matmul(
    visitor: "ToMLIRBase", x: Compiled, y: Compiled, *, init: Compiled = None
):
    verify_memref_tensor_types(x, y)
    if init is not None:
        verify_memref_tensor_types(init)

    # TODO: if init is None, construct a zero tensor
    if init is None:
        raise ValueError("linalg.matmul requires an init value")

    # TODO: allow different element types and casting in the future,
    # need to raise PR in llvm-project to add `cast` parameter to MatMulOp
    if len({x.element_type, y.element_type, init.element_type}) != 1:
        raise TypeError(
            "operands with differing element types used in matmul"
            "linalg operation. Element types must be the same"
        )

    if not issubclass(x.element_type, (Float, Int)):
        raise TypeError(
            f"{x.element_type.__qualname__} is not supported in this "
            f"linalg.matmul operation. Only Float and Int "
            f"are supported"
        )

    if not (len(x.shape) == len(y.shape) == len(init.shape) == 2):
        raise TypeError("operands and init need to have rank of 2")
    if not (
        are_dims_compatible(x.shape[1], y.shape[0])
        and are_dims_compatible(x.shape[0], init.shape[0])
        and are_dims_compatible(y.shape[1], init.shape[1])
    ):
        raise TypeError(
            "operands need to be in matmul form. e.g (m,k)x(k,n) = (m,n)"
        )

    rst = mlir_linalg.matmul(
        lower_single(x), lower_single(y), outs=[lower_single(init)]
    )
    if isinstance(init, Tensor):
        return type(init)(rst)
    else:
        return init


@CallMacro.generate()
def batch_matmul(
    visitor: "ToMLIRBase", x: Compiled, y: Compiled, *, init: Compiled = None
):
    verify_memref_tensor_types(x, y)
    if init is not None:
        verify_memref_tensor_types(init)

    # TODO: if init is None, construct a zero tensor
    if init is None:
        raise ValueError("linalg.batch_matmul requires an init value")

    # TODO: allow different element types and casting in the future,
    # need to raise PR in llvm-project to add `cast` parameter to MatMulOp
    if len({x.element_type, y.element_type, init.element_type}) != 1:
        raise TypeError(
            "operands with differing element types used in batch_matmul"
            "linalg operation. Element types must be the same"
        )

    if not issubclass(x.element_type, (Float, Int)):
        raise TypeError(
            f"{x.element_type.__qualname__} is not supported in this "
            f"linalg.batch_matmul operation. Only Float and Int "
            f"are supported"
        )

    if not (len(x.shape) == len(y.shape) == len(init.shape) == 3):
        raise TypeError("operands need to have rank of 3")
    if not (
        are_dims_compatible(x.shape[0], y.shape[0], init.shape[0])
        and are_dims_compatible(x.shape[1], init.shape[1])
        and are_dims_compatible(x.shape[2], y.shape[1])
        and are_dims_compatible(y.shape[2], init.shape[2])
    ):
        raise TypeError(
            "operands need to be in batch matmul form. e.g (b,m,k)x(b,k,n) = (b,m,n)"
        )

    rst = mlir_linalg.batch_matmul(
        lower_single(x), lower_single(y), outs=[lower_single(init)]
    )
    if isinstance(init, Tensor):
        return type(init)(rst)
    else:
        return init


@CallMacro.generate()
def fill(visitor: "ToMLIRBase", x: Compiled, c: Compiled) -> Tensor | MemRef:
    """
    Fill a MemRef/Tensor with the single value c.
    If x is a MemRef, it is modified in-place.
    If x is a Tensor, a new Tensor is returned.
    """
    verify_memref_tensor_types(x)

    # MLIR also supports casting, but we must cast in PyDSL anyway, to deal
    # with the case when c is a Python constant expression
    c = x.element_type(c)
    rep = mlir_linalg.fill(lower_single(c), outs=[lower_single(x)])

    if isinstance(x, Tensor):
        return type(x)(rep)
    else:
        return x


def verify_reduced_dims(
    in_shape: tuple[int], expected_shape: tuple[int], dims: Iterable[int]
):
    """
    Verifies that dimensions with indices in dims can be removed from in_shape
    to result in expected_shape. Raises a ValueError if dims is invalid, and
    a TypeError if the final dimensions don't match.
    """
    new_shape = []
    prv_dim = -1
    for dim in dims:
        if dim < 0 or dim > len(in_shape):
            raise ValueError(
                f"dimension {dim} is out of bounds, input has rank "
                f"{len(in_shape)}"
            )
        if dim <= prv_dim:
            raise ValueError(f"dims should be in increasing order, got {dims}")
        for i in range(prv_dim + 1, dim):
            new_shape.append(in_shape[i])
        prv_dim = dim

    for i in range(prv_dim + 1, len(in_shape)):
        new_shape.append(in_shape[i])

    if tuple(new_shape) != tuple(expected_shape):
        raise ValueError(
            f"removing dimensions {dims} from the shape {in_shape} results in "
            f"{tuple(new_shape)}, but expected it to be {expected_shape}"
        )


@CallMacro.generate()
def reduce(
    visitor: ToMLIRBase,
    combiner: Compiled,
    x: Compiled,
    *,
    init: Compiled,
    dims: Evaluated,
) -> Tensor | MemRef:
    """
    Reduces x along the given dimensions using the given combiner function.

    combiner should be a function that takes two operands of type
    x.element_type and init.element_type, and returns a single value of type
    init.element_type. Currently, combiner can be an InlineFunction or a
    CallMacro that takes 2 Compiled arguments.

    The output will have the same shape and type as init.
    For MemRefs, init will be modified in-place.
    For Tensors, a new tensor will be returned.

    For each output value, it is initialized to the corresponding value of
    init, then the combiner function is applied repeatedly to the current value
    of the output and an appropriate element of x. Specifically, combiner
    should have signature (cur: init.element_type, new_v: x.element_type) ->
    init.element_type.

    dims should specify the dimensions that will be eliminated from x in
    increasing order.
    """
    verify_memref_tensor_types(x, init)
    verify_reduced_dims(x.shape, init.shape, dims)

    result_types = []
    if isinstance(init, Tensor):
        result_types.append(lower_single(type(init)))

    rep = mlir_linalg.ReduceOp(result_types, lower(x), lower(init), dims)
    in_t = x.element_type
    out_t = init.element_type

    # This feels like it could be done by the MLIR Python binding.
    # This adds a new MLIR "block" with argument types in_t, out_t.
    rep.combiner.blocks.append(lower_single(in_t), lower_single(out_t))
    body = rep.combiner.blocks[0]

    with InsertionPoint(body):
        arg0 = in_t(body.arguments[0])
        arg1 = out_t(body.arguments[1])
        # Swap order of arguments passed to PyDSL function
        res = out_t(combiner(visitor, arg1, arg0))
        mlir_linalg.YieldOp(lower(res))

    if isinstance(init, Tensor):
        return type(init)(rep)
    else:
        return init


@CallMacro.generate()
def broadcast(
    visitor: ToMLIRBase, x: Compiled, *, out: Compiled, dims: Evaluated
) -> Tensor | MemRef:
    """
    Broadcasts elements of x to out by adding dims.

    For MemRefs, out is modified in-place.
    For Tensors, a new Tensor is returned, and only the shape of out is used.

    out should be a Tensor/MemRef such that after removing the dimensions at
    indices specified by dims, its shape becomes identical to the shape of x.

    dims should specify the dimension indices to add to x, in increasing order.

    Currently, x and out must have the same element type.
    """
    verify_memref_tensor_types(x, out)
    verify_reduced_dims(out.shape, x.shape, dims)

    if x.element_type is not out.element_type:
        raise TypeError(
            f"element types of linalg.broadcast operands must be the same, "
            f"got {x.element_type.__qualname__} and "
            f"{out.element_type.__qualname__}"
        )

    rep = mlir_linalg.broadcast(
        lower_single(x), outs=lower(out), dimensions=dims
    )

    if isinstance(out, Tensor):
        # mlir_linalg.broadcast returns an OpResultList of length <= 1
        # I think the next line should really be done by the Python binding
        rep = mlir_linalg._get_op_result_or_value(rep)
        return type(out)(rep)
    else:
        return out
