import ast
import subprocess
import typing

import triton
from triton.backends.compiler import GPUTarget
from triton.runtime.jit import JITFunction

import mlir.ir as mlirir
from mlir.ir import InsertionPoint
from mlir.dialects import func

from pydsl.type import (
    Int,
    Sign,
    Float,
    Bool,
    Number,
    UInt32,
    UInt8,
    Tuple,
    Index,
)
from pydsl.memref import MemRef, DYNAMIC, alloca
from pydsl.protocols import lower_single
from pydsl.compiler import ToMLIRBase


def pydsl_type_to_triton_str(pytype: typing.Any) -> str:
    """
    Converts a PyDSL type to a Triton type string.
    This is used to generate the signature for Triton functions.
    """
    if issubclass(pytype, MemRef):
        return "*" + pydsl_type_to_triton_str(pytype.element_type)
    if issubclass(pytype, Bool):
        return "B"
    if issubclass(pytype, Index):
        raise TypeError(
            f"converting {pytype.__qualname__} to a Triton type is not supported"
        )
    if issubclass(pytype, Int):
        # TODO: This currently ignores signedness, since triton-ascend does not support it.
        sign_val = "i" if pytype.sign == Sign.UNSIGNED else "i"
        return sign_val + str(pytype.width)
    if issubclass(pytype, Float):
        return "fp" + str(pytype.width)
    # Handle constexpr (for BLOCK_SIZE, etc.)
    if issubclass(pytype, Number):
        return "constexpr"
    raise TypeError(
        f"converting {pytype.__qualname__} to a Triton type is not supported"
    )


def combine_mlir_modules(
    main_module: mlirir.Module,
    module2: mlirir.Module,
    triton_funcs: dict,
    current_signature: str,
) -> func.FuncOp:
    """
    Combines two MLIR modules by copying the operations from `module2` into `main_module`.
    checks to see if the function has already been created, and if it has, checks to see if the
    signature has been created. If the function exists with a different signature, add "n" to the
    function name, where "n" is the current number of distinct signatures.
    """
    with InsertionPoint(main_module.body):
        prefix = "__triton_to_pydsl_"
        for op in module2.body.operations:
            if isinstance(op, func.FuncOp):
                signatures = triton_funcs.setdefault(op.sym_name.value, [])
                caller_name = (
                    op.sym_name.value + prefix + str(len(signatures))
                )  # add a number to the end of the function name
                if current_signature in signatures:
                    caller_name = (
                        op.sym_name.value
                        + prefix
                        + str(signatures.index(current_signature))
                    )
                    caller = next(
                        (
                            op2
                            for op2 in main_module.body.operations
                            if isinstance(op2, func.FuncOp)
                            and op2.sym_name.value == caller_name
                        )
                    )
                    break  # this function has already been compiled, we can stop.
                signatures.append(current_signature)
                op.attributes["sym_name"] = mlirir.StringAttr.get(caller_name)
                new_op = op.clone()
                caller = new_op
            else:
                op.clone()
    return caller


def run_and_get_output_stdin(cmds, stdinString):
    result = subprocess.run(
        " ".join([str(c) for c in cmds]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        input=stdinString.encode("utf-8"),
        shell=True,
        check=False,
    )
    return result.stdout.decode("utf-8") if result.stdout else None


def triton_to_linalg(
    module: mlirir.Module,
    triton_func: JITFunction,
    args: list[typing.Any],
    triton_funcs: list,
) -> func.FuncOp:
    """
    Converts triton functions to linalg dialect using triton-adapter-opt.
    returns a tuple containing the arguments and return types of the function
    combined.
    """

    signature = {}
    constexprs = {}
    # TODO: have logic here to automatically assume that the grid indices are 0
    #  if the user doesn't specify them.
    for name, typ in args[:-3]:
        signature[name] = pydsl_type_to_triton_str(type(typ))
        if isinstance(typ, Number):  # handling the constexpr case
            constexprs[name] = typ.value
    triton_src = triton.compiler.ASTSource(
        fn=triton_func,
        signature=signature,
        constexprs=constexprs,
    )
    # fixes the backend for CUDA, so that it can be used with triton-adapter-opt.
    # 80 is the gpu architecture (i.e. sm_80), 32 is the warp size.
    # These particular values aren't important, but they need to be set to some
    # valid CUDA backedn for triton-adapter-opt to work.
    target = GPUTarget("cuda", 80, 32)
    triton_mlir = triton.compile(triton_src, target=target).asm["ttir"]

    mlir_file = run_and_get_output_stdin(
        ["triton-adapter-opt", "--triton-to-linalg"], triton_mlir
    )
    ctx = module.context
    triton_module = mlirir.Module.parse(mlir_file, context=ctx)
    current_signature = " ".join(signature.values())
    func_op = combine_mlir_modules(
        module, triton_module, triton_funcs, current_signature
    )
    # Create a func.FuncOp with the extracted inputs, results, and function name

    return func_op


def handle_TritonCallable(
    visitor: ToMLIRBase, node: ast.Call, x: JITFunction
) -> func.CallOp:
    """
    Handle Triton JIT functions.
    """

    args = [visitor.visit(arg) for arg in node.args]
    memref_type = MemRef.get((DYNAMIC,), UInt8)

    # Use PyDSL's alloca macro to allocate the dummy memref (1 element)
    index_size = Index(1)
    alloc = alloca(
        visitor, memref_type, Tuple.from_values(visitor, index_size)
    )

    call_args = [alloc] + args
    n = len(call_args)

    func_op = triton_to_linalg(
        visitor.module,
        x,
        [(node.args[i].id, args[i]) for i in range(len(args))],
        visitor.triton_funcs,
    )
    final_parameter_number = len(func_op.type.inputs)
    num_extra = final_parameter_number - n
    if num_extra < 0:
        raise RuntimeError(
            f"Too many arguments for function {func_op.sym_name.value}: "
            f"expected {final_parameter_number}, got {n} (after adding memref)"
        )

    # Create the required number of i32 constants. These are added by
    # triton-adapter-opt but we don't use them, so we will just set them to 1.

    # TODO: need to remove these or better understand why triton-adapter-opt
    # adds them
    const_i32_list = []
    for _ in range(num_extra):
        const_val = UInt32(1)
        const_i32_list.append(const_val)

    # Determine insertion position
    if n < 3:
        insert_index = n
    else:
        insert_index = n - 3

    # Insert the constants into call_args
    new_call_args = (
        call_args[:insert_index] + const_i32_list + call_args[insert_index:]
    )

    flattened_args = [
        lower_single(UInt32(arg))
        if isinstance(arg, Number)
        else lower_single(arg)
        for arg in new_call_args
    ]

    # TODO: should wrap this in a PyDSL type. Fix this with
    # https://github.com/Huawei-CPLLab/PyDSL/issues/34
    return func.CallOp(func_op, flattened_args)
