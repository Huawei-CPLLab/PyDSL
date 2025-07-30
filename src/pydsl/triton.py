import ast
import subprocess
import typing
from collections.abc import Iterable

import triton
from triton.backends.compiler import GPUTarget
from triton.runtime.jit import JITFunction

import mlir.ir as mlirir
from mlir.ir import InsertionPoint
from mlir.dialects import func

from pydsl.type import Bool, Float, Index, Int, Number, Sign, UInt32
from pydsl.memref import MemRef
from pydsl.protocols import lower_single, ToMLIRBase


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
        return "i32"
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


def remove_unnecessary_args(func_op: func.FuncOp) -> None:
    """
    Removes unnecessary arguments from the funcOp.
    This is used to clean up the MLIR after triton-adapter-opt.
    """
    # Find the function in the module
    entry_block = func_op.entry_block
    if entry_block is None:
        raise AssertionError(
            f"function {func_op.sym_name.value} has no entry block"
        )

    # Collect indices of unused arguments
    indices_to_remove = []
    for i, arg in enumerate(entry_block.arguments):
        # `arg.uses` is an iterable of OperationUse;
        # if it's empty, the arg is unused. We also check if it's the first
        # or last 6 arguments, since these are the ones added by triton-adapter-opt
        # and we want to keep unused arguments that are user-defined.
        if not list(arg.uses) and (
            i == 0 or i >= len(entry_block.arguments) - 6
        ):
            indices_to_remove.append(i)

    if not indices_to_remove:
        return  # No unused arguments

    indices_to_remove.reverse()

    # Update argument attributes if present
    if "arg_attrs" in func_op.attributes:
        arg_attrs = func_op.arg_attrs
        new_attrs = [
            attr
            for i, attr in enumerate(arg_attrs)
            if i not in indices_to_remove
        ]
        func_op.arg_attrs = mlirir.ArrayAttr.get(new_attrs)

    # Remove unused arguments from the entry block
    for idx in indices_to_remove:
        entry_block.erase_argument(idx)

    # Update function type
    old_func_type = func_op.type
    new_inputs = [
        typ
        for i, typ in enumerate(old_func_type.inputs)
        if i not in indices_to_remove
    ]
    new_func_type = mlirir.FunctionType.get(new_inputs, old_func_type.results)
    func_op.attributes["function_type"] = mlirir.TypeAttr.get(new_func_type)


def combine_mlir_modules(
    main_module: mlirir.Module,
    module2: mlirir.Module,
    triton_funcs: dict,
    current_signature: tuple[str, ...],
) -> func.FuncOp:
    """
    Combines two MLIR modules by copying the operations from `module2` into
    `main_module`. Checks to see if the function has already been created, and
    if it has, checks to see if the signature has been created. If the function
    exists with a different signature, add "n" to the function name, where "n"
    is the current number of distinct signatures.
    """
    with InsertionPoint(main_module.body):
        prefix = "__triton_to_pydsl_"
        callees = []
        for op in module2.body.operations:
            if isinstance(op, func.FuncOp):
                signatures = triton_funcs.setdefault(op.sym_name.value, {})
                callee_name = (
                    op.sym_name.value + prefix + str(len(signatures))
                )  # add a number to the end of the function name
                if current_signature in signatures.keys():
                    callees.append(signatures[current_signature])
                    continue  # this function has already been compiled
                op.attributes["sym_name"] = mlirir.StringAttr.get(callee_name)
                new_op = op.clone()
                remove_unnecessary_args(new_op)
                signatures[current_signature] = new_op
                callees.append(new_op)
            else:
                op.clone()
    assert len(callees) == 1  # make sure there is only one FuncOp in module2
    return callees[0]


def run_and_get_output_stdin(
    cmds: Iterable[str], stdin_string: str
) -> str | None:
    result = subprocess.run(
        " ".join(cmds),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        input=stdin_string.encode("utf-8"),
        shell=True,
        check=True,
    )
    return result.stdout.decode("utf-8") if result.stdout else None


def triton_to_linalg(
    module: mlirir.Module,
    triton_func: JITFunction,
    args: list[tuple[str, typing.Any]],
    triton_funcs: list,
) -> func.FuncOp:
    """
    Converts triton functions to linalg dialect using triton-adapter-opt.
    args is a list of tuples of the form (name, value), where name is the
    name of the triton function argument and value is the PyDSL argument that
    the function has been called with.
    Returns the converted function as a func.FuncOp.
    """

    signature = {}
    constexprs = {}
    for name, val in args:
        signature[name] = pydsl_type_to_triton_str(type(val))
        if isinstance(val, Number):  # handling the constexpr case
            constexprs[name] = val.value

    triton_src = triton.compiler.ASTSource(
        fn=triton_func,
        signature=signature,
        constexprs=constexprs,
    )

    # Fixes the backend for CUDA, so that it can be used with triton-adapter-opt.
    # 80 is the gpu architecture (i.e. sm_80), 32 is the warp size.
    # These particular values aren't important, but they need to be set to some
    # valid CUDA backend for triton-adapter-opt to work.
    target = GPUTarget("cuda", 80, 32)
    triton_mlir = triton.compile(triton_src, target=target).asm["ttir"]

    mlir_module_str = run_and_get_output_stdin(
        ["triton-adapter-opt", "--triton-to-linalg"], triton_mlir
    )
    triton_module = mlirir.Module.parse(mlir_module_str)
    signature_type_str = " ".join(signature.values())
    signature_compressed = (signature_type_str, *constexprs.values())
    func_op = combine_mlir_modules(
        module, triton_module, triton_funcs, signature_compressed
    )
    return func_op


def handle_TritonCallable(
    visitor: ToMLIRBase, node: ast.Call, triton_func: JITFunction
) -> func.CallOp:
    """
    Handle Triton JIT functions.
    """

    args = [visitor.visit(arg) for arg in node.args]
    call_args = []
    for arg in args:
        if not isinstance(arg, Number):
            if isinstance(arg, Index):
                # Convert Index to UInt32 for Triton compatibility
                arg = UInt32(arg)
            call_args.append(arg)
    # TODO: once the grid syntax is redesigned, this check can simply be
    # len(args) != len(triton_func.arg_names)
    if (
        len(args) < len(triton_func.arg_names)
        or len(args) - len(triton_func.arg_names) > 3
    ):
        raise TypeError(
            f"function {triton_func._fn_name} expects {len(triton_func.arg_names)} "
            f"arguments, but got {len(args)}"
        )

    func_op = triton_to_linalg(
        visitor.module,
        triton_func,
        zip(triton_func.arg_names, args),
        visitor.triton_funcs,
    )
    final_parameter_number = len(func_op.type.inputs)
    num_extra = final_parameter_number - len(call_args)
    if num_extra != 0:
        raise TypeError(
            f"function {triton_func._fn_name} expects {final_parameter_number}"
            f"arguments, but got {len(call_args)}"
        )

    flattened_args = [lower_single(arg) for arg in call_args]

    # TODO: should wrap this in a PyDSL type. Fix this with
    # https://github.com/Huawei-CPLLab/PyDSL/issues/34
    return func.CallOp(func_op, flattened_args)
