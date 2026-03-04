from mlir.dialects import bufferization
from pydsl.macro import CallMacro, Compiled
from pydsl.tensor import Tensor
from pydsl.memref import MemRef
from pydsl.protocols import ToMLIRBase, lower_single, SubtreeOut

TensorFactory = Tensor.class_factory


def verify_all_memref(*args):
    """
    Checks that all arguments are MemRef.
    Raises a TypeError otherwise.
    """

    # Collect argument type names for error messages
    arg_type_names = []
    for arg in args:
        arg_type_names.append(type(arg).__qualname__)
    arg_type_str = ", ".join(arg_type_names)

    # Check that every argument is a MemRef
    for arg in args:
        if not isinstance(arg, MemRef):
            raise TypeError(
                "bufferization operation expects arguments of type MemRef, "
                f"got {arg_type_str}"
            )


def verify_all_tensor(*args):
    """
    Checks that all arguments are Tensor.
    Raises a TypeError otherwise.
    """

    # Collect argument type names for error messages
    arg_type_names = []
    for arg in args:
        arg_type_names.append(type(arg).__qualname__)
    arg_type_str = ", ".join(arg_type_names)

    # Check that every argument is a Tensor
    for arg in args:
        if not isinstance(arg, Tensor):
            raise TypeError(
                "bufferization operation expects arguments of type Tensor, "
                f"got {arg_type_str}"
            )


@CallMacro.generate()
def to_tensor(visitor: "ToMLIRBase", x: Compiled) -> SubtreeOut:
    verify_all_memref(x)

    rep = bufferization.to_tensor(
        lower_single(x), restrict=True, writable=True
    )
    static_shape = rep.type.shape
    t_type = TensorFactory(tuple(static_shape), rep.type.element_type)

    return t_type(rep)


@CallMacro.generate()
def materialize_in_destination(
    visitor: "ToMLIRBase", x: Compiled, y: Compiled
):
    verify_all_tensor(x)
    verify_all_memref(y)
    bufferization.MaterializeInDestinationOp(
        None, lower_single(x), lower_single(y), writable=True
    )
    return
