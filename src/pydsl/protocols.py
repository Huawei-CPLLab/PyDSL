import ast
import typing
import numpy as np

from ast import NodeVisitor
from collections.abc import Callable, Iterable
from functools import reduce
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    Self,
    Union,
    runtime_checkable,
)

import mlir.ir as mlir
from mlir.ir import Operation, OpView, Value as MValue

from pydsl.scope import ScopeStack

if TYPE_CHECKING:
    from pydsl.type import Bool as BoolT


@runtime_checkable
class Lowerable(Protocol):
    """
    A protocol where its instances or its class can be lowered down to raw
    MLIR representation.

    Note that all return types of methods in this protocol are wrapped in
    tuple. This is because a Lowerable can potentially compose of multiple MLIR
    types, which would then need to be passed around and returned from function
    in batches.
    """

    def lower(self) -> tuple[MValue]:
        """
        This function should return the MLIR representation of the particular
        instance of the protocol, wrapped within a tuple.
        """
        ...

    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]: ...


SubtreeOut: typing.TypeAlias = Union[Lowerable, OpView, MValue]
"""
The union type of all possible partial results being returned while traversing
the AST.

TODO: This is not well designed and failed to encapsulate the Number type. It
is overly messy and allowed the compiler to throw around raw MLIR Python
binding types which adds unnecessary complexities.

This should eventually be removed in favor of a custom PyDSL Value class.
"""


def lower(
    v: Lowerable | type | OpView | MValue | mlir.Type,
) -> tuple[MValue] | tuple[mlir.Type]:
    """
    Convert a `Lowerable` type, type instance, and other MLIR objects into its
    lowest MLIR representation, as a tuple.

    This function is *not* idempotent.

    Specific behavior:
    - If `v` is a `Lowerable` type, a `mlir.ir.Type` is returned.
    - If `v` is a `Lowerable` type instance, a `mlir.ir.Value` is returned.
    - If `v` is an `mlir.ir.OpView` type instance, then its results (of type
      `mlir.ir.Value`) are returned.
    - If `v` is already an `mlir.ir.Value` or `mlir.ir.Type`, `v` is returned
      enclosed in a tuple.
    - If `v` is not any of the types above, `TypeError` will be raised.

    For example:
    - `lower(Index)` should be equivalent to `(IndexType.get(),)`.
    - `lower(Index(5))` should be equivalent to
      `(ConstantOp(IndexType.get(), 5).results,)`.
    - ```lower(UInt8(4).op_add(UInt8(5)))``` should be equivalent to ::

        tuple(AddIOp(
            ConstantOp(IntegerType.get_signless(8), 4), )
            ConstantOp(IntegerType.get_signless(8), 5)).results)
    """
    match v:
        case OpView() | Operation():
            return tuple(v.results)
        case MValue() | mlir.Type():
            return (v,)
        case type() if issubclass(v, Lowerable):
            # Lowerable class
            return v.lower_class()
        case _ if issubclass(type(v), Lowerable):
            # Lowerable class instance
            return v.lower()
        case _:
            raise TypeError(f"{v} is not Lowerable")


def lower_single(
    v: Lowerable | type | OpView | MValue | mlir.Type,
) -> MValue | mlir.Type:
    """
    lower with the return value stripped of its tuple.
    Lowered output tuple must have length of exactly 1. Otherwise,
    `ValueError` is raised.

    This function is idempotent.
    """

    res = lower(v)
    if len(res) != 1:
        raise ValueError(f"lowering expected single element, got {res}")
    return res[0]


def lower_flatten(li: list[Lowerable]) -> list:
    """
    Apply lower to each element of the list, then unpack the resulting tuples
    within the list.
    """
    # Uses map-reduce
    # Map:    lower each element
    # Reduce: flatten the resulting list of tuples into a list of its
    #         constituents
    return reduce(lambda a, b: a + [*b], map(lower, li), [])


def is_compiled(x: Any) -> bool:
    match x:
        case Lowerable():
            return True
        case Number():
            return True


@runtime_checkable
class HandlesFor(Protocol):
    """
    A protocol where when its instances are inserted as the iterator of a for
    loop, they are responsible for transforming the entire for loop.
    """

    def on_For(visitor: "ToMLIRBase", node: ast.For) -> SubtreeOut:
        """
        Called when the implementer is inserted into a for loop.
        """


@runtime_checkable
class HandlesIf(Protocol):
    """
    A protocol where when its instances are inserted as the test of an if
    statement, they are responsible for transforming the entire statement,
    including the else block.
    """

    def on_If(visitor: "ToMLIRBase", node: ast.If) -> SubtreeOut:
        """
        Called when the implementer is inserted into an if statement.
        """


# TODO: this is a work-in-progress for now. We want to improve call decoupling
# in the future. For now, visit_Call will be a mess!
@runtime_checkable
class CompileTimeCallable(Protocol):
    """
    A protocol where its instances accepts a plain call from the input program
    during compile time.

    on_Call does not need to follow the Protocol's signature, or even be a
    function. As such, CompileTimeCallable allows an implementing class to
    delegate its on_Call behavior to another class by having its instance under
    the on_Call attribute. This delegation often forms an attribute chain.

    Note: this class' behavior is in likeness to Python's Callable, but not
    congruent. Callable performs `type(x).__call__(x, arg1, ...)` uniformly to
    any x.
    This deviation is for the sake of ergonomics. It avoids requiring every
    CompileTimeCallable class to use a metaclass that implements
    metaclass.on_Call(x, arg1, ...) to be equivalent to x.on_Call(arg1, ...)

    If you would like x to be passed into on_Call as the first argument,
    decorate it with @classmethod.

    CompileTimeCallable uses on_Call instead of __call__.
    """

    def on_Call(
        attr_chain: list[Any],
        visitor: "ToMLIRBase",
        node: ast.Call,
        prefix_args: tuple[ast.AST] = tuple(),
    ) -> SubtreeOut:
        """
        Function that deals with the incoming call to the CompileTimeCallable.

        attr_chain: the chain of objects that is attributed when making the
        call. Example: object.hello() will have [object, hello] as its chain.
        This information is for cases when the callable is a class method and
        need context from its constituent class.

        visitor: the ToMLIRBase visitor context that is performing the
        compilation.

        node: the node within the Python AST where the call to the
        CompileTimeCallable is made.
        """
        ...


def handle_CompileTimeCallable(
    visitor: "ToMLIRBase",
    node: ast.Call,
    prefix_args: tuple = tuple(),
):
    """
    Finds and calls the proper compile-time function when given an AST Call.

    `prefix_args` is the list of arguments to be inserted before the
    arguments written in the code. This is used for situations such as
    inserting self into a member function call.
    """

    attr_chain = visitor.scope_stack.resolve_attr_chain(node.func)

    # recursively get the on_Call of the object until it's a Callable or it's
    # unresolvable
    # we avoid getting type(x) if it's Python's `type`, i.e. if x is a class
    while True:
        x = attr_chain[-1]

        if type(x).__name__ == "JITFunction":
            # This is a Triton JIT function
            # Import here, not earlier, in case the user doesn't have Triton
            from pydsl.triton import handle_TritonCallable

            return handle_TritonCallable(visitor, node, x)

        match x:
            case _ if issubclass(type(x), type):
                # if x is a class, fetch its on_Call
                on_Call = x.on_Call
            case _:
                # if x is an instance, fetch its class's on_Call
                on_Call = type(x).on_Call

        match on_Call:
            # Is this object callable using PyDSL's protocol mechanism?
            case CompileTimeCallable():
                attr_chain.append(on_Call)
                continue
            # Is this object callable using Python's built-in mechanism?
            # This will eventually be true since all calls on callable
            # object/"fake functions" ultimately results in a real
            # function call
            case Callable():
                return on_Call(
                    attr_chain, visitor, node, prefix_args=prefix_args
                )
            case _:
                raise TypeError(
                    f"{x} is neither Callable nor CompileTimeCallable"
                )


@runtime_checkable
class CompileTimeSliceable(Protocol):
    """
    A protocol where its instances accept an indexing operation from the input
    program during compile time.

    Implement the methods as you would a Python Sequence type:
    https://docs.python.org/3/reference/datamodel.html#object.__getitem__
    """

    def on_getitem(
        self: typing.Self, visitor: "ToMLIRBase", slice: ast.AST
    ) -> SubtreeOut: ...

    def on_setitem(
        self: typing.Self,
        visitor: "ToMLIRBase",
        slice: ast.AST,
        value: ast.AST | SubtreeOut,
    ) -> SubtreeOut: ...


@runtime_checkable
class CompileTimeClassSliceable(Protocol):
    """
    A protocol where the class itself (and *not necessarily* its instances)
    accept an indexing operation from the input program during compile time.

    This is typically used for generating a specified generic type when
    resolving Python typing objects as concrete classes.

    For example, slicing the MemRef class with [12, 6] would result in a new
    dynamically generated subclass of MemRef with shape of (12, 6).
    """

    @classmethod
    def on_class_getitem(
        cls, visitor: "ToMLIRBase", slice: ast.AST
    ) -> SubtreeOut: ...

    # Note: on_class_setitem is optional since it doesn't make a lot of sense
    # to "write" a new definition to a particular class generic


@runtime_checkable
class CompileTimeTestable(Protocol):
    """
    A protocol where its instances can be converted into a Bool during compile
    time.
    """

    def Bool(self) -> "BoolT": ...


@runtime_checkable
class CompileTimeIterable(Protocol):
    """
    A protocol where its instances can be iterated with elements known at
    compile time.

    This stipulates that the iterable has fixed compile-time length.

    Anything that is CompileTimeIterable can be unpacked as such ::

        a, b = cti  # where cti is a 2-element CompileTimeIterable
    """

    def as_iterable(self, visitor: "ToMLIRBase") -> Iterable[SubtreeOut]: ...


@runtime_checkable
class Returnable(Protocol):
    """
    A protocol where its instances can be placed in a scope and will accept
    any return statement within such scope
    """

    def on_Return(
        self: Self, visitor: "ToMLIRBase", node: ast.Return
    ) -> SubtreeOut: ...


class ToMLIRBase(NodeVisitor):
    """
    This is an empty class which ToMLIR inherits. It serves no
    purpose other than to enable type-hinting ToMLIR without risking
    cyclic import.
    """

    module: Optional[str] = None
    scope_stack: Optional[ScopeStack] = None
    interceptor_stack: list[Callable[[SubtreeOut], SubtreeOut]] = []
    context_stack: list[Any] = []
    catch_comp_error: bool = True

    # Dummy method mostly to change the type hint of `node` to include
    # `Lowerable`
    def visit(self, node: ast.AST | Lowerable) -> SubtreeOut:
        return super().visit(node)


# TODO: what file to put this in? I had it in frontend.py initially, then
# realized many files need to import ArgContainer, which would result in
# cyclic import error, since frontend imports numerous pydsl things.
class ArgContainer:
    """
    A container for storing arguments passed to a function.

    The current usecase is for it to store any numpy.ndarrays that are passed
    to a function, so that when we return from the function and create new
    ndarrays possibly referring to the same memory locations as existing ones,
    we can add a pointer from the new ndarray to the old one to prevent it from
    deallocating the memory.

    In the future, maybe we can expand this class to store more information
    about arguments if it's useful.

    This class is intended to be passed to all to_CType and from_CType calls.
    Note that we could not do input/output ndarray overlap analysis at the
    outer level where we call the function, since function arguments and
    return values can be complicated nested types.
    """

    ndarray_list: list[np.ndarray]

    def __init__(self):
        self.ndarray_list = []

    def add_arg(self, arg: Any):
        if isinstance(arg, np.ndarray):
            self.ndarray_list.append(arg)

    @staticmethod
    def _get_ndarray_root(arr: np.ndarray) -> np.ndarray:
        """
        Try to get the root of an ndarray by repeatedly taking its base
        pointer. This should result in an array that owns its memory and its
        memory is contiguous.
        """
        seen = set()
        while arr.base is not None:
            if id(arr) in seen:
                raise AssertionError("found ndarray with cyclic base pointers")
            seen.add(id(arr))
            arr = arr.base

        if not arr.flags["C_CONTIGUOUS"]:
            raise AssertionError(
                "found ndarray that has no base array but is not C contiguous"
            )

        return arr

    def get_overlap(self, arr: np.ndarray) -> np.ndarray | None:
        """
        Compares arr with any ndarrays stored in this container to check if
        their memories overlap. If any such arrays exist, finds their root
        (the ndarray which allocated the memory) and returns that. Otherwise,
        returns None. If numpy and MLIR work the way we expect them to and the
        user is not doing crazy things, there should be at most one overlapping
        root array, it should have contiguous memory, and it should fully
        contain arr.
        """
        result = None

        for arg_arr in self.ndarray_list:
            if not np.may_share_memory(arr, arg_arr):
                continue

            root = self._get_ndarray_root(arg_arr)
            if not np.may_share_memory(arr, root):
                continue

            if result is None:
                result = root
            elif id(root) != id(result):
                raise AssertionError(
                    "array overlaps with multiple root arrays"
                )

        return result


def canonicalize_args(f: Callable) -> Callable:
    """
    Binds each argument to the corresponding parameter, applies default
    arguments, then calls f. This is useful in combination with
    functools.cache:

    ```
    @canonicalize_args
    @cache
    def f(a, b):
        print(a, b)


    f(1, 2)
    f(1, 2)
    f(1, b=2)
    f(b=2, a=1)
    ```

    Without canonicalize_args, the above code will print 3 times, but with
    canonicalize_args, it will only print once.

    This should be used when it's very important that calling the function with
    the same arguments twice causes it to only run once and return the exact
    same object. If @cache is used just for optimization, then
    canonicalize_args is not necessary.
    """

    def payload(*args, **kwargs):
        bound_args = signature(f).bind(*args, **kwargs)
        bound_args.apply_defaults()
        return f(*bound_args.args, **bound_args.kwargs)

    return payload
