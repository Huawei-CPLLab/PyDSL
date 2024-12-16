from __future__ import annotations

import ast
import builtins
import inspect
import typing
from abc import ABC, abstractmethod
from ast import Expression
from collections.abc import Callable
from inspect import BoundArguments, Signature, signature
from typing import Annotated, Any

from mlir.ir import OpView

from pydsl.protocols import SubtreeOut, ToMLIRBase


class Macro(ABC):
    """
    An abstract class whose instances are evaluated during compile time and
    manipulates the MLIR output.
    """


class IteratorMacro(Macro):
    """
    A macro that is intended to behave like a Python iterator.

    Instead of implementing __next__, subclasses are required to generate the
    necessary MLIR for the various contexts where an iterator may be legally
    used in Python.
    """

    @abstractmethod
    def on_For(visitor: ToMLIRBase, node: ast.For): ...

    @abstractmethod
    def on_ListComp(visitor: ToMLIRBase, node: ast.ListComp): ...

    @abstractmethod
    def _signature(*args: Any, **varargs: Any) -> None:
        """
        The signature of this function determines the signature of the
        iterator. The body of this function is ignored and can be left empty.

        You may use the same type hints supported by `CallMacro.generate`,
        such as `*args: list[Compiled]`.
        However, do not use `visitor: ToMLIRBase` as your first argument as
        that will not be passed in as a prefix argument.

        To obtain the bound arguments, call `iterator_bound_args`
        """
        ...

    def get_target(node: ast.For):
        return node.target

    @classmethod
    def iterator_bound_args(
        cls, visitor: ToMLIRBase, node: ast.For
    ) -> BoundArguments:
        iterator = node.iter

        return CallMacro.parse_args(
            inspect.signature(cls._signature),
            visitor,
            iterator,
        )


class ArgCompiler(ABC):
    """
    A class that take in an argument tree and perform the
    necessary compilation prior to handing it to a Macro.
    """

    @abstractmethod
    def compile(visitor: ToMLIRBase, arg: ast.AST) -> Any: ...

    @staticmethod
    def is_type_ArgCompiler(annotation):
        if typing.get_origin(annotation) != Annotated:
            return False

        metadata = typing.get_args(annotation)[1]

        return issubclass(metadata, ArgCompiler)

    @staticmethod
    def from_type(annotation):
        """
        Take an Annotated type hint and extract its ArgCompiler metadata.

        E.g. The `TypeAlias` Compiled is annotated as `ArgRep.COMPILED`.

        TypeError is excepted if `hint` does not annotate ArgCompiler.
        """
        if not ArgCompiler.is_type_ArgCompiler(annotation):
            raise TypeError(f"{annotation} is not an ArgCompiler")

        return typing.get_args(annotation)[1]


class EvaluatedArg(ArgCompiler):
    """
    This indicates that the argument is a literal or a variable name that is
    evaluated as a compile-time Python value.

    Please note: once the Python runtime is opened, all CallMacros nested
    inside it will demand all arguments in compile-time Python representation.
    PyDSL may no longer be used.
    For instance, if you nest a CallMacro `f` inside a EvaluatedArg
    argument, and `f` demands an UncompiledArg as its first argument,
    then you must pass in an ast.AST Python object, not the
    representation of such tree as-is.

    As an example, writing f(4) in PyDSL is equivalent to writing
    f(Expr(value=Constant(value=4))) in compile-time Python.
    """

    def compile(visitor: ToMLIRBase, arg: ast.AST) -> Any:
        def hydrate(f: Callable | CallMacro):
            """
            Helper function that takes a function and redirects the call to
            _on_Call if it is a subclass of CallMacro.
            """
            if not iscallmacro(f):
                return f

            def hydrated_f(*args, **kwargs):
                ba = f.signature().bind(visitor, *args, **kwargs)
                return f._on_Call(visitor, ba)

            return hydrated_f

        def hydrate_dict(d: dict[str, Any]):
            return {k: hydrate(v) for k, v in d.items()}

        exp_globals = hydrate_dict(visitor.scope_stack.globals())
        exp_locals = hydrate_dict(visitor.scope_stack.locals())

        try:
            return eval(
                compile(
                    Expression(body=arg),
                    visitor.scope_stack.resolve_name("__file__"),
                    "eval",
                ),
                exp_globals,
                exp_locals,
            )
        except Exception as e:
            raise ValueError(
                f"an exception occured while evaluating expression as an "
                f"ArgType.PYTHON: {e}"
            ) from e


class CompiledArg(ArgCompiler):
    """
    This indicates that the argument should be passed in as expressed in
    target language, i.e. MLIR OpViews.
    This accepts any expression that can be outputted by this compiler.
    """

    def compile(visitor: ToMLIRBase, arg: ast.AST) -> SubtreeOut:
        match arg:
            case ast.AST():
                return visitor.visit(arg)
            case _:
                raise TypeError(
                    f"Compiled ArgCompiler expected a single AST node. Got "
                    f"{arg}"
                )


class UncompiledArg(ArgCompiler):
    """
    This indicates that the argument should remain as a Python AST.
    """

    def compile(visitor: ToMLIRBase, arg: ast.AST) -> ast.AST:
        return arg


# These type hints are used by functions annotated with CallMacro.generate
T = typing.TypeVar("T")
Evaluated: typing.TypeAlias = Annotated[T, EvaluatedArg]
Compiled: typing.TypeAlias = Annotated[T, CompiledArg]
Uncompiled: typing.TypeAlias = Annotated[T, UncompiledArg]


def iscallmacro(f: Any) -> bool:
    return issubclass(type(f), type) and issubclass(f, CallMacro)


class CallMacro(Macro):
    """
    A macro that intends to be called as-is without being at the top level of
    a special code body, such as being the iterator of a `for` statement.

    The macro is responsible for transforming the Call node that called this
    function and its constituents.
    Note that if the context of the Call matches those of other Macro classes,
    they may be prioritized as the intended superclass and error may be thrown.

    This function may be called during compile-time, in which case the function
    is expected to return the MLIR that satisfies its behavior.
    The caller is responsible for providing the argument that the macro
    expects, such as Python objects, ASTs, or MLIR operations.
    Outside of any source that's to be compiled, the function always expects
    the MLIR visitor as the first argument.
    If visited within a source that's to be compiled, the arguments will
    automatically be prepended with the ToMLIRBase visitor.
    """

    is_member = False

    @staticmethod
    def parse_args(
        signature: Signature,
        visitor: ToMLIRBase,
        node: ast.Call,
        prefix_args: tuple[ast.AST] = tuple(),
    ) -> BoundArguments:
        args = (*prefix_args, *node.args)
        keywords = {kw.arg: kw.value for kw in node.keywords}
        parameters = signature.parameters

        try:
            # associate each argument value passed into the call with
            # the parameter of the function
            bound_args = signature.bind(*args, **keywords)
            binding = bound_args.arguments
        except TypeError as e:
            raise TypeError(f"error occured when calling a macro: {e}") from e

        for name in binding.keys():
            # compile each argument value according to its argtype hinting
            param = parameters[name]
            ann = param.annotation
            annorigin = typing.get_origin(ann)
            annargs = typing.get_args(ann)

            # Python doesn't have functor abstraction, so we'll have to do
            # functor mapping for each of the common Python data type
            match annorigin:
                case builtins.list if (
                    ArgCompiler.is_type_ArgCompiler(annargs[0])
                ):
                    argcomp = ArgCompiler.from_type(annargs[0])
                    binding[name] = [
                        argcomp.compile(visitor, i) for i in binding[name]
                    ]

                case builtins.tuple if (
                    ArgCompiler.is_type_ArgCompiler(annargs[0])
                ):
                    argcomp = ArgCompiler.from_type(annargs[0])
                    binding[name] = tuple([
                        argcomp.compile(visitor, i) for i in binding[name]
                    ])

                case builtins.dict if (
                    ArgCompiler.is_type_ArgCompiler(annargs[0])
                ):
                    argcomp = ArgCompiler.from_type(annargs[0])
                    binding[name] = {
                        k: argcomp.compile(visitor, v)
                        for (k, v) in binding[name].items()
                    }

                case builtins.set if (
                    ArgCompiler.is_type_ArgCompiler(annargs[0])
                ):
                    argcomp = ArgCompiler.from_type(annargs[0])
                    binding[name] = {
                        argcomp.compile(visitor, i) for i in binding[name]
                    }

                case _ if ArgCompiler.is_type_ArgCompiler(ann):
                    argcomp = ArgCompiler.from_type(ann)

                    # note that this is mutating bound_args
                    binding[name] = argcomp.compile(visitor, binding[name])

        # Apply defaults for unfilled arguments after ArgCompiler pass over
        # the non-empty arguments
        bound_args.apply_defaults()

        return bound_args

    @abstractmethod
    def signature() -> Signature:
        """
        This function must be overwritten to specify the function signature of
        the macro.
        """
        return NotImplemented

    @abstractmethod
    def _on_Call(
        visitor: ast.NodeVisitor,
        args: BoundArguments,
    ) -> Any:
        return NotImplemented

    def on_Call(
        attr_chain,
        visitor: ToMLIRBase,
        node: ast.Call,
        prefix_args: tuple[ast.AST] = tuple(),
    ) -> OpView:
        fn = attr_chain[-1]

        # Prefix the appropriate values to function arguments
        if fn.is_member:
            # if a member function, append the class as a prefix
            assert (
                len(attr_chain) >= 2
            ), f"attribute chain does not contain the class of {fn}"
            prefix_args = (attr_chain[-2], *prefix_args)
        prefix_args = (visitor, *prefix_args)

        bound_args = CallMacro.parse_args(
            fn.signature(), visitor, node, prefix_args=prefix_args
        )

        return fn._on_Call(visitor, bound_args)

    def generate(is_member=False):
        """
        Decorator that converts a function into a CallMacro.

        This decorator may interpret type hinting in the function arguments
        as such:
        - The first argument must be ToMLIRBase (although this is not
          enforced at runtime)
        - The remaining arguments must be type-hinted Annotation[a, b], where a
          can be any type, and b must be an ArgCompiler.
            - For your convenience, you can use Evaluated[T], Compiled, and
              Uncompiled for your remaining arguments.

        If the macro is to be a @classmethod of a class, pass is_member=True.
        The class will be passed as the first argument and it won't need to be
        type-hinted.
        """

        def generate_sub(f: Callable) -> CallMacro:
            def _on_Call(visitor, ba):
                return f(*ba.args, **ba.kwargs)

            # dynamically generate a new subclass of CallMacro that is based
            # on f
            return type(
                f.__name__,
                (CallMacro,),
                {
                    "is_member": is_member,
                    "signature": lambda: signature(f),
                    "_on_Call": _on_Call,
                },
            )

        return generate_sub
