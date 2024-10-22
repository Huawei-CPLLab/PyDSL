from __future__ import annotations

import ast
import inspect
import typing
from abc import ABCMeta, abstractmethod
from ast import Expression
from collections.abc import Callable
from enum import Enum, auto
from typing import Annotated, Any, Self

from mlir.ir import OpView

from python_bindings_mlir.helpers import Lowerable, ToMLIRBase


class Macro(ABCMeta):
    """
    A metaclass whose instances are evaluated during compile time and
    manipulates the MLIR output.

    For all instances of Macro, and for all class methods annotated as
    @abstractmethod within the instance, they must be implemented by their
    subclasses.
    """

    def __new__(cls, name, bases, attr, *args, **varargs):
        new_cls = super().__new__(cls, name, bases, attr, *args, **varargs)

        for base in bases:
            if not hasattr(base, "__abstractmethods__"):
                continue

            for abstract_method in base.__abstractmethods__:
                # check if the abstractmethod is overwritten
                if getattr(new_cls, abstract_method) is getattr(
                    base, abstract_method
                ):
                    raise TypeError(
                        f"macro {new_cls.__name__} did not implement abstract "
                        f"method {abstract_method}, required by base {base}"
                    )

        return new_cls


class IteratorMacro(metaclass=Macro):
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


class ArgRep(Enum):
    EVALUATED = auto()
    """
    This indicates that the argument is a literal or a variable name that is
    evaluated as a compile-time Python value.

    Please note: once the Python runtime is opened, all CallMacros nested
    inside it will demand all arguments in compile-time Python representation.
    python_bindings_mlir may no longer be used.
    For instance, if you nest a CallMacro `f` inside a PYTHON ArgType
    argument, and `f` demands a TREE as its first argument,
    then you must pass in an ast.AST Python object, not writing the
    representation of such tree as-is.

    As an example, writing f(4) in python_bindings_mlir is equivalent to writing
    f(Expr(value=Constant(value=4))) in compile-time Python.
    """

    COMPILED = auto()
    """
    This indicates that the argument should be passed in as expressed in
    target language, i.e. MLIR OpViews.
    This accepts any expression that can be outputted by this compiler.
    """

    UNCOMPILED = auto()
    """
    This indicates that the argument should remain as a Python AST.
    """


# These type hints are used by functions annotated with CallMacro.generate
T = typing.TypeVar("T")
Evaluated: typing.TypeAlias = Annotated[T, ArgRep.EVALUATED]
Compiled: typing.TypeAlias = Annotated[
    typing.Union[OpView, "Lowerable"], ArgRep.COMPILED
]
Uncompiled: typing.TypeAlias = Annotated[ast.AST, ArgRep.UNCOMPILED]


def iscallmacro(f: Any) -> bool:
    return issubclass(type(f), type) and issubclass(f, CallMacro)


def eval_python(visitor: ToMLIRBase, arg: ast.AST) -> Any:
    def hydrate(f):
        """
        Helper function that takes a function and redirects the call to
        _on_Call if it is a subclass of CallMacro.
        """
        if not iscallmacro(f):
            return f

        def hydrated_f(*args):
            return f._on_Call(visitor, [*args])

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


class CallMacro(metaclass=Macro):
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

    @abstractmethod
    def argreps() -> list[ArgRep]:
        """
        This function must be overwritten to specify the type of the
        non-variable arguments that the function accepts.
        """

    def argsreps() -> ArgRep | None:
        """
        This function can be overwritten to specify the type of the variable
        argument.
        """

        return None

    # TODO: enable passing of kwargs as a Dict[str, Any]

    @abstractmethod
    def _on_Call(
        self: Self | None,
        cls: type[Self],
        visitor: ast.NodeVisitor,
        args: list[Any],
    ) -> Any: ...

    @classmethod
    def cast_arg(cls, visitor: ToMLIRBase, arg, argtype):
        """
        Helper function for casting a single argument
        """
        match argtype:
            case ArgRep.EVALUATED:
                return eval_python(visitor, arg)
            case ArgRep.COMPILED:
                match arg:
                    case ast.AST():
                        return visitor.visit(arg)
                    case _:
                        raise TypeError(
                            f"{cls.__name__} expected a single AST node. Got "
                            f"{arg}"
                        )
            case ArgRep.UNCOMPILED:
                return arg
            case _:
                raise TypeError(
                    f"argtype() in {cls.__name__}, must return List[ArgType], "
                    f"got {argtype} as an element in the returned list"
                )

    @classmethod
    def _parse_args(
        cls,
        visitor: ToMLIRBase,
        node: ast.Call,
        prefix_args: list[ast.AST] = [],
    ) -> list[Any]:
        args = prefix_args + node.args

        if cls.argsreps() is None and len(cls.argreps()) != len(args):
            raise ValueError(
                f"{cls.__name__} expected {len(cls.argreps())} arguments, got "
                f"{len(args)}"
            )

        ret = []

        # evaluate regular args
        for arg, argtype in zip(args, cls.argreps(), strict=False):
            ret.append(cls.cast_arg(visitor, arg, argtype))

        # evaluate varargs
        if (varargtype := cls.argsreps()) is not None:
            for arg in args[len(cls.argreps()) :]:
                ret.append(cls.cast_arg(visitor, arg, varargtype))

        return ret

    # TODO: enable passing of kwargs as a Dict[str, Any]
    # Looks like this: def foo(*args: str, **kwds: int): ...

    def on_Call(
        attr_chain,
        visitor: ToMLIRBase,
        node: ast.Call,
        prefix_args: list[ast.AST] = [],
    ) -> OpView:
        cls = attr_chain[-1]

        if cls.is_member:
            assert (
                len(attr_chain) >= 2
            ), f"Attribute chain does not contain the class of {cls}"
            return cls._on_Call(
                attr_chain[-2],
                visitor,
                cls._parse_args(visitor, node, prefix_args=prefix_args),
            )

        return cls._on_Call(
            visitor, cls._parse_args(visitor, node, prefix_args=prefix_args)
        )

    def generate(is_member=False):
        """
        Decorator that converts a function into a CallMacro.

        This decorator enforce strict type hinting in the function arguments,
        as such:
        - The first argument must be ToMLIRBase (although this is not
          enforced at runtime)
        - The remaining arguments must be type-hinted Annotation[a, b], where a
          can be any type, and b must be an ArgRep.
            - For your convenience, you can use Evaluated[T], Compiled, and
              Uncompiled for your remaining arguments.

        WARNING: this decorator does not work with string type hints that are
        not imported at runtime, even if it's done with `if TYPE_CHECKING`. The
        runtime semantics of the type hint are examined to inform its behavior.

        If the macro is to be a @classmethod of a class, pass is_member=True.
        The class will be passed as the first argument and it won't need to be
        type-hinted.
        """

        def generate_sub(f: Callable) -> CallMacro:
            calling_locals = inspect.currentframe().f_back.f_locals
            calling_globals = inspect.currentframe().f_back.f_globals

            (
                args,
                varargs,
                varkw,
                defaults,
                kwonlyargs,
                kwonlydefaults,
                annotations,
            ) = inspect.getfullargspec(f)

            if varargs or varkw or defaults or kwonlyargs or kwonlydefaults:
                raise NotImplementedError(
                    f"CallMacro cannot yet support variable args, keyword "
                    f"args, or default values. Provided by '{f.__name__}'"
                )

            hints = typing.get_type_hints(
                f,
                include_extras=True,
                localns=calling_locals,
                globalns=calling_globals,
            )

            def _on_Call(visitor, args):
                return f(visitor, *args)

            if is_member:
                args = args[1:]

                def _on_Call(cls, visitor, args):
                    return f(cls, visitor, *args)

            try:
                arg_types = [hints[a] for a in args]
            except KeyError:
                raise TypeError(
                    f"CallMacro generation requires ToMLIRBase and following "
                    f"positional arguments of '{f.__name__}' to be "
                    f"type-hinted"
                )

            if len(arg_types) == 0 or arg_types[0] is not ToMLIRBase:
                raise TypeError(
                    f"CallMacro generation requires first positional argument "
                    f"of '{f.__name__}' to be hinted as ToMLIRBase type"
                )

            # this excludes the first argument which is always ToMLIRBase, and
            # extract metadata from Annotation type
            var_argreps = [a.__metadata__ for a in arg_types[1:]]
            if any([
                len(a) != 1 or (type(a[0]) is not ArgRep) for a in var_argreps
            ]):
                raise TypeError(
                    f"CallMacro requires non-first positional argument types "
                    f"of '{f.__name__}' to be annotated with exactly one "
                    f"ArgRep metadata"
                )
            var_argreps = [a[0] for a in var_argreps]

            # dynamically generate a new subclass of CallMacro that is based
            # on f
            return type(
                f.__name__,
                (CallMacro,),
                {
                    "is_member": is_member,
                    "argreps": lambda: var_argreps,
                    "_on_Call": _on_Call,
                },
            )

        return generate_sub
