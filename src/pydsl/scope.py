import ast
import typing
from collections.abc import Iterator
from contextlib import contextmanager
from functools import reduce
from typing import Any


class BindingContext:
    """
    A binding context that represents blocks in if branches, for body, etc.
    """

    context_obj: ast.AST
    f_locals: dict[str, Any]

    def __init__(self, context_obj: ast.AST, f_locals: dict[str, Any]) -> None:
        self.context_obj = context_obj
        self.f_locals = f_locals

    def assign_name(self, name, value) -> None:
        self.f_locals[name] = value


class Scope:
    stack: list[BindingContext]
    scoping_obj: Any
    bound: set[str]

    def __init__(
        self, scoping_obj: Any, f_locals: dict[str, Any], bound: set[str]
    ) -> None:
        self.stack = [BindingContext(scoping_obj, f_locals)]
        self.scoping_obj = scoping_obj
        self.bound = bound

    @property
    def f_locals(self) -> dict[str, Any]:
        return reduce(lambda acc, d: {**acc, **d.f_locals}, self.stack, {})

    def init_global(f_locals) -> "Scope":
        return Scope("<module>", f_locals, set())

    def assign_name(self, name, value) -> None:
        self.stack[-1].assign_name(name, value)

    @contextmanager
    def new_binding_context(
        self, binding_context: BindingContext
    ) -> Iterator[None]:
        self.stack.append(binding_context)
        try:
            yield
        finally:
            self.stack.pop()


class ScopeStack:
    stack: list[Scope]
    global_scope: Scope

    def __init__(self, f_locals) -> None:
        self.global_scope = Scope.init_global(f_locals)
        self.stack = [self.global_scope]

    def globals(self) -> dict[str, Any]:
        """
        Returns the current globals name table, as if the built-in globals()
        is called.
        """
        if self.global_scope is None:
            raise AssertionError(
                "attempted to call globals() on a stack without a global scope"
            )
        return self.global_scope.f_locals

    def locals(self) -> dict[str, Any]:
        """
        Returns the current locals name table, as if the built-in locals() is
        called.
        """
        if len(self.stack) == 0:
            raise AssertionError(
                "attempted to call locals() on an empty stack"
            )

        return self.stack[-1].f_locals

    def bounded(self) -> set[str]:
        """
        Returns variables that are bounded in the current scope.
        """
        if len(self.stack) == 0:
            raise AssertionError(
                "attempted to call bounded() on an empty stack"
            )

        return self.stack[-1].bound

    def find_name(self, name):
        """
        Used during the generation of the initial locals() when entering a
        function scope.
        """
        for scope in reversed(self.stack):
            if name in scope.f_locals:
                return scope.f_locals[name]

        raise NameError(f"name '{name}' is not defined")

    def resolve_name(self, name: str):
        if name in self.locals():
            return self.locals()[name]

        if name in self.bounded():
            raise UnboundLocalError(
                f"cannot access local variable '{name}' where it is not "
                f"associated with a value"
            )

        if name in self.globals():
            return self.globals()[name]

        raise NameError(f"name '{name}' is not defined")

    def resolve_attr_chain(self, chain: ast.Attribute | ast.Name) -> list[Any]:
        """
        Resolve a chain of Attributes in AST format.

        Returns a list of attribute name resolutions starting with the name
        resolution of the leftmost attribute name.

        E.g. `chain=ast.parse("a.b.c").body.value` returns `[a, a.b, a.b.c]`,
        assuming everything in the list exists in the scope stack.

        Name node is assumed to be an attribute chain of size 1.
        """
        match chain:
            case ast.Attribute(attr=attrname):  # chain length > 1
                next_chain = chain.value
                current_result = self.resolve_attr_chain(next_chain)
                return [*current_result, getattr(current_result[-1], attrname)]
            case ast.Name(id=id):  # base case: chain length = 1
                return [self.resolve_name(id)]
            case _:
                raise Exception(
                    f"unexpected AST node type "
                    f"{type(chain.value).__qualname__} in attribute chain"
                )

    def resolve_as_type(self, name, cls):
        result = self.resolve_name(name)

        if not issubclass(result, cls):
            raise TypeError(
                f"expected a {cls.__name__}, got '{name}' which is not a"
                f"subclass of {cls.__name__}"
            )

        return result

    def resolve_as_protocol(
        self, name: str, proto: typing.Type, just_check=False
    ) -> bool | Any:
        """
        Attempt to resolve `name` to a type that implements the protocol
        `proto`.

        If `just_check` is False, error is thrown if it is not an instance.
        Otherwise, a boolean representing whether it is an instance is
        returned.
        """
        result = self.resolve_name(name)
        isproto = isinstance(result, proto)

        if just_check:
            return isproto
        else:
            if not isproto:
                raise TypeError(
                    f"expected an implementation of {proto.__name__}, got "
                    f"'{name}' which does not implement {proto.__name__}"
                )

            return result

    def assign_name(self, name, value):
        if len(self.stack) == 0:
            raise AssertionError(
                "attempted to call assign_name() on an empty stack"
            )

        self.stack[-1].assign_name(name, value)

    def current_scope_node(self) -> Any:
        if len(self.stack) < 1:
            raise AssertionError(
                "attempted to get current scope node on an empty stack"
            )

        return self.stack[-1].scoping_obj

    @contextmanager
    def new_scope(self, scope: Scope) -> Iterator[None]:
        self.stack.append(scope)
        try:
            yield
        finally:
            self.stack.pop()

    @contextmanager
    def new_binding_context(
        self, binding_context: BindingContext
    ) -> Iterator[None]:
        try:
            with self.stack[-1].new_binding_context(binding_context):
                yield
        finally:
            self.stack.pop()
