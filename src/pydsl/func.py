import ast
import enum
import inspect
import itertools
import textwrap
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cache
from inspect import BoundArguments, Parameter, Signature

import mlir.dialects.func as func
import mlir.dialects.transform as transform
import mlir.ir as mlir
from mlir.ir import FunctionType, InsertionPoint, StringAttr, Value

from pydsl.analysis.names import BoundAnalysis, UsedAnalysis
from pydsl.protocols import (
    Lowerable,
    SubtreeOut,
    ToMLIRBase,
    lower,
    lower_flatten,
)
from pydsl.scope import Scope, ScopeStack

ArgsT = typing.TypeVar("ArgsT")
RetT = typing.TypeVar("RetT")


class Visibility(enum.Enum):
    PRIVATE = enum.auto()
    PUBLIC = enum.auto()


class FunctionLike(typing.Generic[ArgsT, RetT], ABC):
    # Subclass-specific fields
    argst: tuple[Lowerable] = None
    """
    The type of the arguments of the function
    """
    rett: Lowerable = None
    """
    The type of the return value of the function
    None represents void return
    """
    _default_name = "unnamed_funclike"
    _default_subclass_name = "FunctionLikeUnnamedSubclass"
    _default_visibility = Visibility.PUBLIC

    # Instance-specific fields
    name: str = None
    _signature: typing.Optional[Signature] = None
    """
    The signature of a specific FunctionLike. This means that it includes:
    - Name of each argument
    - Type annotation of each argument
    - Type annotation of the return argument
    """
    val: func.FuncOp
    """
    The internal MLIR representation
    """
    visibility: Visibility
    """
    Whether this FunctionLike can be accessed by other modules
    """

    def __init__(
        self,
        name=_default_name,
        signature=None,
        visibility=_default_visibility,
    ):
        """
        The initialization of FunctionLike defines the basic fields of the
        FunctionLike.
        """
        self.name = name
        self._signature = signature
        self.visibility = visibility
        self.val = self.init_val()

    @abstractmethod
    def init_val(self) -> mlir.Operation:
        """
        Initializes the internal MLIR operation of the FunctionLike.

        The body of the value may or may not be initialized. This is instead
        done in `init_val_body`.

        The delaying of defining the body allows for e.g. recursion of a
        Function.
        """
        ...

    @abstractmethod
    def init_val_body(
        self: typing.Self, visitor: ToMLIRBase, node: ast.FunctionDef
    ) -> None:
        """
        Initializes the body of the internal MLIR operation of the
        FunctionLike.

        This requires `init_val` to be called first.
        """
        ...

    @abstractmethod
    def _get_new_vars(self) -> dict[str, SubtreeOut]:
        """
        Return a dictionary of variables being defined when the program
        enters the body of the function
        """
        ...

    @abstractmethod
    def lift(
        self: typing.Self,
        rep: tuple[FunctionType],
        name=_default_name,
    ) -> None: ...

    @abstractmethod
    def lower(self: typing.Self) -> tuple[Value]: ...

    @classmethod
    @abstractmethod
    def lower_class(cls) -> tuple[mlir.Type]: ...

    @property
    def signature(self) -> Signature:
        """
        Returns the signature of the FunctionLike instance.

        If the instance doesn't have information on its signature, the class's
        function signature is used instead. Note however that the class
        signature only include information about the argument and return type
        and lacks instance-specific information such as argument names.
        """
        if self._signature is not None:
            return self._signature
        else:
            return self.class_signature()

    @property
    def return_type(self) -> type:
        return self.signature.return_annotation

    @classmethod
    def full_init(
        cls, visitor: ToMLIRBase, node: ast.FunctionDef
    ) -> "Function":
        f = cls.node_to_header(visitor, node)
        f.init_val_body(visitor, node)
        return f

    @classmethod
    @cache
    def node_to_header(
        cls, visitor: ToMLIRBase, node: ast.FunctionDef
    ) -> "Function":
        # It is ESSENTIAL that this function result is cached, as this function
        # can potentially be called multiple times with the same
        # FunctionDef, but every FunctionDef needs to be mapped to a unique
        # Function object

        # This acts as the gatekeeper of maintaining one-to-one correspondence
        # between every FunctionDef and every Function instance

        reg_sig = cls.FunctionDef_to_signature(visitor, node)
        # Get the Function subclass corresponding to the signature
        subcls = cls.signature_to_class(visitor, reg_sig)

        # TODO: no way to change visibility
        return subcls(name=node.name, signature=reg_sig)

    @classmethod
    def on_class_getitem(
        cls, visitor: ToMLIRBase, slice: ast.AST
    ) -> type["Function"]:
        match slice:
            case ast.Tuple(elts=[ast.List(elts=args), ret]):
                pass  # we will use the matched arguments below
            case _:
                raise ValueError(
                    "improper type arguments to Function, format should be "
                    "Function[[*arg_types], ret_type]"
                )

        args = [visitor.resolve_type_annotation(a) for a in args]
        ret = visitor.resolve_type_annotation(ret)

        return cls.class_factory(args, ret)

    @classmethod
    @cache
    def class_factory(
        cls,
        argst: tuple[Lowerable],
        rett: typing.Optional[Lowerable],
        name=_default_subclass_name,
    ):
        """
        Create a new subclass of Function dynamically with the specified
        argument types and return type
        """

        def check(typ: typing.Any) -> None:
            if not isinstance(typ, Lowerable):
                raise TypeError(
                    f"arguments and return type of {cls.__name__} must be "
                    f"lowerable"
                )

        for a in argst:
            check(a)

        if rett is not None:
            check(rett)

        return type(
            name,
            (cls,),
            {
                "argst": tuple(argst),
                "rett": rett,
            },
        )

    @contextmanager
    def _new_scope(
        self, stack: ScopeStack, node: ast.FunctionDef
    ) -> Iterator[None]:
        """
        Establish a new function scope in the scope stack while the context
        is open.
        """
        func_args = self._get_new_vars()

        used = UsedAnalysis.analyze(node)
        bound = BoundAnalysis.analyze(node)

        # Replacing the next line with names_table = {} passes all tests right
        # now. I suspect this is only useful for nested functions
        names_table = {name: stack.find_name(name) for name in (used - bound)}
        names_table.update(func_args)

        with stack.new_scope(Scope(self, names_table, bound)):
            yield

    def on_Return(self, visitor: "ToMLIRBase", node: ast.Return) -> SubtreeOut:
        retv: SubtreeOut = (
            visitor.visit(node.value) if node.value is not None else None
        )
        rett = self.rett if self.rett is not None else type(None)

        # check if return value matches function return type
        if not isinstance(retv, rett):
            try:  # try casting
                retv = rett(retv)
            except TypeError as e:  # if casting failed
                raise TypeError(
                    f"function expected return types {rett}. Got {retv}"
                ) from e

        return func.ReturnOp(lower(retv) if retv is not None else ())

    @classmethod
    def _lower_typing(cls) -> tuple[tuple[SubtreeOut], tuple[SubtreeOut]]:
        """
        Return a tuple of (argument type hints, return type hints), where
        both elements are also tuples.

        This is exactly as it would be accepted by `mlir.func.FuncOp` or
        `mlir.ir.FunctionType.get`.
        """
        lowered_argst = tuple(lower_flatten(cls.argst))
        lowered_rett = (
            tuple(lower_flatten([cls.rett])) if cls.rett is not None else ()
        )
        return lowered_argst, lowered_rett

    @classmethod
    def _lowered_argst(cls) -> tuple[SubtreeOut]:
        return cls._lower_typing()[0]

    @classmethod
    def _lowered_rett(cls) -> tuple[SubtreeOut]:
        return cls._lower_typing()[1]

    @classmethod
    @cache
    def class_signature(cls) -> Signature:
        """
        A backup signature used for situation where a Function is created
        from only an MLIR FunctionType.

        This should be used if Function._signature = None
        """
        return Signature(
            parameters=[
                Parameter(
                    name=f"unnamed_arg_{i}",
                    kind=Parameter.POSITIONAL_ONLY,
                    annotation=argt,
                )
                for i, argt in enumerate(cls.argst)
            ],
            return_annotation=cls.rett,
        )

    @staticmethod
    def _enforced_signature(sig: Signature) -> Signature:
        """
        Convert a regular signature into one specialized for PyDSL's existing
        features, following these rules:
        - All arguments become regular arguments (i.e.
          `Parameter.POSITIONAL_OR_KEYWORD`)
            - Their resulting order is identical to the order in which they
              are defined in the original function
        - If type hinting is missing in any argument or the return annotation,
          a SyntaxError is thrown
        - All default values of arguments are stripped
        - If vararg or kwarg is detected, a NotImplementedError is thrown
        """

        def enforce_param(p: Parameter):
            if p.annotation == Parameter.empty:
                raise SyntaxError(
                    "all arguments of function must be type-annotated"
                )

            match p.kind:
                case Parameter.VAR_POSITIONAL | Parameter.VAR_KEYWORD:
                    raise NotImplementedError(
                        "variable positional/keyword arguments are not "
                        "supported in functions"
                    )
                case _:
                    return p.replace(
                        kind=Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=p.annotation,
                    )

        enforced_params = [enforce_param(p) for p in sig.parameters.values()]

        # TODO: this is still an open design decision. No annotation should
        # mean Any, which is not allowed. But it's more ergonomic for it to
        # just mean None, which is 90% of the case. People are more likely
        # to not add annotation because the function is void rather than
        # because they forgot to enforce a type.
        if sig.return_annotation == Signature.empty:
            sig = sig.replace(return_annotation=None)
            # raise SyntaxError("return of function must be type-annotated")

        sig = sig.replace(
            parameters=enforced_params, return_annotation=sig.return_annotation
        )

        return sig

    @staticmethod
    @cache
    def FunctionDef_to_signature(
        visitor: ToMLIRBase, node: ast.FunctionDef
    ) -> Signature:
        """
        Converts a FunctionDef AST node into a regular signature, with
        some special rules on resolving annotations:
        - All type annotations are resolved into class objects based on
          variables held by visitor's `scope_stack`.
            - E.g. `Tuple[F32, UInt32]` becomes a dynamically generated `Tuple`
              class with elements `F32, UInt32`
            - Exception: lack of annotation becomes
              `Parameter.empty`/`Signature.empty`
        """
        all_args = node.args

        kinds = [
            (
                posonlyargs := all_args.posonlyargs,
                Parameter.POSITIONAL_ONLY,
            ),
            (
                arg := all_args.args,
                Parameter.POSITIONAL_OR_KEYWORD,
            ),
            (
                vararg := (
                    [all_args.vararg] if all_args.vararg is not None else []
                ),
                Parameter.VAR_POSITIONAL,
            ),
            (
                all_args.kwonlyargs,
                Parameter.KEYWORD_ONLY,
            ),
            (
                kwarg := (
                    [all_args.kwarg] if all_args.kwarg is not None else []
                ),
                Parameter.VAR_KEYWORD,
            ),
        ]

        defaults = all_args.defaults
        kw_defaults = all_args.kw_defaults

        # a sequential, flattened list of args, flattened from the following
        # lists in exact order: posonlyargs, args, varargs, kwonlyargs, kwarg
        all_args_flat: list[ast.arg] = itertools.chain(*[
            klist for klist, _ in kinds
        ])

        # the parameter kind of each ast.arg in all_args_flat
        kinds_flat = itertools.chain(*[
            [pkind] * len(klist) for klist, pkind in kinds
        ])

        # the default_value of each ast.arg in all_args_flat
        defaults_flat: list = [
            *(
                [Parameter.empty]
                * (len(posonlyargs) + len(arg) - len(defaults))
            ),
            *defaults,  # right-aligned from end of arg
            *([Parameter.empty] * len(vararg)),  # vararg is empty
            *[d if d is not None else Parameter.empty for d in kw_defaults],
            *([Parameter.empty] * len(kwarg)),  # kwarg is empty
        ]

        def build_param(
            arg: ast.arg, kind, default: typing.Optional[ast.AST]
        ) -> Parameter:
            if default != Parameter.empty:
                default = visitor.visit(default)

            annotation = (
                visitor.resolve_type_annotation(arg.annotation)
                if arg.annotation is not None
                else Parameter.empty
            )

            return Parameter(
                name=arg.arg,
                default=default,
                annotation=annotation,
                kind=kind,
            )

        parameters = [
            build_param(arg, kind, default)
            for arg, kind, default in zip(
                all_args_flat, kinds_flat, defaults_flat, strict=True
            )
        ]

        return_annotation = visitor.resolve_type_annotation(node.returns)
        return_annotation = (
            return_annotation
            if return_annotation is not None
            else Signature.empty
        )

        return Signature(
            parameters=parameters,
            return_annotation=return_annotation,
        )

    @classmethod
    @cache
    def signature_to_class(
        cls, visitor: ToMLIRBase, sig: Signature
    ) -> type["Function"]:
        """
        Generate a new Function subclass based on the provided signature.
        """
        lower_sig: Signature = cls._enforced_signature(sig)
        argst = tuple([p.annotation for p in lower_sig.parameters.values()])
        rett = lower_sig.return_annotation

        assert (
            rett is not Signature.empty
        ), "signature contains Signature.empty after lowered"

        return cls.class_factory(argst, rett)


class Function(FunctionLike):
    _default_name = "unnamed_func"
    _default_subclass_name = "FunctionUnnamedSubclass"

    def init_val(self) -> mlir.Operation:
        match self.visibility:
            case Visibility.PUBLIC:
                func_visibility = StringAttr.get("public")
            case Visibility.PRIVATE:
                func_visibility = StringAttr.get("private")
            case _:
                raise NotImplementedError(
                    f"visibility {self.visibility} is not supported"
                )

        val = func.FuncOp(
            self.name,
            # (input, result)
            self._lower_typing(),
        )

        val.sym_visibility = func_visibility

        return val

    def init_val_body(
        self: typing.Self, visitor: ToMLIRBase, node: ast.FunctionDef
    ) -> None:
        if type(node.body[-1]) is not ast.Return:
            if self._lowered_rett() == ():
                # insert an empty return at the end
                # TODO: transforming AST like this is not allowed in
                # NodeVisitor. This should instead be a NodeTransformer pass
                # that happens before visiting.
                node.body.append(ast.Return())
            else:
                raise ValueError(
                    "return operator must be at the end if function is not "
                    "void (for now)"
                )

        self.val.add_entry_block()
        with (
            InsertionPoint(self.val.entry_block),
            self._new_scope(visitor.scope_stack, node),
        ):
            for n in node.body:
                visitor.visit(n)

    def _get_new_vars(self) -> dict[str, SubtreeOut]:
        return {
            # TODO: change this to using typ.lift(arg) function once that's
            # in Lowerable protocol
            name: typ(arg)
            for arg, typ, name in zip(
                self.val.arguments,
                self.argst,
                self.signature.parameters.keys(),  # argument names
            )
        }

    def lift(
        self: typing.Self,
        rep: tuple[FunctionType],
        name=_default_name,
    ) -> None:
        match rep:
            case (FunctionType(),):
                self.val = rep[0]
                self.name = name

                # perform argst/rett type check
                if self._lowered_argst() != tuple(
                    self.val.inputs()
                ) or self._lowered_rett() != tuple(self.val.results()):
                    raise TypeError(
                        f"{rep} cannot be lifted to Function subclass "
                        f"with inputs {self._lowered_argst()} and results "
                        f"{self._lowered_rett()}"
                    )
            case _:
                raise TypeError(f"Function cannot be defined from {type(rep)}")

    def lower(self: typing.Self) -> tuple[Value]:
        return (self.val,)

    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]:
        if not all([cls.argst, cls.rett]):
            e = TypeError(
                "attempted to lower Function without defined argument types "
                "or return types"
            )
            if (clsname := cls.__name__) != Function._default_subclass_name:
                e.add_note(f"hint: class name is {clsname}")
            raise e

        return mlir.FunctionType.get(*cls._lower_typing())

    def on_Call(
        attr_chain: list[typing.Any],
        visitor: "ToMLIRBase",
        node: ast.Call,
        prefix_args: tuple[ast.AST] = tuple(),
    ) -> SubtreeOut:
        self = attr_chain[-1]
        prefix_args = [visitor.visit(a) for a in prefix_args]
        args = [visitor.visit(a) for a in node.args]

        return func.CallOp(self.val, lower_flatten([*prefix_args, *args]))


class TransformSequence(FunctionLike):
    val: transform.NamedSequenceOp

    _default_name = "__transform_main"
    _default_subclass_name = "TransformSequenceUnnamedSubclass"

    def __init__(self, signature=None, **_):
        super().__init__(
            name=self._default_name,
            signature=signature,
            visibility=Visibility.PRIVATE,
        )

    def init_val(self) -> mlir.Operation:
        val = transform.NamedSequenceOp(
            "__transform_main",
            *self._lower_typing(),
        )

        assert (
            self.visibility == Visibility.PRIVATE
        ), "TransformSequence must be private by nature"

        return val

    def init_val_body(
        self: typing.Self, visitor: ToMLIRBase, node: ast.FunctionDef
    ) -> None:
        with (
            InsertionPoint(self.val.body),
            self._new_scope(visitor.scope_stack, node),
        ):
            for n in node.body:
                visitor.visit(n)

            transform.YieldOp()

    def _get_new_vars(self) -> dict[str, SubtreeOut]:
        return {
            # TODO: change this to using typ.lift(arg) function once that's
            # in Lowerable protocol
            name: typ(arg)
            for arg, typ, name in zip(
                [self.val.bodyTarget],
                self.argst,
                self.signature.parameters.keys(),  # argument names
            )
        }

    def lift(
        self,
        rep: tuple[transform.NamedSequenceOp],
        name=_default_name,
    ) -> None:
        match rep:
            case (transform.NamedSequenceOp(),):
                self.val = rep[0]
                self.name = name

                functype: FunctionType = self.val.function_type

                # perform argst/rett type check
                if self._lowered_argst() != tuple(
                    functype.inputs()
                ) or self._lowered_rett() != tuple(functype.results()):
                    raise TypeError(
                        f"{rep} cannot be lifted to Function subclass "
                        f"with inputs {self._lowered_argst()} and results "
                        f"{self._lowered_rett()}"
                    )
            case _:
                raise TypeError(
                    f"TransformSequence cannot be defined from {type(rep)}"
                )

    def lower(self: typing.Self) -> tuple[Value]:
        return (self.val,)

    @classmethod
    def lower_class(cls) -> tuple[mlir.Type]:
        # FIXME: not implemented
        raise NotImplementedError("FIXME")


class InlineFunction:
    """
    A PyDSL inline function. This behaves similar to a normal PyDSL function,
    but we guarantee that all of the MLIR it produces will be in-line, and will
    not create an MLIR function object.

    Currently, the inline function is able to access variables from the
    caller's scope. This is probably not ideal and should be removed.

    An InlineFunction is somewhat similar to a CallMacro with all Compiled
    arguments. The biggest difference is that the body of an InlineFunction is
    parsed as PyDSL, while the body of a CallMacro is parsed as Python. It
    should be possible to call an InlineFunction from a CallMacro by passing in
    the visitor and a SubtreeOut for each argument.
    """

    # TODO: maybe we can prevent the inline function from accessing variables
    # from an outer function by creating a new visitor object? Might need to do
    # some logic to make sure it receives the right variables from whatever
    # context it's defined in. And if at some point we decide to support having
    # an inline function defined inside a PyDSL function, that will just be
    # painful.
    # TODO: a good fraction of this code is very similar to CallMacro's code.
    # Maybe there is some way to reduce code duplication.

    found_ret: bool = False
    retv: SubtreeOut | None = None
    """
    The return value of the function is stored here.

    TODO: this won't quite work if in the future, we support an inline function
    calling itself recursively (can become possible and not lead to infinite
    recursion with
    https://github.com/Huawei-CPLLab/PyDSL/issues/17#issuecomment-3139943581).
    """

    signature: Signature
    src_ast: ast.FunctionDef

    def __init__(self, signature: Signature, src_ast: ast.FunctionDef):
        self.signature = signature
        self.src_ast = src_ast

    def on_Return(self, visitor: "ToMLIRBase", node: ast.Return) -> SubtreeOut:
        if self.found_ret:
            raise SyntaxError(
                "detected multiple return statements in inline function"
            )

        self.found_ret = True
        self.retv = (
            visitor.visit(node.value) if node.value is not None else None
        )

    def parse_args(self, *args, **kwargs) -> BoundArguments:
        """
        Try to bind arguments, apply defaults, and type cast. Arguments should
        already be compiled to PyDSL types by this point.
        """
        params = self.signature.parameters

        try:
            # Associate each argument value passed into the call with
            # the parameter of the function
            bound_args = self.signature.bind(*args, **kwargs)
            binding = bound_args.arguments
        except TypeError as e:
            raise TypeError(
                f"couldn't bind arguments when calling an inline function: {e}"
            ) from e

        # Apply defaults for unfilled arguments
        bound_args.apply_defaults()

        # Cast arguments with type annotations
        for name in binding.keys():
            ann = params[name].annotation

            if ann is not Parameter.empty and ann is not typing.Any:
                if not isinstance(binding[name], ann):
                    # Mutates bound_args
                    binding[name] = ann(binding[name])

        return bound_args

    @staticmethod
    def on_Call(
        attr_chain: list[typing.Any],
        visitor: ToMLIRBase,
        node: ast.Call,
        prefix_args: tuple[ast.AST] = tuple(),
    ) -> SubtreeOut:
        """
        Reformats arguments to (visitor, *args, **kwargs). Also parses the
        arguments to a SubtreeOut, to be passed into __call__.
        """
        fn: InlineFunction = attr_chain[-1]
        assert isinstance(
            fn, InlineFunction
        ), "InlineFunction on_Call called on not an InlineFunction"

        args = tuple(visitor.visit(arg) for arg in (*prefix_args, *node.args))
        kwargs = {kw.arg: visitor.visit(kw.value) for kw in node.keywords}

        # Equivalent to InlineFunction.__call__(fn, visitor, *args, **kwargs)
        return fn(visitor, *args, **kwargs)

    # NOTE: the important distinction between on_Call and __call__ is that
    # CallMacros and other compiler code access __call__ directly, while
    # PyDSL code accesses on_Call. So anything that is necessary for both
    # should be done in __call__, and any parsing necessary only when called
    # from a PyDSL context should be in on_Call.

    def __call__(self, visitor: ToMLIRBase, *args, **kwargs) -> SubtreeOut:
        """
        Takes in a visitor and a number of arguments, and returns the result
        of applying this function on the arguments. args and kwargs should all
        already be compiled and thus have type SubtreeOut. The new scope will
        be created here.
        """
        bound_args = self.parse_args(*args, **kwargs)

        self.found_ret = False
        self.retv = None

        stack = visitor.scope_stack
        used = UsedAnalysis.analyze(self.src_ast)
        bound = BoundAnalysis.analyze(self.src_ast)
        names_table = {name: stack.find_name(name) for name in (used - bound)}
        # Assign arguments to their respective parameter names
        names_table.update(bound_args.arguments)

        with stack.new_scope(Scope(self, names_table, bound)):
            for node in self.src_ast.body:
                visitor.visit(node)

        retv = self.retv
        rett = self.signature.return_annotation

        if rett is Signature.empty or rett is None:
            if retv is not None:
                raise TypeError(
                    f"expected return type of None for inline function, got "
                    f"{type(retv).__qualname__}. Use typing.Any to indicate "
                    f"that any return type is allowed"
                )
        elif rett is not typing.Any:
            # Try casting
            if not isinstance(retv, rett):
                retv = rett(retv)

        return retv

    @classmethod
    def generate(cls):
        """
        Returns a decorator which converts a function into an inline function.

        An inline function can be defined as:
        @InlineFunction.generate()
        def f(): ...

        We do not use something like @InlineFunction.generate or @inline
        directly, since in the future, we might want to support passing in some
        more arguments to determine how the inline function should be generated
        (similar to how CallMacro.generate has a method_type argument). For
        example, maybe we will want to support passing in a context object
        that specifies the local variables.
        """

        def generate_sub(f: Callable) -> InlineFunction:
            # ast.parse returns an ast.Module, extract only the function
            src_ast = ast.parse(textwrap.dedent(inspect.getsource(f))).body[0]

            if not isinstance(src_ast, ast.FunctionDef):
                raise AssertionError(
                    f"expected AST of inline function to be ast.FunctionDef, "
                    f"got {type(src_ast).__qualname__}"
                )

            # Return a new instance of InlineFunction
            return cls(inspect.signature(f), src_ast)

        return generate_sub
