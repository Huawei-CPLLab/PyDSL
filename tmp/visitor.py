import ast
import inspect
import types
import typing
from collections.abc import Iterable
from contextlib import contextmanager
from functools import cache, reduce
from typing import Callable, Never

from mlir.dialects import func, transform, scf, arith
from mlir.ir import (
    Context,
    InsertionPoint,
    Location,
    Module,
    StringAttr,
    UnitAttr,
)

# BEWARE OF WHAT YOU IMPORT HERE. IMPORTING ANYTHING SUCH AS type OR macro
# WILL CAUSE CYCLIC ERROR, AS ALMOST EVERY MODULE IN python_bindings_mlir RELIES ON compiler
# MODULE!
#
# Classes such as types and CallingMacro have a level of indirection through
# Protocols and duck-typed calling conventions so that they can be used without
# being imported explicitly.
# E.g. CallingMacro implements the CompileTimeCallable protocol.
# E.g. Addition in Int and Float uses the op_add magic function.
from python_bindings_mlir.helpers import (
    CompileTimeSliceable,
    CompileTimeTestable,
    HandlesFor,
    HandlesIf,
    SubtreeOut,
    handle_CompileTimeCallable,
    lower_flatten,
    lower_single,
    ToMLIRBase,
)
from python_bindings_mlir.scope import ScopeStack
from python_bindings_mlir.type import Number


def generate_parent(root):
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            child.parent = node


def generate_next_line(root):
    for node in ast.walk(root):
        if hasattr(node, "body") and isinstance(body := node.body, Iterable):
            for i, child in enumerate(body):
                child.next_line = body[i + 1] if (i + 1) < len(body) else None


class CompilationError(Exception):
    def __init__(
        self,
        msg: str = "compilation failed",
        exception: Exception = None,
        node=None,
        src=None,
    ) -> None:
        super().__init__(msg)
        self.exception = exception
        self.node = node
        self.src = src

    def update_note(self):
        self.__notes__ = [self.programmer_message()]

    _exception: Exception | None = None

    @property
    def exception(self):
        return self._exception

    @exception.setter
    def exception(self, value: Exception) -> None:
        self._exception = value
        self.update_note()

    _node: ast.AST | None = None

    @property
    def node(self):
        return self._node

    @node.setter
    def node(self, value: ast.AST) -> None:
        self._node = value
        self.update_note()

    _src: str | None = None

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, value: str) -> None:
        self._src = value
        self.update_note()

    def programmer_message(self):
        # TODO: File, function info not implemented
        # A typical Python stack trace looks like this
        # File "ast.py", line 52, in parse
        #     return compile(source, filename, mode, flags,
        #         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # File "<unknown>", line 1
        #     a =
        #         ^

        error_info = (
            f"{type(self.exception).__name__}: {self.exception!s}"
            if self.exception
            else ""
        )
        line_info = (
            f"Line {self.node.lineno}"
            if all([self.node, hasattr(self.node, "lineno")])
            else ""
        )

        line_display = []
        if all([
            self.node,
            *(
                hasattr(self.node, attr)
                for attr in [
                    "lineno",
                    "col_offset",
                    "end_lineno",
                    "end_col_offset",
                ]
            ),
            self.src,
        ]):
            lines = self.src.splitlines()[
                self.node.lineno - 1 : self.node.end_lineno
            ]

            for i, l in enumerate(lines):
                col_offset = self.node.col_offset if i == 0 else 0
                end_col_offset = (
                    self.node.end_col_offset
                    if (i == len(lines) - 1)
                    else len(l)
                )

                squiggle = " " * col_offset + "^" * (
                    end_col_offset - col_offset
                )
                line_display.append(f"{l}\n{squiggle}")

        joined = "\n".join(line_display)
        return f"{line_info}\n{joined}\n{error_info}"


DIRECTIVE_ESCAPE = "@"


class ToMLIR(ToMLIRBase):
    mlir = None
    scope_stack: ScopeStack = None
    interceptor_stack = []
    context_stack = []
    """
    The last chain of attributes evaluated. This needs to be stored for cases
    where it is needed by method calls that require self or cls.
    """
    dont_catch = False

    # Results are cached because visiting a node multiple times can result in
    # MLIR programs being generated multiple times
    # ast.NodeVisitor cannot modify the tree it is visiting, so cache will
    # never be outdated
    @cache
    def visit(self, node) -> SubtreeOut:
        try:
            visitation = super().visit(node)

            # compose all interceptors starting from bottom of the stack and
            # apply to visitation
            return reduce(
                lambda x, f: f(x), self.interceptor_stack, visitation
            )

        except Exception as e:
            if self.dont_catch:
                raise e

            if issubclass(type(ce := e), CompilationError):
                raise ce

            ce = CompilationError(exception=e)
            # add node info and raise it further up the call stack
            ce.node = node
            raise ce from e

    @contextmanager
    def new_context(self, flag):
        self.context_stack.append(flag)
        try:
            yield
        finally:
            self.context_stack.pop()

    def visit_with_interception(
        self, node, interceptor: typing.Callable[[SubtreeOut], SubtreeOut]
    ) -> SubtreeOut:
        """
        Visit a part of the tree where each partial result of the tree goes
        through the interceptor before returned. This is useful for recursive
        applications.

        Multiple interceptors can be present at once. Existing interceptors
        will be applied before newer ones.
        """
        self.interceptor_stack.append(interceptor)
        retv = self.visit(node)
        self.interceptor_stack.pop()

        return retv

    def __init__(self, f_locals) -> None:
        super(ToMLIR, self).__init__()
        # f_locals is a dictionary of local variables at the time the function
        # being compiled is defined
        self.f_locals = f_locals
        self.setup()

    def setup(self) -> None:
        self.mlir = None
        self.scope_stack = ScopeStack(self.f_locals)

    def visit_Tuple(self, node: ast.Tuple) -> SubtreeOut:
        return tuple(self.visit(entry) for entry in node.elts)

    def handle_directive(self, node: ast.Expr) -> SubtreeOut:
        val = node.value.value

        if (not hasattr(node, "next_line")) or (
            operand := node.next_line
        ) is None:
            raise ValueError(
                "docstring '@' directive must be placed before a valid "
                "operator"
            )

        directive_expr = val[len(DIRECTIVE_ESCAPE) :]
        directive_ast = ast.parse(directive_expr).body[0]

        if type(directive_ast) is not ast.Expr:
            raise ValueError("docstring '@' directive is not an expression")

        match value := directive_ast.value:
            case ast.Call():
                # adds the AST as the first parameter, similar to
                # how self works in Python
                return handle_CompileTimeCallable(
                    self, value, prefix_args=[operand]
                )

            case _:
                raise TypeError(
                    f"docstring '@' directive expression immediately contains "
                    f"{type(value)}, which is not supported"
                )

    def visit_Expr(self, node: ast.Expr) -> SubtreeOut:
        DIRECTIVE_ESCAPE = "@"

        expr_val = node.value

        match expr_val:
            case ast.Constant():
                val = expr_val.value
                if type(val) is str and val.startswith(DIRECTIVE_ESCAPE):
                    return self.handle_directive(node)
            case _:
                self.visit(expr_val)

    def visit_Constant(self, node: ast.Constant) -> Never:
        return Number(ast.literal_eval(node))

    def visit_Call(self, node: ast.Call) -> SubtreeOut:
        return handle_CompileTimeCallable(self, node)

    # for now, assume all operands are floating point numbers

    def visit_BinOp(self, node: ast.BinOp) -> SubtreeOut:
        left = self.visit(node.left)
        right = self.visit(node.right)

        # TODO: this does not yet support the right variants
        match node.op:
            case ast.Add():
                return left.op_add(right)
            case ast.Sub():
                return left.op_sub(right)
            case ast.Mult():
                return left.op_mul(right)
            case ast.MatMult():
                return left.op_matmul(right)
            case ast.Div():
                return left.op_truediv(right)
            case ast.FloorDiv():
                return left.op_floordiv(right)
            case ast.Mod():
                return left.op_mod(right)
            # divmod must be supported with calling macros
            case ast.Pow():
                return left.op_pow(right)
            case ast.LShift():
                return left.op_lshift(right)
            case ast.RShift():
                return left.op_rshift(right)
            case ast.BitAnd():
                return left.op_and(right)
            case ast.BitXor():
                return left.op_xor(right)
            case ast.BitOr():
                return left.op_xor(right)
            case ast.And():
                return left.on_And(right)
            case ast.Or():
                return left.on_Or(right)
            # TODO: more ops can be added in the future
            case _:
                raise SyntaxError(
                    f"{type(node.op)} is not supported as a binary operator"
                )

    def visit_BoolOp(self, node: ast.BoolOp) -> SubtreeOut:
        values = [self.visit(i) for i in node.values]

        match node.op:
            case ast.And():
                reducer = lambda a, b: a.on_And(b)
            case ast.Or():
                reducer = lambda a, b: a.on_Or(b)
            case _:
                raise SyntaxError(
                    f"{type(node.op)} is not supported as a boolean operator"
                )

        return reduce(reducer, values[1:], values[0])

    def visit_Compare(self, node: ast.Compare) -> SubtreeOut:
        def op_eval(opt, left, right):
            match opt:
                case ast.Lt():
                    return left.op_lt(right)
                case ast.LtE():
                    return left.op_le(right)
                case ast.Eq():
                    return left.op_eq(right)
                case ast.NotEq():
                    return left.op_ne(right)
                case ast.Gt():
                    return left.op_gt(right)
                case ast.GtE():
                    return left.op_ge(right)
                case ast.Is():
                    raise NotImplementedError(
                        "identity comparison is not implemented"
                    )
                case ast.IsNot():
                    raise NotImplementedError(
                        "identity comparison is not implemented"
                    )
                case ast.In():
                    return left.op_contains(right)
                case ast.NotIn():
                    return left.op_contains(right).on_Not()
                case _:
                    raise NotImplementedError(
                        f"the operator {opt} is not implemented"
                    )

        match node:
            case ast.Compare(left=first, ops=operators, comparators=rest):
                operands = [self.visit(n) for n in [first, *rest]]
                partials = [
                    op_eval(op, operands[i], operands[i + 1])
                    for i, op in enumerate(operators)
                ]
                return reduce(
                    lambda l, r: l.on_And(r), partials[1:], partials[0]
                )
            case _:
                raise NotImplementedError(
                    "this format of comparison is not implemented"
                )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> SubtreeOut:
        operand = self.visit(node.operand)

        match node.op:
            case ast.USub():
                return operand.op_neg()
            case ast.UAdd():
                return operand.op_pos()
            # abs must be supported with calling macros
            case ast.Invert():
                return operand.op_invert()
            case ast.Not():
                return operand.on_Not()
            case _:
                raise SyntaxError(
                    f"{type(node.op)} is not supported as an unary operator"
                )

    def visit_Name(self, node: ast.Name) -> SubtreeOut:
        return self.scope_stack.resolve_name(node.id)

    def visit_Return(self, node: ast.Return) -> SubtreeOut:
        try:
            current_func = self.scope_stack.current_scope_node()
        except AssertionError as e:
            raise SyntaxError(
                "attempted to return while not inside of a function"
            ) from e

        if not isinstance(current_func, ast.FunctionDef):
            raise SyntaxError(
                "attempted to return while not inside of a function"
            )

        _, ret_t = self.get_func_typing(current_func)

        match node.value:
            case None:
                ret = ()
            case ast.Tuple():
                ret = self.visit(node.value)
            case _:
                ret = (self.visit(node.value),)

        if not len(ret) == len(ret_t):
            raise TypeError(
                f"expected {len(ret_t)} return values but {len(ret)}"
            )

        if not all([
            isinstance(v, t) for v, t in zip(ret, ret_t, strict=False)
        ]):
            raise TypeError(
                f"function expected return types {ret_t}. Got "
                f"{tuple([type(v) for v in ret])}"
            )

        return func.ReturnOp(lower_flatten([*ret]))

    def visit_Subscript(self, node: ast.Subscript) -> SubtreeOut:
        assert type(node.ctx) is not ast.Store, AssertionError(
            "subscript with a Store context shouldn't be visited!"
        )

        value = self.visit(node.value)
        slice = node.slice
        if isinstance(value, CompileTimeSliceable):
            return value.on_getitem(self, slice)
        raise TypeError(f"{value} does not implement CompileTimeSliceable")

    def visit_Assign(self, node: ast.Assign) -> SubtreeOut:
        # ignore tuple assignment for now, hence the "node.targets[0]"
        match target := node.targets[0]:
            case ast.Name():
                match node.value:
                    case ast.Constant():
                        raise ValueError(
                            "assigning constant without type hinting is "
                            "currently not supported"
                        )
                    case _:
                        rhs = self.visit(node.value)
                        self.scope_stack.assign_name(target.id, rhs)
                        return rhs

            case ast.Subscript():
                value = node.value
                target = self.visit(node.targets[0].value)
                slice = node.targets[0].slice
                if isinstance(target, CompileTimeSliceable):
                    return target.on_setitem(self, slice, value)
                raise TypeError(
                    f"{target} does not implement CompileTimeSliceable"
                )

            case _:
                raise ValueError(
                    f"assigning to {type(node.targets)} is currently not "
                    f"supported."
                )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> SubtreeOut:
        # TODO: cannot deal with cases like a: Index = 40 / 8 or
        # a: F32 = array[i]. It's pretty ugly right now.
        try:
            literal = ast.literal_eval(node.value)
        except ValueError:
            raise NotImplementedError(
                "assigning non-Constant literals to annotated assigns is "
                "currently not supported"
            )

        self.scope_stack.assign_name(
            node.target.id,
            retval := self.scope_stack.resolve_name(node.annotation.id)(
                literal
            ),
        )

        return retval

    def visit_Attribute(self, node: ast.Attribute) -> SubtreeOut:
        # we only care about the last element in the attribute chain
        return self.scope_stack.resolve_attr_chain(node)[-1]

    def get_attr_chain(self, node: ast.Attribute | ast.Name) -> list[str]:
        """
        Take an Attribute or Name node and turn it into a list of attributes
        in the chain. Rightmost attribute naming occurs as the first element
        of the list.

        A Name will simply be evaluated to its id nested in a list.
        """
        match node.value:
            case ast.Attribute(attr=attrname):  # chain length > 1
                next_node = node.value
                return self.get_attr_chain(next_node).insert(0, attrname)
            case ast.Name(id=id):  # chain length = 1
                return [id]

    def visit_For(self, node: ast.For) -> SubtreeOut:
        iterator = node.iter

        # we will not accept any other way to pass in an iterator for now
        assert (
            type(iterator) is ast.Call
        ), "iterator of the for loop must be a Call for now"

        name = iterator.func.id
        iterator = self.scope_stack.resolve_as_protocol(name, HandlesFor)

        return iterator.on_For(self, node)

    @cache
    def get_func_typing(
        self, node: ast.FunctionDef
    ) -> tuple[tuple[SubtreeOut], tuple[SubtreeOut]]:
        """
        Return a tuple of (argument type hint, return type hint)
        """
        arg_annotation = [arg.annotation for arg in node.args.args]

        if None in arg_annotation:
            arg = node.args.args[arg_annotation.index(None)]
            raise CompilationError(
                exception=SyntaxError(
                    f"type not hinted for argument {arg.arg}"
                ),
                node=arg,
            )

        arg_types = tuple([
            self.scope_stack.resolve_name(ann.id) for ann in arg_annotation
        ])

        match node.returns:
            case None | ast.Constant(value=None):
                return_types = ()

            # just a single return type name
            case ast.Name(id=id):
                return_types = tuple([self.scope_stack.resolve_name(id)])

            # TODO: maybe Tuple should be a type defined by the compiler
            # rather than relying on the built-in tuple...

            # a tuple type
            case ast.Subscript(
                value=ast.Name(id=id), slice=(ast.Tuple(elts=elts) | elts)
            ) if typing.get_origin(self.scope_stack.resolve_name(id)) is tuple:
                if not isinstance(elts, Iterable):
                    elts = (elts,)

                if not all([isinstance(e, ast.Name) for e in elts]):
                    raise TypeError("illegal type hint slice for tuple")

                return_types = tuple([
                    self.scope_stack.resolve_name(e.id) for e in elts
                ])

            case _:
                raise TypeError(
                    f"return type hint for {node.name} is not supported"
                )

        return arg_types, return_types

    # TODO: every method in ToMLIR should be using this
    # instead of calling get_func_typing directly
    # This is a refactoring effort worth looking into.
    def get_func_signature(self, node: ast.FunctionDef) -> inspect.Signature:
        arg_names = [arg.arg for arg in node.args.args]
        arg_types, return_type = self.get_func_typing(node)

        # convert these info into an inspect.Signature

        # return type needs to be an object suitable for type hinting
        # most are OK as-is, but tuples must be a GenericAlias
        match return_type:
            case tuple():
                return_annotation = types.GenericAlias(
                    type(return_type), return_type
                )
            case _:
                return_annotation = return_type

        # parameters are inspect.Parameter objects
        parameters = [
            inspect.Parameter(
                name=name,
                annotation=ann,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for (name, ann) in zip(arg_names, arg_types)
        ]

        return inspect.Signature(
            parameters=parameters,
            return_annotation=return_annotation,
        )

    def visit_If(self: typing.Self, node: ast.If) -> SubtreeOut:
        # TODO: This is currently very barebone just to get things out of the
        # way. You need proper use analysis to improve if statements
        scope = self.scope_stack

        match node:
            # Case for when a HandlesIf is passed as the test
            case ast.If(
                test=ast.Call(func=(ast.Name(id=name)))
            ) if scope.resolve_as_protocol(name, HandlesIf, just_check=True):
                handler = scope.resolve_as_protocol(name, HandlesIf)
                return handler.on_If(self, node)

            # Case for when something that can be evaluated is passed as the
            # test
            case ast.If(test=test, body=body, orelse=orelse):
                has_else = len(orelse) != 0

                test = self.visit(test)
                if not isinstance(test, CompileTimeTestable):
                    raise TypeError(
                        f"if statement expected Bool or types supporting "
                        f"Bool, got {test}"
                    )

                if_exp = scf.IfOp(
                    lower_single(test.Bool()),
                    [],  # results are empty for now
                    hasElse=has_else,
                )

                with InsertionPoint(if_exp.then_block):
                    for b in body:
                        self.visit(b)
                    scf.YieldOp([])

                if has_else:
                    with InsertionPoint(if_exp.else_block):
                        for b in orelse:
                            self.visit(b)
                        scf.YieldOp([])

                return if_exp

            case _:
                raise SyntaxError("unexpected form in if statement")

    def visit_IfExp(self, node: ast.IfExp) -> SubtreeOut:
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)

        if not isinstance(test, CompileTimeTestable):
            raise TypeError(
                f"if expression expected Bool or types supporting Bool, got "
                f"{test}"
            )

        if not (body_t := type(body)) is (orelse_t := type(orelse)):
            raise TypeError(
                f"if expression require branches to be of same type, got "
                f"{body_t} and {orelse_t}"
            )

        return body_t(
            arith.SelectOp(
                lower_single(test), lower_single(body), lower_single(orelse)
            ).result
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> SubtreeOut:
        arg_names = [arg.arg for arg in node.args.args]
        arg_types, return_type = self.get_func_typing(node)

        f = func.FuncOp(
            node.name,
            # (input, result)
            (
                lower_flatten(arg_types),
                lower_flatten(return_type) if return_type is not None else [],
            ),
        )

        # we will assume all functions are public for now
        f.sym_visibility = StringAttr.get("public")
        f.add_entry_block()

        if type(node.body[-1]) is not ast.Return:
            if return_type == ():
                # insert an empty return at the end
                node.body.append(ast.Return())
            else:
                raise ValueError(
                    "return operator must be at the end if function is not "
                    "void (for now)"
                )

        with (
            InsertionPoint(f.entry_block),
            self.scope_stack.new_func_scope(
                node,
                {
                    name: arg_types[i](f.arguments[i])
                    for i, name in enumerate(arg_names)
                },
                return_type if return_type is not None else (),
            ),
        ):
            for n in node.body:
                self.visit(n)

        return f

    # TODO: This may not be the best place for this function, but it's very
    # similar to ToMLIR.visit_FunctionDef
    # Maybe have a separate transform-seq-based visitor that inherits
    # the main ToMLIR visitor but with this method overwritten

    def visit_FunctionDef_as_transform_seq(
        self, node: ast.FunctionDef
    ) -> SubtreeOut:
        arg_names = [arg.arg for arg in node.args.args]
        arg_types = [
            self.scope_stack.resolve_name(arg.annotation.id)
            for arg in node.args.args
        ]

        if len(arg_names) != 1:
            raise ValueError(
                f"function {node.name} used as transform.sequence operation "
                f"should have exactly 1 argument, got {len(arg_names)}"
            )

        seq = transform.NamedSequenceOp(
            "__transform_main",
            lower_flatten(arg_types),
            [],
        )

        with InsertionPoint(seq.body):
            with self.scope_stack.new_func_scope(
                node,
                {arg_names[0]: seq.bodyTarget},
                (),  # no return value
            ):
                for n in node.body:
                    self.visit(n)

                transform.YieldOp()

        return seq

    # TODO: consider also having a WithMacro, which supports With-As cluase
    def visit_With(self, node: ast.With) -> SubtreeOut:
        for item in node.items:
            if type(call := item.context_expr) is ast.Call:
                return handle_CompileTimeCallable(
                    self, call, prefix_args=[node.body]
                )
            raise NotImplementedError(
                "with statement currently only allows calls as contexts"
            )

    def compile(
        self,
        node,
        transform_seq: ast.AST | None = None,
        interceptor: Callable[[SubtreeOut], SubtreeOut] | None = None,
    ) -> str:
        # create additional properties in AST nodes that we will need during
        # compilation
        generate_parent(node)
        generate_next_line(node)

        self.setup()

        # pre-initiate symbol objects

        with Context() as _, Location.unknown():
            self.mlir = Module.create()
            with InsertionPoint(self.mlir.body):
                if interceptor is not None:
                    self.visit_with_interception(node, interceptor)
                else:
                    self.visit(node)

                if transform_seq is not None:
                    # TODO: this is hacky, should create a dedicated transform
                    # seq visitor
                    self.mlir.operation.attributes[
                        "transform.with_named_sequence"
                    ] = UnitAttr.get()
                    self.visit_FunctionDef_as_transform_seq(
                        transform_seq.body[0]
                    )

            self.mlir.operation.verify()
            return str(self.mlir)
