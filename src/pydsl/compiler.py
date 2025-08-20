import ast
import dataclasses
import typing
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from functools import cache, reduce
from pathlib import Path
from typing import Any

from mlir.dialects import arith, func, scf
import mlir.ir as mlirir
from mlir.ir import (
    Context,
    InsertionPoint,
    Location,
    UnitAttr,
)

from pydsl.func import Function, TransformSequence

# BEWARE OF WHAT YOU IMPORT HERE. IMPORTING ANYTHING SUCH AS type OR macro
# WILL CAUSE CYCLIC ERROR, AS ALMOST EVERY MODULE IN pydsl RELIES ON compiler
# MODULE!
#
# Classes such as types and CallMacro have a level of indirection through
# Protocols and duck-typed calling conventions so that they can be used without
# being imported explicitly.
# E.g. CalMacro implements the CompileTimeCallable protocol.
# E.g. Addition in Int and Float uses the op_add magic function.
from pydsl.protocols import (
    CompileTimeClassSliceable,
    CompileTimeIterable,
    CompileTimeSliceable,
    CompileTimeTestable,
    HandlesFor,
    HandlesIf,
    Lowerable,
    Returnable,
    SubtreeOut,
    ToMLIRBase,
    handle_CompileTimeCallable,
    lower_single,
)
from pydsl.scope import ScopeStack
from pydsl.type import Index, Number, iscompiled
from pydsl.type import Slice as DSlice
from pydsl.type import Tuple as DTuple


# FIXME: turn these into proper passes that return a dict
def generate_parent(root):
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            child.parent = node


# FIXME: turn these into proper passes that return a dict
def generate_next_line(root):
    for node in ast.walk(root):
        if hasattr(node, "body") and isinstance(body := node.body, Iterable):
            for i, child in enumerate(body):
                child.next_line = body[i + 1] if (i + 1) < len(body) else None


@dataclasses.dataclass
class Source:
    src_str: str
    src_ast: ast.AST
    embeded: bool
    filepath: typing.Optional[Path] = None
    embedee_filepath: typing.Optional[Path] = None

    def init_embeded(src_str: str, filepath: Path, src_ast=None) -> "Source":
        src_ast = src_ast if src_ast is not None else ast.parse(src_str)
        return Source(
            src_str,
            src_ast,
            embeded=True,
            filepath=None,
            embedee_filepath=filepath,
        )

    def init_file(src_str: str, filepath: Path, src_ast=None) -> "Source":
        src_ast = src_ast if src_ast is not None else ast.parse(src_str)
        return Source(
            src_str,
            src_ast,
            embeded=False,
            filepath=filepath,
            embedee_filepath=None,
        )

    @property
    def path(self) -> Path:
        if (not self.embeded) and (self.filepath is not None):
            return self.filepath

        if (self.embeded) and (self.embedee_filepath is not None):
            return self.embedee_filepath

        raise AssertionError("Source appears neither embeded nor non-embeded")


class CompilationError(Exception):
    def __init__(
        self,
        msg: str = "compilation failed",
        exception: Exception = None,
        node: ast.AST = None,
        src: Source = None,
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
    def node(self: typing.Self):
        return self._node

    @node.setter
    def node(self: typing.Self, value: ast.AST) -> None:
        self._node = value
        self.update_note()

    _src: typing.Optional[Source] = None

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, value: Source) -> None:
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

        if self.src is not None:
            file_descriptor = f"{self.src.path}"
            if self.src.embeded:
                file_descriptor = f"<embed in {file_descriptor}>"
        else:
            file_descriptor = "<unknown>"

        if hasattr(self.node, "lineno"):
            line_descriptor = str(self.node.lineno)
        else:
            line_descriptor = "<unknown>"

        line_display = []

        locator = f'File "{file_descriptor}", line {line_descriptor}'

        if all([
            self.node is not None,
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
            lines = self.src.src_str.splitlines()[
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

        joined_line_display = "\n".join(line_display)
        return f"{locator}\n{joined_line_display}\n{error_info}"


DIRECTIVE_ESCAPE = "@"


class ToMLIR(ToMLIRBase):
    mlir = None
    scope_stack: ScopeStack = None
    interceptor_stack = []
    context_stack = []
    catch_comp_error: bool = True
    module: mlirir.Module = None
    triton_funcs: dict[
        str, dict[tuple[str, ...], func.FuncOp]
    ] = {}  # stores all triton functions
    # which have been added during compilation and their signatures
    """
    The last chain of attributes evaluated. This needs to be stored for cases
    where it is needed by method calls that require self or cls.
    """

    # Results are cached because visiting a node multiple times can result in
    # MLIR programs being generated multiple times
    # ast.NodeVisitor cannot modify the tree it is visiting, so cache will
    # never be outdated
    @cache
    def visit(self, node: ast.AST | Lowerable) -> SubtreeOut:
        try:
            match node:
                case ast.AST():
                    visitation = super().visit(node)
                case _ if iscompiled(node):
                    return node
                case _:
                    raise TypeError(f"{type(node).__name__} cannot be visited")

            # compose all interceptors starting from bottom of the stack and
            # apply to visitation
            return reduce(
                lambda x, f: f(x), self.interceptor_stack, visitation
            )

        except Exception as e:
            if not self.catch_comp_error:
                raise

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

    def __init__(self, f_locals, catch_comp_error=True) -> None:
        super(ToMLIR, self).__init__()
        # f_locals is a dictionary of local variables at the time the function
        # being compiled is defined
        self.f_locals = f_locals
        self.catch_comp_error = catch_comp_error
        self.setup()

    def setup(self) -> None:
        self.module = None
        self.scope_stack = ScopeStack(self.f_locals)
        self.triton_funcs = {}

    def visit_Slice(self, node: ast.Slice):
        lo = None if node.lower is None else Index(self.visit(node.lower))
        hi = None if node.upper is None else Index(self.visit(node.upper))
        step = None if node.step is None else Index(self.visit(node.step))
        return DSlice(lo, hi, step)

    def visit_Tuple(self, node: ast.Tuple) -> SubtreeOut:
        return DTuple.from_values(
            self, *[self.visit(entry) for entry in node.elts]
        )

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
                    self, value, prefix_args=(operand,)
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

    def visit_Expression(self, node: ast.Expression) -> SubtreeOut:
        # ast.Expression is output by ast.parse(mode="eval")
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> SubtreeOut:
        # TODO: Constant may not always be Number. It may also be e.g. string.
        # When other forms of constants are supported this needs to be updated.
        return Number(ast.literal_eval(node))

    def visit_Call(self, node: ast.Call) -> SubtreeOut:
        return handle_CompileTimeCallable(self, node)

    # for now, assume all operands are floating point numbers

    def visit_BinOp(self, node: ast.BinOp) -> SubtreeOut:
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self.eval_binop(left, node.op, right)

    # TODO: this does not yet support the right variants
    def eval_binop(
        self, left: SubtreeOut, op: ast.operator, right: SubtreeOut
    ) -> SubtreeOut:
        match op:
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
                return left.op_or(right)
            case ast.And():
                return left.op_and(right)
            case ast.Or():
                return left.op_or(right)
            # TODO: more ops can be added in the future
            case _:
                raise SyntaxError(
                    f"{type(op)} is not supported as a binary operator"
                )

    def visit_BoolOp(self, node: ast.BoolOp) -> SubtreeOut:
        values = [self.visit(i) for i in node.values]

        match node.op:
            case ast.And():

                def reducer(a, b):
                    return a.op_and(b)
            case ast.Or():

                def reducer(a, b):
                    return a.op_or(b)
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
                    return left.op_contains(right).op_not()
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
                    lambda l, r: l.op_and(r), partials[1:], partials[0]
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
            case ast.Invert():
                return operand.op_invert()
            case ast.Not():
                return operand.op_not()
            case _:
                raise SyntaxError(
                    f"{type(node.op)} is not supported as an unary operator"
                )

    def visit_Name(self, node: ast.Name) -> SubtreeOut:
        return self.scope_stack.resolve_name(node.id)

    def visit_Return(self, node: ast.Return) -> SubtreeOut:
        try:
            maybe_returnable: Any = self.scope_stack.current_scope_node()
        except AssertionError as e:
            raise SyntaxError(
                "attempted to return while not inside of a function"
            ) from e

        if not isinstance(maybe_returnable, Returnable):
            raise SyntaxError(
                f"attempted to return while inside of a "
                f"{type(maybe_returnable).__name__} scope, which is not "
                f"Returnable"
            )

        returnable: Returnable = maybe_returnable
        return returnable.on_Return(self, node)

    def visit_Subscript(self, node: ast.Subscript) -> SubtreeOut:
        assert type(node.ctx) is not ast.Store, AssertionError(
            "subscript with a Store context shouldn't be visited!"
        )

        value = self.visit(node.value)
        slice = node.slice

        if isinstance(value, type) and issubclass(
            value, CompileTimeClassSliceable
        ):
            return value.on_class_getitem(self, slice)

        if (not isinstance(value, type)) and isinstance(
            value, CompileTimeSliceable
        ):
            return value.on_getitem(self, slice)

        raise TypeError(
            f"{value} does not implement CompileTimeSliceable nor "
            f"CompileTimeClassSliceable"
        )

    def visit_Assign(self, node: ast.Assign) -> SubtreeOut:
        value = self.visit(node.value)

        # More than one target indicates chained assignment, e.g. a = b = val
        # The assignments should be done left-to-right (a = val, then b = val)
        for target in node.targets:
            self.visit_assignment(target, value)

        return value

    def visit_assignment(
        self, target: ast.AST, value: SubtreeOut
    ) -> SubtreeOut:
        match target:
            case ast.Name():
                self.scope_stack.assign_name(target.id, value)
                return value

            case ast.Subscript():
                ltarget = self.visit(target.value)
                slice = target.slice

                if isinstance(ltarget, CompileTimeSliceable):
                    return ltarget.on_setitem(self, slice, value)
                raise TypeError(
                    f"{ltarget} does not implement CompileTimeSliceable"
                )

            case ast.Tuple():
                if not isinstance(value, CompileTimeIterable):
                    raise TypeError(
                        f"cannot unpack {value}, which is not a "
                        f"CompileTimeIterable"
                    )
                iterable = value.as_iterable(self)

                targets = target.elts

                if len(targets) < len(iterable):
                    raise ValueError(
                        f"too many values to unpack (expected {len(targets)}, "
                        f"got {len(iterable)})"
                    )

                if len(targets) > len(iterable):
                    raise ValueError(
                        f"not enough values to unpack (expected "
                        f"{len(targets)}, got {len(iterable)})"
                    )

                for t, i in zip(targets, iterable, strict=True):
                    self.visit_assignment(t, i)

                return value

            case _:
                raise ValueError(
                    f"assigning to {type(target).__name__} is currently not "
                    f"supported."
                )

    def visit_AugAssign(self, node: ast.AugAssign) -> SubtreeOut:
        # make separate load/store context
        # while ensuring the target is evaluated only once (handles side-effects safely)
        match node.target:
            case ast.Subscript(value=v, slice=s, ctx=_):
                val, slc = self.visit(v), self.visit(s)
                read_t = ast.Subscript(value=val, slice=slc, ctx=ast.Load())
                store_t = ast.Subscript(value=val, slice=slc, ctx=ast.Store())
            case ast.Name(id=i, ctx=_):
                read_t = ast.Name(id=i, ctx=ast.Load())
                store_t = ast.Name(id=i, ctx=ast.Store())
            case other:
                raise TypeError(
                    f"unsupported AugAssign target: {ast.dump(other)}"
                )

        # compute new value
        new_val = self.eval_binop(
            self.visit(read_t), node.op, self.visit(node.value)
        )
        # assign back to the visited target
        self.visit_assignment(store_t, new_val)
        return new_val

    def visit_AnnAssign(self, node: ast.AnnAssign) -> SubtreeOut:
        val = self.visit(node.value)
        ret_type = self.scope_stack.resolve_name(node.annotation.id)
        ret_val = ret_type(val)

        self.scope_stack.assign_name(node.target.id, ret_val)

        return ret_val

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

    def resolve_type_annotation(self, ann: typing.Optional[ast.AST]):
        """
        Convert typing objects into class objects.

        Examples with existing types:
        - UInt32 becomes the UInt32 class
        - MemRef[12, 5] becomes a dynamically generated MemRef class with
        shape (12, 5)
        - Tuple[F32, UInt32] becomes a dynamically generated Tuple class with
        elements F32, UInt32
        - None becomes None, as a special case representing void return
        """
        match ann:
            # When the constant None is used for type hinting, it refers to
            # a void return.
            # We will assume that an empty type hint also refers to void return
            case None | ast.Constant(value=None):
                return None

            # String type hints are treated the same way as regular type hints
            # in Python, except it doesn't make the static checker as angry
            case ast.Constant(value=str()):
                return self.scope_stack.resolve_name(ann.value)

            # Oddly, type-hinting using literals without Literal[]
            # are legal. Whether this makes type checkers happy is a different
            # story and isn't too much of a concern for PyDSL.
            case ast.Constant(value=value):
                return value

            # A single return type name
            case ast.Name(id=id):
                return self.scope_stack.resolve_name(id)

            # A subscripted type
            case ast.Subscript(value=ast.Name(id=id), slice=slice):
                origin = self.scope_stack.resolve_name(id)
                return origin.on_class_getitem(self, slice)

            case _:
                raise TypeError(
                    f"using {type(ann).__name__} for type hinting is not "
                    f"supported"
                )

    def visit_If(self: typing.Self, node: ast.If) -> SubtreeOut:
        # TODO: This is currently very barebone just to get things out of the
        # way. You need proper use analysis to improve if statements
        scope = self.scope_stack

        match node:
            # Case for when a HandlesIf is passed as the test
            case ast.If(test=ast.Call(func=(ast.Name(id=name)))) if (
                scope.resolve_as_protocol(name, HandlesIf, just_check=True)
            ):
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

                if isinstance(test, Number):
                    # we can do constant folding
                    if test.value:
                        for b in body:
                            self.visit(b)
                    else:
                        for b in orelse:
                            self.visit(b)

                    return

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

        if isinstance(test, Number):
            # we can do constant folding
            if test.value:
                return self.visit(node.body)
            else:
                return self.visit(node.orelse)

        body = self.visit(node.body)
        orelse = self.visit(node.orelse)

        if not isinstance(test, CompileTimeTestable):
            raise TypeError(
                f"if expression expected Bool or types supporting Bool, got "
                f"{test}"
            )

        if (body_t := type(body)) is not (orelse_t := type(orelse)):
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
        return Function.full_init(self, node)

    def visit_Module(self, node: ast.Module) -> SubtreeOut:
        module = Module()
        module.build_body(self, node)
        return module

    # TODO: This may not be the best place for this function, but it's similar
    # to the way that a Function is created
    # Maybe have a separate transform-seq-based visitor that inherits
    # the main ToMLIR visitor but with this method overwritten

    def visit_FunctionDef_as_transform_seq(
        self, node: ast.FunctionDef
    ) -> SubtreeOut:
        return TransformSequence.full_init(self, node)

    # TODO: consider also having a WithMacro, which supports With-As cluase
    def visit_With(self, node: ast.With) -> SubtreeOut:
        for item in node.items:
            if type(call := item.context_expr) is ast.Call:
                return handle_CompileTimeCallable(
                    self, call, prefix_args=(node.body,)
                )
            raise NotImplementedError(
                "with statement currently only allows calls as contexts"
            )

    def customize_context(self, ctx: Context):
        """
        This method is guaranteed to be called right after the MLIR Context is
        created. Subclasses can override this method to register their own
        dialects.
        """
        pass

    @contextmanager
    def compile(
        self,
        node,
        transform_seq: ast.AST | None = None,
        interceptor: Callable[[SubtreeOut], SubtreeOut] | None = None,
    ) -> Iterable["Module"]:
        # create additional properties in AST nodes that we will need during
        # compilation
        generate_parent(
            node
            # FIXME: this shouldn't be done as it's an illegal field and it mutates the tree (we are not allowed to mutate)
        )
        generate_next_line(
            node
            # FIXME: this shouldn't be done as it's an illegal field and it mutates the tree (we are not allowed to mutate)
        )

        self.setup()

        # pre-initiate symbol objects

        with Context() as ctx, Location.unknown():
            self.customize_context(ctx)

            if interceptor is not None:
                module: Module = self.visit_with_interception(
                    node, interceptor
                )
            else:
                module: Module = self.visit(node)

            with module.insert():
                if transform_seq is not None:
                    module.set_unit_attr("transform.with_named_sequence")
                    # TODO: this is hacky, should create a dedicated transform
                    # seq visitor
                    self.visit_FunctionDef_as_transform_seq(
                        transform_seq.body[0]
                    )

            module.verify()

            yield module


MLIRNode: typing.TypeAlias = (
    mlirir.Module | mlirir.Operation | mlirir.Region | mlirir.Attribute
)


class Dialect:
    """
    A class that represents a particular dialect in MLIR.

    Note that equality is sensitive to the full path of the dialect. This means
    `transform` != `transform.validator` and that
    `validator` != `transform.validator`.
    """

    # Instantiation is cached based on initialization arguments.
    # So for example, this allows
    # Dialect.from_name("arith") == Dialect.from_name("arith")
    # despite being initialized twice.
    # Whereas not doing caching would result in False as they would be
    # two different objects both with name == "arith".
    @cache
    def __new__(cls, *args, **kwargs):
        return super(Dialect, cls).__new__(cls)

    _name: tuple[str]

    @property
    def name(self) -> str:
        return ".".join(self._name)

    def __init__(self: typing.Self, name: tuple[str]) -> None:
        self._name = name

    def __eq__(self: typing.Self, other: Any) -> bool:
        if isinstance(other, Dialect):
            return self._name == other._name
        else:
            return False

    def __hash__(self: typing.Self) -> int:
        return hash(self._name)

    def __str__(self: typing.Self) -> str:
        return self.name

    @staticmethod
    def from_operation(op: mlirir.Operation) -> "Dialect":
        """
        Gets the dialect of the operation, based on an operation object.

        This ultimately makes use of the operation's name.
        """
        return Dialect.from_operation_name(op.name)

    @staticmethod
    def from_operation_name(name: str) -> "Dialect":
        """
        Gets the dialect of the operation, based on an operation name.
        """
        levels: tuple[str] = tuple(name.split("."))
        if len(levels) < 2:
            raise AssertionError(
                f"encountered an operation name {name} with no `.`"
            )

        return Dialect(levels[:-1])

    @staticmethod
    def from_name(name: str) -> "Dialect":
        """
        Gets the dialect based on the dialect's name.
        """
        levels: tuple[str] = tuple(name.split("."))
        return Dialect(levels)


class MLIRVisitor:
    def visit(self, node: MLIRNode):
        match node:
            case mlirir.Module():
                self.visit_Module(node)
            case mlirir.Operation():
                self.visit_Operation(node)
            case mlirir.OpView():
                self.visit_Operation(node.operation)
            case mlirir.Region():
                self.visit_Region(node)
            case mlirir.Block():
                self.visit_Block(node)
            case mlirir.Attribute():
                self.visit_Attribute(node)

    def visit_Module(self, node: mlirir.Module):
        self.visit(node.operation)

    def visit_Operation(self, node: mlirir.Operation):
        for r in node.regions:
            self.visit(r)

        for a in node.attributes:
            self.visit(a)

    def visit_Region(self, node: mlirir.Region):
        for b in node.blocks:
            self.visit(b)

    def visit_Block(self, node: mlirir.Block):
        for o in node.operations:
            self.visit(o)

    def visit_Attribute(self, node: mlirir.Attribute):
        pass


class DialectsUsed(MLIRVisitor):
    """
    A visitor that analyzes an MLIR structure and returns every dialect used by
    the structure.
    """

    dialects: set[Dialect]

    def __init__(self: typing.Self):
        self.dialects = set()

    def visit_Operation(self: typing.Self, node: mlirir.Operation):
        self.dialects.add(Dialect.from_operation(node))
        return super().visit_Operation(node)

    @staticmethod
    def analyze(module: mlirir.Module) -> set[Dialect]:
        """
        Get a set of all dialects used in `module`.
        """
        du = DialectsUsed()
        du.visit(module)
        return du.dialects


class Module:
    module: mlirir.Module
    _functions: set[Function]

    def __init__(self):
        self._functions = set()

    @property
    def functions(self) -> set[Function]:
        return self._functions

    @property
    def mlir(self) -> str:
        return str(self.module)

    @property
    def dialects(self) -> set[Dialect]:
        return DialectsUsed.analyze(self.module)

    def add_function(self, d: Function) -> None:
        self._functions.add(d)

    def lower(self):
        return self.module

    @classmethod
    def lower_class(cls):
        return mlirir.Module

    def set_attr(self, key, val) -> None:
        self.module.operation.attributes[key] = val

    def set_unit_attr(self, key) -> None:
        self.set_attr(key, UnitAttr.get())

    def verify(self) -> typing.Never | typing.Literal[True]:
        return self.module.operation.verify()

    def build_body(self, visitor: ToMLIR, node: ast.Module) -> Iterable[None]:
        self.module = mlirir.Module.create()
        visitor.module = self.module

        with self.insert():
            # Full-module initialization
            for n in node.body:
                match n:
                    # Functions are exception in that their bodies are
                    # not evaluated before all of their signatures are
                    # defined in the module
                    case ast.FunctionDef(name=name):
                        f = Function.node_to_header(visitor, n)
                        visitor.scope_stack.assign_name(name, f)
                        self.add_function(f)

                    # Everything else is evaluated as normal
                    case _:
                        visitor.visit(n)

            # Member body initialization
            for n in node.body:
                match n:
                    case ast.FunctionDef(name=name):
                        visitor.visit(n)

    @contextmanager
    def insert(self) -> Iterable[None]:
        with InsertionPoint(self.module.body):
            yield
