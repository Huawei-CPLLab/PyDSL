from __future__ import annotations
import builtins
import ast
from abc import abstractmethod
from collections.abc import Iterable
from enum import Enum, auto
from typing import TYPE_CHECKING, Any
import typing

import mlir.ir as mlir
from mlir.dialects import affine, arith, func
from mlir.ir import (
    AffineDimExpr,
    AffineExpr,
    AffineMap,
    AffineSymbolExpr,
    IndexType,
    InsertionPoint,
    IntegerSet,
)

from pydsl.macro import ArgRep, CallMacro, IteratorMacro
from pydsl.protocols import (
    Lowerable,
    SubtreeOut,
    handle_CompileTimeCallable,
    lower_single,
)
from pydsl.type import Index, lower_flatten
from pydsl.protocols import ToMLIRBase

if TYPE_CHECKING:
    # This is for imports for type hinting purposes only
    from pydsl.compiler import ScopeStack


class AffineContext:
    """
    A dummy flag class for indicating whether the visitor is inside of an
    affine context.

    In this context, types may choose to exchange regular operators with their
    affine variants. For example, MemRef can change all LoadOp to AffineLoadOp.
    """


class affine_range(IteratorMacro):
    def on_For(visitor: ToMLIRBase, node: ast.For) -> affine.AffineForOp:
        iter_arg = node.target
        iterator = node.iter

        lb = None
        step = 1

        args = iterator.args

        match len(args):
            case 1:
                ub = args[0]
            case 2:
                lb, ub = args
            case 3:
                lb, ub, step = args[0], args[1], args[2].value
            case _:
                raise ValueError(
                    f"range expects up to 3 arguments, got {len(args)}"
                )

        # TODO: when other max/min operators are implemented in pydsl, they
        # should be included here!
        if type(lb) is ast.Call:
            match visitor.scope_stack.resolve_name(lb.func.id):
                case builtins.max:
                    lb = lb.args
                case builtins.min:
                    raise SyntaxError(
                        "top-level call of a lower-bound cannot be "
                        "builtins.min"
                    )

        if type(ub) is ast.Call:
            match visitor.scope_stack.resolve_name(ub.func.id):
                case builtins.min:
                    ub = ub.args
                case builtins.max:
                    raise SyntaxError(
                        "top-level call of an upper-bound cannot be "
                        "builtins.max"
                    )

        ub_am_expr = AffineMapExprWalk.compile(
            ub, visitor.scope_stack
        ).lowered()

        lb_am_expr = (
            AffineMapExpr.constant(0)
            if lb is None
            else AffineMapExprWalk.compile(lb, visitor.scope_stack).lowered()
        )

        affine_for_op = affine.AffineForOp(
            lb_am_expr.map,
            ub_am_expr.map,
            step,
            iter_args=[],  # TODO: no iter args for now, maybe a feature later
            lower_bound_operands=[*lb_am_expr.dims, *lb_am_expr.syms],
            upper_bound_operands=[*ub_am_expr.dims, *ub_am_expr.syms],
        )

        with (
            InsertionPoint(affine_for_op.body),
            visitor.new_context(AffineContext),
        ):
            visitor.scope_stack.assign_name(
                iter_arg.id, Index(affine_for_op.induction_variable)
            )
            for n in node.body:
                visitor.visit(n)

            # nothing will be yielded for now
            affine.AffineYieldOp([])

        return affine_for_op

    def on_ListComp(node: ast.ListComp):
        raise NotImplementedError(
            "affine_range cannot be used for list comprehension for now"
        )


class integer_set:
    def on_If(visitor: "ToMLIRBase", node: ast.If) -> SubtreeOut:
        # TODO: This is currently very barebone just to get things out of the
        # way. You need proper use analysis to improve if statements
        match node:
            case ast.If(test=ast.Call(args=args), body=body, orelse=orelse):
                has_else = len(orelse) != 0

                if len(args) != 1:
                    raise TypeError(
                        f"integer_set() takes 1 positional argument but "
                        f"{len(args)} were given"
                    )

                if not (
                    isinstance(args[0], ast.Compare)
                    or isinstance(args[0], ast.BoolOp)
                ):
                    raise TypeError(
                        "integer_set() expects affine comparison as its only "
                        "argument"
                    )

                iset = IntegerSetExprWalk.compile(args[0], visitor.scope_stack)

                if_exp = iset.to_affine_if([], has_else)

                with InsertionPoint(if_exp.then_block):
                    for b in body:
                        visitor.visit(b)
                    affine.AffineYieldOp([])

                if has_else:
                    with InsertionPoint(if_exp.else_block):
                        for b in orelse:
                            visitor.visit(b)
                        affine.AffineYieldOp([])

                return if_exp

            case _:
                raise AssertionError("unexpected improper if structure")


class AffineMapExpr:
    """
    An AffineMap with all dimensions and symbols filled
    """

    map: AffineMap
    dims: list[Any]
    syms: list[Any]

    def __init__(
        self, map: AffineMap, dims: list[Any], syms: list[Any]
    ) -> None:
        self.map = map
        self.dims = dims
        self.syms = syms

    @staticmethod
    def constant(x: int) -> "AffineMapExpr":
        return AffineMapExpr(
            AffineMap.get(0, 0, [AffineExpr.get_constant(x)]),
            [],  # no dimension
            [],  # no symbol
        )

    def lowered(self) -> AffineMapExpr:
        # lower all the dims and syms
        return AffineMapExpr(
            self.map, lower_flatten(self.dims), lower_flatten(self.syms)
        )


class IntegerSetExpr:
    """
    An IntegerSet with all dimensions and symbols filled
    """

    set: IntegerSet
    dims: list[Any]
    syms: list[Any]

    def __init__(
        self, set: IntegerSet, dims: list[Any], syms: list[Any]
    ) -> None:
        self.set = set
        self.dims = dims
        self.syms = syms

    def lowered(self) -> IntegerSetExpr:
        # operands to IntegerSet must be Index
        try:
            dims = [Index(d) for d in self.dims]
            syms = [Index(s) for s in self.syms]
        except TypeError as e:
            raise TypeError(
                "all dimensions and symbols of integer set expression must be "
                "Index or castable to Index"
            ) from e

        return IntegerSetExpr(
            self.set, lower_flatten(dims), lower_flatten(syms)
        )

    def to_affine_if(self, results, has_else=False) -> affine.AffineIfOp:
        lowered = self.lowered()
        return affine.AffineIfOp(
            lowered.set,
            results,
            cond_operands=[*lowered.dims, *lowered.syms],
            hasElse=has_else,
        )


class affine_map(CallMacro):
    def argreps():
        return []

    def argsreps() -> ArgRep:
        return ArgRep.UNCOMPILED

    def _on_Call(visitor: ToMLIRBase, args: list[Any]) -> AffineMapExpr:
        return AffineMapExprWalk.compile(args, visitor.scope_stack)


class AffineCallMacro(CallMacro):
    """
    A CallMacro that can only be called in an affine context
    """

    @classmethod
    def _on_Call(cls, visitor: ast.NodeVisitor, args: list[Any]) -> Any:
        if not isinstance(visitor, AffineMapExprWalk):
            raise TypeError(
                f"{cls.__name__} is an AffineCallMacro but was called outside "
                f"of an affine map expression"
            )

        return cls._affine_on_Call(visitor, args)

    @abstractmethod
    def argreps() -> list[ArgRep]:
        pass

    @abstractmethod
    def _affine_on_Call(visitor: AffineMapExprWalk, args: list[Any]) -> Any:
        pass


class dimension(AffineCallMacro):
    """
    A dimension in an affine expression
    """

    def argreps():
        return [ArgRep.UNCOMPILED]

    def _affine_on_Call(visitor: AffineMapExprWalk, args: list[Any]) -> Any:
        arg = args[0]
        match arg:
            case ast.Constant():
                # Having a constant in a dim() has a very specific connotation.
                # We cannot visit the constant as-is.
                #
                # We need to create an ad-hoc ConstantOp to be used as a dim.
                # AffineDimExpr does not accept a AffineConstantExpr.
                if type(arg.value) is not int:
                    raise TypeError(
                        f"dimension expected integer, got {arg.value}"
                    )

                return visitor.add_dim(
                    arith.ConstantOp(IndexType.get(), arg.value)
                )

            case ast.Name():
                return visitor.add_dim(visitor.visit_without_inference(arg))

            case _:
                raise TypeError(f"{type(arg)} type cannot be used in dim")


class symbol(AffineCallMacro):
    """
    A symbol in an affine expression
    """

    def argreps():
        return [ArgRep.UNCOMPILED]

    def _affine_on_Call(visitor: AffineMapExprWalk, args: list[Any]) -> Any:
        arg = args[0]
        match arg:
            case ast.Constant():
                """
                Having a constant in a sym() has a very specific connotation.
                We cannot visit the constant as-is.

                We need to create an ad-hoc ConstantOp to be passed into
                AffineSymbolExpr. AffineSymbolExpr does not accept a
                AffineConstantExpr.
                """
                if type(arg.value) is not int:
                    raise TypeError(
                        f"symbol expected integer, got {arg.value}"
                    )

                return visitor.add_sym(
                    arith.ConstantOp(IndexType.get(), arg.value)
                )

            case ast.Name():
                return visitor.add_sym(visitor.visit_without_inference(arg))

            case _:
                raise TypeError(f"{type(arg)} type cannot be used in sym")


class floordivide(AffineCallMacro):
    def argreps():
        return [ArgRep.UNCOMPILED, ArgRep.UNCOMPILED]

    def _affine_on_Call(visitor: AffineMapExprWalk, args: list[Any]) -> Any:
        left = visitor.visit(args[0])
        right = visitor.visit(args[1])
        return AffineExpr.get_floor_div(left, right)


class ceildivide(AffineCallMacro):
    def argreps():
        return [ArgRep.UNCOMPILED, ArgRep.UNCOMPILED]

    def _affine_on_Call(visitor: AffineMapExprWalk, args: list[Any]) -> Any:
        left = visitor.visit(args[0])
        right = visitor.visit(args[1])
        return AffineExpr.get_ceil_div(left, right)


AffineId: typing.TypeAlias = type[symbol | dimension]


def infer_affine_id(v: SubtreeOut) -> set[AffineId]:
    """
    Attempt to infer what affine identifier (e.g. dimension, symbol) the
    value belongs to

    This function attempts to implement the definitions from MLIR's
    specification, but errs to be exclusive when implementation can't be done:
    https://mlir.llvm.org/docs/Dialects/Affine/#restrictions-on-dimensions-and-symbols

    If the value's identifier cannot be inferred, an empty set is returned
    """

    valid = set()

    if isinstance(v, Lowerable):
        v = lower_single(v)

    match v:
        # For both symbol and dimension:

        # 1. a region argument for an op with trait AffineScope (eg. FuncOp)
        # NOTE: we only do FuncOp because Python binding has no access to trait
        case mlir.BlockArgument(owner=mlir.Block(owner=func.FuncOp())):
            valid.update((
                symbol,
                dimension,
            ))

        # 2. a value defined at the top level of an AffineScope op
        # NOTE: again, only FuncOp
        case mlir.Value(
            owner=mlir.Operation(parent=mlir.Operation(opview=func.FuncOp()))
        ):
            valid.update((
                symbol,
                dimension,
            ))

        # 3. a value that dominates the AffineScope op enclosing the value’s
        #    use
        # TODO: no clue what this means

        # 4. the result of a constant operation
        # NOTE: we will only include arith.ConstantOp
        case mlir.Value(owner=mlir.Operation(opview=arith.ConstantOp())):
            valid.update((
                symbol,
                dimension,
            ))

        # 5. the result of an affine.apply operation that recursively takes as
        #    arguments any valid symbolic identifiers
        # TODO: this is too hard to implement for now. We don't use
        # affine.apply anyways

        # 6. the result of a dim operation on either a memref that is an
        #    argument to a AffineScope op or a memref where the corresponding
        #    dimension is either static or a dynamic one in turn bound to a
        #    valid symbol
        # TODO: this is too hard to implement for now. We don't use memref.dim
        # anyways

        # For only dimensions:

        # 1. induction variables of enclosing affine.for and affine.parallel
        #    operations
        case mlir.BlockArgument(
            owner=mlir.Block(
                owner=(affine.AffineForOp() | affine.AffineParallelOp())
            )
        ):
            valid.update((dimension,))

    # 2. the result of an affine.apply operation (which recursively may use
    #    other dimensions and symbols)
    # TODO: this is too hard to implement for now.

    return valid


class AffineExprWalk(ast.NodeVisitor):
    """
    A helper NodeVisitor subclass that walks a Python subtree assumed to be
    entirely used to be converted into an affine_map.
    """

    no_inference: bool = False
    scope_stack: ScopeStack
    dims: list[Any]
    syms: list[Any]

    def __init__(self, scope_stack: ScopeStack) -> None:
        self.scope_stack = scope_stack
        self.dims = []
        self.syms = []

    def visit_without_inference(self, node: ast.AST):
        self.no_inference = True
        ret = self.visit(node)
        self.no_inference = False
        return ret

    def add_dim(self, x: Any) -> AffineDimExpr:
        if x in self.dims:
            return AffineDimExpr.get(self.dims.index(x))

        self.dims.append(x)
        return AffineDimExpr.get(len(self.dims) - 1)

    def add_sym(self, x: Any) -> AffineSymbolExpr:
        if x in self.syms:
            return AffineSymbolExpr.get(self.syms.index(x))

        self.syms.append(x)
        return AffineSymbolExpr.get(len(self.syms) - 1)

    def visit_Name(self, node: ast.Name) -> Any:
        val = self.scope_stack.resolve_name(node.id)

        if self.no_inference:
            return val

        inferences = infer_affine_id(val)

        if symbol in inferences:
            return self.add_sym(val)

        if dimension in inferences:
            return self.add_dim(val)

        raise SyntaxError(
            f"cannot infer the affine identifier of {node.id}. Please "
            f"indicate whether it's a symbol or dimension manually"
        )

    def visit_Constant(self, node: ast.Constant) -> mlir.AffineConstantExpr:
        return AffineExpr.get_constant(node.value)

    def visit_BinOp(
        self, node: ast.BinOp
    ) -> (
        mlir.AffineAddExpr
        | mlir.AffineMulExpr
        | mlir.AffineFloorDivExpr
        | mlir.AffineModExpr
    ):
        left = self.visit(node.left)
        right = self.visit(node.right)

        match type(node.op):
            case ast.Add:
                return AffineExpr.get_add(left, right)
            case ast.Sub:
                return AffineExpr.get_add(left, AffineExpr.get_mul(-1, right))
            case ast.Mult:
                return AffineExpr.get_mul(left, right)
            case ast.FloorDiv:
                return AffineExpr.get_floor_div(left, right)
            case ast.Mod:
                return AffineExpr.get_mod(left, right)
            case _:
                raise ValueError(
                    f"Ln {node.lineno}: {type(node.op)} is not a valid affine "
                    f"binary operator."
                )

    def visit_Call(self, node: ast.Call) -> Any:
        return handle_CompileTimeCallable(self, node)


class AffineMapExprWalk(AffineExprWalk):
    @staticmethod
    def compile(
        node: ast.AST | Iterable[ast.AST], scope_stack: ScopeStack
    ) -> AffineMapExpr:
        """
        Returns an AffineMapExpr equivalent to the AST.

        If the expression itself is already evaluated to an AffineMapExpr, it
        is returned as-is. This makes the affine map compilation process an
        idempotent operation.
        """

        if type(node) is ast.List or type(node) is ast.Tuple:
            # convert ast List into a proper Python list
            node = node.elts

        if not isinstance(node, Iterable):
            # if only a single element is given, make it iterable
            node = [node]

        walk = AffineMapExprWalk(scope_stack)
        exprs = [walk.visit(elt) for elt in node]

        # if the expression is already an affine map, return as-is
        if len(exprs) == 1 and isinstance(exprs[0], AffineMapExpr):
            return exprs[0]

        amap = AffineMap.get(len(walk.dims), len(walk.syms), exprs)
        return AffineMapExpr(amap, walk.dims, walk.syms)


class IntegerSetConstraint(Enum):
    EQUAL = auto()
    GEQUAL = auto()

    def lower(self):
        return self is IntegerSetConstraint.EQUAL


class IntegerSetExprWalk(AffineExprWalk):
    @staticmethod
    def compile(node: ast.AST, scope_stack: ScopeStack) -> IntegerSetExpr:
        """
        Returns an IntegerSetExpr equivalent to the AST.

        If the expression itself is already evaluated to an integer set, it is
        returned as-is. This makes the integer set compilation process an
        idempotent operation.
        """

        walk = IntegerSetExprWalk(scope_stack)
        exprs = walk.visit(node)

        # if the expression is already an integer set, return as-is
        if len(exprs) == 1 and isinstance(exprs[0], IntegerSetExpr):
            return exprs[0]

        iset = IntegerSet.get(
            len(walk.dims),
            len(walk.syms),
            [e[0] for e in exprs],
            [e[1].lower() for e in exprs],
        )
        return IntegerSetExpr(iset, walk.dims, walk.syms)

    def visit_BoolOp(self, node: ast.BoolOp) -> list[ast.AST]:
        def always_list(a: Any) -> list[Any]:
            return list(a) if isinstance(a, Iterable) else [a]

        match node.op:
            case ast.And():
                # flatten the list of affine expressions
                return sum(
                    [always_list(self.visit(n)) for n in node.values], []
                )
            case _:
                raise SyntaxError(
                    f"{node.op} is not supported in an affine expression"
                )

    def visit_Compare(
        self, node: ast.Compare
    ) -> list[tuple[AffineExpr, IntegerSetConstraint]]:
        # the reduce operator in the list convolution
        def op_eval(
            opt: ast.AST, left: AffineExpr, right: AffineExpr
        ) -> list[tuple[AffineExpr, IntegerSetConstraint]]:
            match opt:
                case ast.Lt():
                    # forall a, b in Int . a < b <=> b - a - 1 >= 0
                    return [(right - left - 1, IntegerSetConstraint.GEQUAL)]
                case ast.LtE():
                    # forall a, b in Int . a <= b <=> b - a >= 0
                    return [(right - left, IntegerSetConstraint.GEQUAL)]
                case ast.Eq():
                    # forall a, b in Int . a == b <=> a - b == 0
                    return [(left - right, IntegerSetConstraint.EQUAL)]
                # ast.NotEq() is not supported
                case ast.Gt():
                    # forall a, b in Int . a > b <=> a - b - 1 >= 0
                    return [(left - right - 1, IntegerSetConstraint.GEQUAL)]
                case ast.GtE():
                    # forall a, b in Int . a >= b <=> a - b >= 0
                    return [(left - right, IntegerSetConstraint.GEQUAL)]
                case _:
                    raise NotImplementedError(
                        f"the operator {opt} is not allowed in affine "
                        f"expression"
                    )

        match node:
            case ast.Compare(left=first, ops=operators, comparators=rest):
                operands = [self.visit(n) for n in [first, *rest]]
                # convolve the operands with width 2, stride 1,
                # then flatten the list of lists into a flat list with sum
                return sum(
                    [
                        op_eval(op, operands[i], operands[i + 1])
                        for i, op in enumerate(operators)
                    ],
                    [],
                )
            case _:
                raise NotImplementedError(
                    "this format of comparison is not implemented"
                )
