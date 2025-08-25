from typing import Any, Generic, Optional, TypeVar
import ast

from mlir.dialects import scf
from mlir.ir import InsertionPoint
import mlir.ir as mlirir

from pydsl.macro import Compiled, IteratorMacro, Uncompiled
from pydsl.protocols import (
    CompileTimeTestable,
    ToMLIRBase,
)
from pydsl.scope import BindingContext, Scope
from pydsl.type import Index, lower_single, Poison


# Generic type variable for node
ASTN = TypeVar("ASTN", bound=ast.AST)
MLIRN = TypeVar("MLIRN", bound=mlirir.OpView)


class SCFOp(Generic[ASTN, MLIRN]):
    """
    Base class for constructing SCF ops (IfOp and ForOp).
    Subclasses implement methods:
    - build_op(op_type, op_args, dummy)
    - build_regions(op)
    - excluded_vars() -> tuple
    """

    def __init__(self, visitor, node):
        self.visitor = visitor
        self.node = node

    def __new__(cls, visitor: ToMLIRBase, node: ASTN):
        self = super().__new__(cls)
        self.__init__(visitor, node)

        scope = visitor.scope_stack
        current_scope = scope.stack[-1]
        bound_prior = current_scope.f_locals

        # 2. Build dummy op
        dummy_op = self.build_op(op_type=[], op_args=[], dummy=True)

        # 3. Build region bindings
        bound_dicts = self.build_regions(dummy_op)

        # 4. Analyze yields
        yield_vars_cand: list[str] = [
            var
            for d in bound_dicts
            for var, val in d.items()
            if val != bound_prior.get(var) and var not in self.excluded_vars()
        ]

        op_type: list[type] = []
        yield_vars = []
        yield_vals_per_region: list[list[Any]] = [[] for _ in bound_dicts]
        poisons: dict[str, Poison] = dict()

        for var in yield_vars_cand:
            dicts = [bound_prior, *bound_dicts]
            types = [type(d[var]) if var in d else None for d in dicts]

            rst_type = cls.verify_types(var, types, node)

            if isinstance(rst_type, Poison) or rst_type is Poison:
                poisons[var] = rst_type
            else:
                yield_vars.append(var)

                types = [x for x in types if x is not None]

                op_type.append(rst_type)
                for i, d in enumerate(bound_dicts):
                    val = d.get(var) or bound_prior[var]
                    yield_vals_per_region[i].append(val)

        # 5. Build final op
        new_op = self.build_op(
            op_type,
            [bound_prior.get(v) for v in yield_vars],
            dummy=False,
        )

        # 6. Move blocks
        for dummy_region, new_region, vals in zip(
            dummy_op.regions, new_op.regions, yield_vals_per_region
        ):
            cls.move_block(
                dummy_region.blocks[0],
                new_region.blocks[0],
                [lower_single(v) for v in vals],
            )

        # 7. Replace operands with block arguments if applicable
        cls._replace_operands_with_block_args(new_op, yield_vars, bound_prior)

        # 8. Assign results back
        for var, typ, val in zip(
            yield_vars, op_type, new_op.results, strict=True
        ):
            scope.assign_name(var, typ(val))

        # 9. assign poison
        for var, poison in poisons.items():
            scope.assign_name(var, poison)

        dummy_op.erase()
        return new_op

    def build_block(
        self,
        scope: Scope,
        block_body: list[ast.stmt],
        dest_block: mlirir.Block,
        assign_targets: dict[str, mlirir.Value] | None = None,
    ):
        """
        Compile an AST block in a new binding context and return the updated local
        bindings.

        Parameters
        ----------
        scope : Scope
            The current scope.
        block_body : list[ast.AST]
            The list of AST nodes (statements) to compile.
        dest_block : mlir.ir.Block
            The MLIR block to insert ops into.
        assign_targets : dict[str, mlirir.Value] | None
            Optional dict of names to assign before executing the block (e.g., induction variable).

        Returns
        -------
        dict[str, mlirir.Value]
            new bindings after the block.
        """

        binding_prior = scope.f_locals

        with (
            scope.new_binding_context(BindingContext(self.node, dict())),
            InsertionPoint(dest_block),
        ):
            # Assign any pre-defined targets (e.g., loop induction variable)
            if assign_targets:
                for name, val in assign_targets.items():
                    scope.assign_name(name, val)

            # Visit all nodes in the block
            for stmt in block_body:
                self.visitor.visit(stmt)

            # Yield empty list for the first pass
            scf.YieldOp([])

            # Return new bindings created by the block
            return {
                var: val
                for var, val in scope.f_locals.items()
                if val != binding_prior.get(var)
            }

    @staticmethod
    def move_block(
        src_block: mlirir.Block,
        dest_block: mlirir.Block,
        yield_vals: list[mlirir.Value],
    ):
        with InsertionPoint(dest_block):
            clone_map = {}
            for r1, r2 in zip(src_block.arguments, dest_block.arguments):
                r1.replace_all_uses_with(r2)
                clone_map[r1] = r2
            for op in src_block.operations:
                if not isinstance(op, scf.YieldOp):
                    cloned = op.clone()
                    for r1, r2 in zip(op.results, cloned.results, strict=True):
                        r1.replace_all_uses_with(r2)
                        clone_map[r1] = r2
            scf.YieldOp([clone_map.get(v, v) for v in yield_vals])

    def build_op(
        self,
        op_type: Optional[list] = None,
        op_args: Optional[list] = None,
        dummy: bool = True,
    ) -> MLIRN:
        raise NotImplementedError

    def build_regions(self, op: MLIRN) -> tuple[dict[str, Any], ...]:
        raise NotImplementedError

    @staticmethod
    def verify_types(
        var: str, types: list[type | None], node: ASTN
    ) -> type | Poison:
        raise NotImplementedError

    def excluded_vars(self) -> tuple[str, ...]:
        return ()

    @staticmethod
    def replace_value_in_region(region, old_val, new_val):
        for block in region.blocks:
            for op in block.operations:
                for i, operand in enumerate(op.operands):
                    if operand == old_val:
                        op.operands[i] = new_val
                for nested_region in op.regions:
                    SCFOp.replace_value_in_region(
                        nested_region, old_val, new_val
                    )

    @staticmethod
    def _replace_operands_with_block_args(
        op: MLIRN, yield_vars: list[str], bound_prior: dict[str, Any]
    ):
        """Default: do nothing. Override in ops like ForOp that have iter_args as block args."""
        pass


class IfOp(SCFOp[ast.If, scf.IfOp]):
    def __init__(self, visitor, node):
        super().__init__(visitor, node)
        self.test = visitor.visit(node.test)
        if not isinstance(self.test, CompileTimeTestable):
            raise TypeError("if test must be bool-like")

    def build_op(
        self,
        op_type: Optional[list] = None,
        op_args: Optional[list] = None,
        dummy=True,
    ) -> scf.IfOp:
        if op_type is None:
            op_type = []
        vals = [] if dummy else [lower_single(t) for t in op_type]
        return scf.IfOp(lower_single(self.test.Bool()), vals, hasElse=True)

    def build_regions(
        self, op: scf.IfOp
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        bound_if = self.build_block(
            self.visitor.scope_stack.stack[-1],
            self.node.body,
            op.then_block,
        )
        bound_else = self.build_block(
            self.visitor.scope_stack.stack[-1],
            self.node.orelse,
            op.else_block,
        )
        return (bound_if, bound_else)

    @staticmethod
    def verify_types(
        var: str, types: list[type | None], node: ast.If
    ) -> type | Poison:
        type_prior, type_if, type_else = types

        # Check for branch conflict
        if type_if is not None and type_else is not None:
            if type_if != type_else:
                return Poison(
                    TypeError(
                        f"inconsistant type of variable '{var}' in if branches:\n"
                        f"types {type_if} and {type_else}\n"
                        f"if statement:\n"
                        f"{ast.unparse(node)}"
                    )
                )
            else:
                # just shadow the prior binding
                return type_if

        if type_prior is not None:
            if type_if is not None and type_if != type_prior:
                return Poison(
                    TypeError(
                        f"type of variable '{var}' changed in 'if' branch:\n"
                        f"before branch: {type_prior}, inside 'if': {type_if}\n"
                        f"statement: {ast.unparse(node)}"
                    )
                )

            if type_else is not None and type_else != type_prior:
                return Poison(
                    TypeError(
                        f"type of variable '{var}' changed in 'else' branch:\n"
                        f"before branch: {type_prior}, inside 'else': {type_else}\n"
                        f"statement: {ast.unparse(node)}"
                    )
                )

            return type_prior

        # No prior type, only one branch has type → error
        if type_if is not None:
            return Poison(
                TypeError(
                    f"Variable '{var}' not initialized in `else` branch \n"
                    f"Statement: {ast.dump(node)}"
                )
            )
        if type_else is not None:
            return Poison(
                TypeError(
                    f"Variable '{var}' not initialized in `if` branch\n"
                    f"Statement: {ast.unparse(node)}"
                )
            )

    def excluded_vars(self):
        return ()


class ForOp(SCFOp[ast.For, scf.ForOp]):
    start: Index
    stop: Index
    step: Index
    for_target: ast.AST

    def __init__(self, visitor, node):
        super().__init__(visitor, node)

        ba = range.iterator_bound_args(visitor, node)

        self.start, self.stop, self.step = range._obtain_boundary(
            visitor, ba.args
        )
        self.for_target = range.get_target(node)

    def build_op(
        self,
        op_type: Optional[list] = None,
        op_args: Optional[list] = None,
        dummy=True,
    ) -> scf.ForOp:
        if op_args is None:
            op_args = []
        op_args = [] if dummy else [lower_single(v) for v in op_args]
        return scf.ForOp(
            lower_single(self.start),
            lower_single(self.stop),
            lower_single(self.step),
            op_args,
        )

    def build_regions(self, op: scf.ForOp) -> tuple[dict[str, Any]]:
        bound_for = self.build_block(
            self.visitor.scope_stack.stack[-1],
            self.node.body,
            op.body,
            {self.for_target.id: Index(op.induction_variable)},
        )
        return (bound_for,)

    @staticmethod
    def verify_types(
        var: str, types: list[type | None], node: ast.For
    ) -> type | Poison:
        type_prior, type_for = types
        if type_prior != type_for:
            return Poison(
                TypeError(
                    f"inconsistent type for variable '{var}' in loop statement:\n"
                    f"{ast.unparse(node)}\n"
                    f"type before loop: {type_prior}\n"
                    f"type inside loop: {type_for}"
                )
            )

        return type_prior

    def excluded_vars(self):
        return (self.for_target.id,)

    @staticmethod
    def _replace_operands_with_block_args(
        op: scf.ForOp, yield_vars: list[str], bound_prior
    ):
        """Replace prior values with ForOp block arguments (iter_args)."""
        for idx, var in enumerate(yield_vars):
            old_val = lower_single(bound_prior[var])
            new_val = op.body.arguments[
                1 + idx
            ]  # first arg is induction variable
            SCFOp.replace_value_in_region(op.body.region, old_val, new_val)


class range(IteratorMacro):
    def _signature(
        *args: list[Compiled],
    ): ...

    def _obtain_boundary(
        visitor: ToMLIRBase, args
    ) -> tuple[Index, Index, Index]:
        # default values for start and step

        match len(args):
            case 1:
                start, stop, step = None, args[0], None
            case 2:
                start, stop, step = args[0], args[1], None
            case 3:
                start, stop, step = args[0], args[1], args[2]
            case _:
                raise ValueError(
                    f"range expects up to 3 arguments, got {len(args)}"
                )

        start = start if start is not None else Index(0)
        step = step if step is not None else Index(1)

        return (
            Index(start),
            Index(stop),
            Index(step),
        )

    def on_For(visitor: ToMLIRBase, node: ast.For) -> scf.ForOp:
        return ForOp(visitor, node)

    def on_ListComp(node: ast.ListComp):
        raise NotImplementedError(
            "range cannot be used for list comprehension for now"
        )
