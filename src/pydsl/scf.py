import ast

from mlir.dialects import scf
from mlir.ir import InsertionPoint

from pydsl.macro import Compiled, IteratorMacro, Uncompiled
from pydsl.protocols import Lowerable, SubtreeOut, ToMLIRBase
from pydsl.type import Index, lower_single


class range(IteratorMacro):
    def _signature(
        *args: list[Compiled],
        yields: Uncompiled = ast.Tuple(elts=[]),
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

    def _obtain_yields_id(visitor, kwargs) -> list[str]:
        yields = kwargs["yields"]

        # make sure everything is a name
        for n in yields.elts:
            if not isinstance(n, ast.Name):
                raise TypeError(
                    f"range yields expected tuple of names, got {type(n)}"
                )

        return [n.id for n in yields.elts]

    def _obtain_id_value(visitor: ToMLIRBase, ids) -> list[SubtreeOut]:
        return [visitor.scope_stack.resolve_name(i) for i in ids]

    def _lower_ids(visitor: ToMLIRBase, ids: list[str]) -> list[SubtreeOut]:
        values = [visitor.scope_stack.resolve_name(i) for i in ids]

        for v in values:
            if not isinstance(v, Lowerable):
                raise TypeError(
                    f"value {v} of type {type(v).__name__} cannot be yielded "
                    f"by for statement as it's not lowerable.\n"
                    f"Hint: consider using a type with concrete MLIR "
                    f"representation."
                )

        return [lower_single(visitor.scope_stack.resolve_name(i)) for i in ids]

    def on_For(visitor: ToMLIRBase, node: ast.For) -> scf.ForOp:
        target = range.get_target(node)

        ba = range.iterator_bound_args(visitor, node)
        args = ba.args
        kwargs = ba.kwargs

        start, stop, step = range._obtain_boundary(visitor, args)
        yield_ids = range._obtain_yields_id(visitor, kwargs)
        starting_yield_types = [
            type(i) for i in range._obtain_id_value(visitor, yield_ids)
        ]

        for_op = scf.ForOp(
            lower_single(start),
            lower_single(stop),
            lower_single(step),
            iter_args=range._lower_ids(visitor, yield_ids),
        )

        assert type(target) is not ast.Tuple
        with InsertionPoint(for_op.body):
            # TODO: This does not throw error when an unyielded variable is
            # referenced outside of the loop. Requires dedicated analysis.
            visitor.scope_stack.assign_name(
                target.id, Index(for_op.induction_variable)
            )

            # Swap out the yielded id's values with those usable within the for
            # loop
            for yid, syt, iia in zip(
                yield_ids, starting_yield_types, for_op.inner_iter_args
            ):
                # Each starting yield type "raises" the raw MLIR yield values
                # back into the type they were before entering ForOp
                visitor.scope_stack.assign_name(yid, syt(iia))

            for n in node.body:
                visitor.visit(n)

            ending_yield_types = [
                type(i) for i in range._obtain_id_value(visitor, yield_ids)
            ]

            scf.YieldOp(range._lower_ids(visitor, yield_ids))

        # Assign the yielded results of the for statement back to the yielded
        # ids
        for yid, eyt, res in zip(
            yield_ids, ending_yield_types, for_op.results_
        ):
            visitor.scope_stack.assign_name(yid, eyt(res))

        return for_op

    def on_ListComp(node: ast.ListComp):
        raise NotImplementedError(
            "range cannot be used for list comprehension for now"
        )
