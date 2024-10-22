import ast

from mlir.dialects import arith, scf
from mlir.ir import IndexType, InsertionPoint

from python_bindings_mlir.helpers import ToMLIRBase
from python_bindings_mlir.macro import IteratorMacro
from python_bindings_mlir.type import Index, lower_single


class range(IteratorMacro):
    def on_For(visitor: ToMLIRBase, node: ast.For) -> scf.ForOp:
        iter_arg = node.target
        iterator = node.iter

        # default values for start and step
        start = arith.ConstantOp(IndexType.get(), 0)
        step = arith.ConstantOp(IndexType.get(), 1)

        args = [lower_single(Index(visitor.visit(a))) for a in iterator.args]

        match len(args):
            case 1:
                (stop,) = args
            case 2:
                start, stop = args
            case 3:
                start, stop, step = args
            case _:
                raise ValueError(
                    f"range expects up to 3 arguments, got {len(args)}"
                )

        for_op = scf.ForOp(start, stop, step)

        assert type(iter_arg) is not ast.Tuple
        with InsertionPoint(for_op.body):
            visitor.scope_stack.assign_name(
                iter_arg.id, Index(for_op.induction_variable)
            )

            for n in node.body:
                visitor.visit(n)

            scf.YieldOp([])

        return for_op

    def on_ListComp(node: ast.ListComp):
        raise NotImplementedError(
            "range cannot be used for list comprehension for now"
        )
