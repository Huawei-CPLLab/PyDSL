import ast
from mlir.ir import *
import mlir.dialects.arith as arith
import mlir.dialects.scf as scf

from pydsl.metafunc import IteratorMetafunction
from pydsl.type import Index, lower_single

class range(IteratorMetafunction):

    def on_For(visitor: ast.NodeVisitor, node: ast.For) -> scf.ForOp:
        iter_arg = node.target
        iterator = node.iter

        start = arith.ConstantOp(IndexType.get(), 0)
        step = arith.ConstantOp(IndexType.get(), 1)

        args = [lower_single(visitor.visit(a)) for a in iterator.args]

        match len(args):
            case 1:
                stop, = args
            case 2:
                start, stop = args
            case 3:
                start, stop, step = args
            case _:
                raise ValueError(f"range expects up to 3 arguments, got {len(args)}")
        
        for_op = scf.ForOp(start, stop, step)
        
        assert(type(iter_arg) is not ast.Tuple)
        with InsertionPoint(for_op.body):
            with visitor.variable_stack.new_scope({iter_arg.id: Index(for_op.induction_variable)}):
                for n in node.body:
                    visitor.visit(n)

                # nothing will be yielded for now
                scf.YieldOp([])

        
        return for_op     

    
    def on_ListComp(node: ast.ListComp):
        raise NotImplementedError("range cannot be used for list comprehension for now")