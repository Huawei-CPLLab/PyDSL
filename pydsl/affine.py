from __future__ import annotations

import ast
from abc import abstractmethod
from typing import Any, Iterable, List
from functools import reduce
from mlir.ir import *
import mlir.dialects.arith as arith
import mlir.dialects.affine as affine

from pydsl.metafunc import CallingMetafunction, IteratorMetafunction, SubscriptingMetafunction
from pydsl.type import Index, lower, lower_flatten

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # This is for imports for type hinting purposes only and which can result in cyclic imports.
    from pydsl.compiler import VariableStack, ToMLIR


class affine_range(IteratorMetafunction):

    def on_For(visitor: ToMLIR, node: ast.For) -> affine.AffineForOp:
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
                raise ValueError(f"range expects up to 3 arguments, got {len(args)}")

        if type(lb) is ast.Call:
            match lb.func.id:
                case 'min':
                    raise SyntaxError(f"min is not permitted as the outer-most function of the lower-bound affine map")
                case 'max':
                    lb = lb.args

        if type(ub) is ast.Call:
            match ub.func.id:
                case 'min':
                    ub = ub.args    
                case 'max':
                    raise SyntaxError(f"max is not permitted as the outer-most function of the upper-bound affine map")

        ub_am_expr = AffineMapExprWalk.compile(ub, visitor.variable_stack).lowered()

        lb_am_expr =                    \
            AffineMapExpr.constant(0)   \
            if lb is None else          \
            AffineMapExprWalk.compile(lb, visitor.variable_stack).lowered()
        
        affine_for_op = affine.AffineForOp(
            lb_am_expr.map, ub_am_expr.map, step, 
            iter_args=[], # TODO: no iter args for now, maybe a feature later
            lower_bound_operands=[*lb_am_expr.dims, *lb_am_expr.syms],
            upper_bound_operands=[*ub_am_expr.dims, *ub_am_expr.syms]
        )

        with InsertionPoint(affine_for_op.body):
            with visitor.variable_stack.new_scope(
                {iter_arg.id: Index(affine_for_op.induction_variable)}
            ):
                for n in node.body:
                    visitor.visit(n)

                # nothing will be yielded for now
                affine.AffineYieldOp([])

        return affine_for_op
    

    def on_ListComp(node: ast.ListComp):
        raise NotImplementedError("affine_range cannot be used for list comprehension for now")


class AffineMapExpr:
    """
    An AffineMap with all dimensions and symbols filled
    """

    map: AffineMap
    dims: List[Any]
    syms: List[Any]

    def __init__(self, map: AffineMap, dims: List[Any], syms: List[Any]) -> None:
        self.map = map
        self.dims = dims
        self.syms = syms


    @staticmethod
    def constant(x: int):
        return AffineMapExpr(
            AffineMap.get(0, 0, [AffineExpr.get_constant(x)]), 
            [], # no dimension
            []  # no symbol
        )

    
    def lowered(self) -> AffineMapExpr:
        # lower all the dims and syms
        return AffineMapExpr(self.map, lower_flatten(self.dims), lower_flatten(self.syms))


class affine_map(CallingMetafunction):

    def argtypes():
        return []

    def varargtype() -> CallingMetafunction.ArgType:
        return CallingMetafunction.ArgType.TREE

    # TODO: make the return tuple a proper type
    def _on_Call(visitor: ToMLIR, args: List[Any]) -> AffineMapExpr:
        # map, dims, syms = AffineMapExprWalk.compile(call.args, visitor.variable_stack)
        # return affine.AffineStoreOp(value, target, indices=[*dims, *syms], map=map)
        return AffineMapExprWalk.compile(args, visitor.variable_stack)


class AffineMetafunction(CallingMetafunction):
    """
    A calling metafunction that can only be called in an affine context
    """

    @classmethod
    def _on_Call(cls, visitor: ast.NodeVisitor, args: List[Any]) -> Any:
        if not isinstance(visitor, AffineMapExprWalk):
            raise TypeError(f"{cls.__name__} is an affine metafunction but was called outside of an affine map expression")
        
        return cls._affine_on_Call(visitor, args)
    
    @abstractmethod
    def argtypes() -> List[AffineMetafunction.ArgType]:
        pass
        

    @abstractmethod
    def _affine_on_Call(visitor: AffineMapExprWalk, args: List[Any]) -> Any:
        pass


class dimension(AffineMetafunction):
    """
    A dimension in an affine expression
    """

    def argtypes():
        return [AffineMetafunction.ArgType.TREE]
    
    def _affine_on_Call(visitor: AffineMapExprWalk, args: List[Any]) -> Any:
        arg = args[0]
        match arg:
            case ast.Constant():
                # Having a constant in a dim() has a very specific connotation. We cannot visit the constant as-is.
                # We need to create an ad-hoc ConstantOp to be used as a dim. AffineDimExpr does not accept a AffineConstantExpr.
                if type(arg.value) is not int: raise TypeError(f"dimension expected integer, got {arg.value}")
                
                return visitor.add_dim(arith.ConstantOp(IndexType.get(), arg.value))
            
            case ast.Name():
                return visitor.add_dim(visitor.visit(arg))
            
            case _:
                raise TypeError(f"{type(arg)} type cannot be used in dim")
    

class symbol(AffineMetafunction):
    """
    A symbol in an affine expression
    """

    def argtypes():
        return [AffineMetafunction.ArgType.TREE]
    

    def _affine_on_Call(visitor: AffineMapExprWalk, args: List[Any]) -> Any:
        arg = args[0]
        match arg:
            case ast.Constant():
                # Having a constant in a sym() has a very specific connotation. We cannot visit the constant as-is.
                # We need to create an ad-hoc ConstantOp to be passed into AffineSymbolExpr. AffineSymbolExpr does not accept a AffineConstantExpr.
                if type(arg.value) is not int: raise TypeError(f"symbol expected integer, got {arg.value}")
                
                return visitor.add_sym(arith.ConstantOp(IndexType.get(), arg.value))
            
            case ast.Name():
                return visitor.add_sym(visitor.visit(arg))
            
            case _:
                raise TypeError(f"{type(arg)} type cannot be used in sym")
            

class floordivide(AffineMetafunction):

    def argtypes():
        return [AffineMetafunction.ArgType.TREE, AffineMetafunction.ArgType.TREE]

    def _affine_on_Call(visitor: AffineMapExprWalk, args: List[Any]) -> Any:
        left = visitor.visit(args[0])
        right = visitor.visit(args[1])
        return AffineExpr.get_floor_div(left, right)
    

class ceildivide(AffineMetafunction):

    def argtypes():
        return [AffineMetafunction.ArgType.TREE, AffineMetafunction.ArgType.TREE]

    def _affine_on_Call(visitor: AffineMapExprWalk, args: List[Any]) -> Any:
        left = visitor.visit(args[0])
        right = visitor.visit(args[1])
        return AffineExpr.get_ceil_div(left, right)


class AffineMapExprWalk(ast.NodeVisitor):
    """
    A helper NodeVisitor subclass that walks a Python subtree assumed to be entirely used to be converted into an affine_map
    """

    def __init__(self, variable_stack: VariableStack) -> None:
        self.variable_stack = variable_stack
        self.dims = []
        self.syms = []


    def add_dim(self, x: Any):
        if x in self.dims:
            return AffineDimExpr.get(self.dims.index(x))

        self.dims.append(x)
        return AffineDimExpr.get(len(self.dims) - 1)
    

    def add_sym(self, x: Any):
        if x in self.syms:
            return AffineSymbolExpr.get(self.syms.index(x))
        
        self.syms.append(x)
        return AffineSymbolExpr.get(len(self.syms) - 1)
    

    def visit_Name(self, node: ast.Name) -> Any:
        return self.variable_stack.resolve_name(node.id)


    def visit_Constant(self, node: ast.Constant) -> Any:
        return AffineExpr.get_constant(node.value)


    def visit_BinOp(self, node: ast.BinOp) -> Any:
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
                raise ValueError(f"Ln {node.lineno}: {type(node.op)} is not a valid affine binary operator.")

    
    def visit_Call(self, node: ast.Call) -> Any:
        mf = self.variable_stack.resolve_name_as_metafunc(node.func.id, AffineMetafunction)
        mf: AffineMetafunction
        return mf.on_Call(self, node)
    

    @staticmethod
    def compile(node: ast.AST | Iterable[ast.AST], name_resolver: VariableStack) -> AffineMapExpr:
        """
        Returns a tuple of (affine map, dim arguments, sym arguments)
        """

        if type(node) is ast.List:
            # convert ast List into a proper Python list
            node = node.elts

        if not isinstance(node, Iterable):
            # if only a single element is given, make it iterable
            node = [node]

        walk = AffineMapExprWalk(name_resolver)
        exprs = [walk.visit(elt) for elt in node]

        amap = AffineMap.get(len(walk.dims), len(walk.syms), exprs)

        return AffineMapExpr(amap, walk.dims, walk.syms)
 