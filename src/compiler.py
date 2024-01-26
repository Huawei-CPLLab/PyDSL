import ast
import typing
from typing import Any, Dict, Never
from contextlib import contextmanager
from functools import cache

import mlir.ir as mlir
from mlir.ir import *
import mlir.dialects.arith as arith
import mlir.dialects.func as func
import mlir.dialects.memref as memref
import mlir.dialects.transform as transform

from mlir.dialects.transform import structured

from pydsl.transform import Transform
from pydsl.metafunc import CallingMetafunction, IteratorMetafunction, SubscriptingMetafunction
from pydsl.type import Lowerable, lower, lower_single, lower_flatten

def generate_parent(root):
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            child.parent = node


def generate_next_line(root):
    for node in ast.walk(root):
        if hasattr(node, 'body'):
            for i, child in enumerate(node.body):
                child.next_line = node.body[i + 1] if (i+1) < len(node.body) else None 


class VariableStack:

    names_frames_stack: typing.List[Dict[str, Any]] = []

    def __init__(self, f_locals) -> None:
        self.names_frames_stack = []
        self.names_frames_stack.append(f_locals)

    
    def resolve_name(self, name):
        for d in reversed(self.names_frames_stack):
            if name in d:
                return d[name]
        
        raise NameError(f"name '{name}' is not defined")
    
    
    def resolve_name_as_metafunc(self, name, metafunc):
        result = self.resolve_name(name)

        if not issubclass(result, metafunc):
            raise TypeError(f"Expected a {metafunc.__name__}, got '{name}' which is not a subclass of {metafunc.__name__}")
        
        return result
    

    def assign_name(self, name, value):
        self.names_frames_stack[-1][name] = value

    
    @contextmanager
    def new_scope(self, new_variables: Dict[str, Any]):
        self.names_frames_stack.append(new_variables)
        try:
            yield
        finally:
            self.names_frames_stack.pop()


class CompilationException(Exception):
    pass

class ToMLIR(ast.NodeVisitor):

    mlir = None
    variable_stack: VariableStack = None
    transforms = []

    # Results are cached because visiting a node multiple times can result in MLIR programs being generated multiple times
    # ast.NodeVisitor cannot modify the tree it is visiting, so cache will never be outdated
    @cache
    def visit(self, node):
        # try:
        return super().visit(node)
        # except Exception as e:
        #     if issubclass(type(e), CompilationException):
        #         raise e

        #     if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
        #         raise CompilationException(f"Exception when compiling Line {node.lineno}, Col {node.col_offset}") from e
        #     else:
        #         raise CompilationException(f"Exception occured while compiling") from e

    def __init__(self, f_locals) -> None:
        super(ToMLIR, self).__init__()
        # f_locals is a dictionary of local variables at the time the function being compiled is defined
        self.f_locals = f_locals

    
    def handle_CallingMetafunction(self, call: ast.Call) -> OpView:
        func_name = call.func.id
        calling_func: CallingMetafunction = self.variable_stack.resolve_name_as_metafunc(func_name, CallingMetafunction)
        return calling_func.on_Call(self, call)

    
    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(entry) for entry in node.elts)
    

    def visit_Expr(self, node: ast.Expr) -> OpView:
        DIRECTIVE_ESCAPE = "@"
        
        expr_val = node.value

        match expr_val:
            case ast.Constant():
                val = expr_val.value
                if type(val) is str and val.startswith(DIRECTIVE_ESCAPE):
                    if  (not hasattr(node, 'next_line')) or             \
                        (operand := node.next_line) is None:

                        raise ValueError("Docstring '@' directive must be placed before a valid operator")

                    directive_expr = val[len(DIRECTIVE_ESCAPE):]
                    directive_ast = ast.parse(directive_expr).body[0]

                    if type(directive_ast) is not ast.Expr:
                        raise ValueError("Docstring '@' directive is not an expression")

                    match value := directive_ast.value:
                        case ast.Call():
                            # adds the AST as the first parameter, similar to how self works in Python
                            value.args.insert(0, operand)
                            return self.handle_CallingMetafunction(value)
                        
                        case _:
                            raise TypeError(f"Docstring '@' directive expression immediately contains {type(value)}, which is not supported")

            case _:
                self.visit(expr_val)


    def visit_Constant(self, node: ast.Constant) -> Never:
        raise ValueError(f"Using the constant literal {node.value} directly is currently not supported as we cannot infer its MLIR type")
    

    def visit_Call(self, node: ast.Call) -> Any:
        # for now, the only kind of call that is relevant are CallingMetafunction
        return self.handle_CallingMetafunction(node)


    # for now, assume all operands are floating point numbers
    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)

        match node.op:
            case ast.Add():
                return left.__add__(right)
            case ast.Sub():
                return left.__sub__(right)
            case ast.Mult():
                return left.__mul__(right)
            case ast.Div():
                return left.__truediv__(right)
            case ast.FloorDiv():
                return left.__floordiv__(right)
            # TODO: more ops can be added in the future
            case _:
                raise ValueError(f"Ln {node.lineno}: {type(node.op)} is currently not supported as a binary operator")
    

    def visit_Name(self, node: ast.Name) -> Any:
        return self.variable_stack.resolve_name(node.id)
    

    def visit_Return(self, node: ast.Return) -> Any:
        return func.ReturnOp(lower_flatten([self.visit(node.value)]))
    

    def visit_Subscript(self, node: ast.Subscript) -> OpView | Lowerable:
        assert type(node.ctx) is not ast.Store, AssertionError("Subscript with a Store context shouldn't be visited!")

        value = self.visit(node.value)
        slice = self.visit(node.slice)
        return value.__getitem__(slice)


    def visit_Assign(self, node: ast.Assign) -> OpView:
        # ignore tuple assignment for now, hence the "node.targets[0]"

        # the match/case here is getting out of hand...
        match node.targets[0]:
            case ast.Name():
                match node.value:
                    case ast.Constant():
                        raise ValueError(f"Assigning constant without type hinting is currently not supported")
                    case _:
                        rhs = self.visit(node.value)
                        self.variable_stack.assign_name(node.targets[0].id, rhs)
                        return rhs
            
            case ast.Subscript():
                value   = self.visit(node.value)
                target  = self.visit(node.targets[0].value)
                slice   = self.visit(node.targets[0].slice)
                return target.__setitem__(slice, value)
                
            case _:
                raise ValueError(f"Assigning to {type(node.targets)} is currently not supported.")
        

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        self.variable_stack.assign_name(node.target.id, self.variable_stack.resolve_name(node.annotation.id)(node.value.value))


    def visit_For(self, node: ast.For) -> Any:
        iterator = node.iter
        name = iterator.func.id
        
        # we will not accept any other way to pass in an iterator for now
        assert type(iterator) is ast.Call, "Iterator of the for loop must be a Call for now"        
        iterator = self.variable_stack.resolve_name_as_metafunc(name, IteratorMetafunction)

        return iterator.on_For(self, node)


    def visit_FunctionDef(self, node: ast.FunctionDef) -> OpView:
        arg_names = [arg.arg for arg in node.args.args]
        arg_types = [self.variable_stack.resolve_name(arg.annotation.id) for arg in node.args.args]
        return_type = self.variable_stack.resolve_name(node.returns.id)

        f = func.FuncOp(node.name, \
                        (lower_flatten(arg_types), lower_flatten([return_type]))) # (input, result)
        f.sym_visibility = StringAttr.get("public") # we will assume all functions are public for now
        f.add_entry_block()

        # TODO: a requirement just for now. We want to support None (void) as well.
        if type(node.body[-1]) is not ast.Return:
            raise ValueError(f"Function must end with return operator (for now)")
        
        with InsertionPoint(f.entry_block):
            with self.variable_stack.new_scope({name: arg_types[i](f.arguments[i]) for i, name in enumerate(arg_names)}):
                for n in node.body:
                    self.visit(n)

        return f
    

    # TODO: This may not be the best place for this function, but it's very similar to ToMLIR.visit_FunctionDef 
    def visit_FunctionDef_as_transform_seq(self, node: ast.FunctionDef) -> OpView:
        arg_names = [arg.arg for arg in node.args.args]
        arg_types = [self.variable_stack.resolve_name(arg.annotation.id) for arg in node.args.args]

        if len(arg_names) != 1:
            raise ValueError(f"Function {node.name} used as transform.sequence operation should have exactly 1 argument, got {len(arg_names)}")

        seq = transform.SequenceOp(
            transform.FailurePropagationMode.Propagate, # for now, assume failure mode is always propagate
            [],
            lower_single(arg_types[0]),
        )

        with InsertionPoint(seq.body):
            with self.variable_stack.new_scope({arg_names[0]: seq.bodyTarget}):
                for n in node.body:
                    self.visit(n)
                    
                transform.YieldOp()

        return seq
            

    def compile(self, node, transform_seq: ast.AST | None = None):
        # create additional properties in AST nodes that we will need during compilation
        generate_parent(node)
        generate_next_line(node)
        self.mlir = None
        self.variable_stack = VariableStack(self.f_locals)
        self.transforms = []
        self.cache = {}

        with Context() as ctx, Location.unknown():
            self.mlir = Module.create()
            with InsertionPoint(self.mlir.body):
                self.visit(node)
                if transform_seq is not None:
                    # TODO: this is hacky, should create a dedicated transform seq visitor
                    self.visit_FunctionDef_as_transform_seq(transform_seq.body[0])

            self.mlir.operation.verify()
            return str(self.mlir)
