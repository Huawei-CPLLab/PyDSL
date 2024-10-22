import ast
from collections.abc import Iterable
from functools import cache

"""
The Used and Bound analysis attempts to emulate Python's name resolution behavior as closely as
possible.
It is NOT a runtime frames stack, but rather a stack of nested function scope
and the variables that they define.

The two useful results from the data structure are emulated globals() and
locals().

TODO: This module cannot yet handle global, local, nonlocal statement behavior.

Here's some helpful links that explains the LEGB name resolution rule:
https://stackoverflow.com/questions/291978/short-description-of-the-scoping-rules
https://discuss.python.org/t/how-name-resolution-works-in-python/26452
https://pythontutor.com

Rough algorithm to determine Global(l) and Local(l):

1. Go through the function f and see if a variable is assigned to (not
   including use). This DOES NOT include assignments in nested functions. Note
   that this includes all function arguments as they are assigned to when the
   function is called. The set of all variables that's assigned to is Bound(f).
2. Go through the function f and see if a variable is used (not including
   assignment). This DOES include variable usage in nested functions which are
   NOT BOUND. The set of all variables that's used is Used(f).
3. Define Find(f, var), which goes outward from the scope of f looking if var
   is in Local(L(f)), where L(f) is the last line left off in f.
    - If not, go outward by one scope and repeat.
    - If reached top, raise NameError(f"name '{name}' is not defined").
    - If value is found but is UNBOUND, then raise NameError(f"cannot access
        free variable '{var}' where it is not associated with a value in
        enclosing scope")
3. For all l in the instructions of f, Let Global(l) = the locals at the top
   level of the module which f is in
4. Define all Local(l) recursively as such, for all l_i in instructions of f,
   where l_1 is the first line in l:
    Local(l_1) = ({var: Find(f, var) for var in Used(f) \\ Bound(f)} <-
    {UNBOUND for var in Bound(f)} <- Arguments(f) (Where \\ is set subtraction,
    <- updates one associative set (i.e. map) with entries from another, i.e.
    the .update() method for Python dict)
    Local(l_i+1) = Local(l_(i)) <- Assignments(l), where Assignments(l) are
    assignments that happened on l
if f is the top scope, this is instead the definition:
    Local(l_1) = {}
    Local(l_i+1) = Local(l_(i)) <- Assignments(l)

5. When resolving variable, use this algorithm:

def resolve(name, l):
    if name in Local(l):
        val = Local(l)[name]
    elif name in Global(l):
        val = Global(l)[name]
    else:
        raise NameError(f"name '{name}' is not defined")

    if val == UNBOUND:
        raise UnboundLocalError(f"cannot access local variable '{name}' where \
        it is not associated with a value")

    return val

Special case to note:
- Nested functions include lambdas


TODO: Since UNBOUND doesn't actually shows up in Python's locals(), rewrite
this pseudocode so that each scope keeps track of bound variables and prevent
them from accessing globals()
"""


class BoundAnalysis(ast.NodeVisitor):
    bound: set

    def __init__(self) -> None:
        super().__init__()
        self.bound = set()

    def visit_Name(self, node: ast.Name) -> None:
        if type(node.ctx) is ast.Store:
            self.bound.add(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # We don't visit the body because we don't want to count in bound
        # variables from within nested functions
        # Everything else (e.g. function name, type hints, decorators, etc.)
        # on the same level though
        self.bound.add(node.name)

        for dec in node.decorator_list:
            self.visit(dec)

        self.visit(node.args)
        self.visit(node.returns)

        for param in node.type_params:
            self.visit(param)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # We don't visit lambda at all. Nothing is on the same level.
        pass

    @cache
    def analyze(func: ast.FunctionDef | ast.Lambda) -> set[str]:
        ba = BoundAnalysis()

        # names of arguments are also bound in the function
        ba.bound.update([a.arg for a in func.args.args])

        body = func.body if isinstance(func.body, Iterable) else [func.body]

        for b in body:
            ba.visit(b)

        return ba.bound


class UsedAnalysis(ast.NodeVisitor):
    used: set

    def __init__(self) -> None:
        super().__init__()
        self.used = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Only variables which are used and not bound in a nested function are
        # considered used in the parent function
        used = UsedAnalysis.analyze(node)
        bound = BoundAnalysis.analyze(node)
        self.used.update(used - bound)

    visit_Lambda = visit_FunctionDef

    def visit_Name(self, node: ast.Name):
        if type(node.ctx) is ast.Load:
            self.used.add(node.id)

    @cache
    def analyze(func: ast.FunctionDef | ast.Lambda) -> set[str]:
        ua = UsedAnalysis()

        body = func.body if isinstance(func.body, Iterable) else [func.body]

        for b in body:
            ua.visit(b)

        return ua.used
