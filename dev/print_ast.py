import ast

code = r"""
def f(a: Int32 = 123) -> UInt64:
    a = b = 5
    return a + b
"""

tree = ast.parse(code)
print(ast.dump(tree, indent=4))
