from pydsl.frontend import compile
from pydsl.scf import range
from pydsl.memref import MemRefFactory
from pydsl.type import F32, F64, Index

Memref64 = MemRefFactory((40, 40), F64)


@compile(locals())
def function_example(a: F32, b: F32) -> F32:
    d: F32 = 12.0
    l: Index = 5

    for i in range(l):
        e: F32 = 3.0
        f = e + d  # noqa: F841

    return (a / b) + d


retval = function_example(25, 3)  # this now calls the compiled library
print(25 / 3 + 12)
print(retval)
