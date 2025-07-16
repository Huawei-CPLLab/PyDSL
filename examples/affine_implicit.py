from pydsl.affine import affine_range as arange
from pydsl.frontend import compile
from pydsl.memref import MemRefFactory
from pydsl.type import F32, Index

n = 2200
m = 1800
MemRefF32NM = MemRefFactory((n, m), F32)


@compile(dump_mlir=True)
def affine_implicit_example(
    v0: Index,
    v1: Index,
    A: MemRefF32NM,
) -> F32:
    b: F32 = 0.0
    for i in arange(v0, v0 + 8, 2):
        A[i, v1 + 5] = b

    return b
