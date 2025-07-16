from pydsl.affine import (
    affine_map as am,
    affine_range as arange,
    dimension as D,
    symbol as S,
)
from pydsl.frontend import compile
from pydsl.memref import MemRefFactory
from pydsl.type import F32, Index

n = 2200
m = 1800
MemRefF32NM = MemRefFactory((n, m), F32)


@compile(locals(), dump_mlir=True)
def affine_explicit_example(
    v0: Index,
    v1: Index,
    A: MemRefF32NM,
) -> F32:
    b: F32 = 0.0
    for i in arange(am(S(v0)), am(S(v0) + 8), 2):
        A[am(D(i), S(v1) + 5)] = b

    return b
