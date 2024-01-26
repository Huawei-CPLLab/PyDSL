import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.type import UInt32, F32, Index
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile
from pydsl.affine import    \
    affine_range as arange, \
    affine_map as am,       \
    dimension as D,         \
    symbol as S

MemRefF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)

@compile(locals(), dump_mlir=True)
def jacobi(T: Index, N: Index, a: MemRefF32, b: MemRefF32) -> UInt32:
    dummy: UInt32 = 5

    for _ in arange(S(T)):
        for i in arange(1, S(N) - 1):
            for j in arange(1, S(N) - 1):
                const:F32 = 0.2
                b[am(D(i), D(j))] = (a[am(D(i), D(j))] +        \
                                     a[am(D(i), D(j) - 1)] +    \
                                     a[am(D(i), D(j) + 1)] +    \
                                     a[am(D(i) - 1, D(j))] +    \
                                     a[am(D(i) + 1, D(j))]) * const

        
        for i in arange(1, S(N) - 1):
            for j in arange(1, S(N) - 1):
                const: F32 = 0.2
                a[am(D(i), D(j))] = (b[am(D(i), D(j))] +        \
                                     b[am(D(i), D(j) - 1)] +    \
                                     b[am(D(i), D(j) + 1)] +    \
                                     b[am(D(i) - 1, D(j))] +    \
                                     b[am(D(i) + 1, D(j))]) * const

    return dummy