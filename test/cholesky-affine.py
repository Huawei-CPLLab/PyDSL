import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.type import UInt32, F32, Index
from pydsl.frontend import compile
from pydsl.affine import            \
    affine_range as arange,         \
    affine_map as am,               \
    dimension as D,                 \
    symbol as S                     

memref_f32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)

@compile(locals(), dump_mlir=True)
def cholesky(N: Index, A: memref_f32) -> Index:
    a: UInt32 = 5
    const: F32 = 9.0
    for i in arange(1, D(N)):
        for j in arange(D(i)):
            for k in arange(D(j)):
                A[am(D(i), D(j))] = A[am(D(i), D())] - A[am(D(i), D(k))] * A[am(D(j), D(k))]     
            A[am(D(i), D(j))] = A[am(D(i), D(j))] / A[am(D(j), D(j))]
        for k in arange(D(i)):
            A[am(D(i), D(i))] = A[am(D(i), D(i))] - A[am(D(i), D(k))] * A[am(D(i), D(k))]
        A[am(D(i), D(i))] = A[am(D(i), D(i))] ** 0.5 # won't work

    return a

li = [(i % 39) for i in range(40 * 40)]
retval = cholesky(40, li)
print(f"Returned value:\t{retval}")
print(f"Modified li:\t{li[:20]}")