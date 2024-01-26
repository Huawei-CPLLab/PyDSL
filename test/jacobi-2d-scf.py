import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.type import UInt32, F32, Index
from pydsl.frontend import compile
from pydsl.scf import range
from pydsl.memref import MemRefFactory, DYNAMIC


MemRefF32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)

@compile(locals(), dump_mlir=True, auto_build=False)
def jacobi(T: Index, N: Index, a: MemRefF32, b: MemRefF32) -> UInt32:
    dummy: UInt32 = 5
    c1: Index = 1

    for _ in range(T):
        for i in range(N - c1):
            for j in range(N - c1):
                const: F32 = 0.2
                b[i, j] = (a[i, j]                      \
                        + a[i, j - c1] + a[i, j + c1]   \
                        + a[i - c1, j] + a[i + c1, j])  \
                        * const
        
        for i in range(c1, N - c1):
            for j in range(c1, N - c1):
                const: F32 = 0.2
                a[i, j] = (b[i, j]                      \
                        + b[i, j - c1] + b[i, j + c1]   \
                        + b[i - c1, j] + b[i + c1, j])  \
                        * const

    return dummy