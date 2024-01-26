import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.type import F32, Index
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile
from pydsl.affine import            \
    affine_range as arange,         \
    affine_map as am,               \
    dimension as D,                 \
    symbol as S                     

memref_f32 = MemRefFactory((DYNAMIC, DYNAMIC, DYNAMIC), F32)

@compile(locals(), dump_mlir=True, auto_build=False)
def heat(T: Index, N: Index, a: memref_f32, b: memref_f32) -> Index:
    const1: F32 = 0.125
    const2: F32 = 2.0
    v0: Index = 5
    for t in arange(S(T)):
        for i in arange(1, D(N)):
            for j in arange(1, D(N)):
                for k in arange(1, D(N)):
                    b[am(D(i), D(j), D(k))] = const1 * (a[am(D(i) + 1, D(j), D(k))] - const2 * a[am(D(i), D(j), D(k))] + a[am(D(i) - 1, D(j), D(k))]) \
                                            + const1 * (a[am(D(i), D(j) + 1, D(k))] - const2 * a[am(D(i), D(j), D(k))] + a[am(D(i), D(j) - 1, D(k))]) \
                                            + const1 * (a[am(D(i), D(j), D(k) + 1)] - const2 * a[am(D(i), D(j), D(k))] + a[am(D(i), D(j), D(k) - 1)]) \
                                            + a[am(D(i), D(j), D(k))]
        
        for i in arange(1, D(N)):
            for j in arange(1, D(N)):
                for k in arange(1, D(N)):
                    a[am(D(i), D(j), D(k))] = const1 * (b[am(D(i) + 1, D(j), D(k))] - const2 * b[am(D(i), D(j), D(k))] + b[am(D(i) - 1, D(j), D(k))]) \
                                            + const1 * (b[am(D(i), D(j) + 1, D(k))] - const2 * b[am(D(i), D(j), D(k))] + b[am(D(i), D(j) - 1, D(k))]) \
                                            + const1 * (b[am(D(i), D(j), D(k) + 1)] - const2 * b[am(D(i), D(j), D(k))] + b[am(D(i), D(j), D(k) - 1)]) \
                                            + b[am(D(i), D(j), D(k))]
                

    return v0

li = [(float(i % 9)) for i in range(10 * 10 * 10)]
li_2 = [(float(i % 9)) for i in range(10 * 10 * 10)]
retval = heat(10, 10, li, li_2)
print(f"Returned value:\t{retval}")
print(f"Modified li:\t{li[:20]}")