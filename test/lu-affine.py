import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.type import UInt32, F64, Index
from pydsl.memref import MemRefFactory
from pydsl.frontend import compile
from pydsl.affine import            \
    affine_range as arange,         \
    affine_map as am,               \
    dimension as D,                 \
    symbol as S                     

MemRefF64 = MemRefFactory((40, 40), F64)

@compile(locals(), dump_mlir=True, auto_build=False)
def lu(v0: Index, arg1: MemRefF64) -> Index:
    a: UInt32 = 5

    for arg2 in arange(S(v0)):
        for arg3 in arange(D(arg2)):
            for arg4 in arange(D(arg3)):
                arg1[am(D(arg2), D(arg3))] =        \
                    arg1[am(D(arg2), D(arg3))]      \
                    - (arg1[am(D(arg2), D(arg4))]   \
                    * arg1[am(D(arg4), D(arg3))])
            
            arg1[am(D(arg2), D(arg3))] =        \
                arg1[am(D(arg2), D(arg3))]      \
                / arg1[am(D(arg3), D(arg3))]

        for arg3 in arange(D(arg2), S(v0)):
            for arg4 in arange(D(arg2)):
                arg1[am(D(arg2), D(arg3))] =        \
                    arg1[am(D(arg2), D(arg3))]      \
                    - (arg1[am(D(arg2), D(arg4))]   \
                    * arg1[am(D(arg4), D(arg3))])
                
    return v0


# li = [(i % 39) for i in range(40 * 40)]
# retval = lu(40, li)
# print(f"Returned value:\t{retval}")
# print(f"Modified li:\t{li[:20]}")