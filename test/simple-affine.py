import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.type import F64, Index
from pydsl.memref import MemRefFactory
from pydsl.frontend import compile
from pydsl.affine import            \
    affine_range as arange,         \
    affine_map as am,               \
    dimension as D                     

MemRefF64 = MemRefFactory((40,), F64)

@compile(locals(), dump_mlir=True, auto_build=False)
def lu(v0: F64, arg1: MemRefF64) -> Index:
    a: Index = 40
    c1: Index = 1

    for i in arange(D(a)):
        arg1[am(D(i))] = arg1[am(D(i))] + v0

    return c1 + a