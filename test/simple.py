import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.type import F64, Index
from pydsl.memref import MemRefFactory
from pydsl.frontend import compile
from pydsl.scf import range

MemRefF64 = MemRefFactory((40,), F64)

@compile(locals(), dump_mlir=True, auto_build=False)
def example(v0: F64, arg1: MemRefF64) -> Index:
    a: Index = 40
    c1: Index = 1

    for i in range(a):
        arg1[i] = arg1[i] + v0

    return c1 + a