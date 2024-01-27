import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.type import F32, F64, Index, UInt16
from pydsl.memref import MemRefFactory
from pydsl.scf import range
from pydsl.frontend import compile

Memref64 = MemRefFactory((40, 40), F64)

@compile(locals(), dump_mlir=True, auto_build=False)
def hello(a: F32, b: F32) -> F32:
    d: F32 = 12.0
    l: Index = 5

    for i in range(l):
        e: F32 = 3.0
        f = e + d
    
    return (a / b) + d


# retval = hello(25, 3) # this now calls the .so library
# print(retval)
