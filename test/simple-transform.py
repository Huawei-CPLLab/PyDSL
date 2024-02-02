import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.type import F64, Index, AnyOp
from pydsl.memref import MemRefFactory
from pydsl.frontend import compile
from pydsl.scf import range
from pydsl.transform import tag, loop_coalesce, match_tag as match

MemRefF64 = MemRefFactory((40,50,), F64)

def transform_seq(targ: AnyOp):
    loop_coalesce(match(targ, 'coalesce'))

@compile(locals(), transform_seq=transform_seq, dump_mlir=True, auto_build=False)
def simple_case(v0: F64, arg1: MemRefF64) -> Index:
    N: Index = 40
    M: Index = 50
    c1: Index = 1

    """@tag("coalesce")"""
    for i in range(N):
        for j in range(M):
            arg1[i, j] = arg1[i, j] + v0
    
    return c1 + N + M