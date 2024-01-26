import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile
from pydsl.type import UInt32, F32, Index, AnyOp
from pydsl.memref import MemRefFactory, DYNAMIC
from pydsl.frontend import compile
from pydsl.affine import            \
    affine_range as arange,         \
    affine_map as am,               \
    dimension as D,                 \
    symbol as S                     

memref_f32 = MemRefFactory((DYNAMIC, DYNAMIC), F32)

def transform_seq(targ: AnyOp):
    fuse_into(
    fuse_into(
    fuse_into(
    fuse_into(
        match(targ, 'fuse_1'), 
        match(targ, 'fuse_target1')),
        match(targ, 'fuse_target2')), 
        match(targ, 'fuse_target3')), 
        match(targ, 'fuse_target4'))

    fuse(match(targ, 'fuse_4'), match(targ, 'fuse_3'), 2)

    tile(match(targ, 'tile'), [32, 32, 32], 6)

@compile(locals(), dump_mlir=True)
def seidal(t: Index, N: Index, A: memref_f32) -> Index:
    a: UInt32 = 5
    const: F32 = 9.0
    v0: Index = 1
    for _ in arange(S(t)):
        for i in arange(1, D(N)):
            for j in arange(1, D(N)):
                A[am(D(i), D(j))] = (A[am(D(i) - 1, D(j) - 1)] + A[am(D(i) - 1, D(j))] + A[am(D(i) - 1 , D(j) + 1)] \
                                     + A[am(D(i), D(j) - 1)] + A[am(D(i), D(j))] + A[am(D(i), D(j) + 1)] \
                                     + A[am(D(i) + 1, D(j) - 1)] + A[am(D(i) + 1, D(j))] + A[am(D(i) + 1, D(j) + 1)]) / const                

    return v0

li = [(i % 39) for i in range(40 * 40)]
retval = seidal(40, 40, li)
print(f"Returned value:\t{retval}")
print(f"Modified li:\t{li[:20]}")