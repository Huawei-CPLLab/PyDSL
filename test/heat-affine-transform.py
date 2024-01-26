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

memref_f32 = MemRefFactory((DYNAMIC, DYNAMIC, DYNAMIC), F32)

def transform_seq(targ: AnyOp):
    fuse(match(targ, 'fuse_1'), match(targ, 'fuse_2'), 2)

    tile(match(targ, 'tile'), [16, 8, 8, 8], 6)

@compile(locals(), transform_seq=transform_seq, dump_mlir=True)
def heat(T: Index, N: Index, a: memref_f32, b: memref_f32) -> Index:
    const1: F32 = 0.125
    const2: F32 = 2.0
    v0: Index = 5
    """@tag(tile)"""
    for t in arange(S(T)):
        """@tag("fuse_1")"""
        for i in arange(1, D(N)):
            for j in arange(1, D(N)):
                for k in arange(1, D(N)):
                    b[am(D(i), D(j), D(k))] = const1 * (a[am(D(i) + 1, D(j), D(k))] - const2 * a[am(D(i), D(j), D(k))] + a[am(D(i) - 1, D(j), D(k))]) \
                                            + const1 * (a[am(D(i), D(j) + 1, D(k))] - const2 * a[am(D(i), D(j), D(k))] + a[am(D(i), D(j) - 1, D(k))]) \
                                            + const1 * (a[am(D(i), D(j), D(k) + 1)] - const2 * a[am(D(i), D(j), D(k))] + a[am(D(i), D(j), D(k) - 1)]) \
                                            + a[am(D(i), D(j), D(k))]
        """@tag("fuse_2")"""
        for i in arange(1, D(N)):
            for j in arange(1, D(N)):
                for k in arange(1, D(N)):
                    a[am(D(i), D(j), D(k))] = const1 * (b[am(D(i) + 1, D(j), D(k))] - const2 * b[am(D(i), D(j), D(k))] + b[am(D(i) - 1, D(j), D(k))]) \
                                            + const1 * (b[am(D(i), D(j) + 1, D(k))] - const2 * b[am(D(i), D(j), D(k))] + b[am(D(i), D(j) - 1, D(k))]) \
                                            + const1 * (b[am(D(i), D(j), D(k) + 1)] - const2 * b[am(D(i), D(j), D(k))] + b[am(D(i), D(j), D(k) - 1)]) \
                                            + b[am(D(i), D(j), D(k))]
                

    return v0

li = [(i % 9) for i in range(10 * 10 * 10)]
li_2 = [(i % 9) for i in range(10 * 10 * 10)]
retval = heat(10, 10, li, li_2)
print(f"Returned value:\t{retval}")
print(f"Modified li:\t{li[:20]}")