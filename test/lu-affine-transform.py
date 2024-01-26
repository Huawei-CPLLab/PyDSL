import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

from pydsl.transform import tag, fuse_into, match_tag as match, fuse, tile
from pydsl.type import UInt32, F64, Index, AnyOp
from pydsl.memref import MemRefFactory
from pydsl.frontend import compile
from pydsl.affine import            \
    affine_range as arange,         \
    affine_map as am,               \
    dimension as D,                 \
    symbol as S

MemrefF64 = MemRefFactory((40, 40), F64)

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


# This example requires specific pass arguments to mlir-opt, which is currently not supported. Hence auto_build is set to false for now.
@compile(locals(), transform_seq=transform_seq, dump_mlir=True, auto_build=False)
def lu(v0: Index, arg1: MemrefF64) -> UInt32:
    a: UInt32 = 5

    """@tag("tile")"""
    for arg2 in arange(S(v0)):

        """@tag("fuse_4")"""
        for arg3 in arange(D(arg2)):

            """@tag("fuse_1")"""
            for arg4 in arange(D(arg3)):
                arg1[am(D(arg2), D(arg3))] =    \
                    arg1[am(D(arg2), D(arg3))]  \
                    - (arg1[am(D(arg2), D(arg4))] 
                    * arg1[am(D(arg4), D(arg3))])
            
            """@tag("fuse_target1")"""
            v1 = arg1[am(D(arg3), D(arg3))]

            """@tag("fuse_target2")"""
            v2 = arg1[am(D(arg2), D(arg3))]

            """@tag("fuse_target3")"""
            v3 = v2 / v1

            """@tag("fuse_target4")"""
            arg1[am(D(arg2), D(arg3))] = v3

        """@tag("fuse_3")"""
        for arg3 in arange(D(arg2), S(v0)):
            for arg4 in arange(D(arg2)):
                arg1[am(D(arg2), D(arg3))] =    \
                    arg1[am(D(arg2), D(arg3))]  \
                    - (arg1[am(D(arg2), D(arg4))] 
                    * arg1[am(D(arg4), D(arg3))])

    return a


def weird_function(b: bool):
    a = 5

    if b:
        a = "h"
    
    # both str and int can *= 5
    a *= 5

    print(a) # 25 or "hhhhh"?