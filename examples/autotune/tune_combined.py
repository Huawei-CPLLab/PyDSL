from timeit import Timer

import numpy as np

from pydsl.type import Index, F16, F32, F64, AnyOp
from pydsl.memref import MemRefFactory
from pydsl.frontend import CTarget
from pydsl.poly_target import PolyCTarget
from pydsl.affine import affine_range as arange
from pydsl.transform import decorate_next, tile, tag, match_tag as match

from pydsl.autotune import autotune, Var, Setting, TestingData

## Getting rid of some annoying log messages
import logging

logging.basicConfig(level=logging.ERROR)

N = 1000
MemF32 = MemRefFactory((N, N), F32)

A = np.ones((N, N), dtype=np.float32)


def t_seq(targ: AnyOp):
    tile(match(targ, "tile"), tile_factor, 4)


autotune_values = Setting.transform_seq([t_seq]) * Setting.target_class([
    PolyCTarget
])
autotune_values *= Var("tile_factor", [[2, 2], [8, 8], [32, 32]])
autotune_values *= Var("DATA_TYPE", [F32, F64])
autotune_values *= Var("NUM", [100, 200, 600])
autotune_values *= TestingData([[N, A]])


@autotune(autotune_values)
def heat_fuse_tile(n: Index, A: MemF32):
    num = Index(NUM)
    for _ in arange(num):
        decorate_next(tag("tile"))
        for i in arange(n):
            for j in arange(n):
                t = DATA_TYPE(A[i, j])
                x = (-1.5) * t * t + (2.5) * t + 1
                A[i, j] = x
