import numpy as np

from pydsl.transform import (
    decorate_next,
    tile,
    parallel,
    tag,
    match_tag as match,
)
from pydsl.type import Index, AnyOp, F32
from pydsl.memref import MemRef
from pydsl.poly_target import PolyCTarget
from pydsl.affine import affine_range as arange

from pydsl.autotune import autotune, Var, Setting, TestingData

## Getting rid of some annoying log messages
import logging

logging.basicConfig(level=logging.ERROR)

# Testing arguments
N = 5000
A = np.ones((N, N), dtype=np.float32)


# Possible transformation Sequences
def t_seq_tile(targ: AnyOp):
    tile(match(targ, "loop_outter"), tile_factor, 4)


def t_parallel(targ: AnyOp):
    parallel(match(targ, "loop_outter"))


def t_parallel_double(targ: AnyOp):
    parallel(match(targ, "loop_outter"))
    parallel(match(targ, "loop_inner"))


# Building the autotune values
autotune_values = Var(
    "tile_factor", [[2, 2], [8, 8], [32, 32]]
) * Setting.transform_seq([t_seq_tile])
autotune_values += Setting.transform_seq([t_parallel, t_parallel_double])
autotune_values *= TestingData([[N, A]])
autotune_values *= Setting.target_class([PolyCTarget])


# Autotuning
@autotune(autotune_values)
def heat_fuse_tile(n: Index, A: MemRef[F32, N, N]):
    for _ in arange(100):
        decorate_next(tag("loop_outter"))
        for i in arange(n):
            decorate_next(tag("loop_inner"))
            for j in arange(n):
                A[i, j] = (-1.5) * A[i, j] * A[i, j] + (2.5) * A[i, j] + 1
