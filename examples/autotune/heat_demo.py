import numpy as np

from pydsl.transform import (
    decorate_next,
    fuse,
    get_loop,
    tag,
    tile,
    parallel,
    match_tag as match,
)
from pydsl.type import Index, AnyOp, F32
from pydsl.memref import MemRef
from pydsl.frontend import PolyCTarget
from pydsl.affine import affine_range as arange

from pydsl.autotune import autotune, Var, Setting, TestingData
from copy import deepcopy

## Getting rid of some annoying log messages
import logging

logging.basicConfig(level=logging.ERROR)

N = 200


# ---------------------------------------------------- Transformation Sequences ----------------------------------------------------
def transform_seq_fuse_tile(targ: AnyOp):
    fuse_rez = fuse(match(targ, "fuse_1"), match(targ, "fuse_2"), 3)
    tile(match(targ, "tile"), tile_factor, 8)
    parallel(get_loop(fuse_rez, 0), False)


def transform_seq_parallel(targ: AnyOp):
    parallel(match(targ, "fuse_1"), False)
    parallel(match(targ, "fuse_2"), False)


# ----------------------------------------------------------- Input Data -----------------------------------------------------------
t = 200
A = np.fromfunction(
    lambda i, j, k: (i + j + (t - k)) * 10 / (t), (N, N, N), dtype=np.float32
)
B = deepcopy(A)

# ----------------------------------------------- Building Auto Tune Configurations ------------------------------------------------

autotune_values = Var(
    "tile_factor", [[2, 2, 2, 2], [8, 8, 8, 8], [8, 8, 32, 32]]
) * Setting.transform_seq([transform_seq_fuse_tile])
autotune_values += Setting.transform_seq([transform_seq_parallel])
autotune_values *= Setting.target_class([PolyCTarget]) * TestingData([
    [t, N, A, B]
])


@autotune(autotune_values)
def heat_fuse_tile(
    tsteps: Index, n: Index, A: MemRef[F32, N, N, N], B: MemRef[F32, N, N, N]
):
    a: F32 = 2.0
    b: F32 = 0.125
    decorate_next(tag("tile"))
    for _ in arange(tsteps):
        decorate_next(tag("fuse_1"))
        for i in arange(1, n - 1):
            for j in arange(1, n - 1):
                for k in arange(1, n - 1):
                    B[i, j, k] = A[i, j, k] + b * (
                        A[i + 1, j, k]
                        - a * A[i, j, k]
                        + A[i - 1, j, k]
                        + A[i, j + 1, k]
                        - a * A[i, j, k]
                        + A[i, j - 1, k]
                        + A[i, j, k + 1]
                        - a * A[i, j, k]
                        + A[i, j, k - 1]
                    )
        decorate_next(tag("fuse_2"))
        for i in arange(1, n - 1):
            for j in arange(1, n - 1):
                for k in arange(1, n - 1):
                    A[i, j, k] = B[i, j, k] + b * (
                        B[i + 1, j, k]
                        - a * B[i, j, k]
                        + B[i - 1, j, k]
                        + B[i, j + 1, k]
                        - a * B[i, j, k]
                        + B[i, j - 1, k]
                        + B[i, j, k + 1]
                        - a * B[i, j, k]
                        + B[i, j, k - 1]
                    )
