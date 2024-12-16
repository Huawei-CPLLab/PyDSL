import numpy as np

from pydsl.type import Index, F32, F64
from pydsl.memref import MemRef
from pydsl.affine import affine_range as arange

from pydsl.autotune import autotune, Var, TestingData

## Getting rid of some annoying log messages
import logging

logging.basicConfig(level=logging.ERROR)

## Getting rid of some annoying log messages
import logging

logging.basicConfig(level=logging.ERROR)

N = 1000
Af32 = np.ones((N, N), dtype=np.float32)
Af64 = np.ones((N, N), dtype=np.float64)

autotune_values = Var("DATA_TYPE", [F32, F64]) * TestingData([[N, Af32]])


@autotune(autotune_values)
def heat_fuse_tile(n: Index, A: MemRef[F32, N, N]):
    for _ in arange(1000):
        for i in arange(n):
            for j in arange(n):
                t = DATA_TYPE(A[i, j])
                x = (-1.5) * t * t + (2.5) * t + 1
                A[i, j] = x
