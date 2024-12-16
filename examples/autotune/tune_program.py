import numpy as np

from pydsl.type import Index, F32
from pydsl.memref import MemRef
from pydsl.affine import affine_range as arange, symbol as S

from pydsl.autotune import autotune, Var, TestingData

## Getting rid of some annoying log messages
import logging

logging.basicConfig(level=logging.ERROR)

N = 1000

A = np.ones((N, N), dtype=np.float32)

autotune_values = Var("NUM", [100, 200, 600]) * TestingData([[N, A]])


@autotune(autotune_values)
def heat_fuse_tile(n: Index, A: MemRef[F32, N, N]):
    v = Index(NUM)
    for _ in arange(S(v)):
        for i in arange(n):
            for j in arange(n):
                # -3/2 x^2 + 5/2 x + 1
                A[i, j] = (-1.5) * A[i, j] * A[i, j] + (2.5) * A[i, j] + 1
