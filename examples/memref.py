import numpy as np
from pydsl.frontend import compile
from pydsl.memref import MemRefFactory
from pydsl.type import UInt64
from pydsl.scf import range

MemRef64 = MemRefFactory((1,), UInt64)


@compile(dump_mlir=True)
def memref_example(m: MemRef64) -> MemRef64:
    for i in range(3, 7, 2):
        m[0] = m[0] + i

    return m


n = np.asarray([0], dtype=np.uint64)

memref_example(n)
print(n)  # [8]
