from pydsl.frontend import compile
from pydsl.memref import MemRefFactory
from pydsl.type import F64
from pydsl.transform import int_attr, recursively
from pydsl.affine import affine_range as arange

MemRef40 = MemRefFactory((40,), F64)
MemRef40x40 = MemRefFactory((40, 40), F64)


@compile(dump_mlir=True)
def with_recursively_example(
    s: MemRef40, r: MemRef40, q: MemRef40, p: MemRef40, A: MemRef40x40
):
    for i in arange(40):
        for j in arange(40):
            with recursively(lambda x: int_attr(x, "set", 1)):
                s[j] = (s[j] + r[i]) * A[i, j]

            with recursively(lambda x: int_attr(x, "set", 2)):
                q[i] = (q[i] + p[i]) * A[i, j]
