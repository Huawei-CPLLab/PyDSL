import numpy as np

from pydsl.frontend import compile
from pydsl.memref import MemRefFactory
from pydsl.type import Index, UInt32, Bool
from pydsl.scf import range as srange

from helper import run

MemRefSingle = MemRefFactory((1,), UInt32)


def test_range_basic():
    @compile(globals(), dump_mlir=True)
    def f(n: Index, m: MemRefSingle):
        c1: Index = 1
        c2: Index = 2
        for i in srange(c1, n, c2):
            m[0] = m[0] + i

    for n in range(10):
        m = np.asarray([0], dtype=np.uint32)
        f(n, m)
        assert m[0] == sum(range(1, n, 2))


def test_range_implicit_type():
    @compile(globals())
    def f(m: MemRefSingle):
        for i in srange(1, 10, 2):
            m[0] = m[0] + i

    m = np.asarray([0], dtype=np.uint32)
    f(m)
    assert m[0] == sum(range(1, 10, 2))


def test_if():
    @compile(globals())
    def f(m: MemRefSingle, b: Bool):
        if b:
            m[Index(0)] = UInt32(5)

    n = np.asarray([0], dtype=np.uint32)
    f(n, False)
    assert n[0] == 0

    f(n, True)
    assert n[0] == 5


def test_if_else():
    @compile(globals())
    def f(m: MemRefSingle, b: Bool):
        if b:
            m[Index(0)] = UInt32(5)
        else:
            m[Index(0)] = UInt32(10)

    n = np.asarray([0], dtype=np.uint32)
    f(n, False)
    assert n[0] == 10

    f(n, True)
    assert n[0] == 5


def test_const_if():
    @compile()
    def f(m: MemRefSingle):
        if True:
            m[0] = 123
        else:
            m[0] = 456

    n1 = np.asarray([0], dtype=np.uint32)
    f(n1)
    assert n1[0] == 123


if __name__ == "__main__":
    run(test_range_basic)
    run(test_range_implicit_type)
    run(test_if)
    run(test_if_else)
    run(test_const_if)
