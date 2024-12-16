import math

import numpy as np
import pytest

from pydsl.affine import affine_range as arange
from pydsl.frontend import compile
from pydsl.memref import DYNAMIC, Dynamic, MemRef, MemRefFactory, alloca, alloc

# from pydsl.memref import alloc
from pydsl.type import F64, Bool, Index, Tuple, UInt32


def test_load_implicit_index_uint32():
    @compile(globals())
    def f(m2: MemRef[UInt32, 2], m2x2: MemRef[UInt32, 2, 2]):
        m2[1] = UInt32(5)
        m2x2[1, 1] = UInt32(5)

    n2 = np.empty((2,), dtype=np.uint32)
    n2x2 = np.empty((2, 2), dtype=np.uint32)
    f(n2, n2x2)
    assert n2[1] == 5
    assert n2x2[1, 1] == 5


def test_load_implicit_index_f64():
    @compile(globals())
    def f(m2: MemRef[F64, 2], m2x2: MemRef[F64, 2, 2]) -> Tuple[Bool, Bool]:
        m2[1] = 5
        m2x2[1, 1] = 5.1
        return m2[0] == 3, m2x2[0, 0] == 3.1

    n2 = np.zeros((2,), dtype=np.float64)
    n2x2 = np.zeros((2, 2), dtype=np.float64)
    n2[0] = 3
    n2x2[0, 0] = 3.1

    print(f(n2, n2x2))

    assert math.isclose(n2[1], 5)
    assert math.isclose(n2x2[1, 1], 5.1)


def test_load_wrong_shape():
    @compile(globals())
    def f(m: MemRef[F64, 2, 2]):
        pass

    wrong_shape = np.zeros((2, 3), dtype=np.float64)

    with pytest.raises(TypeError):
        f(wrong_shape)


def test_return_memref():
    @compile(globals())
    def f(m: MemRef[F64, 2, 2]) -> MemRef[F64, 2, 2]:
        return m

    m = np.asarray([[1, 2], [3, 4]], dtype=np.float64)
    assert (f(m) == m).all()


def test_return_tuple_memref():
    @compile(globals())
    def f(m: MemRef[F64, 2, 2]) -> Tuple[MemRef[F64, 2, 2]]:
        return (m,)

    m = np.asarray([[1, 2], [3, 4]], dtype=np.float64)
    assert (f(m)[0] == m).all()


def test_alloca_scalar():
    @compile()
    def f() -> UInt32:
        m_scalar = alloca(MemRef[UInt32, 1])

        for i in arange(10):
            m_scalar[0] = i

        return m_scalar[0]

    assert f() == 9


def test_alloca_dynamic():
    @compile()
    def f(a: Index, b: Index) -> Tuple[UInt32, UInt32]:
        m = alloca(MemRef[UInt32, Dynamic, 2, Dynamic], (a, b))

        m[0, 0, 0] = 1
        m[a - 1, 1, b - 1] = 2

        return m[0, 0, 0], m[a - 1, 1, b - 1]

    for i in range(1, 3):
        assert f(i, i) == (1, 2)


def test_alloc_scalar():
    @compile()
    def f() -> MemRef[UInt32, 1]:
        m_scalar = alloc(MemRef[UInt32, 1])
        m_scalar[0] = 1
        return m_scalar

    assert (f() == np.asarray([1], dtype=np.uint32)).all()
