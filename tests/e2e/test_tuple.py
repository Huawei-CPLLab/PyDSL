import numpy as np

from pydsl.frontend import compile
from pydsl.affine import affine_range as arange
from pydsl.memref import MemRef
from pydsl.type import Index, Tuple, UInt32


def test_return_empty_tuple():
    @compile()
    def f() -> Tuple[()]:
        return ()

    assert f() == (), f"expected (), got {f()}"


def test_return_tuple_with_memref():
    @compile()
    def f(
        ind: Index, m: MemRef[UInt32, 10]
    ) -> Tuple[Index, MemRef[UInt32, 10]]:
        for i in arange(ind, ind + 3):
            m[i] = 1

        return ind, m

    m1 = np.zeros([10], np.uint32)
    a1 = 3

    a2, m2 = f(a1, m1)

    assert a1 == a2
    assert (m1 == np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])).all()

    # in-place MemRef result should be the same as returned MemRef result
    assert (m2 == m1).all()
