import numpy as np

from pydsl.affine import (
    affine_range as arange,
    integer_set as iset,
    symbol as S,
    dimension as D,
)
from pydsl.frontend import compile
from pydsl.memref import MemRefFactory
from pydsl.type import (
    Index,
    UInt32,
)
from helper import compilation_failed_from, run

MemRefCompare1 = MemRefFactory((1,), UInt32)
MemRefCompare5 = MemRefFactory((5,), UInt32)

MemRef10UInt32 = MemRefFactory((10,), UInt32)


def test_explicit_affine_range():
    @compile()
    def f(ind: Index, m: MemRef10UInt32):
        for i in arange(S(ind), D(ind) + 3):
            m[i] = 1

    m1 = np.zeros([10], np.uint32)
    f(3, m1)
    assert (m1 == np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])).all()


def test_implicit_affine_range():
    @compile()
    def f(ind: Index, m: MemRef10UInt32):
        for i in arange(ind, ind + 3):
            m[i] = 1

    m1 = np.zeros([10], np.uint32)
    f(3, m1)
    assert (m1 == np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])).all()


def test_affine_range_max_min():
    @compile()
    def f(ind: Index, m: MemRef10UInt32):
        for i in arange(max(4, ind), min(7, ind + 3, 200)):
            m[i] = 5

    m1 = np.zeros([10], np.uint32)
    f(3, m1)
    assert (m1 == np.array([0, 0, 0, 0, 5, 5, 0, 0, 0, 0])).all()

    m2 = np.zeros([10], np.uint32)
    f(5, m2)
    assert (m2 == np.array([0, 0, 0, 0, 0, 5, 5, 0, 0, 0])).all()


def test_illegal_affine_range():
    with compilation_failed_from(SyntaxError):

        @compile()
        def f(ind: Index, m: MemRef10UInt32):
            for i in arange(max(4, ind), max(7, ind + 3, 200)):
                m[i] = 5.0

    with compilation_failed_from(SyntaxError):

        @compile()
        def f(ind: Index, m: MemRef10UInt32):
            for i in arange(min(4, ind), min(7, ind + 3, 200)):
                m[i] = 5.0


# Re-enable when we switch to LLVM 20.
# def test_affine_if_cmp():
#     @compile(globals())
#     def compare(m: MemRefCompare5, x: Index):
#         if iset(x < 8):
#             m[Index(0)] = UInt32(1)
#         else:
#             m[Index(0)] = UInt32(0)

#         if iset(x <= 8):
#             m[Index(1)] = UInt32(1)
#         else:
#             m[Index(1)] = UInt32(0)

#         if iset(x == 8):
#             m[Index(2)] = UInt32(1)
#         else:
#             m[Index(2)] = UInt32(0)

#         if iset(x > 8):
#             m[Index(3)] = UInt32(1)
#         else:
#             m[Index(3)] = UInt32(0)

#         if iset(x >= 8):
#             m[Index(4)] = UInt32(1)
#         else:
#             m[Index(4)] = UInt32(0)

#     def gold_compare(x):
#         return [x < 8, x <= 8, x == 8, x > 8, x >= 8]

#     n = np.full([5], 2, dtype=np.uint32)

#     for x in [6, 7, 8, 9, 10]:
#         compare(n, x)
#         assert all(n == gold_compare(x))


# def test_affine_if_conjunction():
#     @compile(globals())
#     def compare(m: MemRefCompare1, x: Index):
#         if iset(6 <= x < 8 and x > 6 and 9 >= x):
#             m[Index(0)] = UInt32(1)
#         else:
#             m[Index(0)] = UInt32(0)

#     n = np.full([1], 2, dtype=np.uint32)

#     for x in [5, 6, 7, 8, 9, 10]:
#         compare(n, x)
#         assert n[0] == (6 <= x < 8 and x > 6 and 9 >= x)


if __name__ == "__main__":
    run(test_explicit_affine_range)
    run(test_implicit_affine_range)
    run(test_affine_range_max_min)
    run(test_illegal_affine_range)
    # run(test_affine_if_cmp)
    # run(test_affine_if_conjunction)
