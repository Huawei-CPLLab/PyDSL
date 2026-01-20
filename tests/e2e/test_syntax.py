from typing import Annotated
import numpy as np
from pydsl.frontend import compile
from pydsl import llvm
from pydsl.memref import MemRef
from pydsl.type import Index, SInt16, Tuple, UInt8, UInt16
from helper import compilation_failed_from, run


def test_annassign():
    @compile(globals())
    def _():
        a: UInt16 = 2
        b: SInt16 = -2


def test_illegal_annassign():
    with compilation_failed_from(ValueError):

        @compile(globals())
        def _():
            a: UInt16 = -2


def test_non_literal_annassign():
    @compile()
    def f() -> UInt16:
        a: UInt8 = 50 * 2
        b: UInt16 = a + a
        b = b + 100
        return b

    assert f() == 300


def test_assign_implicit_type():
    @compile(globals())
    def assign():
        a = 5

        # The add is just to make sure that `5` gets converted into a concrete
        # type. We don't care if the result of the addition is correct or not.
        UInt16(8) + a

    mlir = assign.emit_mlir()

    assert r"arith.constant 5 : i16" in mlir


def test_assign_tuple():
    @compile()
    def f() -> Tuple[SInt16, SInt16]:
        a, b = 7, 3
        return a, b

    assert f() == (7, 3)


def test_chain_assign():
    @compile()
    def f() -> Tuple[SInt16, UInt16]:
        a = b = 8
        return a, b

    assert f() == (8, 8)


def test_chain_assign_mixed():
    @compile()
    def f(m1: MemRef[SInt16, 5]) -> Tuple[SInt16, SInt16]:
        a, b = (m1[2], m1[4]) = c = 2, 8
        d, e = c
        res1 = a + m1[2] + d
        res2 = b + m1[4] + e
        return res1, res2

    assert f(np.zeros((5,), dtype=np.int16)) == (2 + 2 + 2, 8 + 8 + 8)


def test_attribute():
    @compile()
    def f(m: Annotated[MemRef[UInt8], llvm.nonnull, llvm.align(8)]):
        pass

    mlir = f.emit_mlir()
    assert r"memref<i8> {llvm.align = 8 : i64, llvm.nonnull}" in mlir


def test_minus_eq():
    @compile()
    def f(a: SInt16) -> SInt16:
        a -= 2
        return a

    assert f(1) == 1 - 2


def test_plus_eq_memref():
    @compile()
    class Mod:
        def f(m: MemRef[UInt8, 2, 3]) -> None:
            m[1, 2] += 3

    m = np.zeros((2, 3), dtype=np.uint8)

    Mod.f(m)
    expected = np.zeros_like(m)
    expected[1, 2] = 3
    assert np.array_equal(m, expected)


def test_plus_eq_side_effect():
    @compile()
    class Mod:
        def f(m2: MemRef[Index, 1]) -> Index:
            m2[0] += 1
            return m2[0]

        def g(m1: MemRef[UInt8, 2, 3], m2: MemRef[Index, 1]) -> None:
            # f has to be evaluated only once
            m1[1, f(m2)] += 1

    m1 = np.zeros((2, 3), dtype=np.uint8)
    m2 = np.zeros((1,), dtype=np.uint64)

    Mod.g(m1, m2)
    expected = np.zeros_like(m1)
    expected[1, 1] = 1
    assert m2[0] == 1
    assert (m1 == expected).all()


if __name__ == "__main__":
    run(test_annassign)
    run(test_illegal_annassign)
    run(test_non_literal_annassign)
    run(test_assign_implicit_type)
    run(test_assign_tuple)
    run(test_chain_assign)
    run(test_chain_assign_mixed)
    run(test_attribute)
    run(test_minus_eq)
    run(test_plus_eq_memref)
    run(test_plus_eq_side_effect)
