import numpy as np
from pydsl.frontend import compile
from pydsl.memref import MemRef
from pydsl.type import SInt16, Tuple, UInt8, UInt16
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


if __name__ == "__main__":
    run(test_annassign)
    run(test_illegal_annassign)
    run(test_non_literal_annassign)
    run(test_assign_implicit_type)
    run(test_assign_tuple)
    run(test_chain_assign)
    run(test_chain_assign_mixed)
