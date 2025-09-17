import numpy as np

from pydsl.frontend import compile
from pydsl.memref import MemRefFactory
from pydsl.type import Index, F16, UInt16, UInt32, Bool, SInt32, Tuple, Any
from pydsl.scf import range as srange

from helper import compilation_failed_from, run

MemRefSingle = MemRefFactory((1,), UInt32)


def test_range_basic():
    @compile(globals())
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


def test_fold():
    @compile()
    def func1() -> Bool:
        a: Bool = False
        if True:
            a = True
        else:
            a = False
        return a

    assert "scf.if" not in func1.emit_mlir()
    assert func1()

    @compile()
    def func2() -> Bool:
        a: Bool = True
        if True:
            a = False
        else:
            a = 3
        return a

    assert "scf.if" not in func2.emit_mlir()
    assert not func2()

    @compile()
    def func3() -> Bool:
        return 3 if False else False

    assert "scf.if" not in func3.emit_mlir()
    assert not func3()

    @compile()
    def func4() -> Bool:
        return True if True else False

    assert "scf.if" not in func4.emit_mlir()
    assert func4()


def test_if_local_type_match():
    @compile()
    def f(b: Bool) -> UInt32:
        a: UInt32 = 2
        if b:
            a: UInt32 = 1

        return a

    assert f(True) == 1
    assert f(False) == 2


def test_if_local_type_mismatch_poison_not_read():
    @compile()
    def f(b: Bool):
        a: UInt32 = 2
        if b:
            a: SInt32 = 1

        # the type of `a` is unknown here

        # shadows previous `a`, so this can compile
        a: F16 = 3.14


def test_if_local_type_mismatch_poison_read():
    with compilation_failed_from(TypeError):

        @compile()
        def f(b: Bool):
            a: UInt32 = 2
            if b:
                a: SInt32 = 1

            # the type of `a` is unknown here so this cannot compile
            b = b + a


def test_if_else_local_type_mismatch_poison_read():
    with compilation_failed_from(TypeError):

        @compile()
        def f(b: Bool) -> UInt32:
            if b:
                a: UInt16 = 1
            else:
                a: UInt32 = 2

            return a


def test_if_local2():
    @compile()
    def f(b: Bool, a: UInt32) -> UInt32:
        if b:
            a: UInt32 = 1

        return a

    assert f(True, 2) == 1
    assert f(False, 2) == 2


def test_if_else_local():
    @compile()
    def f(b: Bool) -> Tuple[SInt32, SInt32]:
        if b:
            a: SInt32 = 1
            c: SInt32 = -1
        else:
            c: SInt32 = -2
            a: SInt32 = 2

        return a, c

    assert f(True) == (1, -1)
    assert f(False) == (2, -2)


def test_if_else_local_nested():
    @compile()
    def f(b1: Bool, b2: Bool) -> Tuple[SInt32, SInt32]:
        c: SInt32 = -4
        if b1:
            if b2:
                a: SInt32 = 1
                c: SInt32 = -1
            else:
                c: SInt32 = -3
                a: SInt32 = 3
        else:
            a: SInt32 = 4
            if b2:
                a: SInt32 = 2
                c: SInt32 = -2

        return a, c

    assert f(True, True) == (1, -1)
    assert f(True, False) == (3, -3)
    assert f(False, True) == (2, -2)
    assert f(False, False) == (4, -4)


def test_range_yield():
    @compile()
    def f() -> UInt32:
        a: UInt32 = 0
        for i in srange(5):
            a = a + 1
        return a

    assert f() == 5


def test_range_type_mismatch_poison_not_read():
    @compile()
    def f() -> UInt32:
        a: UInt32 = 0
        for i in srange(5):
            a: SInt32 = SInt32(i)

        # the type of `a` is unknown here

        # shadows previous `a`, so this can compile
        a: F16 = 3.14


def test_range_type_mismatch_poison_not_read():
    with compilation_failed_from(TypeError):

        @compile()
        def f() -> UInt32:
            a: UInt32 = 0
            for i in srange(5):
                a: UInt16 = UInt16(i)

            # the type of `a` is unknown here so this cannot compile
            return a


if __name__ == "__main__":
    # automatically find and run all test functions
    for name, fn in dict(globals()).items():
        if name.startswith("test_") and callable(fn):
            run(fn)
