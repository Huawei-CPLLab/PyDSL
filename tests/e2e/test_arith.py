import itertools
import math

import pytest

from pydsl.frontend import compile
from pydsl.math import abs as p_abs
from pydsl.type import (
    F32,
    F64,
    Bool,
    Index,
    SInt8,
    SInt64,
    Tuple,
    UInt8,
    UInt64,
)
from tests.e2e.helper import compilation_failed_from, f32_edges, f32_isclose


def test_illegal_unfit_Int_input():
    with pytest.raises(TypeError):

        @compile(globals())
        def f(_: UInt8):
            pass

        f(1 << 8)


def test_illegal_Int_sign_input():
    with pytest.raises(ValueError):

        @compile(globals())
        def f(_: UInt8):
            pass

        f(-1)


def test_cast_UInt8_to_Floats():
    @compile(globals())
    def cast(a: UInt8) -> Tuple[F32, F64]:
        return F32(a), F64(a)

    for ui16 in [0, 1, 5, (1 << 8) - 1]:
        f32, f64 = cast(ui16)
        assert isinstance(f32, float)
        assert isinstance(f64, float)
        assert math.isclose(ui16, f32)
        assert math.isclose(ui16, f64)


def test_cast_UInt64_to_Floats():
    @compile(globals())
    def cast(a: UInt64) -> Tuple[F32, F64]:
        return F32(a), F64(a)

    for ui64 in [0, 1, 5, (1 << 64) - 1]:
        f32, f64 = cast(ui64)
        assert isinstance(f32, float)
        assert isinstance(f64, float)
        assert math.isclose(ui64, f32)
        assert math.isclose(ui64, f64)


def test_cast_SInt8_to_Floats():
    @compile(globals())
    def cast(a: SInt8) -> Tuple[F32, F64]:
        return F32(a), F64(a)

    for si8 in [0, 1, 5, (1 << 7) - 1, -1, -5, -(1 << 7)]:
        f32, f64 = cast(si8)
        assert isinstance(f32, float)
        assert isinstance(f64, float)
        assert math.isclose(si8, f32)
        assert math.isclose(si8, f64)


def test_cast_SInt64_to_Floats():
    @compile(globals())
    def cast(a: SInt64) -> Tuple[F32, F64]:
        return F32(a), F64(a)

    for si64 in [0, 1, 5, (1 << 63) - 1, -1, -5, -(1 << 63)]:
        f32, f64 = cast(si64)
        assert isinstance(f32, float)
        assert isinstance(f64, float)
        assert math.isclose(si64, f32)
        assert math.isclose(si64, f64)


def test_cast_F32_to_Floats():
    @compile(globals())
    def cast(a: F32) -> Tuple[F32, F64]:
        return F32(a), F64(a)

    # math.nan is not included because it's not close to itself
    for i in f32_edges:
        f32, f64 = cast(i)
        assert isinstance(f32, float)
        assert isinstance(f64, float)
        assert f32_isclose(i, f32)
        assert f32_isclose(i, f64)


def test_cast_F64_to_Floats():
    @compile(globals())
    def cast(a: F64) -> Tuple[F32, F64]:
        return F32(a), F64(a)

    # math.nan is not included because it's not close to itself
    for i in f32_edges:
        f32, f64 = cast(i)
        assert isinstance(f32, float)
        assert isinstance(f64, float)
        assert f32_isclose(i, f32)
        assert f32_isclose(i, f64)


def test_cast_Index_to_Floats():
    @compile(globals())
    def cast(a: Index) -> Tuple[F32, F64]:
        return F32(a), F64(a)

    for ind in [0, 1, 5, (1 << 63) - 1]:
        f32, f64 = cast(ind)
        assert isinstance(f32, float)
        assert isinstance(f64, float)
        assert f32_isclose(ind, f32)
        assert f32_isclose(ind, f64)


def test_UInt8_addition():
    @compile(globals())
    def add(a: UInt8, b: UInt8) -> UInt8:
        return a + b

    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1), (5, 3), ((1 << 8) - 2, 1)]:
        assert add(a, b) == a + b


def test_UInt64_addition():
    @compile(globals())
    def add(a: UInt64, b: UInt64) -> UInt64:
        return a + b

    for a, b in [(0, 0), (0, 1), (1, 0), (1, 1), (5, 3), ((1 << 64) - 2, 1)]:
        assert add(a, b) == a + b


def test_illegal_different_sign_addition():
    with compilation_failed_from(TypeError):

        @compile(globals())
        def _(a: UInt64, b: SInt64) -> UInt64:
            return a + b


def test_negf_F32():
    @compile(globals())
    def negf(a: F32) -> F32:
        return -a

    for f in f32_edges:
        assert f32_isclose(negf(f), -f)


def test_select():
    @compile(globals())
    def select(test: Bool, a: UInt64, b: UInt64) -> UInt64:
        return a if test else b

    for test in [True, False]:
        assert select(test, 4, 23) == (4 if test else 23)


def test_cmp_uint():
    @compile(globals())
    def cmpu(a: UInt64, b: UInt64) -> Tuple[Bool, Bool, Bool, Bool, Bool]:
        return (a < b, a <= b, a == b, a >= b, a > b)

    for a, b in itertools.product(range(2), range(2)):
        assert cmpu(a, b) == (a < b, a <= b, a == b, a >= b, a > b)


def test_cmp_sint():
    @compile(globals())
    def cmpu(a: SInt64, b: SInt64) -> Tuple[Bool, Bool, Bool, Bool, Bool]:
        return (a < b, a <= b, a == b, a >= b, a > b)

    for a, b in itertools.product(range(-1, 2), range(-1, 2)):
        assert cmpu(a, b) == (a < b, a <= b, a == b, a >= b, a > b)


def test_cmp_float():
    @compile(globals())
    def cmpu(a: F64, b: F64) -> Tuple[Bool, Bool, Bool, Bool, Bool]:
        return (a < b, a <= b, a == b, a >= b, a > b)

    for a, b in itertools.product(range(-1, 2), range(-1, 2)):
        a, b = a * 0.5, b * 0.5
        assert cmpu(a, b) == (a < b, a <= b, a == b, a >= b, a > b)


def test_cmp_chained():
    @compile(globals())
    def cmp_chain(a: SInt64, b: SInt64, c: SInt64) -> Bool:
        return a < b <= c

    for a, b, c in [(1, 2, 3), (1, 0, 3), (1, 2, 1), (0, 1, 1), (0, 0, 1)]:
        assert cmp_chain(a, b, c) == (a < b <= c)


def test_and():
    @compile(globals())
    def f_and(a: Bool, b: Bool, c: Bool) -> Bool:
        return a and b and c

    for a, b, c in itertools.product(*([[True, False]] * 3)):
        assert f_and(a, b, c) == (a and b and c)


def test_or():
    @compile(globals())
    def f_or(a: Bool, b: Bool, c: Bool) -> Bool:
        return a or b or c

    for a, b, c in itertools.product(*([[True, False]] * 3)):
        assert f_or(a, b, c) == (a or b or c)


def test_not():
    @compile(globals())
    def f_not(a: Bool) -> Bool:
        return not a

    for a in [True, False]:
        assert f_not(a) == (not a)


def test_implicit_arith_UInt64_lhs():
    @compile(globals())
    def impui_l(a: UInt64) -> Tuple[UInt64, UInt64, UInt64, UInt64]:
        return a + 5, a - 5, a * 5, a // 5

    for a in range(5, 8):
        # Note that if a < 5, then a - 5 will wrap around to the higest ints
        # for MLIR, but not for Python
        assert impui_l(a) == (a + 5, a - 5, a * 5, a // 5)


def test_implicit_arith_UInt64_rhs():
    @compile(globals())
    def impui_r(a: UInt64) -> Tuple[UInt64, UInt64, UInt64, UInt64]:
        return 5 + a, 1000 - a, 5 * a, 5 // a

    for a in range(5, 8):
        # Note that if a > 1000, then 1000 - a will wrap around to the higest
        # ints for MLIR, but not for Python
        assert impui_r(a) == (5 + a, 1000 - a, 5 * a, 5 // a)


def test_implicit_arith_Index_lhs():
    @compile(globals())
    def impui_l(a: Index) -> Tuple[Index, Index, Index, Index]:
        return a + 5, a - 5, a * 5, a // 5

    for a in range(5, 8):
        # Note that if a < 5, then a - 5 will wrap around to the higest ints
        # for MLIR, but not for Python
        assert impui_l(a) == (a + 5, a - 5, a * 5, a // 5)


def test_implicit_arith_Index_rhs():
    @compile(globals())
    def impui_r(a: Index) -> Tuple[Index, Index, Index, Index]:
        return 5 + a, 1000 - a, 5 * a, 5 // a

    for a in range(5, 8):
        # Note that if a > 1000, then 1000 - a will wrap around to the higest
        # ints for MLIR, but not for Python
        assert impui_r(a) == (5 + a, 1000 - a, 5 * a, 5 // a)


def test_implicit_compare_UInt64_rhs():
    @compile(globals())
    def impui_cmp(a: UInt64) -> Tuple[Bool, Bool, Bool, Bool, Bool]:
        return 5 < a, 5 <= a, 5 == a, 5 >= a, 5 > a

    for a in range(4, 6):
        assert impui_cmp(a) == (5 < a, 5 <= a, 5 == a, 5 >= a, 5 > a)


def test_implicit_compare_implicit():
    @compile(globals())
    def imp_cmp(a: UInt64) -> Tuple[Bool, Bool, Bool, Bool, Bool]:
        return 5 < 5, 5 <= 5, 5 == 5, 5 >= 5, 5 > 5

    for a in range(4, 6):
        assert imp_cmp(a) == (5 < 5, 5 <= 5, 5 == 5, 5 >= 5, 5 > 5)


def test_chained_implicit():
    @compile(globals())
    def chained_imp(a: UInt64) -> UInt64:
        return a + (6 * 2) // (12 - a)

    for a in range(5, 8):
        # a must be less than 12 for this to work
        assert chained_imp(a) == (a + (6 * 2) // (12 - a))


def test_illegal_implicit():
    with compilation_failed_from(TypeError):

        @compile(globals())
        def illegal_cast(a: UInt64) -> None:
            a + 5.2


# TODO: unary not is not implemented for now
def test_SInt_unary():
    @compile(globals())
    def SInt_un() -> Tuple[SInt64, SInt64, SInt64, SInt64]:
        a: SInt64 = 5
        return -a, +a, p_abs(a), ~a

    assert SInt_un() == (-5, +5, abs(5), ~5)


# TODO: unary not is not implemented for now
def test_Number_unary():
    @compile(globals())
    def imp_un() -> Tuple[SInt64, SInt64, SInt64, SInt64]:
        a = 5
        return -a, +a, p_abs(a), ~a

    assert imp_un() == (-5, +5, abs(5), ~5)
