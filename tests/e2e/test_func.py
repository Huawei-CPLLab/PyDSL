from typing import Any, TypeAlias, TypeVar

import numpy as np
from pydsl.frontend import compile
from pydsl.type import F32, Index, SInt16, Tuple, UInt16
from helper import compilation_failed_from, f32_isclose, run
from pydsl.memref import MemRef, MemRefFactory


def test_compile_identity():
    @compile(globals())
    def identity(a: Index) -> Index:
        return a

    for i in [0, 1, 5]:
        assert identity(i) == i


def test_compile_multiple():
    @compile(globals())
    def identity_1(a: Index) -> Index:
        return a

    @compile(globals())
    def identity_2(
        a: Index,
        b: Index,
    ) -> Index:
        return b

    assert identity_1(5) == 5
    assert identity_2(6, 7) == 7


def test_illegal_no_arg_type():
    with compilation_failed_from(SyntaxError):

        @compile(globals())
        def _(a) -> Index:
            return a


def test_illegal_type_hint_arg():
    with compilation_failed_from(TypeError):

        @compile(globals())
        def _(a: Any) -> Index:
            return a


def test_illegal_type_hint_ret():
    with compilation_failed_from(TypeError):

        @compile(globals())
        def _(a: Index) -> Any:
            return a


def test_illegal_void_on_single_return():
    with compilation_failed_from(TypeError):

        @compile(globals())
        def _(a: Index) -> Index:
            return


def test_illegal_Tuple_on_single_return():
    with compilation_failed_from(TypeError):

        @compile(globals())
        def _(a: Index) -> Index:
            return a, a


def test_illegal_single_on_Tuple_return():
    with compilation_failed_from(TypeError):

        @compile(globals())
        def _(a: Index) -> Tuple[Index, Index]:
            return a


def test_illegal_single_mismatch_return():
    with compilation_failed_from(TypeError):

        @compile(globals())
        def _(a: Index) -> Index:
            return F32(1.2)


def test_illegal_Tuple_mismatch_return():
    with compilation_failed_from(TypeError):

        @compile(globals())
        def _(a: Index) -> Tuple[Index, Index]:
            return a, F32(1.2)


def test_illegal_high_level_mismatch_return():
    """
    This test is particularly important as both signed and unsigned integers
    use the same MLIR signless integer type on the lower level
    """
    with compilation_failed_from(TypeError):

        @compile(globals())
        def _(a: SInt16) -> UInt16:
            return a


def test_void_func_with_type_hint():
    @compile(globals())
    def _(_: Index) -> None:
        return

    @compile(globals())
    def _(_: Index) -> None:
        pass


def test_void_func_without_type_hint():
    @compile(globals())
    def _(_: Index):
        return

    @compile(globals())
    def _(_: Index):
        pass


def test_illegal_composite_type_hint():
    T = TypeVar("T")
    BadType: TypeAlias = Any | T

    with compilation_failed_from(NameError):

        @compile(globals())
        def _(a: Index) -> BadType[Index]:
            return a


def test_Tuple_func():
    @compile(globals())
    def f(a: Index, b: Index) -> Tuple[Index, Index, Index]:
        return a, b, a

    assert all([
        f(a, b) == (a, b, a)
        for a, b in [
            (0, 0),
            (0, 1),
            (1, 0),
            (5, 3),
            (4, 7),
        ]
    ])


def test_single_Tuple_func():
    @compile(globals())
    def f(a: Index) -> Tuple[Index]:
        return (a,)

    assert all([f(a) == (a,) for a in [0, 1, 2]])


def test_name_shadowing():
    @compile(globals())
    def f(a: Index) -> Index:
        b: Index = 5
        b = a
        return b

    for i in [0, 1, 5]:
        assert f(i) == i


def test_illegal_nonexistent_name():
    with compilation_failed_from(NameError):

        @compile(globals())
        def _() -> Index:
            return nonexistent_name  # type: ignore


def test_type_hint_definition_swap():
    """
    This test checks if compile() always uses the latest namespace to identify
    the type hints of a function.

    The latest type hint should be used for:
    - the full MLIR compilation process
    - the frontend ctype conversion of input Python values
    """

    # can be anything that doesn't accept CorrectType
    WrongType = MemRefFactory((2,), Index)
    CorrectType: TypeAlias = Index

    # Start off with the wrong type.
    ChangingType: TypeAlias = WrongType

    def id(a: ChangingType) -> CorrectType:
        return a

    # Now we swap from a wrong type to a correct type to see whether
    # compile() follow suit.
    # If does, it should now compile correctly and accept 5.2 without error
    ChangingType = CorrectType
    try:
        id = compile()(id)
        assert id(5) == 5
    except TypeError as e:
        raise AssertionError(
            "compile() seems to have not used the correct type hint"
        ) from e


def test_return_empty_tuple():
    def cast() -> Tuple[()]:
        return ()

    emp = cast()
    assert emp == ()


def test_return_casting_tuple_1():
    def cast() -> Tuple[F32]:
        return 3.2

    extra = cast()
    assert f32_isclose(extra, 3.2)


def test_return_casting_tuple_2():
    def cast(a: UInt16) -> Tuple[F32, F32]:
        return a, 3.2

    for i in range(10):
        i_casted, extra = cast(i)
        assert f32_isclose(i_casted, i)
        assert f32_isclose(extra, 3.2)


def test_module_call_basic():
    @compile()
    class Mod:
        def f1() -> UInt16:
            return f2()

        def f2() -> UInt16:
            return 12

    assert Mod.f1() == 12


def test_body_func():
    @compile(body_func="my_body_func")
    class CallTest:
        def my_body_func():
            a = 12

        def f1() -> UInt16:
            return f2()

        def f2() -> UInt16:
            return a

    assert CallTest.f1() == 12


def test_recursion():
    @compile()
    def recursion(i: Index, m: MemRef[UInt16, 5]):
        if i < 5:
            m[i] = 1
            recursion(i + 1, m)

    m = np.zeros(5, dtype=np.uint16)
    recursion(0, m)

    assert (m == np.ones(5, dtype=np.uint16)).all()


if __name__ == "__main__":
    run(test_compile_identity)
    run(test_compile_multiple)
    run(test_illegal_no_arg_type)
    run(test_illegal_type_hint_arg)
    run(test_illegal_type_hint_ret)
    run(test_illegal_void_on_single_return)
    run(test_illegal_Tuple_on_single_return)
    run(test_illegal_single_on_Tuple_return)
    run(test_illegal_single_mismatch_return)
    run(test_illegal_Tuple_mismatch_return)
    run(test_illegal_high_level_mismatch_return)
    run(test_void_func_with_type_hint)
    run(test_void_func_without_type_hint)
    run(test_illegal_composite_type_hint)
    run(test_Tuple_func)
    run(test_single_Tuple_func)
    run(test_name_shadowing)
    run(test_illegal_nonexistent_name)
    run(test_type_hint_definition_swap)
    run(test_return_empty_tuple)
    run(test_return_casting_tuple_1)
    run(test_return_casting_tuple_2)
    run(test_module_call_basic)
    run(test_body_func)
    run(test_recursion)
