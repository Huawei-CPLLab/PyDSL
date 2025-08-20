from typing import Any, TypeAlias, TypeVar

import numpy as np
from pydsl.frontend import compile
from pydsl.func import InlineFunction
from pydsl.macro import CallMacro, Compiled
from pydsl.memref import MemRef, MemRefFactory
from pydsl.protocols import SubtreeOut, ToMLIRBase
from pydsl.type import Bool, F32, Index, SInt16, Tuple, UInt16, UInt32, UInt64
from helper import compilation_failed_from, f32_isclose, run


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
    use the same MLIR signless integer type on the lower level.
    """
    with compilation_failed_from(TypeError):

        @compile(globals())
        def _(a: SInt16) -> UInt16:
            return a


def test_void_func_with_type_hint():
    @compile(globals())
    def _(_: Index) -> None:
        return


def test_void_func_without_type_hint():
    @compile(globals())
    def _(_: Index):
        return


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


def test_arg_cast():
    @compile()
    class Mod:
        def f(a: UInt32) -> UInt64:
            return a

        def g(a: UInt16) -> UInt64:
            return f(a)

    assert Mod.g(42) == 42


def test_ret_cast():
    @compile()
    class Mod:
        def f(a: UInt32) -> UInt32:
            return a

        def g(a: UInt32) -> UInt64:
            return f(a)

    assert Mod.g(42) == 42


def test_ret_tuple():
    @compile()
    class Mod:
        def f(a: UInt32) -> Tuple[UInt32, UInt32]:
            return a, a + 1

        def g(a: UInt32) -> UInt64:
            (b, c) = f(a)
            return c

    assert Mod.g(42) == 43


def test_multi_func_cast():
    @compile()
    class Mod:
        def my_add(a: UInt16, b: UInt16) -> UInt64:
            return a + b

        def triple_add(a: UInt16, b: UInt16, c: UInt16) -> UInt64:
            return my_add(a, b) + c

    assert Mod.triple_add(20000, 30000, 40000) == 20000 + 30000 + 40000


def test_recursion():
    @compile()
    def recursion(i: Index, m: MemRef[UInt16, 5]):
        if i < 5:
            m[i] = 1
            recursion(i + 1, m)

    m = np.zeros(5, dtype=np.uint16)
    recursion(0, m)

    assert (m == np.ones(5, dtype=np.uint16)).all()


def test_inline_func_basic():
    @InlineFunction.generate()
    def my_add(a, b: Any) -> Any:
        a = a + 1
        b = b - 1
        return a + b

    @compile()
    def f(a: SInt16, b: SInt16) -> Tuple[SInt16, SInt16, SInt16]:
        c = my_add(a, b)
        return a, b, c

    assert f(12, 34) == (12, 34, 12 + 34)


def test_inline_func_cast():
    @InlineFunction.generate()
    def my_add(a: UInt32, b: UInt16) -> UInt64:
        return a + b

    @compile()
    def f() -> UInt64:
        return my_add(60000, 9000) + (1 << 60)

    assert f() == 60000 + 9000 + (1 << 60)


def test_inline_func_call_macro():
    @InlineFunction.generate()
    def inline_add1(a, b) -> Any:
        return a + b

    @CallMacro.generate()
    def macro_add(visitor: ToMLIRBase, a: Compiled, b: Compiled) -> SubtreeOut:
        return inline_add1(visitor, a, b)

    @InlineFunction.generate()
    def inline_add2(a, b) -> Any:
        return macro_add(a, b)

    @compile()
    def f(x: SInt16, y: SInt16) -> SInt16:
        return macro_add(x, y)

    @compile()
    def g(a: UInt16, b: UInt16) -> UInt16:
        return inline_add2(a, b)

    assert f(12, -34) == 12 - 34
    assert g(123, 456) == 123 + 456


# This test can be modified in the future once we support multiple returns.
# For now, this test tries to make sure we detect multiple returns correctly
# and throw an error.
def test_inline_func_multiple_returns():
    # Bad
    @InlineFunction.generate()
    def inline_f1(a, b) -> Any:
        return UInt16(5)
        return a + b

    # Bad
    @InlineFunction.generate()
    def inline_f2(a, b) -> Any:
        if Bool(True):
            return a
        else:
            return b

    # Ok
    @InlineFunction.generate()
    def inline_f3(a, b) -> Any:
        if Bool(True):
            c = 0
        else:
            d = 1

        return a + b

    # Bad
    @InlineFunction.generate()
    def inline_f4(a, b):
        if Bool(True):
            return
        else:
            return

    with compilation_failed_from(SyntaxError):

        @compile()
        def f1(x: UInt16, y: UInt16) -> UInt16:
            return inline_f1(x, y)

    with compilation_failed_from(SyntaxError):

        @compile()
        def f2(x: UInt16, y: UInt16) -> UInt16:
            return inline_f2(x, y)

    @compile()
    def f3(x: UInt16, y: UInt16) -> UInt16:
        return inline_f3(x, y)

    assert f3(1, 2) == 1 + 2

    with compilation_failed_from(SyntaxError):

        @compile()
        def f4(x: UInt16, y: UInt16):
            inline_f4(x, y)


def test_inline_func_tuple_return():
    @InlineFunction.generate()
    def inline_add_mul(a, b) -> Any:
        return a + b, a * b

    @compile()
    def f(a: SInt16, b: SInt16) -> Tuple[SInt16, SInt16]:
        x, y = inline_add_mul(a, b)
        return x + 1, y - 1

    assert f(-10, 3) == (-10 + 3 + 1, -10 * 3 - 1)


def test_inline_func_kw_args():
    @InlineFunction.generate()
    def inline_f(a, b) -> Any:
        return a * 2 + b

    @compile()
    def f(x: UInt32, y: UInt32) -> UInt32:
        return inline_f(b=x, a=y)

    assert f(12, 34) == 34 * 2 + 12


def test_inline_func_bad_kw_arg():
    @InlineFunction.generate()
    def inline_f(a, b) -> Any:
        return a + b

    with compilation_failed_from(TypeError):

        @compile()
        def f(x: UInt32, y: UInt32) -> UInt32:
            return inline_f(x, a=y)


def test_inline_func_pos_as_kw_only():
    @InlineFunction.generate()
    def inline_f(a, *, b) -> Any:
        return a * 2 + b

    with compilation_failed_from(TypeError):

        @compile()
        def f(x: UInt32, y: UInt32) -> UInt32:
            return inline_f(x, y)


# In the future, we should make this NameError, but for now, this is correct
def test_inline_func_scope():
    @InlineFunction.generate()
    def inline_f(a, b) -> Any:
        d = a + b + c
        return d * 3

    @compile()
    def f(a: UInt32, b: UInt32, c: UInt32) -> UInt32:
        return inline_f(a, b)

    assert f(12, 34, 56) == (12 + 34 + 56) * 3


def test_inline_func_unbounded_local():
    # Yes, adding c = 1 is supposed to make this fail because Python is a
    # well-designed language
    @InlineFunction.generate()
    def inline_f(a, b) -> Any:
        d = a + b + c
        c = 1
        return d * 3

    with compilation_failed_from(UnboundLocalError):

        @compile()
        def f(a: UInt32, b: UInt32, c: UInt32) -> UInt32:
            return inline_f(a, b)


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
    run(test_arg_cast)
    run(test_ret_cast)
    run(test_ret_tuple)
    run(test_multi_func_cast)
    run(test_recursion)
    run(test_inline_func_basic)
    run(test_inline_func_cast)
    run(test_inline_func_call_macro)
    run(test_inline_func_multiple_returns)
    run(test_inline_func_tuple_return)
    run(test_inline_func_kw_args)
    run(test_inline_func_bad_kw_arg)
    run(test_inline_func_pos_as_kw_only)
    run(test_inline_func_scope)
    run(test_inline_func_unbounded_local)
