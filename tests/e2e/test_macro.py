import ast

from pydsl.frontend import compile
from pydsl.macro import CallMacro, Compiled, Evaluated, MethodType, Uncompiled
from pydsl.protocols import ToMLIRBase
from pydsl.type import Int, Number, Sign, UInt32
from helper import compilation_failed_from, run


def test_Compiled_ArgCompiler():
    @CallMacro.generate()
    def bind(visitor: ToMLIRBase, a: Compiled) -> None:
        assert isinstance(a, Number), "a wasn't compiled"

    @compile()
    def _() -> None:
        bind(5)


def test_Evaluated_ArgCompiler():
    @CallMacro.generate()
    def bind(visitor: ToMLIRBase, a: Evaluated) -> None:
        assert a == 5, "a wasn't evaluated"

    @compile()
    def _() -> None:
        bind(5)


def test_Uncompiled_ArgCompiler():
    @CallMacro.generate()
    def bind(visitor: ToMLIRBase, a: Uncompiled) -> None:
        assert isinstance(a, ast.Constant), "a wasn't uncompiled"

    @compile()
    def _() -> None:
        bind(5)


def test_ArgCompiler_fmap():
    @CallMacro.generate()
    def bind(
        visitor: ToMLIRBase,
        i: Compiled,
        *li: list[Compiled],
        **di: dict[Compiled],
    ) -> None:
        assert i == li[0]
        assert tuple(li) == (
            di["a"],
            di["b"],
            di["c"],
            di["d"],
        )

    @compile()
    def _() -> None:
        a: UInt32 = 5
        b: UInt32 = 6
        c: UInt32 = 7
        d: UInt32 = 8
        bind(a, a, b, c, d, b=b, a=a, d=d, c=c)


def test_param_nested():
    @CallMacro.generate()
    def add(visitor: ToMLIRBase, a: Compiled, b: Compiled):
        return a.op_add(b)

    @CallMacro.generate()
    def mul(visitor: ToMLIRBase, a: Compiled, b: Compiled):
        return a.op_mul(b)

    @compile()
    def f(a: UInt32, b: UInt32, c: UInt32) -> UInt32:
        return add(mul(a, b), c)

    assert f(4, 5, 6) == 4 * 5 + 6


def test_body_nested():
    @CallMacro.generate()
    def add(visitor: ToMLIRBase, a: Compiled, b: Compiled):
        return a.op_add(b)

    @CallMacro.generate()
    def fma(visitor: ToMLIRBase, a: Compiled, b: Compiled, c: Compiled):
        return add(visitor, a.op_mul(b), c)

    @compile()
    def f(a: UInt32, b: UInt32, c: UInt32) -> UInt32:
        return fma(a, b, c)

    assert f(7, 8, 9) == 7 * 8 + 9


def test_instance_method():
    class MyInt(Int, width=32, sign=Sign.UNSIGNED):
        @CallMacro.generate(method_type=MethodType.INSTANCE)
        def my_add(visitor: ToMLIRBase, self, rhs: Compiled):
            return self.op_add(rhs)

    # Calling from instance
    @compile()
    def f(a: UInt32, b: UInt32) -> UInt32:
        a = MyInt(a)
        return a.my_add(b)

    assert f(4, 7) == 4 + 7

    # Calling from class
    with compilation_failed_from(TypeError):

        @compile()
        def g(a: UInt32, b: UInt32) -> UInt32:
            return MyInt.my_add(b)


def test_class_method():
    class MyInt(Int, width=21, sign=Sign.UNSIGNED):
        @CallMacro.generate(method_type=MethodType.CLASS)
        def add_width(visitor: ToMLIRBase, cls, rhs: Compiled):
            return UInt32(cls.width).op_add(rhs)

    # Calling from instance
    @compile()
    def f(x: UInt32) -> UInt32:
        a = MyInt(123)
        return a.add_width(x)

    assert f(17) == 17 + 21

    # Calling from class
    @compile()
    def g(x: UInt32) -> UInt32:
        return MyInt.add_width(x)

    assert g(123) == 123 + 21


def test_class_only_method():
    class MyInt(Int, width=57, sign=Sign.UNSIGNED):
        @CallMacro.generate(method_type=MethodType.CLASS_ONLY)
        def add_width(visitor: ToMLIRBase, cls, rhs: Compiled):
            return UInt32(cls.width).op_add(rhs)

    # Calling from instance
    with compilation_failed_from(TypeError):

        @compile()
        def f(x: UInt32) -> UInt32:
            a = MyInt(123)
            return a.add_width(x)

    # Calling from class
    @compile()
    def g(x: UInt32) -> UInt32:
        return MyInt.add_width(x)

    assert g(200) == 200 + 57


def test_static_method():
    class MyInt(Int, width=40, sign=Sign.UNSIGNED):
        @CallMacro.generate(method_type=MethodType.STATIC)
        def double(visitor: ToMLIRBase, rhs: Compiled):
            return rhs.op_add(rhs)

    # Calling from instance
    @compile()
    def f(x: UInt32) -> UInt32:
        a = MyInt(55)
        return a.double(x)

    assert f(987) == 987 + 987

    # Calling from class
    @compile()
    def g(x: UInt32) -> UInt32:
        return MyInt.double(x)

    assert g(0) == 0 + 0


def test_default_args():
    ast_123 = ast.parse("UInt32(123)", mode="eval")

    @CallMacro.generate()
    def add_macro(
        visitor: ToMLIRBase, x: Uncompiled = ast_123, y: Evaluated = 456
    ) -> UInt32:
        x = visitor.visit(x)
        return x.op_add(y)

    @compile()
    def f() -> UInt32:
        return add_macro()

    assert f() == 123 + 456


if __name__ == "__main__":
    run(test_Compiled_ArgCompiler)
    run(test_Evaluated_ArgCompiler)
    run(test_Uncompiled_ArgCompiler)
    run(test_ArgCompiler_fmap)
    run(test_param_nested)
    run(test_body_nested)
    run(test_instance_method)
    run(test_class_method)
    run(test_class_only_method)
    run(test_static_method)
    run(test_default_args)
