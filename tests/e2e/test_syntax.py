from pydsl.frontend import compile
from pydsl.type import SInt16, UInt16
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


def test_assign_implicit_type():
    @compile(globals())
    def assign():
        a = 5

        # The add is just to make sure that `5` gets converted into a concrete
        # type. We don't care if the result of the addition is correct or not.
        UInt16(8) + a

    mlir = assign.emit_mlir()

    assert r"arith.constant 5 : i16" in mlir


if __name__ == "__main__":
    run(test_annassign)
    run(test_illegal_annassign)
    run(test_assign_implicit_type)
